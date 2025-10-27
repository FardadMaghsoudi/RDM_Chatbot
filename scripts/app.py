from __future__ import annotations
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import gradio as gr
import config
from mistral_model import get_mistral_model, build_pipe, generate_answer
from data_preprocessing import preprocess_data

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

# --- Backend loading state (will be populated by background loader) ---
import threading

# shared state for loader and UI
backend_status = {"state": "starting", "message": "starting up"}
status_lock = threading.Lock()

# placeholders for resources filled by the loader thread
combined_chunks = None
vector_store = None
mistral_model = None
mistral_pipe = None

def _set_status(state: str, message: str = ""):
    with status_lock:
        backend_status["state"] = state
        backend_status["message"] = message

def get_backend_status_str() -> str:
    with status_lock:
        return f"{backend_status.get('state')} - {backend_status.get('message', '')}"

def load_backend():
    """Load preprocessing and model in background to avoid blocking the UI startup."""
    global combined_chunks, vector_store, mistral_model, mistral_pipe
    try:
        _set_status("loading_preprocess", "Preprocessing data and building vector store...")
        combined_chunks, vector_store = preprocess_data()

        _set_status("loading_model", "Loading Mistral model...")
        mistral_model = get_mistral_model(config.MODEL_NAME, config.QUANT_MODEL_NAME, config.HF_TOKEN)
        mistral_pipe = build_pipe(mistral_model)

        _set_status("ready", "Backend ready")
    except Exception as e:
        _set_status("error", f"{type(e).__name__}: {e}")

# start loader in background so Gradio UI can come up immediately
loader_thread = threading.Thread(target=load_backend, daemon=True)
loader_thread.start()

HELP_TEXT = """\
**Dizzi commands**
- `/help` – show this help
- `/time` – current server time
- `/echo some text` – echo back text
"""

def stream_text(text: str, chunk_size: int = 10):
    """Yield text in larger chunks to improve visibility during streaming."""
    for i in range(0, len(text), chunk_size):
        yield text[:i + chunk_size]
        time.sleep(0.1)  # slightly longer delay for better effect

def small_talk_response(message: str) -> Optional[str]:
    """Tiny rule-based responses just for demo polish."""
    lower = message.lower().strip()
    if any(g in lower for g in ("hello", "hi", "hey")):
        return "Hey! 👋 How can I help?"
    if "thank" in lower:
        return "You’re welcome! 😊"
    if "name" in lower and "your" in lower:
        return "I’m **DemoBot**. Nice to meet you!"
    return None

def reload_backend_trigger() -> str:
    """Trigger a reload of the backend in the background (if not already loading).
    Returns the new status string immediately."""
    with status_lock:
        state = backend_status.get("state")
        if state in ("loading_preprocess", "loading_model"):
            return get_backend_status_str()
        # mark as reloading and start a new loader thread
        backend_status["state"] = "reloading"
        backend_status["message"] = "User triggered reload"

    t = threading.Thread(target=load_backend, daemon=True)
    t.start()
    return get_backend_status_str()

def read_backend_status() -> str:
    return get_backend_status_str()

def bot_fn(
    message: gr.ChatMessage, history: List[gr.ChatMessage]
):
    """
    Gradio ChatInterface handler.
    - message: the latest user message (with optional files)
    - history: list of prior ChatMessage objects (role='user'|'assistant')
    """
    # Normalize the incoming message into a plain string (supports str, dict, ChatMessage-like)
    # print(type(message), message)
    if isinstance(message, str):
        user_text = message.strip()
    elif isinstance(message, dict):
        user_text = (message.get("content") or "").strip()
    elif hasattr(message, "content"):
        user_text = (getattr(message, "content") or "").strip()
    else:
        user_text = str(message or "").strip()

    # Commands
    if user_text.startswith("/help"):
        yield gr.ChatMessage(role="assistant", content=HELP_TEXT)
        return

    if user_text.startswith("/time"):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        yield gr.ChatMessage(role="assistant", content=f"Server time: **{now}**")
        return

    if user_text.startswith("/echo"):
        echoed = user_text[len("/echo"):].strip() or "…(nothing to echo)"
        yield gr.ChatMessage(role="assistant", content=echoed)
        return

    # Tiny rule-based small talk first
    canned = small_talk_response(user_text)
    if canned:
        yield gr.ChatMessage(role="assistant", content=canned)
        return

    # Ensure the backend is ready before attempting generation
    with status_lock:
        curr_state = backend_status.get("state")
    if curr_state != "ready":
        yield gr.ChatMessage(role="assistant", content=f"Backend not ready: {get_backend_status_str()}")
        return

    try:
        answer = generate_answer(user_text, vector_store, mistral_pipe)
    except Exception as e:
        # Surface an error message to the user rather than crashing the UI
        yield gr.ChatMessage(role="assistant", content=f"Error generating answer: {type(e).__name__}: {e}")
        return

    # Stream the reply for a nice UX
    yield from stream_text(answer)

def on_clear():
    # Optional hook when the user clicks Clear — here we do nothing.
    return None

with gr.Blocks(title="Dizzi — Gradio Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🤖 Dizzi
A minimal chatbot UI built with **Gradio**.

- Supports message history & markdown
- Streams responses
- Accepts file uploads
- Handy commands: `/help`, `/time`, `/echo`
        """
    )

    # Status area
    with gr.Row():
        status_text = gr.Textbox(value=read_backend_status(), label="Backend status", interactive=False)
        refresh_btn = gr.Button("Refresh status")
        reload_btn = gr.Button("Reload backend")

    # wire up status buttons
    refresh_btn.click(fn=read_backend_status, inputs=None, outputs=status_text)
    reload_btn.click(fn=reload_backend_trigger, inputs=None, outputs=status_text)

    demo.load(fn=read_backend_status, inputs=None, outputs=status_text, every=1)
    
    chat = gr.ChatInterface(
        fn=bot_fn,
        type="messages",                 # use structured ChatMessage objects
        cache_examples=False,
        save_history=True,
        chatbot=gr.Chatbot(
            type="messages",
            show_copy_button=True,
            avatar_images=(None, None),   # set custom avatar image paths if you like
            height=500,
        ),
    )

    # Optional: respond to Clear button
    chat.clear()

if __name__ == "__main__":
    # share=True if you want a temporary public link
    demo.launch(share=os.environ.get("GRADIO_SHARE", "false") in ("1", "true", "yes"))
