from __future__ import annotations
import os
import time
import threading
from datetime import datetime
from typing import List
import queue

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

import config
from mistral_model import get_mistral_model, generate_answer
from data_preprocessing import preprocess_data

# ──────────────────────────────────────────────
#  Shared state
# ──────────────────────────────────────────────
combined_chunks = None
vector_store = None
mistral_model = None

backend_status = {"state": "starting", "message": "Starting up…"}
status_lock = threading.Lock()


def _set_status(state: str, message: str = ""):
    with status_lock:
        backend_status["state"] = state
        backend_status["message"] = message


def get_backend_status_str() -> str:
    with status_lock:
        return f"{backend_status['state']} – {backend_status.get('message', '')}"


# ──────────────────────────────────────────────
#  Background loader (shared by API + Gradio)
# ──────────────────────────────────────────────
def load_backend():
    global combined_chunks, vector_store, mistral_model
    try:
        _set_status("loading_preprocess", "Preprocessing data…")
        combined_chunks, vector_store = preprocess_data()

        _set_status("loading_model", "Loading Mistral model…")
        mistral_model = get_mistral_model()

        _set_status("ready", "Backend ready")
    except Exception as e:
        _set_status("error", f"{type(e).__name__}: {e}")


loader_thread = threading.Thread(target=load_backend, daemon=True)
loader_thread.start()


# ──────────────────────────────────────────────
#  FastAPI  –  REST endpoint
# ──────────────────────────────────────────────
app = FastAPI(title="Dizzy")


class Query(BaseModel):
    question: str


@app.post("/chat")
def chat(query: Query):
    with status_lock:
        if backend_status["state"] != "ready":
            return {"error": f"Backend not ready: {get_backend_status_str()}"}
    answer = generate_answer(query.question, vector_store, mistral_model)
    return {"response": answer}


# ──────────────────────────────────────────────
#  Gradio  –  Chat UI
# ──────────────────────────────────────────────
WELCOME_MESSAGE = """\
**Hello! I am Dizzy.** 🤖

I am your TU Delft RDM assistant. I can help you with:
* Data Management Plans (DMPs)
* Storage & Security policies
* Archiving & Publishing data

*How can I assist you today?*
"""

HELP_TEXT = """\
**Dizzy commands**
- `/help` – show this help
- `/time` – current server time
- `/echo` – echo back text
"""


def generate_response(message: str):
    user_text = message.strip()

    if user_text.startswith("/help"):
        return HELP_TEXT
    if user_text.startswith("/time"):
        return f"Server time: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
    if user_text.startswith("/echo"):
        return user_text[len("/echo"):].strip() or "…(nothing to echo)"

    with status_lock:
        if backend_status["state"] != "ready":
            return f"Backend not ready: {get_backend_status_str()}"

    try:
        return generate_answer(user_text, vector_store, mistral_model)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def clear_and_lock_input(message):
    return gr.update(value="", interactive=False), message


def unlock_input():
    return gr.update(interactive=True, placeholder="Type a message…")


def chat_generation_loop(message: str, history: List[gr.ChatMessage]):
    history.append(gr.ChatMessage(role="user", content=message))
    yield history

    result_queue: queue.Queue = queue.Queue()
    gen_thread = threading.Thread(
        target=lambda: result_queue.put(generate_response(message))
    )
    gen_thread.start()

    start_time = time.time()
    while gen_thread.is_alive():
        elapsed = time.time() - start_time
        loading_msg = gr.ChatMessage(
            role="assistant",
            content=f"🧠 *Thinking…* ({elapsed:.1f}s)",
        )
        history.append(loading_msg)
        yield history
        history.pop()
        time.sleep(0.2)

    gen_thread.join()
    raw_response = result_queue.get()
    total_time = time.time() - start_time

    final_content = f"{raw_response}\n\n_Generated in {total_time:.2f}s_"
    history.append(gr.ChatMessage(role="assistant", content=final_content))
    yield history


def check_status_and_update_ui():
    current_status = get_backend_status_str()
    is_ready = backend_status.get("state") == "ready"
    if is_ready:
        input_update = gr.Textbox(interactive=True, placeholder="Type a message…")
    else:
        input_update = gr.Textbox(
            interactive=False,
            placeholder=f"System loading… ({current_status})",
        )
    return current_status, input_update


def reset_chat():
    return [gr.ChatMessage(role="assistant", content=WELCOME_MESSAGE)]


# --- Gradio Blocks ---
with gr.Blocks(title="Dizzy", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Dizzy")

    with gr.Row():
        status_display = gr.Textbox(label="System Status", interactive=False)

    chatbot = gr.Chatbot(
        value=reset_chat(),
        type="messages",
        height=500,
        show_copy_button=True,
    )

    chat_input = gr.Textbox(
        interactive=False,
        placeholder="Initializing system…",
        show_label=False,
    )

    clear_btn = gr.Button("Clear Chat")
    saved_msg = gr.State()

    chat_input.submit(
        fn=clear_and_lock_input,
        inputs=[chat_input],
        outputs=[chat_input, saved_msg],
        queue=False,
    ).then(
        fn=chat_generation_loop,
        inputs=[saved_msg, chatbot],
        outputs=chatbot,
    ).then(
        fn=unlock_input,
        inputs=None,
        outputs=chat_input,
    )

    clear_btn.click(fn=reset_chat, inputs=None, outputs=chatbot, queue=False)

    timer = gr.Timer(1.0)
    timer.tick(
        fn=check_status_and_update_ui,
        inputs=[],
        outputs=[status_display, chat_input],
    )


# ──────────────────────────────────────────────
#  Mount Gradio onto FastAPI
# ──────────────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/ui")


# ──────────────────────────────────────────────
#  Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True, forwarded_allow_ips="127.0.0.1")
