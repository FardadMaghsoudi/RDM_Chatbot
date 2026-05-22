"""
app.py — Dizzy: TU Delft RDM Chatbot
=====================================
Main entry point for the Gradio-based chat interface.

Responsibilities:
  - Loads the backend vector store (data index) and Mistral (AI) model in a background thread
  - Serves the Gradio UI with a live status indicator
  - Handles user inputs/messages, slash commands, and AI response streaming
"""
# -------------------------------------------------------------
# Dependecies and Imports 
# (load environment variables from .env file (API keys, secrets, etc.))
# -------------------------------------------------------------
from __future__ import annotations
import os
import time                                     #Measures generation duration and drives the "Thinking..." animation in the UI.
import threading                                #Runs backend loading and AI generation without blocking the UI thread, allowing for a responsive experience.
from datetime import datetime                   #Used by the /time slash command to show current server time in a human-readable format.
from typing import Any, List
import queue                                    #Safely passes the AI result from the generation thread back to the main thread for UI updates.
import gradio as gr                             #Builds the chat UI and handles browser communication. 
import config
from mistral_model import get_mistral_model, generate_answer
from data_preprocessing import preprocess_data
from dotenv import load_dotenv                  #loads secrets from .env file into the environment.                 
load_dotenv()
# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
WELCOME_MESSAGE = """\
**Hello! I am Dizzy.** 🤖

I am your TU Delft RDM assistant. I can help you with:
* Data Management Plans (DMPs)
* Storage & Security policies
* Archiving & Publishing data

*How can I assist you today?*
"""
HELP_TEXT = """\
**Dizzy Commands**
- `/help` — show this help message
- `/time` — display the current server time
- `/echo <text>` — repeat back whatever you type after `/echo`
"""

THINKING_INTERVAL = 0.2  #seconds between "Thinking..." timer updates (default: 0.2s)
# ---------------------------------------------------------------------------
# Global backend state (Placeholders for the loaded model and vector store, and the backend status)
# ---------------------------------------------------------------------------
# These are intentionally module-level so the loader thread can populate 
# them and the request handler can read them without passing objects around.

combined_chunks: Any = None                     # Document chunks (reserved for future use)
vector_store: Any = None                        # Searchable index of document embeddings
mistral_model: Any = None                       # Loaded Mistral LLM instance
backend_status: dict[str, str] = {"state": "starting", "message": "Starting up…"}
status_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Backend status helper functions
# ---------------------------------------------------------------------------

def _set_status(state: str, message: str = "") -> None: # Thread-safely update the backend loading status.
    with status_lock:
        backend_status["state"] = state
        backend_status["message"] = message

def get_backend_status_str() -> str:            # Return a human-readable status string (thread-safe).
        return f"{backend_status.get('state')} - {backend_status.get('message', '')}"

def _is_ready() -> bool:                        # Return true only when the backend has fully loaded.
    with _status_lock:
        return _backend_status["state"] == "ready"
    
# ---------------------------------------------------------------------------
# Background loader
# Loads the vector store and Mistral model without blocking the UI, and updates the status for the user.
# ---------------------------------------------------------------------------
def load_backend():
    global combined_chunks, vector_store, mistral_model   #Runs in a daemon thread at startup. Loads the document index and the Mistral model sequentially, updating the shared status at each stage.
    try:
        _set_status("loading_preprocess", "Preprocessing data...")
        combined_chunks, vector_store = preprocess_data() #Time.sleep(2) # Fake delay for demonstration

        _set_status("loading_model", "Loading Mistral model...")
        mistral_model = get_mistral_model()               #Time.sleep(2) # Fake delay for demonstration

        _set_status("ready", "Backend ready")

    except Exception as e:                                #Surface the error type and message in the UI status bar
        _set_status("error", f"{type(e).__name__}: {e}")

# Start loader immediately so the model is ready as early as possible
loader_thread = threading.Thread(target=load_backend, daemon=True) 
loader_thread.start()

# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------
def generate_response(message: str) -> str | None:
    user_text = message.strip()

    # Commands
    if user_text.startswith("/help"):
        return HELP_TEXT
    if user_text.startswith("/time"):
        return f"Server time: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
    if user_text.startswith("/echo"):
        payload = user_text[len("/echo"):].strip()
        return user_text[len("/echo"):].strip() or "…(nothing to echo)"

# Backend check
    with status_lock:
        curr_state = backend_status.get("state")
    if curr_state != "ready":
        return f"Backend not ready: {get_backend_status_str()}"

    # Real Generation
    try:
        answer = generate_answer(user_text, vector_store, mistral_model)
        #answer = f"Simulated answer to: {user_text}" # Placeholder
        return answer
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def clear_and_lock_input(message):
    """Immediately clears the input and stops user from typing during generation."""
    return gr.update(value="", interactive=False), message

def unlock_input():
    """Re-enables the input box once the answer is ready."""
    return gr.update(interactive=True, placeholder="Type a message...")

def chat_generation_loop(message: str, history: List[gr.ChatMessage]):
    """Handles the thinking loop and chatbot updates ONLY."""
    # 1. Append User Message
    history.append(gr.ChatMessage(role="user", content=message))
    yield history 

    result_queue = queue.Queue()
    
    # Run heavy generation in thread
    gen_thread = threading.Thread(
        target=lambda: result_queue.put(generate_response(message))
    )
    gen_thread.start()

    start_time = time.time()
    
    # 2. Live Timer Loop (Now only targeting the chatbot)
    while gen_thread.is_alive():
        elapsed = time.time() - start_time
        loading_msg = gr.ChatMessage(
            role="assistant", 
            content=f"🧠 *Thinking...* ({elapsed:.1f}s)"
        )
        
        history.append(loading_msg)
        yield history # No input box yielded here = no flicker
        history.pop() 
        time.sleep(0.2)

    # 3. Final Processing
    gen_thread.join()
    raw_response = result_queue.get()
    total_time = time.time() - start_time
    
    final_content = f"{raw_response}\n\n_Generated in {total_time:.2f}s_"
    history.append(gr.ChatMessage(role="assistant", content=final_content))
    yield history

def check_status_and_update_ui():
    """
    Called by Timer.
    Updates the Status Textbox AND enables/disables the Chat Input.
    """
    current_status = get_backend_status_str()
    is_ready = backend_status.get("state") == "ready"
    
    # Configure the input box based on status
    if is_ready:
        # We assume status text is hidden or minimal once ready, 
        # but here we keep updating it just so you know it's working.
        input_update = gr.Textbox(interactive=True, placeholder="Type a message...")
    else:
        input_update = gr.Textbox(interactive=False, placeholder=f"System loading... ({current_status})")
        
    return current_status, input_update

def reset_chat():
    """Returns the chat history reset to just the welcome message."""
    return [gr.ChatMessage(role="assistant", content=WELCOME_MESSAGE)]

# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Dizzi", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Dizzy")

    # Clean UI: Just the chat and a status indicator
    with gr.Row():
        # Interactive=False means user can't type in it, it's just for display
        status_display = gr.Textbox(label="System Status", interactive=False)

    chatbot = gr.Chatbot(
        value=reset_chat(),
        type="messages", 
        height=500, 
        show_copy_button=True
    )
    
    # Input Area (Started as Disabled)
    chat_input = gr.Textbox(
        interactive=False, 
        placeholder="Initializing system...", 
        show_label=False,
        autofocus=True,
    )
    
    clear_btn = gr.Button("Clear Chat", variant="secondary")
# Hidden state: holds the submitted message between the two submit steps
    saved_msg = gr.State()

# -----------------------------------------------------------------------
#  # Event wiring
# -----------------------------------------------------------------------

    # 1. Chat Submission (lock → generate → unlock )
    msg_event = chat_input.submit(
        fn=clear_and_lock_input,
        inputs=[chat_input],
        outputs=[chat_input, saved_msg],
        queue=False 
    ).then(
        fn=chat_generation_loop,
        inputs=[saved_msg, chatbot],
        outputs=chatbot
    ).then(
        fn=unlock_input,
        inputs=None,
        outputs=chat_input
    )

    # 2. Clear History
    clear_btn.click(fn=reset_chat, inputs=None, outputs=chatbot, queue=False)

    # 3. Timer: Updates Status text AND enables Chat Input when ready
    # This replaces the need for a manual "Refresh" button
    timer = gr.Timer(1.0)
    timer.tick(
        fn=check_status_and_update_ui, 
        inputs=[], 
        outputs=[status_display, chat_input]
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=True)
