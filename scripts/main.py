from __future__ import annotations
import os
import re
import time
import threading
from datetime import datetime
from typing import List, Literal
import queue
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from pydantic import BaseModel
import gradio as gr

import admin
from rag.mistral_model import get_mistral_model, generate_answer, validate_input, SAFE_RESPONSE
from ingestion.data_preprocessing import preprocess_data

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


def get_backend_status_dict() -> dict:
    with status_lock:
        return dict(backend_status)


admin.register_status_provider(get_backend_status_dict)


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
        admin.log_error("backend_loader", f"{type(e).__name__}: {e}")


_loader_thread: threading.Thread | None = None
_loader_lock = threading.Lock()


def start_backend_loader():
    """Start loading the data and model in a background thread (only once), so the
    server can already answer requests (with a "not ready" status) while it loads."""
    global _loader_thread
    with _loader_lock:
        if _loader_thread is None:
            _loader_thread = threading.Thread(target=load_backend, daemon=True)
            _loader_thread.start()


# ──────────────────────────────────────────────
#  FastAPI  –  REST endpoint
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_backend_loader()
    yield


app = FastAPI(title="Dizzy", lifespan=lifespan)
app.include_router(admin.router)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Query(BaseModel):
    question: str
    history: List[HistoryMessage] = []


@app.post("/chat")
def chat(query: Query, request: Request):
    ip = request.client.host if request.client else None
    user_id = ip or "unknown"

    if get_backend_status_dict()["state"] != "ready":
        return {"error": f"Backend not ready: {get_backend_status_str()}"}

    matched = admin.detect_malicious(query.question)
    input_is_safe, _ = validate_input(query.question)
    if not input_is_safe:
        matched.append("model-filter:input")
    user_chat_id = admin.log_chat(user_id=user_id, role="user", content=query.question, ip=ip, matched_keywords=matched)

    try:
        history = [m.model_dump() for m in query.history]
        answer = generate_answer(query.question, vector_store, mistral_model, history=history)
    except Exception as e:
        admin.log_error("api_chat", f"{type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"}

    if input_is_safe and answer == SAFE_RESPONSE:
        admin.append_flag(user_chat_id, ["model-filter:output"])

    admin.log_chat(user_id=user_id, role="assistant", content=answer, ip=ip)
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


TIMING_FOOTER = re.compile(r"\n\n_Generated in [\d.]+s_$")


def to_model_history(messages) -> list[dict]:
    """Convert Gradio chat messages into the {"role", "content"} dicts generate_answer expects,
    dropping the welcome message and the timing footer added to each answer."""
    history = []
    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg.get("role")
        content = msg.content if hasattr(msg, "content") else msg.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or content == WELCOME_MESSAGE:
            continue
        history.append({"role": role, "content": TIMING_FOOTER.sub("", content)})
    return history


def generate_response(message: str, user_id: str = "anonymous", ip: str | None = None, history: list[dict] | None = None):
    user_text = message.strip()

    matched = admin.detect_malicious(user_text)
    input_is_safe, _ = validate_input(user_text)
    if not input_is_safe:
        matched.append("model-filter:input")
    user_chat_id = admin.log_chat(user_id=user_id, role="user", content=user_text, ip=ip, matched_keywords=matched)

    if user_text.startswith("/help"):
        answer = HELP_TEXT
    elif user_text.startswith("/time"):
        answer = f"Server time: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
    elif user_text.startswith("/echo"):
        answer = user_text[len("/echo"):].strip() or "…(nothing to echo)"
    else:
        with status_lock:
            ready = backend_status["state"] == "ready"
        if not ready:
            answer = f"Backend not ready: {get_backend_status_str()}"
        else:
            try:
                answer = generate_answer(user_text, vector_store, mistral_model, history=history)
            except Exception as e:
                admin.log_error("gradio_chat", f"{type(e).__name__}: {e}")
                answer = f"Error: {type(e).__name__}: {e}"

    if input_is_safe and answer == SAFE_RESPONSE:
        admin.append_flag(user_chat_id, ["model-filter:output"])

    admin.log_chat(user_id=user_id, role="assistant", content=answer, ip=ip)
    return answer


def clear_and_lock_input(message):
    return gr.update(value="", interactive=False), message


def unlock_input():
    return gr.update(interactive=True, placeholder="Type a message…")


def chat_generation_loop(message: str, history: List[gr.ChatMessage], request: gr.Request):
    previous_turns = to_model_history(history)
    history.append(gr.ChatMessage(role="user", content=message))
    yield history

    user_id = request.session_hash if request else "anonymous"
    ip = request.client.host if request and request.client else None

    result_queue: queue.Queue = queue.Queue()
    gen_thread = threading.Thread(
        target=lambda: result_queue.put(generate_response(message, user_id, ip, previous_turns))
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


def create_app() -> FastAPI:
    """The FastAPI app with the Gradio chat UI mounted at /ui (call once)."""
    return gr.mount_gradio_app(app, demo, path="/ui")


# ──────────────────────────────────────────────
#  Run
# ──────────────────────────────────────────────
# SHARE=true swaps who owns the server: instead of mounting Gradio onto our
# FastAPI app and serving it with uvicorn, we let Gradio own the server via
# demo.launch(share=True) and mount our routes (admin panel, /chat) onto
# Gradio's app instead. That's required for Gradio's public tunnel to expose
# /admin and /chat alongside the chat UI on the same shared link. Leave
# SHARE unset for normal deployments.
SHARE = os.getenv("SHARE", "false").strip().lower() in ("1", "true", "yes")

if __name__ == "__main__":
    if SHARE:
        start_backend_loader()
        gradio_app, local_url, share_url = demo.launch(
            server_name=os.getenv("HOST", "0.0.0.0"),
            server_port=int(os.getenv("PORT", 8000)),
            share=True,
            prevent_thread_lock=True,
        )
        gradio_app.include_router(admin.router)
        gradio_app.post("/chat")(chat)

        print(f"Admin panel (local):  {local_url}admin")
        if share_url:
            print(f"Admin panel (shared): {share_url}/admin")

        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            demo.close()
    else:
        import uvicorn
        uvicorn.run(
            create_app(),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 8000)),
            proxy_headers=True,
            forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        )

