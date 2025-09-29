from __future__ import annotations
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

SYSTEM_PROMPT = (
    "You are Dizzi — a friendly helper. Keep replies concise, use markdown, "
    "and acknowledge any files the user uploads. "
    "If asked something you don't know, say so honestly."
)

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

def bot_fn(
    message: gr.ChatMessage, history: List[gr.ChatMessage]
):
    """
    Gradio ChatInterface handler.
    - message: the latest user message (with optional files)
    - history: list of prior ChatMessage objects (role='user'|'assistant')
    """
    # print(type(message), message)
    user_text = (message or "").strip()

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

    # Load the Hugging Face model and tokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare the input for the model
    input_text = f"{SYSTEM_PROMPT}\n\nUser: {user_text}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate the response
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    # Decode the response
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(reply)

    # Extract the assistant's response (after "Assistant:")
    assistant_reply = reply.split("Assistant:")[-1].strip()

    # Stream the reply for a nice UX
    yield from stream_text(assistant_reply)

def on_clear():
    # Optional hook when the user clicks Clear — here we do nothing.
    return None

with gr.Blocks(title="DemoBot — Gradio Chatbot", theme=gr.themes.Soft()) as demo:
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

    chat = gr.ChatInterface(
        fn=bot_fn,
        type="messages",                 # use structured ChatMessage objects
        fill_height=True,
        cache_examples=False,
        submit_btn="Send",
        stop_btn="Stop",
        # retry_btn="Regenerate",
        # undo_btn="Delete last",
        # clear_btn="Clear",
        chatbot=gr.Chatbot(
            type="messages",
            show_copy_button=True,
            avatar_images=(None, None),   # set custom avatar image paths if you like
            height=500,
        ),
        # file_count="multiple",
        # upload=True,
        # concurrency_limit=16,
    )

    # Optional: respond to Clear button
    chat.clear()

if __name__ == "__main__":
    # share=True if you want a temporary public link
    demo.launch()
