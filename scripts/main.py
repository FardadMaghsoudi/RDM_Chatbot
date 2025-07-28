import os
import pickle
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from vector_store import split_text, SimpleVectorStore
from pdf_utils import load_all_pdfs, download_pdfs_from_webpage, save_or_load_pdf_text, save_or_load_pdf_chunks
from web_utils import scrape_webpage, save_or_load_web_chunks
from mistral_model import load_mistral_model, generate_answer
import config
from data_preprocessing import preprocess_data

# Preprocess data and create vector store
combined_chunks, vector_store = preprocess_data()

# Load Mistral model
print("Loading Mistral model...")
mistral_pipe = load_mistral_model(config.MODEL_NAME, config.HF_TOKEN)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    answer = generate_answer(query.question, vector_store, mistral_pipe)
    return {"response": answer}

if __name__ == "__main__":
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() in {"exit", "quit"}:
            break
        answer = generate_answer(question, vector_store, mistral_pipe)
        print("\nAnswer:", answer)
        print("-" * 50) 