import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from .vector_store import split_text, SimpleVectorStore
from .pdf_utils import load_all_pdfs, download_pdfs_from_webpage
from .web_utils import scrape_webpage
from .mistral_model import load_mistral_model, generate_answer
import config

# NOTE: Uncomment the next line if you want to download PDFs again
# pdf_folder = download_pdfs_from_webpage(config.POLICIES_URL, config.PDF_FOLDER)

# Load PDFs and split text
pdf_text = load_all_pdfs(config.PDF_FOLDER)
pdf_chunks = split_text(pdf_text)

# Scrape web pages and split text
web_chunks = []
for url in config.WEB_URLS:
    web_text = scrape_webpage(url)
    web_chunks.extend(split_text(web_text))

combined_chunks = pdf_chunks + web_chunks
vector_store = SimpleVectorStore(combined_chunks)

# Load Mistral model
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