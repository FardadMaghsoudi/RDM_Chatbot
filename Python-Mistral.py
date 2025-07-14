# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:49:23 2025

@author: fmaghsoudimoud
"""

#####################     Libraries      #################### 
from fastapi import FastAPI
from pydantic import BaseModel
import PyPDF2
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
import torch

#############     Text Splitting & Vector Store      ###############
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

class SimpleVectorStore:
    def __init__(self, texts):
        self.texts = texts
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(texts)

    def similarity_search(self, query, k=4):
        query_vec = self.model.encode([query])
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_k_idx = np.argsort(sims)[::-1][:k]
        return [self.texts[i] for i in top_k_idx]

#####################     Load PDFs      #################### 
def load_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def load_all_pdfs(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            all_text += load_pdf_text(full_path) + "\n"
    return all_text

#####################     Scrape Webpages      #################### 
def scrape_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

#####################     Combine PDFs + Web     #################### 
pdf_folder = r"C:\Users\fmaghsoudimoud\OneDrive - Delft University of Technology\Desktop\GenAI\Training Documents\TU"
pdf_text = load_all_pdfs(pdf_folder)
pdf_chunks = split_text(pdf_text)

web_urls = [
    "https://www.tudelft.nl/library",
    "https://www.tudelft.nl/en/about-tu-delft"
]

web_chunks = []
for url in web_urls:
    web_text = scrape_webpage(url)
    web_chunks.extend(split_text(web_text))

combined_chunks = pdf_chunks + web_chunks

vector_store = SimpleVectorStore(combined_chunks)

query = "When was TU Delft founded?"
results = vector_store.similarity_search(query)
for i, r in enumerate(results, 1):
    print(f"Result {i}:\n{r}\n{'-'*40}")
    
#####################     Load Mistral & FastAPI     #################### 
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
hf_token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16, device_map="auto")
mistral_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)


#####################     Answer Generator     #################### 
def generate_answer(query, vector_store):
    docs = vector_store.similarity_search(query, k=4)
    context = "\n\n".join(docs)
    prompt = f"""Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    result = mistral_pipe(prompt)[0]["generated_text"]
    return result.split("Answer:")[-1].strip()


#####################     FastAPI App     #################### 
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    answer = generate_answer(query.question, vector_store)
    return {"response": answer}


url = "http://localhost:8000/chat"
question_payload = {
    "question": "When was TU Delft founded?"
}

response = requests.post(url, json=question_payload)
print("Response:", response.json())


response = requests.post("http://localhost:8000/chat", json={"question": "When was TU Delft founded?"})
print(response.json())


# Replace with the actual URL where your FastAPI app is running
url = "http://localhost:8000/chat"

# Your question
question_payload = {
    "question": "When was TU Delft founded?"
}

# Send POST request
response = requests.post(url, json=question_payload)

# Print the chatbot's response
print("Response:", response.json())


#####################     Local Run     #################### 

if __name__ == "__main__":
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() in {"exit", "quit"}:
            break
        answer = generate_answer(question, vector_store)
        print("\nAnswer:", answer)
        print("-" * 50)


question = "Who are the data stewards?"
answer = generate_answer(question, vector_store)
print("Answer:", answer)












