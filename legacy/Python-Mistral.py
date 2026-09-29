# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:49:23 2025

@author: fmaghsoudimoud
"""

#NOTE:
# This file has been split into multiple modules for better organization.
# Please use main.py as the entry point for the application.


# #####################     Libraries      #################### 
# from fastapi import FastAPI
# from pydantic import BaseModel
# import PyPDF2
# import requests
# from bs4 import BeautifulSoup
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import os
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
# import torch
# import re
# import time

# #############     Text Splitting & Vector Store      ###############
# def split_text(text, chunk_size=500, overlap=50):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - overlap
#     return chunks

# class SimpleVectorStore:
#     def __init__(self, texts):
#         self.texts = texts
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.embeddings = self.model.encode(texts)

#     def similarity_search(self, query, k=4):
#         query_vec = self.model.encode([query])
#         sims = cosine_similarity(query_vec, self.embeddings)[0]
#         top_k_idx = np.argsort(sims)[::-1][:k]
#         return [self.texts[i] for i in top_k_idx]

# #####################     Load PDFs      #################### 
# def load_pdf_text(pdf_path):
#     text = ""
#     with open(pdf_path, "rb") as f:
#         reader = PyPDF2.PdfReader(f)
#         for page in reader.pages:
#             text += page.extract_text() + "\n"
#     return text

# def load_all_pdfs(folder_path):
#     all_text = ""
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pdf"):
#             full_path = os.path.join(folder_path, filename)
#             all_text += load_pdf_text(full_path) + "\n"
#     return all_text

# #####################     Scrape Webpages      #################### 
# def scrape_webpage(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, "html.parser")
#     return soup.get_text()

# #####################     Download PDFs from TU Delft      #################### 

# def download_pdfs_from_webpage(url, download_folder="tudelft_policies"): 
#     os.makedirs(download_folder, exist_ok=True)
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, "html.parser")
#     pdf_links = set()
#     zenodo_links = set()
#     # Direct Zenodo PDF pattern: /files/*.pdf?download=1
#     for a_tag in soup.find_all("a", href=True):
#         href = a_tag["href"]
#         # Direct Zenodo PDF download
#         if ("/files/" in href and href.lower().endswith(".pdf?download=1")):
#             if not href.startswith("http") and not href.startswith("https"):
#                 href = requests.compat.urljoin(url, href)
#             # Remove '?download=1' to get the direct PDF URL
#             href = href.rsplit("?download=1", 1)[0]
#             pdf_links.add(href)
#         elif href.lower().endswith(".pdf"):
#             if not href.startswith("http") and not href.startswith("https"):
#                 href = requests.compat.urljoin(url, href)
#             pdf_links.add(href)
#         elif "zenodo" in href:
#             if not href.startswith("http") and not href.startswith("https"):
#                 href = requests.compat.urljoin(url, href)
#             zenodo_links.add(href)
#     local_paths = []
#     for link in pdf_links:
#         filename = os.path.join(download_folder, os.path.basename(link.split("?")[0]))
#         if not os.path.exists(filename):
#             print(f"Downloading {link} ...")
#             try:
#                 r = requests.get(link)
#                 with open(filename, "wb") as f:
#                     f.write(r.content)
#             except Exception as e:
#                 print(f"Failed to download {link}: {e}")
#         local_paths.append(filename)
#     # For each Zenodo link, fetch the page and download the main PDF
#     for zenodo_url in zenodo_links:
#         print(f"Processing Zenodo page: {zenodo_url}")
#         try:
#             zenodo_resp = requests.get(zenodo_url)
#             zenodo_soup = BeautifulSoup(zenodo_resp.content, "html.parser")
#             # Find any <a> with href ending in .pdf or .pdf?download=1
#             pdf_a = zenodo_soup.find("a", href=re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE))
#             if pdf_a:
#                 pdf_href = pdf_a["href"]
#                 if not pdf_href.startswith("http") and not pdf_href.startswith("https"):
#                     pdf_href = requests.compat.urljoin(zenodo_url, pdf_href)
#                 filename = os.path.join(download_folder, os.path.basename(pdf_href.split("?")[0]))
#                 if not os.path.exists(filename):
#                     print(f"Downloading Zenodo PDF: {pdf_href}")
#                     r = requests.get(pdf_href)
#                     with open(filename, "wb") as f:
#                         f.write(r.content)
#                 local_paths.append(filename)
#             else:
#                 print(f"No PDF found on Zenodo page: {zenodo_url}")
#         except Exception as e:
#             print(f"Failed to process Zenodo page {zenodo_url}: {e}")
#     return download_folder

# # Download PDFs from the TU Delft faculty policies page
# policies_url = "https://www.tudelft.nl/en/library/data-management/research-data-management/tu-delft-faculty-policies-for-research-data"
# pdf_folder = download_pdfs_from_webpage(policies_url)
# print("PDFs downloaded")

# pdf_text = load_all_pdfs(pdf_folder)
# pdf_chunks = split_text(pdf_text)

# web_urls = [
#     "https://www.tudelft.nl/library",
#     "https://www.tudelft.nl/en/about-tu-delft"
# ]

# web_chunks = []
# for url in web_urls:
#     web_text = scrape_webpage(url)
#     web_chunks.extend(split_text(web_text))

# combined_chunks = pdf_chunks + web_chunks

# vector_store = SimpleVectorStore(combined_chunks)

# query = "When was TU Delft founded?"
# results = vector_store.similarity_search(query)
# for i, r in enumerate(results, 1):
#     print(f"Result {i}:\n{r}\n{'-'*40}")
    
# #####################     Load Mistral & FastAPI     #################### 
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# hf_token = "HF_TOKEN"
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
# # Set device for torch
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32, device_map="auto" if DEVICE.type == "cuda" else None)
# mistral_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)


# #####################     Answer Generator     #################### 
# def generate_answer(query, vector_store):
#     docs = vector_store.similarity_search(query, k=4)
#     context = "\n\n".join(docs)
#     prompt = f"""Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
#     result = mistral_pipe(prompt)[0]["generated_text"]
#     return result.split("Answer:")[-1].strip()


# #####################     FastAPI App     #################### 
# app = FastAPI()

# class Query(BaseModel):
#     question: str

# @app.post("/chat")
# def chat(query: Query):
#     answer = generate_answer(query.question, vector_store)
#     return {"response": answer}


# url = "http://localhost:8000/chat"
# question_payload = {
#     "question": "When was TU Delft founded?"
# }

# response = requests.post(url, json=question_payload)
# print("Response:", response.json())


# response = requests.post("http://localhost:8000/chat", json={"question": "When was TU Delft founded?"})
# print(response.json())


# # Replace with the actual URL where your FastAPI app is running
# url = "http://localhost:8000/chat"

# # Your question
# question_payload = {
#     "question": "When was TU Delft founded?"
# }

# # Send POST request
# response = requests.post(url, json=question_payload)

# # Print the chatbot's response
# print("Response:", response.json())


# #####################     Local Run     #################### 

# if __name__ == "__main__":
#     while True:
#         question = input("Ask a question (or type 'exit' to quit): ")
#         if question.lower() in {"exit", "quit"}:
#             break
#         answer = generate_answer(question, vector_store)
#         print("\nAnswer:", answer)
#         print("-" * 50)


# question = "Who are the data stewards?"
# answer = generate_answer(question, vector_store)
# print("Answer:", answer)












