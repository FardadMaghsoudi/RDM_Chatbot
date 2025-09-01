# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:13:42 2025
@author: fmaghsoudimoud
"""

# Libraries 
import os
import pickle
import config
from vector_store import split_text, SimpleVectorStore
#from pdf_utils import save_or_load_pdf_chunks
#from web_utils import download_pdfs_from_webpage
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Helper: Crawl one site
def crawl_website(base_url, max_pages=10, delay=1.0):
    visited = set()
    to_visit = [base_url]
    all_pages = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            all_pages.append((url, page_text))

            # Follow links within same domain
            for link in soup.find_all("a", href=True):
                new_url = urljoin(url, link["href"])
                if urlparse(new_url).netloc == urlparse(base_url).netloc:
                    if new_url not in visited:
                        to_visit.append(new_url)

        except Exception as e:
            print(f" Error fetching {url}: {e}")
            continue
    return all_pages

# Preprocess Data
def preprocess_data():
    all_chunks = []
    store = SimpleVectorStore()

    # Step 1: Crawl all mother links from config
    crawled_pages = []
    for start_url in config.WEB_START_URLS:
        print(f" Crawling from: {start_url}")
        crawled_pages.extend(crawl_website(start_url, max_pages=10))

    print(f" Crawled {len(crawled_pages)} pages in total")

    # Step 2: Split text into chunks
    for url, text in crawled_pages:
        chunks = split_text(text)
        all_chunks.extend(chunks)

    # Step 3: Save chunks to intermediate file
    os.makedirs(config.PREPROCESSED_DATA_DIR, exist_ok=True)
    with open(config.WEB_CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    # Step 4: Add chunks to vector store
    store.add(all_chunks)

    print(f" Stored {len(all_chunks)} chunks in vector store")
    print(f" Saved web chunks to {config.WEB_CHUNKS_PATH}")

    return store, all_chunks, crawled_pages

if __name__ == "__main__":
    store, chunks, pages = preprocess_data()
    global_store = store
    global_chunks = chunks
    global_pages = pages

    print("Preview:")
    print("First URL:", global_pages[0][0])
    print("First 300 chars of text:", global_pages[0][1][:300])
