import requests
from bs4 import BeautifulSoup
from web_crawling import crawl_website, scrape_webpage
import pickle
import os

def scrape_webpage_old(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    return clean_text(text)

def save_or_load_web_chunks(web_chunks_path, web_urls, split_text_func, web_crawling_func=crawl_website):
    if os.path.exists(web_chunks_path):
        with open(web_chunks_path, "rb") as f:
            return pickle.load(f)

    unique_pages = {}

    print(f"Starting discovery from {len(web_urls)} entry points...")

    # 1. Crawl and Collect (Single Pass)
    for start_url in web_urls:
        # web_crawling_func (crawl_website) returns a list of tuples: [(url, clean_text), ...]
        # It has already filtered out pages that don't have the "On this page" section.
        crawled_data = web_crawling_func(start_url)

        for url, text in crawled_data:
            if url not in unique_pages:
                unique_pages[url] = text

    print(f"Total unique valid pages collecting for processing: {len(unique_pages)}")

    # 2. Chunk the text
    web_chunks = []
    for url, text in unique_pages.items():
        # We process the text we already have; no need to fetch again.
        if text:
            chunks = split_text_func(text)
            web_chunks.extend(chunks)

    # 3. Save to disk
    # Ensure directory exists
    os.makedirs(os.path.dirname(web_chunks_path), exist_ok=True)

    with open(web_chunks_path, "wb") as f:
        pickle.dump(web_chunks, f)

    print(f"Saved {len(web_chunks)} chunks to {web_chunks_path}")
    return web_chunks
