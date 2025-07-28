import requests
from bs4 import BeautifulSoup
import pickle
import os

def scrape_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

def save_or_load_web_chunks(web_chunks_path, web_urls, split_text_func, scrape_webpage_func):
    if os.path.exists(web_chunks_path):
        with open(web_chunks_path, "rb") as f:
            return pickle.load(f)
    else:
        web_chunks = []
        for url in web_urls:
            web_text = scrape_webpage_func(url)
            web_chunks.extend(split_text_func(web_text))
        with open(web_chunks_path, "wb") as f:
            pickle.dump(web_chunks, f)
        return web_chunks 