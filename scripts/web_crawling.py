# Libraries 
import os
import pickle
import config
import time
import warnings
from vector_store import split_text, SimpleVectorStore
import requests
from bs4 import BeautifulSoup, NavigableString, Tag, MarkupResemblesLocatorWarning
from urllib.parse import urljoin, urlparse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

def create_session():
    """Creates a session with the correct headers to avoid bot detection."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session

def scrape_webpage(soup):
    """
    TARGETED MODE: Extracts text ONLY from divs with specific TU Delft classes:
    't3ce frame-type-text frame-space-before-40'
    """
    # 1. Safety & Conversion
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    if isinstance(soup, str):
        soup = BeautifulSoup(soup, "html.parser")
    
    extracted_text = []

    # 2. Specific CSS Selector
    # The dot (.) syntax ensures it matches divs having ALL these classes, regardless of order.
    target_selector = "div.t3ce.frame-type-text"
    
    content_blocks = soup.select(target_selector)

    if not content_blocks:
        # Debug print to help you see if the selector missed
        print(f"  -> Debug: No blocks found with class '{target_selector}'")
        return ""

    if len(content_blocks) > 2:
        content_blocks = content_blocks[:-2]

    print(f"  -> Debug: Found {len(content_blocks)} valid content blocks.")

    # 3. Extract text from Paragraphs <p> inside these blocks
    for block in content_blocks:
        # Search for both paragraphs and list items
        # recursive=True is default, so it finds nested items too
        elements = block.find_all(['p', 'li'])
        
        for element in elements:
            # AVOID DUPLICATES: 
            # If an <li> contains a <p>, the <p> will be found separately.
            # We should ignore the <li> to avoid adding the text twice.
            if element.name == 'li' and element.find('p'):
                continue

            text = element.get_text(separator=' ', strip=True)
            
            # Basic filter (keep len > 5 to avoid bullets or empty spacers)
            if len(text) > 5:
                extracted_text.append(text)

    if extracted_text:
        return "\n\n".join(extracted_text)
    
    return ""

# Helper: Crawl one site
def crawl_website(base_url, max_pages=20, delay=1.5):
    visited = set()
    to_visit = [base_url]
    all_pages = []

    session = create_session()

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        print(f"Fetching: {url}")
        visited.add(url)

        try:
            response = session.get(url,timeout=10)
            if response.status_code != 200:
                continue
            
            # 1. Check if the response is valid HTML
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                print(f"  -> Skipping: Content-Type is {content_type} (not HTML)")
                continue

            # 2. Check if content is empty
            if not response.content:
                print("  -> Skipping: Empty response content")
                continue

            #print(f"RESPONSE CONTENT {response.content}")

            soup_obj = BeautifulSoup(response.content, "html.parser")
            
            if isinstance(soup_obj, str):
                print("ERROR: BeautifulSoup returned a string instead of an object.")
                continue

            # Extract Content
            page_text = scrape_webpage(soup_obj)
            # print(page_text)

            if page_text:
                all_pages.append((url, page_text))

            # Link Discovery
            for link in soup_obj.find_all("a", href=True):
                new_url = urljoin(url, link["href"]).split('#')[0]
                
                # Stay on the same domain and filter extensions
                if urlparse(new_url).netloc == urlparse(base_url).netloc:
                    if not any(ext in new_url.lower() for ext in ['.pdf', '.jpg', '.png', '.zip', '.docx']):
                        if new_url not in visited and new_url not in to_visit:
                            to_visit.append(new_url)
            
            # CRITICAL: Respectful delay to avoid bot detection
            time.sleep(delay)

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
        crawled_pages.extend(crawl_website(start_url, max_pages=20))

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
