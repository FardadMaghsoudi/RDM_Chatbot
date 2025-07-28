import os
from vector_store import split_text, SimpleVectorStore
from pdf_utils import save_or_load_pdf_text, save_or_load_pdf_chunks
from web_utils import scrape_webpage, save_or_load_web_chunks
import config

def preprocess_data():
#    NOTE: Uncomment the next line if you want to download PDFs again
#    pdf_folder = download_pdfs_from_webpage(config.POLICIES_URL, config.PDF_FOLDER)

    # Ensure preprocessed-data directory exists
    os.makedirs(config.PREPROCESSED_DATA_DIR, exist_ok=True)

    # Load PDFs and split text (with intermediate saving/loading)
    print("Loading PDFs...")
    pdf_text = save_or_load_pdf_text(config.PDF_TEXT_PATH, config.PDF_FOLDER)
    pdf_chunks = save_or_load_pdf_chunks(config.PDF_CHUNKS_PATH, pdf_text, split_text)

    # Scrape web pages and split text (with intermediate saving/loading)
    print("Scraping web pages...")
    web_chunks = save_or_load_web_chunks(config.WEB_CHUNKS_PATH, config.WEB_URLS, split_text, scrape_webpage)

    print("Creating vector store...")
    combined_chunks = pdf_chunks + web_chunks
    vector_store = SimpleVectorStore(combined_chunks)
    return combined_chunks, vector_store 