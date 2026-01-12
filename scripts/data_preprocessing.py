import os
from vector_store import split_text, SimpleVectorStore, split_text_by_sentences
from pdf_utils import save_or_load_pdf_chunks, download_pdfs_from_webpage
from web_utils import save_or_load_web_chunks
import config

def preprocess_data():
    # NOTE: Uncomment the next line if you want to download PDFs again
    # download_pdfs_from_webpage(config.POLICIES_URL, config.PDF_FOLDER)

    # Ensure preprocessed-data directory exists
    os.makedirs(config.PREPROCESSED_DATA_DIR, exist_ok=True)

    # Load PDFs and split text (with intermediate saving/loading)
    print("Loading PDFs...")
    pdf_chunks = save_or_load_pdf_chunks(config.PDF_CHUNKS_PATH, config.PDF_FOLDER, split_text_by_sentences)

    # Scrape web pages and split text (with intermediate saving/loading)
    print("Scraping web pages...")
    web_chunks = save_or_load_web_chunks(config.WEB_CHUNKS_PATH, config.WEB_URLS, split_text_by_sentences)
    print("Creating vector store...")
    combined_chunks = pdf_chunks + web_chunks
    vector_store = SimpleVectorStore(combined_chunks)
    return combined_chunks, vector_store 

if __name__ == "__main__":
    preprocess_data()
    cc, vs = preprocess_data()
    print(f"Total chunks processed: {len(cc)}")
    # Test similarity search
    query = "What are the data management policies at TU Delft?"
    print(f"Performing similarity search for query: '{query}'")
    results = vs.similarity_search(query, k=3)
    print("Top 3 similar chunks:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}...")  # Print first 200 characters of each chunk