import os
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter, Transformation
from PyPDF2._page import PageObject
import argparse
import pdfplumber
import requests
from bs4 import BeautifulSoup
import re
import pickle

def load_pdf_text(pdf_path, header_margin=60, footer_margin=50):
    """
    Loads text from a PDF while cropping out headers and footers.
    
    Args:
        pdf_path (str): Path to the PDF file.
        header_margin (int): Height of the top area to ignore (default 60).
        footer_margin (int): Height of the bottom area to ignore (default 50).
        
    Returns:
        str: Cleaned text content.
    """
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # pdfplumber crop bbox: (x0, top, x1, bottom) from top-left origin
            cropped = page.crop((0, header_margin, page.width, page.height - footer_margin))
            page_text = cropped.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def load_prdw_pdf_text(pdf_path):
    """
    Extracts text from the PRDW PDF while filtering out repeating headers,
    footers, and navigation elements.
    """
    # Known repeating navigation/footer phrases specific to this document
    NAV_PHRASES = [
        "USING THIS GUIDE", "NAVIGATION", "THE PDRW+", "THE RISKS", "THE MITIGATIONS",
        "INDEX", "Q & A", "Q&A", "TIMINGS",
        "BACK", "NEXT", "BACK TO NAVIGATION",
    ]

    # Compile a regex pattern to detect navigation keywords
    nav_pattern = re.compile(r"(" + "|".join([re.escape(p) for p in NAV_PHRASES]) + r")", re.IGNORECASE)

    page_texts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text()
            if not raw_text:
                continue

            lines = raw_text.splitlines()
            cleaned_lines = []

            for line in lines:
                stripped = line.strip()

                # Skip empty lines and pure page numbers
                if not stripped or re.match(r"^\d+$", stripped):
                    continue

                # Normalize extra spaces (common in PDF text extraction)
                normalized = re.sub(r"\s+", " ", stripped)

                # Filter out lines that are dominated by navigation keywords
                nav_matches = nav_pattern.findall(normalized)
                # If a line contains 3 or more distinct navigation keywords, it's a UI element
                if len(nav_matches) >= 3:
                    continue

                cleaned_lines.append(normalized)

            if cleaned_lines:
                page_texts.append("\n".join(cleaned_lines))

    # Join pages with a clear separator
    return "\n\n--- PAGE BREAK ---\n\n".join(page_texts)

def load_all_pdfs(folder_path, prdw_path):
    all_text = []
    for root, dirs, files in os.walk(folder_path):
        print(f"Number of files in folder: {len(files)}")
        for filename in files:
            if filename.endswith(".pdf"):
                full_path = os.path.join(root, filename)
                raw_text = load_pdf_text(full_path)
                if raw_text:
                    print(f"Found text in {filename}")
                    all_text.append(raw_text)

    prdw_text = load_prdw_pdf_text(prdw_path)
    all_text.append(prdw_text)

    return all_text

def remove_last_n_pages(src: str, dst: str, n: int = 1) -> None:
    """
    Write *dst* as a copy of *src* with the last *n* pages removed.
    """
    reader = PyPDF2.PdfReader(src)
    total  = len(reader.pages)
 
    if n >= total:
        raise ValueError(
            f"Cannot remove {n} page(s) from '{os.path.basename(src)}' "
            f"which only has {total} page(s)."
        )
 
    writer = PyPDF2.PdfWriter()
    for page in reader.pages[: total - n]:
        writer.add_page(page)
 
    _write(writer, dst)
    print(f"  ✓  Removed last {n} page(s): {os.path.basename(src)}  "
          f"({total} → {total - n} pages)")



# TODO: add support for other file types
def download_pdfs_from_webpage(url, download_folder="../policies/tudelft_policies"): 
    os.makedirs(download_folder, exist_ok=True)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    pdf_links = set()
    zenodo_links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if ("/files/" in href and href.lower().endswith(".pdf?download=1")):
            if not href.startswith("http") and not href.startswith("https"):
                href = requests.compat.urljoin(url, href)
            href = href.rsplit("?download=1", 1)[0]
            pdf_links.add(href)
        elif href.lower().endswith(".pdf"):
            if not href.startswith("http") and not href.startswith("https"):
                href = requests.compat.urljoin(url, href)
            pdf_links.add(href)
        elif "zenodo" in href:
            if not href.startswith("http") and not href.startswith("https"):
                href = requests.compat.urljoin(url, href)
            zenodo_links.add(href)
    local_paths = []
    for link in pdf_links:
        filename = os.path.join(download_folder, os.path.basename(link.split("?")[0]))
        if not os.path.exists(filename):
            print(f"Downloading {link} ...")
            try:
                r = requests.get(link)
                with open(filename, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"Failed to download {link}: {e}")
        local_paths.append(filename)
    for zenodo_url in zenodo_links:
        print(f"Processing Zenodo page: {zenodo_url}")
        try:
            zenodo_resp = requests.get(zenodo_url)
            zenodo_soup = BeautifulSoup(zenodo_resp.content, "html.parser")
            pdf_a = zenodo_soup.find("a", href=re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE))
            if pdf_a:
                pdf_href = pdf_a["href"]
                if not pdf_href.startswith("http") and not pdf_href.startswith("https"):
                    print(f"Found relative PDF link on Zenodo page: {pdf_href}")
                    pdf_href = requests.compat.urljoin("https://zenodo.org", pdf_href)
                filename = os.path.join(download_folder, os.path.basename(pdf_href.split("?")[0]))
                if not os.path.exists(filename):
                    print(f"Downloading Zenodo PDF: {pdf_href}")
                    r = requests.get(pdf_href)
                    with open(filename, "wb") as f:
                        f.write(r.content)
                local_paths.append(filename)
            else:
                print(f"No PDF found on Zenodo page: {zenodo_url}")
        except Exception as e:
            print(f"Failed to process Zenodo page {zenodo_url}: {e}")
    return download_folder 

def save_or_load_pdf_chunks(pdf_chunks_path, pdf_folder, split_text_func, prdw_path):
    if os.path.exists(pdf_chunks_path):
        with open(pdf_chunks_path, "rb") as f:
            return pickle.load(f)
    else:
        pdf_docs = load_all_pdfs(pdf_folder, prdw_path)
        all_chunks = []

        for doc_text in pdf_docs:
            if doc_text: # Ensure text is not empty
                # split_text_func expects a string and returns a list of chunks
                doc_chunks = split_text_func(doc_text)

                # 3. Add these chunks to our master list (flattening the list)
                all_chunks.extend(doc_chunks)

        with open(pdf_chunks_path, "wb") as f:
            pickle.dump(all_chunks, f)
        return all_chunks 

def _write(writer: PyPDF2.PdfWriter, dst: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "wb") as fh:
        writer.write(fh)
