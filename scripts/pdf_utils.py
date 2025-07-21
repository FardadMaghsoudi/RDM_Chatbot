import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
import re

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

def download_pdfs_from_webpage(url, download_folder="tudelft_policies"): 
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
                    pdf_href = requests.compat.urljoin(zenodo_url, pdf_href)
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