import os
# Configuration for the RDM Chatbot project

POLICIES_URL = "https://www.tudelft.nl/en/library/data-management/research-data-management/tu-delft-faculty-policies-for-research-data"
WEB_URLS = [
    # main URL of the RDM TU Delft Library
    "https://www.tudelft.nl/en/library/data-management#c1634274"
    # data management plan
    "https://www.tudelft.nl/en/library/data-management/data-management-plan"
    # data management grant proposal
    "https://www.tudelft.nl/en/library/data-management/research-data-management/data-management-sections-for-grant-proposals"
    # about data stewardship
    "https://www.tudelft.nl/en/library/data-management/get-support-on-data-management/data-stewardship-about"
    # library
    "https://www.tudelft.nl/library",
    "https://www.tudelft.nl/en/about-tu-delft"
    # research data management
    "https://www.tudelft.nl/en/library/data-management/research-data-management/the-goal-of-research-data-management",
    # security
    "https://tud365.sharepoint.com/:u:/r/sites/SecurityPrivacyTUD/SitePages/en/Informatiebeveiliging-Nieuws.aspx?csf=1&web=1&e=q8hYWr",
    # privacy
    "https://tud365.sharepoint.com/:u:/r/sites/SecurityPrivacyTUD/SitePages/en/Privacy.aspx?csf=1&web=1&e=kGMFIQ",
    # working safely
    "https://tud365.sharepoint.com/sites/SecurityPrivacyTUD/SitePages/en/Awareness.aspx"
]
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.environ.get("HF_TOKEN")
PDF_FOLDER = "policies"

# Intermediate data file names
PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocessed-data")

PDF_TEXT_PATH = os.path.join(PREPROCESSED_DATA_DIR, f"intermediate_pdf_text.pkl")
PDF_CHUNKS_PATH = os.path.join(PREPROCESSED_DATA_DIR, f"intermediate_pdf_chunks.pkl")
WEB_CHUNKS_PATH = os.path.join(PREPROCESSED_DATA_DIR, f"intermediate_web_chunks.pkl") 