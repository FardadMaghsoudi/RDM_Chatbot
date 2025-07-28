import os
# Configuration for the RDM Chatbot project

POLICIES_URL = "https://www.tudelft.nl/en/library/data-management/research-data-management/tu-delft-faculty-policies-for-research-data"
WEB_URLS = [
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