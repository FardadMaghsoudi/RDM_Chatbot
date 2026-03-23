import os
# Configuration for the RDM Chatbot project

POLICIES_URL = "https://www.tudelft.nl/en/library/data-management/research-data-management/tu-delft-faculty-policies-for-research-data"
WEB_URLS = {
    # main URL of the RDM TU Delft Library
    "RDM overview": "https://www.tudelft.nl/en/library/data-management#c1634274",
    
    # data management plan
    "Data management plans": "https://www.tudelft.nl/en/library/data-management/research-data-management/data-management-plans",
    "RDM 101 training": "https://www.tudelft.nl/library/data-management/trainingen/trainingen-voor-onderzoekers-en/research-data-management-101",
    "DMPOnline tool": "https://www.tudelft.nl/en/library/data-management/research-data-management/dmponline-tool-for-research-data",
    "Data management costs": "https://www.tudelft.nl/en/library/data-management/research-data-management/data-management-costs",
    "Ownership of research data": "https://www.tudelft.nl/en/library/data-management/research-data-management/ownership-of-research-data",
    "Funder policies for research data": "https://www.tudelft.nl/en/library/data-management/research-data-management/funders-policies-for-research-data",
    "Data management sections for grant proposals": "https://www.tudelft.nl/en/library/data-management/research-data-management/data-management-sections-for-grant-proposals",
    
    # about data stewardship
    "Data stewardship at TU Delft": "https://www.tudelft.nl/en/library/data-management/get-support-on-data-management/data-stewardship-at-tu-delft",
    "Contact the data stewards": "https://www.tudelft.nl/en/library/data-management/get-support-on-data-management/contact-the-data-stewards",
    "Research data services team": "https://www.tudelft.nl/en/library/data-management/get-support-on-data-management/research-data-services-team",
    "Strategic framework for data stewardship": "https://www.tudelft.nl/en/library/data-management/get-support-on-data-management/strategic-framework-for-data-stewardship",
    "Data management FAQs": "https://www.tudelft.nl/en/library/data-management/get-support-on-data-management/faqs",

    # research data management
    "Goal of research data management": "https://www.tudelft.nl/en/library/data-management/research-data-management/the-goal-of-research-data-management",
    "Research data storage": "https://www.tudelft.nl/en/library/data-management/research-data-management/research-data-storage",
    "Manage confidential data – personal data": "https://www.tudelft.nl/en/library/data-management/research-data-management/manage-confidential-data-personal-data",
    "Extended personal research data workflow": "https://www.tudelft.nl/en/library/data-management/research-data-management/a-guide-to-the-extended-personal-research-data-workflow",
    "Manage confidential data – non-personal data": "https://www.tudelft.nl/en/library/data-management/research-data-management/manage-confidential-data-non-personal-data",
    "Collect and document research data": "https://www.tudelft.nl/en/library/data-management/research-data-management/collect-and-document-research-data",
    "Electronic lab notebook": "https://www.tudelft.nl/en/library/data-management/research-data-management/electronic-lab-notebook-for-research-data-management",

    # publication
    "Prepare research data for publication": "https://www.tudelft.nl/en/library/data-management/research-data-management/prepare-research-data-for-publication",
    "Publish research data": "https://www.tudelft.nl/en/library/data-management/research-data-management/publish-research-data",
    "PhD data sharing – current and new candidates": "https://www.tudelft.nl/en/library/data-management/research-data-management/publish-your-phd-data-guidance-on-data-sharing-for-current-and-new-doctoral-candidates",
    "PhD data sharing – completing studies": "https://www.tudelft.nl/en/library/data-management/research-data-management/publish-your-phd-data-guidance-for-doctoral-candidates-completing-their-studies",
    "Publish research software": "https://www.tudelft.nl/en/library/data-management/research-data-management/publish-research-software",
    "Cite your research data": "https://www.tudelft.nl/en/library/data-management/research-data-management/cite-your-research-data",

    # TU Delft iRODS
    "iRODS research data management": "https://www.tudelft.nl/en/2023/library/easy-fast-accessible-and-secure-research-data-management", 
    "iRODS technical guide": "https://hackmd.io/@fardadmaghsoudi/By5RxKF_h",
    
    # security
    "Information security news": "https://tud365.sharepoint.com/:u:/r/sites/SecurityPrivacyTUD/SitePages/en/Informatiebeveiliging-Nieuws.aspx?csf=1&web=1&e=q8hYWr",
    "Data classification": "https://tud365.sharepoint.com/sites/SecurityPrivacyTUD/SitePages/en/Noodzakelijkheid-van-Dataclassificatie.aspx",
    "Security and privacy training": "https://tud365.sharepoint.com/sites/SecurityPrivacyTUD/SitePages/en/security-privacy-training.aspx",
    
    # privacy
    "Privacy at TU Delft": "https://tud365.sharepoint.com/:u:/r/sites/SecurityPrivacyTUD/SitePages/en/Privacy.aspx?csf=1&web=1&e=kGMFIQ",
    
    # security and privacy awareness
    "Privacy policy": "https://tud365.sharepoint.com/sites/SecurityPrivacyTUD/SitePages/en/Beleid-privacy.aspx",
    "Security and privacy awareness": "https://tud365.sharepoint.com/sites/SecurityPrivacyTUD/SitePages/en/Awareness.aspx",
    
    # library
    "TU Delft Library": "https://www.tudelft.nl/library",
    "About TU Delft": "https://www.tudelft.nl/en/about-tu-delft"
}

HF_TOKEN = os.environ.get("HF_TOKEN")
PDF_FOLDER = "policies"

# Intermediate data file names
## In case Line 42 doesn't work, use the following two lines instead:
PREPROCESSED_DATA_DIR = os.path.join(os.getcwd(), "preprocessed-data")
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
#PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocessed-data")
PDF_TEXT_PATH = os.path.join(PREPROCESSED_DATA_DIR, f"intermediate_pdf_text.pkl")
PDF_CHUNKS_PATH = os.path.join(PREPROCESSED_DATA_DIR, f"intermediate_pdf_chunks.pkl")
WEB_CHUNKS_PATH = os.path.join(PREPROCESSED_DATA_DIR, f"intermediate_web_chunks.pkl") 
