# RDM Chatbot
Intelligent RDM Support

## Setup Instructions

1. **Create the conda environment:**

   ```bash
   conda create -n rdm_chatbot python=3.10
   ```

2. **Activate the environment:**

   ```bash
   conda activate rdm_chatbot
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```


## Project Structure

- `scripts/` — All main Python modules and entry point
- `tudelft_policies/` — Downloaded PDFs
- `requirements.txt` — Python dependencies
- `README.md` — This file

## Usage

Run the chatbot API or CLI from the `scripts/` folder:

```bash
python scripts/main.py
```

## Modules
- `scripts/main.py`: FastAPI app and CLI entry point
- `scripts/vector_store.py`: Text splitting and vector store logic
- `scripts/pdf_utils.py`: PDF loading and downloading logic
- `scripts/web_utils.py`: Web scraping logic
- `scripts/mistral_model.py`: Mistral model loading and answer generation
- `scripts/config.py`: Configuration (URLs, model names, tokens, etc.)

---

The `Python-Mistral.py` file is kept for reference only.
