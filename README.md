# Dizzy – TU Delft RDM Chatbot

Dizzy is a chatbot for **Research Data Management (RDM)** questions at TU Delft. Researchers can ask it about data
management plans (DMPs), storage and security policies, personal data, archiving and publishing data, iRODS, and
human research ethics (HREC).

It is a **Retrieval-Augmented Generation (RAG)** system:

1. **Ingestion**: the TU Delft RDM policy PDFs, the Personal Research Data Workflow (PRDW) guide, the TU Delft
   library / security / HREC web pages and several TU Delft Jupyter Books are scraped, cleaned and split into
   chunks. The chunks are cached in `preprocessed-data/`.
2. **Retrieval**: the chunks are embedded with `sentence-transformers/multi-qa-mpnet-base-cos-v1` and indexed in
   a FAISS inner-product index. For each question the 5 most similar chunks are retrieved.
3. **Generation**: [`mistralai/Ministral-3-3B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16),
   loaded in 4-bit and merged with a LoRA adapter fine-tuned on RDM Q&A pairs, answers the question using the
   retrieved context and the conversation history.
4. **Safety**: user input and model output are checked against regex patterns (configured in `.env`) to block
   prompt-injection attempts and system-prompt disclosure.

The app serves a **Gradio chat UI**, a **REST endpoint** and a password-protected **admin panel**. The admin panel
shows chat history, flagged requests, errors and system stats, and stores them in a local SQLite database.

---

## Project structure

```
RDM_Chatbot/
├── scripts/                                  # All source code of the chatbot
│   ├── main.py                               # Entry point: starts the server with the Gradio chat UI, the /chat REST endpoint and the admin panel
│   ├── admin.py                              # Admin panel: SQLite chat/error logging, malicious-keyword detection, stats API and HTML dashboard
│   ├── config.py                             # Web/Jupyter Book URLs to scrape and all project paths (data, cache, adapter), resolved from the project root
│   ├── ingestion/                            # Building the knowledge base
│   │   ├── __init__.py                       # Marks the folder as a Python package
│   │   ├── data_preprocessing.py             # Loads (or builds and caches) the PDF and web chunks and creates the vector store; runnable as a script
│   │   ├── pdf_utils.py                      # Downloads the policy PDFs, extracts their text (header/footer cropping, PRDW clean-up) and caches the chunks
│   │   ├── web_utils.py                      # Crawls all configured start URLs and Jupyter Books, chunks the pages and caches the chunks
│   │   └── web_crawling.py                   # Crawlers and HTML text extractors for TU Delft pages and Jupyter Books
│   ├── rag/                                  # Retrieval and generation
│   │   ├── __init__.py                       # Marks the folder as a Python package
│   │   ├── vector_store.py                   # Text cleaning, chunking and the FAISS vector store with sentence-transformer embeddings
│   │   └── mistral_model.py                  # Loads the 4-bit Ministral model + LoRA adapter, builds the prompt, generates answers, runs the safety checks
│   └── finetuning/                           # LoRA fine-tuning of the LLM
│       ├── __init__.py                       # Marks the folder as a Python package
│       ├── finetune.py                       # Trains a LoRA adapter on docs/train_qnas and evaluates it (ROUGE, BLEU, METEOR, BERTScore) in W&B
│       └── experiments.yaml                  # Weights & Biases sweep: grid of hyper-parameters to run finetune.py with
├── tests/                                    # pytest test suite (see "Tests")
│   ├── conftest.py                           # Shared helpers: generates small test PDFs, skips e2e tests unless `-m e2e` is given
│   ├── test_app.py                           # Fast app tests with a fake model: chat page, /chat, history, safety filters, Gradio chat, admin panel
│   ├── test_e2e.py                           # End-to-end tests: starts the real app on the GPU and chats via REST and the Gradio UI
│   ├── test_pdf_utils.py                     # PDF text extraction, _clean versions replacing originals, chunk caching/de-duplication, PDF download
│   ├── test_web_utils.py                     # HTML text extraction, per-start-URL crawl limit, chunk caching/de-duplication, live scraping
│   ├── test_vector_store.py                  # Chunk-level text cleaning and sentence-based chunking (sizes, overlap, paragraph breaks)
│   └── test_chunks_quality.py                # Sanity checks on the cached chunks (count, empty/short/oversized chunks, duplicates)
├── legacy/                                   # Old scripts kept for reference only (not maintained, not updated for this layout)
│   ├── app.py                                # Previous Gradio-only version of main.py
│   ├── Python-Mistral.py                     # Original single-file prototype (fully commented out)
│   ├── finetune.py                           # Earlier fine-tuning script based on TRL's SFTTrainer and 4-bit QLoRA
│   ├── finetune_offloaded_version.py         # Earlier 4-bit fine-tuning variant with CPU weight offloading
│   ├── experiments.yaml                      # W&B sweep config for finetune_offloaded_version.py
│   └── gpu_load_smoke_test.py                # Loads Mistral-7B in 4-bit to check GPU memory and run one forward pass
├── preprocessed-data/                        # Cached knowledge base (tracked in git)
│   ├── intermediate_pdf_chunks.pkl           # Text chunks from the policy PDFs and the PRDW guide
│   └── intermediate_web_chunks.pkl           # Text chunks from the crawled web pages and Jupyter Books
├── Ministral-3-3B-Instruct-2512-BF16-full-r16-lr0.0001-ep5-bs1-test0.1.zip  # The LoRA adapter used by the chatbot (Git LFS); unzip into results/
├── .gitattributes                            # Stores *.zip files with Git LFS
├── .gitignore                                # Files and folders kept out of git
├── pytest.ini                                # pytest settings: test folder, import path (scripts/) and test markers
├── requirements.txt                          # Pinned Python dependencies
└── README.md                                 # This file
```

All paths in `config.py` are resolved relative to the project root, so every command below can be run from any
directory. The commands are written as if run from the project root.

---

## Setup

### 1. Requirements

- Python 3.12 (tested; 3.10+ should work)
- An NVIDIA GPU with CUDA. The chatbot (4-bit 3B model) needs about 4 GB of VRAM. Fine-tuning loads the model in
  bf16 and was run on a 16 GB GPU.
- [Git LFS](https://git-lfs.com/), to fetch the adapter zip file

### 2. Install

```bash
git lfs pull                       # download the adapter zip
python -m venv .venv               # or: conda create -n rdm_chatbot python=3.12
source .venv/bin/activate          # or: conda activate rdm_chatbot
pip install -r requirements.txt
```

### 3. Model adapter

The chatbot loads the LoRA adapter from the folder set as `ADAPTER_DIR` in `scripts/config.py`
(`results/Ministral-3-3B-Instruct-2512-BF16-full-r16-lr0.0001-ep5-bs1-test0.1/`). Unzip the adapter shipped with the repository:

```bash
mkdir -p results
unzip Ministral-3-3B-Instruct-2512-BF16-full-r16-lr0.0001-ep5-bs1-test0.1.zip -d results/
```

The base model is downloaded from Hugging Face on the first run. You need to accept the model's terms on
Hugging Face and set `HF_TOKEN` (see below).

### 4. Environment variables (`.env`)

Create a `.env` file in the project root:

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | yes | Hugging Face access token (needed to download the base model) |
| `SYSTEM_PROMPT` | yes | System prompt for the model. Must contain the placeholder `{url_ref}` (replaced by the list of source URLs); `\n` is converted to a newline |
| `FORBIDDEN_INPUT_PATTERNS` | yes | Regex patterns, separated by `\|\|`, that block suspicious user input |
| `DISCLOSURE_OUTPUT_PATTERNS` | yes | Regex patterns, separated by `\|\|`, that block model output leaking the prompt |
| `SAFE_RESPONSE` | yes | Answer returned when input or output is blocked |
| `ADMIN_USERNAME` | no | Admin panel user (default `admin`) |
| `ADMIN_PASSWORD` | no | Admin panel password. If not set, a random one is printed at startup |
| `HOST` / `PORT` | no | Server address (default `0.0.0.0:8000`) |
| `FORWARDED_ALLOW_IPS` | no | Trusted proxy IPs for `X-Forwarded-For` (default `127.0.0.1`) |
| `SHARE` | no | `true` creates a public Gradio share link (see below) |
| `ADMIN_DB_PATH` | no | Location of the admin SQLite database (default `data/admin.db`) |

Variables set in the shell override the ones in `.env`, e.g. `PORT=8080 python scripts/main.py`.

---

## 1. Data processing

**Script:** `scripts/ingestion/data_preprocessing.py`

```bash
python scripts/ingestion/data_preprocessing.py                  # build the missing chunk files, then test a search
python scripts/ingestion/data_preprocessing.py --download-pdfs  # first re-download the policy PDFs
```

### What it does

1. **(Optional, `--download-pdfs`) Download PDFs.** Downloads every PDF linked on the TU Delft faculty policies
   page (`POLICIES_URL` in `config.py`), including PDFs behind Zenodo links, into `docs/policies/`. Files that
   already exist are not downloaded again.
2. **PDF chunks.** If `preprocessed-data/intermediate_pdf_chunks.pkl` exists, it is loaded as it is. Otherwise every
   `*.pdf` in `docs/policies/` is read with pdfplumber (the top 60 and bottom 50 points of each page, i.e. headers and
   footers, are cut off), and `docs/PRDW.pdf` is read with a special cleaner that removes its navigation
   bar and page numbers. Some PDFs needed manual cleaning (e.g. removing the first or last pages); the cleaned copy is
   saved as `<name>_clean.pdf`, and when both versions are in the folder only the `_clean` one is used. The text is
   split into chunks, duplicate chunks are removed, and the chunks are saved to the `.pkl` file.
3. **Web chunks.** If `preprocessed-data/intermediate_web_chunks.pkl` exists, it is loaded as it is. Otherwise:
   - every URL in `WEB_URLS` (`config.py`) is crawled: the page is fetched, the text of its
     `div.t3ce.frame-type-text` content blocks is extracted, and links to other pages on the same domain are
     followed, breadth-first, up to 20 pages per start URL (`max_pages` in `crawl_website()`), with a 1.5 s pause
     between requests. A page reached from several start URLs is only fetched once. The full crawl takes about
     10–15 minutes;
   - every Jupyter Book in `JUPYTER_BOOK_URLS` is read page by page by following its "next" links;
   - duplicate pages are removed, and the text is split into chunks; duplicate chunks are removed and the chunks
     are saved to the `.pkl` file.
4. **Vector store.** All chunks are embedded and put into the FAISS index. When run as a script, a sample
   similarity search is printed at the end.

The chunking is done by `split_text_by_sentences()` in `scripts/rag/vector_store.py`:

- The text is split into sentences at `.`, `!` or `?` followed by a capital letter, and at blank lines (web pages
  separate paragraphs and list items with them). Single line breaks (line wraps in PDFs) do not end a sentence.
- Each sentence is cleaned with `clean_text()`: unicode normalisation (e.g. the ligature "ﬁ" becomes "fi"), soft
  hyphens (which one PDF uses instead of spaces) become spaces, symbol-font bullet glyphs become "•", and whitespace
  is collapsed. Punctuation and stopwords are kept, since the chunks are passed to the model as context.
  The `--- PAGE BREAK ---` separators of the PRDW text are dropped.
- Sentences are grouped into chunks of about 2,000 characters, with about 200 characters (whole sentences) of
  overlap between consecutive chunks. A single sentence longer than 2,000 characters is split at word boundaries.
- A last chunk shorter than 50 characters is merged into the previous one. A page or document whose whole text is
  shorter than 50 characters (e.g. only "Filter by:") is dropped.

Only the chunking step cleans text this way. The PDF-specific cleaning (header/footer cropping, the PRDW cleaner and
the manually cleaned `_clean.pdf` files) happens before it, when the text is extracted.

### When to run it

The chatbot calls the same `preprocess_data()` function at startup, so it always uses the cached `.pkl` files. If
they are missing, the app builds them itself on startup, which takes a long time (crawling). Run the script
yourself when the sources have changed:

- **Web sources changed** (edited `WEB_URLS` / `JUPYTER_BOOK_URLS` in `config.py`, or the pages were updated):
  delete `preprocessed-data/intermediate_web_chunks.pkl` and run the script.
- **PDFs changed** (added/removed files in `docs/policies/`, or used `--download-pdfs`):
  delete `preprocessed-data/intermediate_pdf_chunks.pkl` and run the script.

Then restart the chatbot and, if the new chunks should be shared, commit the updated `.pkl` files. You can check
the new chunks with `pytest -m data`.

---

## 2. Fine-tuning

**Script:** `scripts/finetuning/finetune.py`, **sweep config:** `scripts/finetuning/experiments.yaml`

### Before you start

- `docs/train_qnas/*.jsonl` must exist. Each JSON object in them is one training example:
  `{"query": "...", "context": "...", "answer": "..."}`. All `.jsonl` files in the folder are used.
- `.env` must contain `HF_TOKEN` and `SYSTEM_PROMPT`. The training prompt is built with the same
  `build_prompt()` template as the chatbot (with the Q&A's `context` as context, and without the URL list and
  chat history).
- Log in to Weights & Biases once: `wandb login` (or set `WANDB_API_KEY`). Every run is logged to W&B.

### Single training run

```bash
python scripts/finetuning/finetune.py --lora_r 16 --epochs 5 --learning_rate 1e-4
```

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | Hugging Face base model (Ministral-3 models and other causal LMs are supported) |
| `--lora_r` | `8` | LoRA rank (`lora_alpha` is set to `2 * lora_r`) |
| `--learning_rate` | `1e-4` | Peak learning rate (cosine schedule, 10% warm-up) |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `1` | Batch size per device (gradients are accumulated over 8 steps) |
| `--test_size` | `0.1` | Fraction of the Q&As held out for evaluation (fixed seed 42) |
| `--project_name` | `RDM_Chatbot-scripts` | W&B project name (all previous runs are in `Dizzy-TUDelft/RDM_Chatbot-scripts`) |

What the script does:

1. Loads all Q&As and splits them into train and test sets.
2. Loads the base model in bf16 and adds LoRA adapters to the attention `q_proj` and `v_proj` layers
   (dropout 0.05). Examples are padded to 2,048 tokens. The prompt tokens are masked out of the labels, so the
   loss is only computed on the answer (and its end-of-sequence token).
3. Trains with gradient checkpointing. After every epoch it evaluates the loss and token accuracy on the test
   set (on the answer tokens), and at the end it keeps the epoch with the lowest evaluation loss.
4. Deletes the intermediate checkpoints, then generates an answer for every test question and logs
   ROUGE-1, ROUGE-L, BLEU, METEOR and BERTScore F1 to the W&B run summary (`eval_gen/...`).
5. Saves the adapter to `results/<model>-full-r<lora_r>-lr<learning_rate>-ep<epochs>-bs<batch_size>-test<test_size>/`,
   e.g. `results/Ministral-3-3B-Instruct-2512-BF16-full-r16-lr0.0001-ep5-bs1-test0.1/`. Every swept
   hyper-parameter is in the name, so sweep runs don't overwrite each other.

### Hyper-parameter sweep with Weights & Biases

`experiments.yaml` defines a **grid** sweep: W&B runs `finetune.py` once for every combination of the values
listed under `parameters`. To try more settings, add values to the lists, e.g. `lora_r: values: [8, 16, 32]`.

```bash
# 1. Create the sweep (run from the project root). It prints the sweep ID and the agent command.
wandb sweep --project RDM_Chatbot-scripts scripts/finetuning/experiments.yaml

# 2. Start an agent with the printed ID. It runs the combinations one after the other.
#    Run it from the project root, because the sweep starts `python scripts/finetuning/finetune.py`.
wandb agent Dizzy-TUDelft/RDM_Chatbot-scripts/<sweep-id>

# Optional: stop after N runs, or run agents in parallel on several GPUs
wandb agent --count 2 Dizzy-TUDelft/RDM_Chatbot-scripts/<sweep-id>
CUDA_VISIBLE_DEVICES=1 wandb agent Dizzy-TUDelft/RDM_Chatbot-scripts/<sweep-id>
```

Compare the runs in the W&B dashboard (evaluation loss and the `eval_gen/*` metrics).

### Using a new adapter in the chatbot

1. Set `ADAPTER_DIR` in `scripts/config.py` to the new folder in `results/`.
2. Restart the chatbot.
3. To share the adapter, zip the folder
   (`cd results && zip -r ../<run_name>.zip <run_name>`). The zip is stored with Git LFS when committed.

---

## 3. Running the chatbot

**Script:** `scripts/main.py`

Checklist: dependencies installed, adapter unzipped into `results/`, `.env` filled in, chunk files present in
`preprocessed-data/`.

```bash
python scripts/main.py                      # serve on http://0.0.0.0:8000
PORT=8080 python scripts/main.py            # use another port
SHARE=true python scripts/main.py           # also create a public *.gradio.live link
```

### What happens at startup

1. The server starts right away and the chat page can be opened immediately.
2. In a background thread, the chunks are loaded and embedded, then the base model is loaded in 4-bit, the LoRA
   adapter is merged in, and the model is compiled. On the first run the base model is downloaded from Hugging Face
   first. After that, loading takes about 30 seconds.
3. Until the backend is ready, the chat page shows the loading status and the input box is locked, and `/chat`
   returns `{"error": "Backend not ready: ..."}`. Check the admin panel's status box to see when it is ready.

Stop the server with `Ctrl+C`.

### URLs

| URL | Description |
|---|---|
| `http://localhost:8000/ui` | Gradio chat UI. Supports the commands `/help`, `/time` and `/echo <text>` |
| `POST http://localhost:8000/chat` | REST endpoint (see below) |
| `http://localhost:8000/admin` | Admin dashboard (log in with `ADMIN_USERNAME` / `ADMIN_PASSWORD`) |

With `SHARE=true` Gradio serves the app instead: the chat UI is at the root of the local and public links, and
`/admin` and `/chat` are available on both links. The admin URLs are printed at startup.

### REST API

The chat UI passes the conversation so far to the model, so follow-up questions work. The REST endpoint is
stateless: send earlier turns in the optional `history` field (the last 10 messages are used):

```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{
  "question": "Who can help me write one?",
  "history": [
    {"role": "user", "content": "What is a DMP?"},
    {"role": "assistant", "content": "A Data Management Plan is ..."}
  ]
}'
# -> {"response": "..."}   or   {"error": "Backend not ready: ..."} while the model is still loading
```

### Admin panel

`/admin` shows the server status (CPU, memory, uptime) and the chatbot status, active users, the recent chat
history, flagged requests and errors. A request is flagged when it contains one of the malicious keywords in
`admin.py`, or when the input/output safety patterns from `.env` block it. All chats are stored in `data/admin.db`.

### Running behind a reverse proxy

Set `HOST=127.0.0.1` and `FORWARDED_ALLOW_IPS` to the proxy's IP, so that the admin panel logs the users' real IP
addresses from the `X-Forwarded-For` header.

---

## Tests

```bash
pytest                      # all tests except the end-to-end tests (~20 s)
pytest -m "not network"     # offline only (~5 s)
pytest -m e2e               # end-to-end tests with the real model on the GPU (~1-2 min)
```

| File | What it checks | Needs |
|---|---|---|
| `tests/test_app.py` | Chat page opens, `/chat` answers, history reaches the model, "not ready" during startup, safety filters, Gradio chat function, admin login | nothing (uses a fake model) |
| `tests/test_pdf_utils.py` | PDF text extraction, `_clean` versions replacing originals, chunk caching and de-duplication; downloading the policy PDFs | internet for the download test |
| `tests/test_web_utils.py` | HTML text extraction, the per-start-URL crawl limit, chunk caching and de-duplication; scraping a live TU Delft page | internet for the live tests |
| `tests/test_vector_store.py` | Chunk-level cleaning (punctuation kept, soft hyphens, bullets) and chunking (sentence/paragraph splitting, sizes, overlap, tiny chunks) | nothing |
| `tests/test_chunks_quality.py` | The cached chunks in `preprocessed-data/` are sane (warns about short/oversized/duplicate chunks) | the `.pkl` files |
| `tests/test_e2e.py` | Starts the real app, opens the chat page, chats via REST and the Gradio UI, follow-up questions, blocked input | GPU, `.env`, adapter; only with `-m e2e` |

Markers (defined in `pytest.ini`): `network`, `data`, `e2e`.

---

## Known limitations

- **Sources without content.** Eight URLs in `WEB_URLS` currently give no text: "Research data services team"
  returns a 404; the six SharePoint pages (security and privacy) need a TU Delft login, which the crawler doesn't
  have; and the iRODS HackMD guide uses a page layout the TU Delft scraper doesn't recognise (the iRODS policy
  PDFs cover this topic).
- **Dutch pages.** The crawler follows links on the same domain, so a few Dutch versions of English pages end up in
  the knowledge base.
- **Training vs. chat prompt.** Fine-tuning prompts contain the Q&A's context only; in the chatbot the prompt also
  contains the list of source URLs and the conversation history.
- **Answers are sampled** (temperature 0.8), so the same question can get different answers, and the generation
  metrics of two identical training runs differ slightly.

---

## Legacy

`legacy/` contains scripts that are no longer used (see the project structure above). They are kept for reference
and are not updated for the current project structure, so they may not run.
