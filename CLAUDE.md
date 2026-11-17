# Dizzy – development phase 2 (branch `Kimiadev`)

Dizzy is a RAG chatbot for TU Delft Research Data Management: crawl/parse sources → chunk → embed (FAISS) → retrieve top-k → Ministral-3 3B (+ optional LoRA adapter) → FastAPI/Gradio UI + admin panel.

## Ground rules
- Work only on the `Kimiadev` branch (or `Kimiadev-*` sub-branches). Never commit to, merge into, rebase or push `main`. `main` is the working version people are testing.
- Never force-push, rewrite history, or delete tags. `v1-baseline` marks the current working version.
- Before changing code, explain the plan and wait for approval. Keep each commit to one kind of change, with a clear message.
- Don't delete files: move superseded code to `archive/` and say so.
- Never commit secrets (`.env`), source documents (`docs/`), built indexes, pickles or model adapters. Add them to `.gitignore` if needed.
- Run everything from the repo root.
- Confidential data (source PDFs, training Q&A, eval sets built from them) exists only on the company server. It never goes into the repo or into a Claude session.
- Laptop work uses public web sources and sample PDFs only.
- Never copy, deploy or push anything to the server. The owner moves code there themselves.

## This machine
- Laptop, no CUDA GPU. Use it for crawling, extraction, chunking, validation, indexing and retrieval evaluation (embedding model on CPU is fine).
- Don't judge answer quality here. The real model runs on the company server (16 GB GPU); LoRA training happens there too.
- Dependencies: `requirements-base.txt` (all machines) + `requirements-gpu.txt` (server only).

## Roadmap for this phase (do in order)
1. Get the project installing and running locally; split requirements.
2. Recover missing data: `docs/PRDW.pdf`, `docs/policies/`, `docs/train_qnas/*.jsonl` (ask the owner; don't fake them).
3. Rebuild ingestion: `sources.yaml` registry (type, URLs, crawl limit, `department`), a page budget per start URL, generic extraction, heading-aware chunks (200–500 tokens) of the original text, metadata (url, title, section, department, crawl date, content hash), stages saved as JSONL, persistent index, `ingest --source X [--rebuild]`.
4. Validation gates + a short report per ingest run.
5. Evaluation harness: gold Q&A per department, retrieval recall@k, answer correctness; `eval --dept X`; record a baseline.
6. Add the new department through the pipeline.
7. Improve retrieval and generation (hybrid search, reranker, chat template, history, citations).
8. Optional LoRA round with training prompts that match inference.
9. Clean up, README, security.

## Known bugs (from the repo review)
- `vector_store.py`: `clean_text` strips punctuation/stopwords before sentence splitting, so whole documents become one chunk.
- `web_crawling.py`: one shared `visited` set and 20-page budget, so only the first start URL gets crawled.
- `main.py` calls `generate_answer` without chat history.
- Inference prompt differs from training (URL list, history, double BOS, literal `</s>`, `[INST]` instead of the chat template).
- `finetune_new.py` truncates the end of the prompt (cuts the question); same split used for checkpoint choice and final metrics.

## Status / next session
Last updated 2026-11-17. Step 1 in progress; no project code changed yet.

### Checked so far
- `requirements.txt` is a full freeze from a Python 3.11+ environment. On 3.10 it fails to resolve (`numpy 2.3.5`, `scipy 1.16.3`, `networkx 3.5`, `contourpy 1.3.3` need 3.11+). On 3.11 it resolves unchanged (dry run, GPU packages excluded).
- `faiss-cpu 1.13.2` only has macOS 14+ wheels. For uv dry runs, set `MACOSX_DEPLOYMENT_TARGET=15.0` (the laptop runs macOS 15).
- Laptop Pythons: 3.11.9 at `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`; `uv` at `~/.local/bin/uv`. No venv created yet.
- torch and peft must be in base: `main.py` imports `mistral_model.py`, which imports both, and sentence-transformers needs torch to embed on CPU.
- Packages in the freeze that nothing in `scripts/` imports: `llama_cpp_python`, `diffusers`, `tyro`, `msgspec`, `ruff`, `hf_transfer`. `dotenv==0.9.9` is a shim for `python-dotenv`.
- Env vars the code reads: `HF_TOKEN`, `SYSTEM_PROMPT`, `SAFE_RESPONSE`, `FORBIDDEN_INPUT_PATTERNS`, `DISCLOSURE_OUTPUT_PATTERNS` (`mistral_model.py`); `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_DB_PATH` (`admin.py`); `HOST`, `PORT`, `SHARE`, `FORWARDED_ALLOW_IPS` (`main.py`). If `SYSTEM_PROMPT` is unset, `mistral_model.py` crashes on import (`None.replace`).
- Tracked files that must not be in the repo: `Ministral-3-3B-Instruct-2512-BF16-full-r16-test0.1.zip` (LoRA adapter, Git LFS) and `preprocessed-data/*.pkl`.

### Decisions
- Python 3.11, pins unchanged.
- Stop tracking the adapter zip and the pickles with `git rm --cached` (local files stay). The pickles contain text from confidential PDFs. Do this in its own commit and add them to `.gitignore`.
- No history rewrite. The files stay in past commits on `main` and `Kimiadev`.

### Remaining for step 1
1. Run `git lfs pull` and confirm the adapter zip is the real file, not an LFS pointer.
2. Split the requirements. Base = the transitive closure of the direct base packages at the freeze's pins (resolve with `uv pip compile -c requirements.txt` for macOS arm64 and linux x86_64). GPU = everything else (`-r requirements-base.txt` + CUDA packages + training packages + the unused ones above, commented). Move `requirements.txt` to `archive/`.
3. Write `.env.example` with every variable above, with no real values.
4. Update the README setup section (uv venv on 3.11, base vs GPU install, `cp .env.example .env`).
5. Make a separate commit that untracks the zip and pickles and updates `.gitignore`.
6. Create a `.venv` on 3.11, install the base requirements, and smoke-test: import every module and start `main.py`. Report the result; don't judge answer quality.
