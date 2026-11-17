# Dizzy – development phase 2 (branch `Kimiadev`)

Dizzy is a RAG chatbot for TU Delft Research Data Management: crawl/parse sources → chunk → embed (FAISS) → retrieve top-k → Ministral-3 3B (+ optional LoRA adapter) → FastAPI/Gradio UI + admin panel.

## Ground rules
- Work only on the `Kimiadev` branch (or `Kimiadev-*` sub-branches). Never commit to, merge into, rebase or push `main`. `main` is the working version people are testing.
- Never force-push, rewrite history, or delete tags. `v1-baseline` marks the current working version.
- Before changing code, explain the plan and wait for approval. Keep each commit to one kind of change, with a clear message.
- Don't delete files: move superseded code to `archive/` and say so.
- Never commit secrets (`.env`), source documents (`docs/`), built indexes, pickles or model adapters. Add them to `.gitignore` if needed.
- Run everything from the repo root.

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
