#!/usr/bin/env python3
"""
Quality checks for preprocessed PDF and web chunk pickle files.

Run from the scripts/ directory:
    python test_chunks_quality.py

Both pickle files must already exist (run data_preprocessing.py first).
"""

import os
import pickle
import sys
import config

MIN_CHUNK_LEN = 50       # chunks shorter than this are flagged as too short
MAX_CHUNK_LEN = 5000     # chunks longer than this are flagged as oversized
MIN_TOTAL_CHUNKS = 10    # sanity floor for each source


def load_pickle(path, label):
    if not os.path.exists(path):
        print(f"[{label}] MISSING: {path}")
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[{label}] Loaded {len(data)} chunks from {path}")
    return data


def check_chunks(chunks, label):
    print(f"\n{'=' * 60}")
    print(f"  Quality report: {label}")
    print(f"{'=' * 60}")

    if not chunks:
        print("  ERROR: chunk list is empty!")
        return False

    total = len(chunks)
    lengths = [len(c) for c in chunks]
    avg_len = sum(lengths) // total
    min_len = min(lengths)
    max_len = max(lengths)

    print(f"  Total chunks   : {total}")
    print(f"  Avg length     : {avg_len:,} chars")
    print(f"  Min length     : {min_len:,} chars")
    print(f"  Max length     : {max_len:,} chars")

    # --- individual checks ---
    passed = True

    if total < MIN_TOTAL_CHUNKS:
        print(f"  FAIL: only {total} chunks (expected >= {MIN_TOTAL_CHUNKS})")
        passed = False
    else:
        print(f"  PASS: chunk count ({total}) >= {MIN_TOTAL_CHUNKS}")

    empty = [i for i, c in enumerate(chunks) if not c or not c.strip()]
    if empty:
        print(f"  FAIL: {len(empty)} empty/whitespace-only chunk(s) at indices {empty[:5]}")
        passed = False
    else:
        print(f"  PASS: no empty chunks")

    too_short = [i for i, c in enumerate(chunks) if 0 < len(c.strip()) < MIN_CHUNK_LEN]
    if too_short:
        print(f"  WARN: {len(too_short)} chunk(s) shorter than {MIN_CHUNK_LEN} chars "
              f"(first 5 indices: {too_short[:5]})")
    else:
        print(f"  PASS: all chunks >= {MIN_CHUNK_LEN} chars")

    oversized = [i for i, c in enumerate(chunks) if len(c) > MAX_CHUNK_LEN]
    if oversized:
        print(f"  WARN: {len(oversized)} chunk(s) longer than {MAX_CHUNK_LEN} chars "
              f"(first 5 indices: {oversized[:5]})")
    else:
        print(f"  PASS: no oversized chunks (threshold {MAX_CHUNK_LEN} chars)")

    seen = set()
    dupes = []
    for i, c in enumerate(chunks):
        if c in seen:
            dupes.append(i)
        seen.add(c)
    if dupes:
        print(f"  WARN: {len(dupes)} duplicate chunk(s) found (first 5 indices: {dupes[:5]})")
    else:
        print(f"  PASS: no duplicate chunks")

    # --- length distribution buckets ---
    buckets = {"< 100": 0, "100-500": 0, "500-1000": 0, "1000-2000": 0, "> 2000": 0}
    for ln in lengths:
        if ln < 100:
            buckets["< 100"] += 1
        elif ln < 500:
            buckets["100-500"] += 1
        elif ln < 1000:
            buckets["500-1000"] += 1
        elif ln < 2000:
            buckets["1000-2000"] += 1
        else:
            buckets["> 2000"] += 1
    print("\n  Length distribution:")
    for bucket, count in buckets.items():
        bar = "#" * (count * 30 // max(total, 1))
        print(f"    {bucket:>12}  {count:4d}  {bar}")

    # --- content spot-check ---
    print("\n  Content preview (first chunk):")
    print("  " + "-" * 56)
    preview = chunks[10].replace("\n", " ")
    print(f"  {preview}")
    print("  " + "-" * 56)

    return passed


def main():
    all_passed = True

    pdf_chunks = load_pickle(config.PDF_CHUNKS_PATH, "PDF chunks")
    web_chunks = load_pickle(config.WEB_CHUNKS_PATH, "Web chunks")

    if pdf_chunks is not None:
        ok = check_chunks(pdf_chunks, "PDF chunks")
        all_passed = all_passed and ok
    else:
        all_passed = False

    if web_chunks is not None:
        ok = check_chunks(web_chunks, "Web chunks")
        all_passed = all_passed and ok
    else:
        all_passed = False

    print(f"\n{'=' * 60}")
    if all_passed:
        print("  Overall: ALL CHECKS PASSED")
    else:
        print("  Overall: SOME CHECKS FAILED — review output above")
    print(f"{'=' * 60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

