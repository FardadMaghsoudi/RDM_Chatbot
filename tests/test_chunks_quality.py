"""
Quality checks for the preprocessed PDF and web chunk pickle files.

The pickle files must already exist (run scripts/ingestion/data_preprocessing.py first);
otherwise the tests are skipped.
"""

import os
import pickle
import warnings

import pytest

import config

pytestmark = pytest.mark.data

MIN_CHUNK_LEN = 50       # chunks shorter than this are flagged as too short
MAX_CHUNK_LEN = 5000     # chunks longer than this are flagged as oversized
MIN_TOTAL_CHUNKS = 10    # sanity floor for each source

CHUNK_FILES = {
    "pdf": config.PDF_CHUNKS_PATH,
    "web": config.WEB_CHUNKS_PATH,
}


@pytest.fixture(params=list(CHUNK_FILES), ids=list(CHUNK_FILES))
def chunks(request):
    path = CHUNK_FILES[request.param]
    if not os.path.exists(path):
        pytest.skip(f"{path} not found - run data_preprocessing.py first")
    with open(path, "rb") as f:
        return pickle.load(f)


def test_chunks_are_list_of_strings(chunks):
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)


def test_minimum_chunk_count(chunks):
    assert len(chunks) >= MIN_TOTAL_CHUNKS


def test_no_empty_chunks(chunks):
    empty = [i for i, c in enumerate(chunks) if not c.strip()]
    assert not empty, f"{len(empty)} empty/whitespace-only chunk(s) at indices {empty[:5]}"


def test_chunk_length_and_duplicates(chunks):
    """Soft checks: reported as warnings, they do not fail the test."""
    too_short = [i for i, c in enumerate(chunks) if 0 < len(c.strip()) < MIN_CHUNK_LEN]
    oversized = [i for i, c in enumerate(chunks) if len(c) > MAX_CHUNK_LEN]
    duplicates = len(chunks) - len(set(chunks))

    if too_short:
        warnings.warn(f"{len(too_short)} chunk(s) shorter than {MIN_CHUNK_LEN} chars (first: {too_short[:5]})")
    if oversized:
        warnings.warn(f"{len(oversized)} chunk(s) longer than {MAX_CHUNK_LEN} chars (first: {oversized[:5]})")
    if duplicates:
        warnings.warn(f"{duplicates} duplicate chunk(s)")
