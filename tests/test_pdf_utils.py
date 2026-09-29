"""Tests for PDF downloading, text extraction and chunk caching (ingestion/pdf_utils.py)."""

import os

import pytest

import config
from ingestion.pdf_utils import download_pdfs_from_webpage, load_all_pdfs, load_pdf_text, save_or_load_pdf_chunks


def simple_split_text(text, chunk_size=500):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks or [""]


def test_load_pdf_text_extracts_text(tmp_path, make_pdf):
    pdf = make_pdf(tmp_path / "policy.pdf", ["Research data must be archived for ten years."])
    assert "archived for ten years" in load_pdf_text(str(pdf))


def test_save_or_load_pdf_chunks_creates_and_reloads_pickle(tmp_path, make_pdf):
    pdf_folder = tmp_path / "policies"
    pdf_folder.mkdir()
    make_pdf(pdf_folder / "policy.pdf", ["Faculty policy on research data storage."])
    prdw_path = make_pdf(tmp_path / "PRDW.pdf", ["Personal research data workflow guide."])
    pdf_chunks_path = tmp_path / "pdf_chunks.pkl"

    # First call extracts the policy PDFs + the PRDW guide and writes the pickle
    chunks_first = save_or_load_pdf_chunks(str(pdf_chunks_path), str(pdf_folder), simple_split_text, str(prdw_path))
    assert pdf_chunks_path.exists()
    text = " ".join(chunks_first)
    assert "research data storage" in text
    assert "workflow guide" in text

    # Second call must load straight from the pickle, without touching the PDFs
    for pdf in [*pdf_folder.iterdir(), prdw_path]:
        pdf.unlink()
    chunks_second = save_or_load_pdf_chunks(str(pdf_chunks_path), str(pdf_folder), simple_split_text, str(prdw_path))
    assert chunks_first == chunks_second


def test_clean_version_replaces_original_pdf(tmp_path, make_pdf):
    pdf_folder = tmp_path / "policies"
    pdf_folder.mkdir()
    make_pdf(pdf_folder / "Policy.pdf", ["Original version with a cover page."])
    make_pdf(pdf_folder / "Policy_clean.pdf", ["Cleaned version without the cover page."])
    make_pdf(pdf_folder / "Other.pdf", ["A policy that has no clean version."])
    prdw_path = make_pdf(tmp_path / "PRDW.pdf", ["Personal research data workflow guide."])

    text = " ".join(load_all_pdfs(str(pdf_folder), str(prdw_path)))

    assert "Cleaned version" in text
    assert "Original version" not in text
    assert "has no clean version" in text
    assert "workflow guide" in text


def test_save_or_load_pdf_chunks_removes_duplicate_chunks(tmp_path, make_pdf):
    pdf_folder = tmp_path / "policies"
    pdf_folder.mkdir()
    make_pdf(pdf_folder / "a.pdf", ["Identical text in two documents."])
    make_pdf(pdf_folder / "b.pdf", ["Identical text in two documents."])
    prdw_path = make_pdf(tmp_path / "PRDW.pdf", ["Workflow guide."])

    chunks = save_or_load_pdf_chunks(str(tmp_path / "chunks.pkl"), str(pdf_folder), lambda text: [text], str(prdw_path))
    assert chunks.count("Identical text in two documents.") == 1


@pytest.mark.network
def test_download_and_extract_policy_pdfs(tmp_path):
    download_folder = tmp_path / "policies"

    result_folder = download_pdfs_from_webpage(config.POLICIES_URL, str(download_folder))
    assert result_folder == str(download_folder)

    pdf_files = [f for f in os.listdir(download_folder) if f.endswith(".pdf")]
    assert pdf_files, "no PDFs were downloaded"

    failed = []
    for pdf_file in pdf_files:
        try:
            text = load_pdf_text(os.path.join(download_folder, pdf_file))
        except Exception as e:
            failed.append(f"{pdf_file}: {type(e).__name__}: {e}")
            continue
        if not text.strip():
            failed.append(f"{pdf_file}: no text extracted")
    assert not failed, "\n".join(failed)
