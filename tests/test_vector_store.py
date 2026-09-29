"""Tests for chunk-level text cleaning and sentence-based chunking (rag/vector_store.py)."""

from rag.vector_store import clean_text, split_into_sentences, split_text_by_sentences


def test_clean_text_keeps_punctuation_and_stopwords():
    text = "Research data must be stored securely.\nIt is the researcher's responsibility!"
    assert clean_text(text) == "Research data must be stored securely. It is the researcher's responsibility!"


def test_clean_text_normalises_unicode_and_whitespace():
    assert clean_text("  data ﬁles \n\n in   storage ") == "data files in storage"


def test_clean_text_replaces_soft_hyphens_and_symbol_bullets():
    assert clean_text("Research\u00addata\u00admanagement") == "Research data management"
    assert clean_text("\uf0b7 Store data safely") == "• Store data safely"


def test_sentences_split_on_punctuation_and_paragraph_breaks():
    text = "First sentence. Second one? Third!\n\nA list item without a full stop\n\nAnother item"
    assert split_into_sentences(text) == [
        "First sentence.", "Second one?", "Third!", "A list item without a full stop", "Another item",
    ]


def test_pdf_line_wraps_do_not_end_a_sentence():
    assert split_into_sentences("This sentence is wrapped\nover two lines.") == ["This sentence is wrapped over two lines."]


def test_page_break_marker_is_not_in_chunks():
    text = "End of page one.\n\n--- PAGE BREAK ---\n\nStart of page two."
    assert split_into_sentences(text) == ["End of page one.", "Start of page two."]


def test_chunks_respect_target_size_and_overlap():
    sentences = [f"Sentence number {i} about research data management." for i in range(200)]
    chunks = split_text_by_sentences(" ".join(sentences), target_chunk_size=500, overlap_size=100)

    assert len(chunks) > 1
    assert all(len(c) <= 600 for c in chunks)
    # consecutive chunks share their boundary sentence(s)
    assert chunks[0].split(". ")[-1] in chunks[1]
    # no content is lost
    assert all(any(s in c for c in chunks) for s in sentences)


def test_very_long_sentence_is_split_at_word_boundaries():
    words = [f"word{i}" for i in range(2000)]
    chunks = split_text_by_sentences(" ".join(words), target_chunk_size=500, overlap_size=0)

    assert len(chunks) > 1
    # a short tail (< min_chunk_size=50) is merged into the last chunk instead of being dropped
    assert all(len(c) <= 500 + 50 for c in chunks)
    assert " ".join(chunks).split() == words


def test_tiny_chunks_are_dropped():
    assert split_text_by_sentences("Filter by:") == []
    assert split_text_by_sentences("Filter by:", min_chunk_size=0) == ["Filter by:"]
