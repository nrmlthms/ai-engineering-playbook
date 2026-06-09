"""
Chunker tests — no network, no LLM calls.
"""

import pytest
from chunker import Chunk, chunk_by_sentences, chunk_by_tokens, chunk_by_tokens_sliding

LONG_TEXT = " ".join([f"Sentence number {i} is here." for i in range(80)])
SHORT_TEXT = "Hello world. This is a test."


# ── chunk_by_tokens ───────────────────────────────────────────────────────────


def test_chunk_by_tokens_returns_chunks() -> None:
    chunks = chunk_by_tokens(LONG_TEXT, chunk_size=50)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_by_tokens_indices_are_sequential() -> None:
    chunks = chunk_by_tokens(LONG_TEXT, chunk_size=50)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_by_tokens_respects_chunk_size() -> None:
    chunks = chunk_by_tokens(LONG_TEXT, chunk_size=50)
    for c in chunks[:-1]:  # last chunk may be smaller
        assert c.token_count <= 50


def test_chunk_by_tokens_preserves_doc_id() -> None:
    chunks = chunk_by_tokens(LONG_TEXT, chunk_size=50, doc_id="readme.md")
    assert all(c.doc_id == "readme.md" for c in chunks)


def test_chunk_by_tokens_empty_text() -> None:
    assert chunk_by_tokens("", chunk_size=100) == []


def test_chunk_by_tokens_short_text_single_chunk() -> None:
    chunks = chunk_by_tokens(SHORT_TEXT, chunk_size=100)
    assert len(chunks) == 1


# ── chunk_by_tokens_sliding ───────────────────────────────────────────────────


def test_chunk_sliding_produces_more_chunks_than_fixed() -> None:
    fixed = chunk_by_tokens(LONG_TEXT, chunk_size=50)
    sliding = chunk_by_tokens_sliding(LONG_TEXT, chunk_size=50, overlap=10)
    assert len(sliding) >= len(fixed)


def test_chunk_sliding_token_counts_bounded() -> None:
    chunks = chunk_by_tokens_sliding(LONG_TEXT, chunk_size=50, overlap=10)
    for c in chunks[:-1]:
        assert c.token_count <= 50


def test_chunk_sliding_overlap_equal_to_size_raises() -> None:
    with pytest.raises(ValueError):
        chunk_by_tokens_sliding(LONG_TEXT, chunk_size=50, overlap=50)


def test_chunk_sliding_overlap_larger_than_size_raises() -> None:
    with pytest.raises(ValueError):
        chunk_by_tokens_sliding(LONG_TEXT, chunk_size=50, overlap=60)


def test_chunk_sliding_preserves_doc_id() -> None:
    chunks = chunk_by_tokens_sliding(LONG_TEXT, chunk_size=50, overlap=10, doc_id="doc.txt")
    assert all(c.doc_id == "doc.txt" for c in chunks)


def test_chunk_sliding_empty_text() -> None:
    assert chunk_by_tokens_sliding("", chunk_size=50, overlap=10) == []


# ── chunk_by_sentences ────────────────────────────────────────────────────────


def test_chunk_by_sentences_returns_chunks() -> None:
    chunks = chunk_by_sentences(LONG_TEXT, max_tokens=80)
    assert len(chunks) > 0


def test_chunk_by_sentences_token_counts_bounded() -> None:
    chunks = chunk_by_sentences(LONG_TEXT, max_tokens=80)
    for c in chunks:
        # allow one sentence's worth of overage on the last-added sentence
        assert c.token_count <= 100


def test_chunk_by_sentences_covers_all_sentences() -> None:
    chunks = chunk_by_sentences(LONG_TEXT, max_tokens=80)
    combined = " ".join(c.text for c in chunks)
    for i in range(0, 10):
        assert f"Sentence number {i}" in combined


def test_chunk_by_sentences_indices_sequential() -> None:
    chunks = chunk_by_sentences(LONG_TEXT, max_tokens=80)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_by_sentences_empty_text() -> None:
    assert chunk_by_sentences("") == []


def test_chunk_by_sentences_single_sentence() -> None:
    chunks = chunk_by_sentences("Just one sentence here.", max_tokens=100)
    assert len(chunks) == 1
    assert "Just one sentence here." in chunks[0].text
