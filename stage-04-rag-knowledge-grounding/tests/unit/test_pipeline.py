"""
Pipeline tests — offline only.
format_context_block and Document are fully testable.
build_rag_prompt raises NotImplementedError until Exercise 1 is complete.
"""

import pytest
from pipeline import Document, build_rag_prompt, format_context_block
from store import SearchResult


def _result(
    doc_id: str = "doc.md", text: str = "Some content.", score: float = 0.9
) -> SearchResult:
    return SearchResult(text=text, score=score, doc_id=doc_id, rank=1)


# ── format_context_block ──────────────────────────────────────────────────────


def test_format_block_includes_index() -> None:
    block = format_context_block(_result(), index=1)
    assert "[1]" in block


def test_format_block_includes_doc_id() -> None:
    block = format_context_block(_result(doc_id="readme.md"), index=1)
    assert "readme.md" in block


def test_format_block_includes_text() -> None:
    block = format_context_block(_result(text="The quick brown fox."), index=2)
    assert "The quick brown fox." in block


def test_format_block_no_doc_id_still_has_index() -> None:
    r = SearchResult(text="text", score=0.9, doc_id="", rank=1)
    block = format_context_block(r, index=3)
    assert "[3]" in block


def test_format_block_different_indices() -> None:
    r = _result()
    assert "[1]" in format_context_block(r, index=1)
    assert "[5]" in format_context_block(r, index=5)


# ── Document ──────────────────────────────────────────────────────────────────


def test_document_fields() -> None:
    doc = Document(text="content", doc_id="test.md")
    assert doc.text == "content"
    assert doc.doc_id == "test.md"
    assert doc.metadata == {}


def test_document_with_metadata() -> None:
    doc = Document(text="content", doc_id="test.md", metadata={"source": "web"})
    assert doc.metadata["source"] == "web"


# ── build_rag_prompt (stub) ───────────────────────────────────────────────────


def test_build_rag_prompt_stub_raises() -> None:
    # Confirms the stub is in place — implement in Exercise 1
    with pytest.raises(NotImplementedError):
        build_rag_prompt("query", [_result()])


def test_build_rag_prompt_stub_raises_with_empty_results() -> None:
    with pytest.raises(NotImplementedError):
        build_rag_prompt("query", [])
