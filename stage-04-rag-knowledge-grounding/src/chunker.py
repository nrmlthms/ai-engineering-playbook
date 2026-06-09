"""
Text chunking for RAG ingestion.

Why chunking matters
─────────────────────
Retrieval works at chunk granularity: a whole 50-page document cannot be a single
chunk (context window) and cannot be meaningfully embedded as one vector. Chunking
is the single biggest lever on RAG quality — more than the choice of embedding model.

Three strategies, each trading off context integrity vs. chunk predictability:

  Fixed-size (chunk_by_tokens)
    Split every N tokens regardless of sentence boundaries.
    Pro: predictable chunk sizes, simple implementation.
    Con: can cut mid-sentence, losing context at boundaries.

  Sliding window (chunk_by_tokens_sliding)
    Fixed-size chunks with O tokens of overlap between neighbours.
    Pro: facts that span chunk boundaries appear in two chunks → less retrieval miss.
    Con: more chunks, higher storage/embedding cost.

  Sentence-aware (chunk_by_sentences)
    Accumulate sentences until the chunk would exceed max_tokens.
    Pro: respects natural language boundaries.
    Con: variable chunk sizes, slightly more complex.

Rule of thumb: start with sliding window at chunk_size=512, overlap=64.
If you see many "chunk boundary" misses in eval, increase overlap or switch to
sentence-aware. Compare strategies empirically — see notebook 01.
"""

import re
from dataclasses import dataclass, field

import tiktoken

_ENCODING = "cl100k_base"  # good proxy for all current models


def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING)


@dataclass
class Chunk:
    text: str
    index: int        # 0-based position within the source document
    token_count: int
    doc_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def chunk_by_tokens(
    text: str,
    chunk_size: int = 512,
    doc_id: str = "",
) -> list[Chunk]:
    """Fixed-size token chunks. Simple baseline — no overlap."""
    enc = _enc()
    token_ids = enc.encode(text)
    if not token_ids:
        return []

    chunks: list[Chunk] = []
    for i in range(0, len(token_ids), chunk_size):
        window = token_ids[i : i + chunk_size]
        chunks.append(
            Chunk(
                text=enc.decode(window),
                index=len(chunks),
                token_count=len(window),
                doc_id=doc_id,
            )
        )
    return chunks


def chunk_by_tokens_sliding(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    doc_id: str = "",
) -> list[Chunk]:
    """Sliding window chunks. Overlap reduces context loss at boundaries."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    enc = _enc()
    token_ids = enc.encode(text)
    if not token_ids:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for i in range(0, len(token_ids), step):
        window = token_ids[i : i + chunk_size]
        if not window:
            break
        chunks.append(
            Chunk(
                text=enc.decode(window),
                index=len(chunks),
                token_count=len(window),
                doc_id=doc_id,
            )
        )
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation. Heuristic — good enough for English prose."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def chunk_by_sentences(
    text: str,
    max_tokens: int = 512,
    doc_id: str = "",
) -> list[Chunk]:
    """
    Group sentences into chunks not exceeding max_tokens.
    Preserves sentence boundaries — no sentence is ever split in half.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    enc = _enc()
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))

        if current_sentences and current_tokens + sentence_tokens > max_tokens:
            chunks.append(
                Chunk(
                    text=" ".join(current_sentences),
                    index=len(chunks),
                    token_count=current_tokens,
                    doc_id=doc_id,
                )
            )
            current_sentences = []
            current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(
            Chunk(
                text=" ".join(current_sentences),
                index=len(chunks),
                token_count=current_tokens,
                doc_id=doc_id,
            )
        )

    return chunks
