"""
Retriever tests — metric functions are pure (no store needed);
retrieval tests use DeterministicEmbedder + ChromaDB EphemeralClient (no server).
"""

import pytest
from embedder import DeterministicEmbedder
from retriever import Retriever, mean_reciprocal_rank, recall_at_k
from store import VectorStore

# ── Pure metric functions ─────────────────────────────────────────────────────


def test_recall_perfect() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_recall_partial() -> None:
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_zero() -> None:
    assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0


def test_recall_at_k_respects_k_limit() -> None:
    # relevant doc "a" is rank 2, not in top-1
    assert recall_at_k(["x", "a"], {"a"}, k=1) == 0.0


def test_recall_at_k_relevant_in_top_k() -> None:
    assert recall_at_k(["x", "a"], {"a"}, k=2) == 1.0


def test_recall_empty_relevant() -> None:
    assert recall_at_k(["a", "b"], set(), k=3) == 0.0


def test_mrr_first_hit() -> None:
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_hit() -> None:
    assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)


def test_mrr_third_hit() -> None:
    assert mean_reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_mrr_no_hit() -> None:
    assert mean_reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


def test_mrr_empty_relevant() -> None:
    assert mean_reciprocal_rank(["a", "b"], set()) == 0.0


# ── Retriever with in-memory ChromaDB ─────────────────────────────────────────

DOCS = [
    ("doc0", "The sky is blue."),
    ("doc1", "Cats are mammals."),
    ("doc2", "Python is a programming language."),
    ("doc3", "The ocean is vast and deep."),
]


@pytest.fixture
def populated_retriever() -> Retriever:
    embedder = DeterministicEmbedder(dimension=64)
    store = VectorStore()  # EphemeralClient — in-memory
    ids = [d[0] for d in DOCS]
    texts = [d[1] for d in DOCS]
    store.add(ids=ids, embeddings=embedder.embed(texts), texts=texts)
    return Retriever(store=store, embedder=embedder)


def test_retrieve_returns_requested_count(populated_retriever: Retriever) -> None:
    results = populated_retriever.retrieve("query text", top_k=2)
    assert len(results) == 2


def test_retrieve_scores_in_valid_range(populated_retriever: Retriever) -> None:
    for r in populated_retriever.retrieve("query", top_k=4):
        assert -1.0 <= r.score <= 1.0


def test_retrieve_ranks_are_sequential(populated_retriever: Retriever) -> None:
    results = populated_retriever.retrieve("query", top_k=3)
    assert [r.rank for r in results] == [1, 2, 3]


def test_retrieve_all_doc_ids_present(populated_retriever: Retriever) -> None:
    results = populated_retriever.retrieve("query", top_k=4)
    assert {r.doc_id for r in results} == {"doc0", "doc1", "doc2", "doc3"}


def test_retrieve_top_k_capped_at_store_size(populated_retriever: Retriever) -> None:
    results = populated_retriever.retrieve("query", top_k=100)
    assert len(results) == len(DOCS)


def test_retrieve_empty_store_returns_empty() -> None:
    embedder = DeterministicEmbedder(dimension=64)
    store = VectorStore()
    retriever = Retriever(store=store, embedder=embedder)
    assert retriever.retrieve("anything") == []
