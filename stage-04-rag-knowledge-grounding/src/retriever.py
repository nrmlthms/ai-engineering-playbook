"""
Retrieval metrics and the Retriever class.

Retrieval evaluation metrics
──────────────────────────────
Given a query, you have a set of "relevant" documents (ground truth) and a ranked
list of retrieved documents. Key metrics:

  Recall@k
    What fraction of all relevant documents appear in the top-k results?
    Perfect retrieval = 1.0. Missing all relevant docs = 0.0.
    Use when you want to maximise coverage (don't miss relevant docs).

    recall@k = |retrieved_top_k ∩ relevant| / |relevant|

  MRR (Mean Reciprocal Rank)
    How early does the FIRST relevant document appear?
    Rank 1 → 1.0, Rank 2 → 0.5, Rank 3 → 0.33, ...
    Use when you want the first result to be relevant (single-answer tasks).

    RR = 1 / rank_of_first_relevant
    MRR = mean(RR) over all queries in your eval set

  Both are computed over a list of retrieved doc IDs vs. a set of known-relevant IDs.
  Test your retriever on a held-out eval set with labeled (query, relevant_ids) pairs.
"""

from embedder import Embedder
from store import SearchResult, VectorStore


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k retrieved results."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant result. 0.0 if no relevant result is found."""
    if not relevant_ids:
        return 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


class Retriever:
    """Embeds a query and searches the vector store."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Embed query and return top-k results from the vector store."""
        embedding = self.embedder.embed_one(query)
        return self.store.query(embedding, top_k=top_k)

    def recall_at_k(self, query: str, relevant_ids: list[str], k: int = 5) -> float:
        """Convenience wrapper: retrieve then compute recall@k."""
        results = self.retrieve(query, top_k=k)
        return recall_at_k([r.doc_id for r in results], set(relevant_ids), k)

    def mrr(self, query: str, relevant_ids: list[str]) -> float:
        """Convenience wrapper: retrieve then compute reciprocal rank."""
        results = self.retrieve(query, top_k=max(len(relevant_ids) * 2, 10))
        return mean_reciprocal_rank([r.doc_id for r in results], set(relevant_ids))
