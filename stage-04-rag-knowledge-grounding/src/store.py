"""
Vector store — ChromaDB-backed retrieval.

What is HNSW?
─────────────
Hierarchical Navigable Small World (HNSW) is the graph-based ANN index used
by ChromaDB (and most production vector DBs). Instead of brute-force cosine
similarity over all N vectors (O(N)), HNSW builds a multi-layer graph and
navigates it in O(log N) — fast enough for millions of vectors.

The trade-off: HNSW is approximate. It may miss the true nearest neighbour.
You tune recall vs. speed with `hnsw:ef_search` (higher = more accurate, slower).
For <10k vectors, the difference is negligible; brute force would also be fast.

Cosine space
────────────
ChromaDB stores cosine *distance* (0 = identical, 1 = orthogonal).
We convert to similarity: score = 1 − distance, so 1 = identical.

In-memory vs. persistent
────────────────────────
  VectorStore()                     EphemeralClient — in-memory, lost on process exit
  VectorStore(persist_path="./db")  PersistentClient — survives restarts
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb


@dataclass
class SearchResult:
    text: str
    score: float  # cosine similarity: 1 = identical, 0 = orthogonal
    doc_id: str
    rank: int  # 1-based rank in result list
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(
        self,
        collection_name: str | None = None,
        persist_path: str | None = None,
    ) -> None:
        # Default to a UUID so each VectorStore() gets an isolated collection.
        # ChromaDB's EphemeralClient is a process-level singleton; sharing a
        # collection name across instances would cause test cross-contamination.
        name = collection_name or str(uuid.uuid4())
        if persist_path:
            self._client: chromadb.ClientAPI = chromadb.PersistentClient(path=persist_path)
        else:
            self._client = chromadb.EphemeralClient()

        self._col = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents with pre-computed embeddings."""
        self._col.add(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    def query(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return top-k results for a query embedding, sorted by similarity."""
        n = min(top_k, self._col.count())
        if n == 0:
            return []

        raw = self._col.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=n,
        )

        ids: list[str] = raw["ids"][0]
        docs: list[str] = raw["documents"][0]  # type: ignore[index]
        distances: list[float] = raw["distances"][0]  # type: ignore[index]
        metas: list[dict[str, Any]] = raw["metadatas"][0]  # type: ignore[index]

        return [
            SearchResult(
                text=doc,
                score=1.0 - dist,  # distance → similarity
                doc_id=doc_id,
                rank=i + 1,
                metadata=meta or {},
            )
            for i, (doc_id, doc, dist, meta) in enumerate(zip(ids, docs, distances, metas))
        ]

    def count(self) -> int:
        return self._col.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(self._col.name)
