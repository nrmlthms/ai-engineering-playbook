# ruff: noqa: F704, E402
# %% [markdown]
# # 02 — Embeddings and Vector Search
#
# What embeddings are, how similarity works, and how ChromaDB finds nearest neighbours.
# Section 1 is offline. Sections 2–3 require the sentence-transformers model
# (downloads ~90 MB on first use). Section 4 requires OPENAI_API_KEY.

# %%
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))

from embedder import DeterministicEmbedder

# %% [markdown]
# ## 1. Cosine similarity — what "close" means in embedding space
#
# Two vectors are similar if they point in the same direction, regardless of magnitude.
# Cosine similarity = dot_product / (|a| × |b|)
#   1.0 = identical direction (same meaning)
#   0.0 = orthogonal (unrelated)
#  -1.0 = opposite direction (antonyms — rare in practice)


# %%
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


# Demo: DeterministicEmbedder has no semantic meaning — just for illustration
e = DeterministicEmbedder(dimension=128)
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy hound.",  # paraphrase
    "Machine learning is a subset of artificial intelligence.",  # unrelated
]

vecs = e.embed(texts)
print(f"{'Pair':<55} {'Cosine':>7}")
print("-" * 65)
pairs = [
    (0, 1, "original vs paraphrase"),
    (0, 2, "original vs unrelated"),
    (1, 2, "paraphrase vs unrelated"),
]
for i, j, label in pairs:
    sim = cosine(vecs[i], vecs[j])
    print(f"  {label:<52} {sim:>7.4f}")

print("\n(Note: DeterministicEmbedder is hash-based — no semantic meaning.)")
print("Real embeddings would show original≈paraphrase >> original vs unrelated.")

# %% [markdown]
# ## 2. SentenceTransformer embeddings (downloads ~90 MB on first use)


# %%
def demo_semantic_similarity() -> None:
    from embedder import SentenceTransformerEmbedder

    st = SentenceTransformerEmbedder()
    print(f"Model dimension: {st.dimension}")

    pairs_text = [
        ("The sky is blue.", "The sky has a blue colour.", "near-duplicate"),
        ("I love pizza.", "Pizza is my favourite food.", "paraphrase"),
        ("The stock market fell today.", "Machine learning models need data.", "unrelated"),
        ("Paris is the capital of France.", "France's capital city is Paris.", "near-duplicate"),
    ]

    print(f"\n{'Pair':<60} {'Type':<15} {'Sim':>6}")
    print("-" * 85)
    for a, b, kind in pairs_text:
        va, vb = st.embed([a, b])
        sim = cosine(va, vb)
        print(f"  {a[:35]!r:<37} / {b[:20]!r:<22} {kind:<15} {sim:>6.3f}")


# Uncomment to run (downloads model on first call):
# demo_semantic_similarity()
print("(Cell ready — uncomment to run; downloads ~90 MB on first use)")

# %% [markdown]
# ## 3. Building a vector store and querying it


# %%
def demo_vector_search() -> None:
    from embedder import SentenceTransformerEmbedder
    from retriever import Retriever
    from store import VectorStore

    embedder = SentenceTransformerEmbedder()
    store = VectorStore()  # in-memory (EphemeralClient)

    # Index some AI engineering facts
    docs = [
        ("rag-001", "RAG combines retrieval with generation to reduce hallucination."),
        ("rag-002", "Chunking strategy matters more for RAG quality than embedding model choice."),
        ("rag-003", "HNSW is an approximate nearest-neighbour algorithm used by most vector DBs."),
        ("llm-001", "Temperature controls randomness: lower = more deterministic output."),
        ("llm-002", "Prompt caching reduces cost by reusing previously computed KV cache."),
        ("eval-001", "Recall@k: fraction of relevant documents found in top-k results."),
        ("eval-002", "MRR (Mean Reciprocal Rank): rewards finding the first relevant result."),
    ]

    ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    store.add(ids=ids, embeddings=embedder.embed(texts), texts=texts)

    # Query
    queries = [
        "How does RAG reduce hallucination?",
        "What metrics evaluate retrieval quality?",
        "What affects performance in a RAG system?",
    ]

    for query in queries:
        retriever = Retriever(store=store, embedder=embedder)
        results = retriever.retrieve(query, top_k=3)
        print(f"\nQuery: {query!r}")
        for r in results:
            print(f"  [{r.rank}] {r.score:.3f}  [{r.doc_id}]  {r.text[:70]}")


# Uncomment to run:
# demo_vector_search()
print("(Cell ready — uncomment to run)")

# %% [markdown]
# ## 4. OpenAI embeddings vs local (requires OPENAI_API_KEY)
#
# text-embedding-3-small: 1536 dims, ~$0.02 / 1M tokens
# all-MiniLM-L6-v2:       384 dims, free (local inference)
#
# Quality difference is small for English text. Use local for cost-sensitive
# applications; OpenAI for multilingual or highest-quality requirements.


# %%
def compare_embedders() -> None:
    from embedder import OpenAIEmbedder, SentenceTransformerEmbedder

    test_pairs = [
        ("Dogs are mammals.", "Cats are mammals."),  # related
        ("Dogs are mammals.", "The stock market fell."),  # unrelated
    ]

    for embedder_cls, label in [
        (SentenceTransformerEmbedder, "MiniLM-L6-v2"),
        (OpenAIEmbedder, "text-embedding-3-small"),
    ]:
        try:
            emb = embedder_cls()
            print(f"\n{label} (dim={emb.dimension}):")
            for a, b in test_pairs:
                va, vb = emb.embed([a, b])
                sim = cosine(va, vb)
                print(f"  {a[:30]!r} ≈ {b[:30]!r}  →  {sim:.3f}")
        except Exception as exc:
            print(f"  {label}: skipped ({exc})")


# Uncomment to run (requires OPENAI_API_KEY for OpenAIEmbedder):
# compare_embedders()
print("(Cell ready — uncomment to run)")
