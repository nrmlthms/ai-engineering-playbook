# Stage 04 — RAG & Knowledge Grounding

> Build a retrieval pipeline that grounds LLM answers in your own documents —
> and evaluate whether it actually works.

---

## What you build

A production-grade RAG stack (`src/`) with three chunking strategies, a protocol-based
embedder abstraction, a ChromaDB-backed vector store, retrieval metrics, and an
end-to-end pipeline that ingests documents and answers questions grounded in them.

---

## Concepts

### Why RAG?

LLMs have two knowledge problems: their training data has a cutoff date, and they
hallucinate when asked about things they haven't seen. RAG solves both by fetching
relevant text at query time and instructing the model to answer from that text.

```
Documents → chunk → embed → vector DB
                                ↓
Query → embed → ANN search → top-k chunks → grounding prompt → LLM → answer
                                       ↓ (optional)
                                   reranker
```

---

### Chunking

Chunking is the single biggest lever on RAG quality — more than embedding model choice.

| Strategy      | How it works                          | Pros                   | Cons                          |
|---------------|---------------------------------------|------------------------|-------------------------------|
| Fixed-size    | Split every N tokens                  | Predictable, simple    | Can cut mid-sentence          |
| Sliding window| Fixed-size + O tokens of overlap      | Recovers boundary facts| More chunks, higher cost      |
| Sentence-aware| Accumulate sentences ≤ max_tokens     | Natural boundaries     | Variable chunk sizes          |

**Rule of thumb:** start with sliding window at `chunk_size=512, overlap=64`.

---

### Embeddings

A dense embedding maps text to a point in high-dimensional space. Similar texts →
similar directions → high cosine similarity.

| Embedder                      | Dimension | Cost         | Use when                    |
|-------------------------------|-----------|--------------|-----------------------------|
| `DeterministicEmbedder`       | any       | Free (hash)  | Offline unit tests only     |
| `SentenceTransformerEmbedder` | 384       | Free (local) | No API budget; local GPU    |
| `OpenAIEmbedder`              | 1536/3072 | ~$0.02/M tok | Highest quality; budget ok  |

**Hybrid search** (dense + BM25) outperforms either alone: dense handles semantic
similarity; BM25 handles exact technical terms.

---

### Vector store (ChromaDB)

ChromaDB uses HNSW for approximate nearest-neighbour search in O(log N).

```python
from store import VectorStore

store = VectorStore()                           # in-memory (tests, prototypes)
store = VectorStore(persist_path="./chroma_db") # persistent (production)

store.add(ids=["doc1"], embeddings=[vec], texts=["content"])
results = store.query(query_vec, top_k=5)
# result.score: cosine similarity (1 = identical, 0 = orthogonal)
```

---

### Retrieval metrics

Evaluate retrieval **separately** from generation — a bad retriever can never be
fixed by a better prompt.

```
Recall@k = |retrieved_top_k ∩ relevant| / |relevant|
MRR      = 1 / rank_of_first_relevant
```

```python
from retriever import recall_at_k, mean_reciprocal_rank

retrieved = ["doc3", "doc1", "doc7"]
relevant  = {"doc1", "doc5"}

recall_at_k(retrieved, relevant, k=3)     # → 0.5 (found doc1, missed doc5)
mean_reciprocal_rank(retrieved, relevant) # → 0.5 (doc1 at rank 2)
```

---

### Grounding prompt

How you instruct the model to use retrieved context instead of parametric memory.
Three approaches — see `src/pipeline.py:build_rag_prompt` (Exercise 1):

| Approach   | Model instruction                            | Best for                      |
|------------|----------------------------------------------|-------------------------------|
| Strict     | "Answer only from context"                   | Compliance, factual Q&A       |
| Augmented  | "Context is primary; note when filling gaps" | General assistant tasks       |
| Cited      | "Cite [N] for each claim"                    | Research, auditable pipelines |

---

## Module map

```
src/
  chunker.py    chunk_by_tokens, chunk_by_tokens_sliding, chunk_by_sentences
  embedder.py   Embedder protocol, DeterministicEmbedder, SentenceTransformerEmbedder, OpenAIEmbedder
  store.py      SearchResult, VectorStore (ChromaDB EphemeralClient / PersistentClient)
  retriever.py  recall_at_k, mean_reciprocal_rank (pure functions), Retriever
  pipeline.py   Document, format_context_block, build_rag_prompt*, RAGPipeline
```

`*` = user contribution stub — Exercise 1.

---

## Running tests

```bash
# Install RAG dependencies first (one-time)
uv sync --extra rag

# 47 unit tests — no API key, no model downloads
make test-stage STAGE=04
```

| File                  | What it covers                                            |
|-----------------------|-----------------------------------------------------------|
| `test_chunker.py`     | Fixed / sliding / sentence strategies, overlap validation |
| `test_embedder.py`    | DeterministicEmbedder dimension, unit vectors, consistency|
| `test_retriever.py`   | Metric functions (pure), Retriever with in-memory ChromaDB|
| `test_pipeline.py`    | format_context_block, Document, build_rag_prompt stub     |

---

## Exercises

1. **Implement `build_rag_prompt()`** — `src/pipeline.py` has the stub and the
   docstring describing all three grounding approaches. Pick one, implement it
   (~10 lines), then run `notebooks/03_rag_pipeline.py` to test it end-to-end.

2. **Compare chunk sizes** — In notebook 01, try `CHUNK_SIZE` values of 128, 256,
   512, and 1024. Ingest the READMEs at each size and measure `recall_at_k` on
   the mini eval set in notebook 03 section 6. Where does performance plateau?

3. **Add a reranker** — After retrieving top-20 with the bi-encoder, add a
   `CrossEncoder` from sentence-transformers to rescore and re-sort to top-5.
   Measure the recall@5 improvement vs. bi-encoder-only.

4. **Hybrid search** — Implement BM25 retrieval (`rank-bm25`) alongside dense
   search. Merge results with Reciprocal Rank Fusion (RRF). Test on queries
   with exact technical terms like "HNSW" or "arXiv:2307.03172".

5. **Faithfulness check** — For each answer, verify every factual claim is
   supported by a retrieved chunk using an LLM-as-judge call:
   `"Is this claim supported by the provided context? yes/no."`
