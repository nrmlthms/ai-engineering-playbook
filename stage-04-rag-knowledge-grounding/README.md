# Stage 04 — RAG & Knowledge Grounding

## Concepts

- Embedding models: dense vs sparse, dimensions, similarity
- Vector databases: ChromaDB, indexing, HNSW
- Retrieval pipelines: chunking, embedding, top-k, reranking
- Grounding prompts with retrieved context
- Evaluating retrieval: recall@k, MRR, faithfulness

## Key ideas

```
Documents → chunk → embed → vector DB
                               ↓
Query → embed → ANN search → top-k chunks → prompt → LLM → answer
                                ↓
                           Reranker (optional)
```

Chunking strategy dominates retrieval quality more than the choice of embedding
model. Semantic chunking (split on topic boundaries) outperforms fixed-size
chunking on long documents.

## What's in `src/`

| File | Purpose |
|------|---------|
| `ingest.py` | Load, chunk, and embed documents into ChromaDB |
| `retriever.py` | Query the vector DB and optionally rerank |
| `pipeline.py` | End-to-end RAG: retrieve → prompt → generate |

## Exercises

1. Ingest the project READMEs and build a Q&A bot over them
2. Compare chunk sizes (256 / 512 / 1024 tokens) on retrieval recall
3. Add a cross-encoder reranker and measure faithfulness with DeepEval
