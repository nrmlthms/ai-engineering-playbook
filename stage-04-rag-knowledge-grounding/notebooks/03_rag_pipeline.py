# ruff: noqa: F704, E402
# %% [markdown]
# # 03 — End-to-End RAG Pipeline
#
# Ingest the stage READMEs → build a Q&A bot over them.
# Requires: ANTHROPIC_API_KEY and the sentence-transformers model.
# Exercise 1 must be complete (build_rag_prompt implemented).

# %%
import sys
from pathlib import Path

# Add stage-04 src
sys.path.insert(0, str(Path().resolve().parent / "src"))
# Add stage-02 src for AnthropicClient
sys.path.insert(0, str(Path().resolve().parent.parent / "stage-02-llm-fundamentals" / "src"))

from embedder import SentenceTransformerEmbedder
from pipeline import Document, RAGPipeline
from retriever import Retriever
from store import VectorStore

# %% [markdown]
# ## 1. Ingest all stage READMEs

# %%
STAGE_ROOT = Path().resolve().parent.parent
readme_paths = sorted(STAGE_ROOT.glob("stage-*/README.md"))
print(f"Found {len(readme_paths)} stage READMEs to ingest")

documents = [
    Document(
        text=p.read_text(),
        doc_id=p.parent.name,
        metadata={"path": str(p)},
    )
    for p in readme_paths
]

for doc in documents:
    tokens_approx = len(doc.text) // 4
    print(f"  {doc.doc_id:<40} ~{tokens_approx:>5} tokens")

# %% [markdown]
# ## 2. Build the retriever (downloads model ~90 MB on first run)


# %%
def build_retriever() -> Retriever:
    embedder = SentenceTransformerEmbedder()
    store = VectorStore()
    return Retriever(store=store, embedder=embedder)


# Uncomment to build:
# retriever = build_retriever()
# print(f"Embedder dimension: {retriever.embedder.dimension}")
print("(Cell ready — uncomment to build retriever)")

# %% [markdown]
# ## 3. Ingest documents


# %%
async def ingest_docs() -> RAGPipeline:
    from llm.anthropic_client import AnthropicClient

    retriever = build_retriever()
    client = AnthropicClient()
    pipeline = RAGPipeline(retriever=retriever, client=client, top_k=5)

    total_chunks = pipeline.ingest(documents, chunk_size=384)
    print(f"Ingested {len(documents)} documents → {total_chunks} chunks")
    print(f"Store size: {retriever.store.count()} vectors")
    return pipeline


# Uncomment to ingest:
# pipeline = await ingest_docs()
print("(Cell ready — uncomment after implementing build_rag_prompt)")

# %% [markdown]
# ## 4. Retrieval — inspect what gets retrieved before generation


# %%
def inspect_retrieval(retriever: Retriever, query: str, top_k: int = 5) -> None:
    """Show what chunks would be retrieved for a query."""
    results = retriever.retrieve(query, top_k=top_k)
    print(f"Query: {query!r}\n")
    for r in results:
        print(f"  [{r.rank}] score={r.score:.3f}  source={r.doc_id}")
        print(f"       {r.text[:120].replace(chr(10), ' ')}…\n")


# Uncomment to run (after building retriever above):
# inspect_retrieval(retriever, "What sampling parameters does Anthropic support?")
# inspect_retrieval(retriever, "How do I implement chain-of-thought prompting?")
print("(Cell ready — uncomment after building retriever)")

# %% [markdown]
# ## 5. Full RAG query (requires build_rag_prompt + ANTHROPIC_API_KEY)

# %%
EVAL_QUESTIONS = [
    "What is the 'lost in the middle' problem in LLMs?",
    "How do I use few-shot prompting with the FewShotFormatter?",
    "What is BPE tokenization and why does it matter?",
    "When should I use reasoning models vs chat models?",
    "What metrics evaluate retrieval quality in RAG?",
]


async def run_rag_eval() -> None:
    """Run each eval question through the RAG pipeline and print answers."""
    p = await ingest_docs()

    for question in EVAL_QUESTIONS:
        print(f"\n{'─' * 70}")
        print(f"Q: {question}")

        try:
            response = await p.query(question)
            print(f"A: {response.answer[:300]}")
            print(f"   Sources: {[r.doc_id for r in response.sources[:3]]}")
        except NotImplementedError:
            print("   ⚠ build_rag_prompt() not implemented yet — complete Exercise 1 first.")
            break


# Uncomment to run full eval:
# await run_rag_eval()
print("(Cell ready — implement build_rag_prompt() first, then uncomment)")

# %% [markdown]
# ## 6. Retrieval metrics on a mini eval set

# %%
EVAL_SET = [
    # (query, relevant_doc_ids)
    ("What is temperature in sampling?", ["stage-02-llm-fundamentals"]),
    ("How do XML tags help with structured output?", ["stage-03-prompt-engineering"]),
    ("What are recall@k and MRR?", ["stage-04-rag-knowledge-grounding"]),
]


def evaluate_retrieval(retriever: Retriever) -> None:
    from retriever import mean_reciprocal_rank, recall_at_k

    print(f"{'Query':<55} {'R@3':>5} {'MRR':>5}")
    print("-" * 70)
    for query, relevant in EVAL_SET:
        results = retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.doc_id for r in results]
        r3 = recall_at_k(retrieved_ids, set(relevant), k=3)
        mrr = mean_reciprocal_rank(retrieved_ids, set(relevant))
        print(f"  {query[:52]:<55} {r3:>5.2f} {mrr:>5.2f}")


# Uncomment after building retriever and ingesting docs:
# evaluate_retrieval(retriever)
print("(Cell ready — uncomment after ingestion)")
