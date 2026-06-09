# ruff: noqa: F704, E402
# %% [markdown]
# # 01 — Chunking Strategies
#
# Chunking is the single biggest lever on RAG quality.
# This notebook compares fixed-size, sliding window, and sentence-aware chunking
# on the same text and measures the trade-offs empirically.
# No API key needed — all offline.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))

from chunker import chunk_by_sentences, chunk_by_tokens, chunk_by_tokens_sliding

# %% [markdown]
# ## 1. Load a real document (stage READMEs)

# %%
STAGE_ROOT = Path().resolve().parent.parent
readmes = sorted(STAGE_ROOT.glob("stage-*/README.md"))
print(f"Found {len(readmes)} README files:")
for p in readmes:
    print(f"  {p.parent.name}")

# Use the stage-02 README as a test document — it's detailed and prose-heavy
sample_path = STAGE_ROOT / "stage-02-llm-fundamentals" / "README.md"
sample_text = sample_path.read_text()
print(f"\nDocument: {sample_path.name}")
print(f"Characters: {len(sample_text):,}")

# %% [markdown]
# ## 2. Compare three chunking strategies

# %%
CHUNK_SIZE = 256   # tokens
OVERLAP = 32       # tokens (for sliding window)

fixed = chunk_by_tokens(sample_text, chunk_size=CHUNK_SIZE)
sliding = chunk_by_tokens_sliding(sample_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
sentence = chunk_by_sentences(sample_text, max_tokens=CHUNK_SIZE)

print(f"{'Strategy':<20} {'Chunks':>8} {'Min tok':>8} {'Max tok':>8} {'Avg tok':>8}")
print("-" * 56)
for label, chunks in [("Fixed", fixed), ("Sliding", sliding), ("Sentence", sentence)]:
    sizes = [c.token_count for c in chunks]
    print(
        f"{label:<20} {len(chunks):>8} {min(sizes):>8} {max(sizes):>8} "
        f"{sum(sizes)/len(sizes):>8.0f}"
    )

# %% [markdown]
# ## 3. Inspect chunk boundary quality
#
# A bad chunk boundary cuts in the middle of a concept.
# Look for chunks that start/end mid-sentence.

# %%
print("=== Fixed-size: boundary samples ===")
for i, chunk in enumerate(fixed[3:6], start=3):
    text_preview = chunk.text[:100].replace("\n", " ")
    print(f"  Chunk {i}: …{text_preview}…")

print("\n=== Sentence-aware: boundary samples ===")
for i, chunk in enumerate(sentence[3:6], start=3):
    text_preview = chunk.text[:100].replace("\n", " ")
    print(f"  Chunk {i}: …{text_preview}…")

# %% [markdown]
# ## 4. Overlap effect: can we recover a cross-boundary fact?
#
# Fact: "Lost in the Middle (arXiv:2307.03172)" spans a heading and body text.
# With fixed chunking it might fall across a boundary. With overlap, it appears
# in two consecutive chunks → at least one will be retrieved.

# %%
TARGET = "2307.03172"

fixed_hits = [c for c in fixed if TARGET in c.text]
sliding_hits = [c for c in sliding if TARGET in c.text]
sentence_hits = [c for c in sentence if TARGET in c.text]

print(f"Chunks containing '{TARGET}':")
print(f"  Fixed:    {len(fixed_hits)}")
print(f"  Sliding:  {len(sliding_hits)}")
print(f"  Sentence: {len(sentence_hits)}")

# %% [markdown]
# ## 5. How chunk size affects retrieval granularity
#
# Small chunks → precise retrieval, but you lose surrounding context.
# Large chunks → more context, but you retrieve noise alongside the answer.
# Typical sweet spot: 256–512 tokens with ~15% overlap.

# %%
for size in [128, 256, 512, 1024]:
    chunks = chunk_by_tokens(sample_text, chunk_size=size)
    print(f"  chunk_size={size:>5}: {len(chunks):>3} chunks")
