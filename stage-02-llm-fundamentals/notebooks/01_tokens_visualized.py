# ruff: noqa: F704, E402
# %% [markdown]
# # 01 — Tokens Visualized
#
# Tokenization is the first transformation your text undergoes before reaching
# a model. Understanding it prevents several common bugs:
#   - Sending more tokens than you think (cost overruns, context truncation)
#   - Splitting words mid-token (surprising model behaviour on edge cases)
#   - Non-Latin text costing 3-5x more per character than English
#
# Run this notebook cell by cell.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))

from llm.tokens import count_tokens, estimate_messages_tokens, tokenize

# %% [markdown]
# ## 1. Basic counting

# %%
texts = [
    "Hello, world!",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "日本語のテキスト",  # Japanese
    "مرحبا بالعالم",  # Arabic
    "The quick brown fox jumps over the lazy dog. " * 20,  # repeated English
]

print(f"{'Text':<55} {'Tokens':>6} {'Chars':>6} {'Ch/Tok':>7}")
print("-" * 76)
for text in texts:
    n = count_tokens(text, model="gpt-4o")
    ratio = len(text) / n if n else 0
    preview = text[:52] + "…" if len(text) > 52 else text
    print(f"{preview:<55} {n:>6} {len(text):>6} {ratio:>7.1f}")

# %% [markdown]
# **Notice:** Japanese and Arabic have ~1 character per token (low efficiency).
# English prose has ~4 characters per token. Code is in between (~2–3).
# This directly affects cost: sending 1000 Chinese characters costs ~3–5x more
# than 1000 English characters worth of equivalent content.

# %% [markdown]
# ## 2. Token boundaries


# %%
def visualise_tokens(text: str, model: str = "gpt-4o") -> None:
    """Print each token on its own line with its ID."""
    spans = tokenize(text, model=model)
    print(f"Text ({len(spans)} tokens): {repr(text)}\n")
    for span in spans:
        # Represent the token text so control chars are visible
        display = repr(span.text)[1:-1]  # strip outer quotes
        print(f"  [{span.index:>3}] id={span.token_id:>6}  {display!r}")


visualise_tokens("unhappiness")

# %%
# Interesting cases — run these one at a time and read the output
visualise_tokens("ChatGPT")
visualise_tokens("1234567890")
visualise_tokens("    " * 4)  # leading spaces affect tokenization!

# %% [markdown]
# ## 3. Counting messages (OpenAI format)

# %%
messages = [
    {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "And Germany?"},
]

estimate = estimate_messages_tokens(messages)
actual = sum(count_tokens(m["content"]) for m in messages)

print(f"Estimate (with overhead): {estimate}")
print(f"Content only (no overhead): {actual}")
print(f"Overhead (role delimiters, reply primer): {estimate - actual}")

# %% [markdown]
# ## 4. Context window budget planning


# %%
def budget_report(
    system_prompt: str,
    example_messages: list[dict[str, str]],
    context_window: int = 200_000,
    reserved_output: int = 4_096,
) -> None:
    """Show how much context window remains after fixed content."""
    system_tokens = count_tokens(system_prompt)
    example_tokens = estimate_messages_tokens(example_messages)
    overhead = system_tokens + example_tokens + reserved_output
    remaining = context_window - overhead
    pct = remaining / context_window * 100

    print(f"Context window:    {context_window:>7,} tokens")
    print(f"System prompt:     {system_tokens:>7,} tokens")
    print(f"Few-shot examples: {example_tokens:>7,} tokens")
    print(f"Reserved output:   {reserved_output:>7,} tokens")
    print("─────────────────────────────────────")
    print(f"Available for user:{remaining:>7,} tokens  ({pct:.1f} %)")


budget_report(
    system_prompt="You are a senior Python engineer. Answer questions about code quality, "
    "testing, and architecture. Be concise." * 10,
    example_messages=[
        {"role": "user", "content": "What is dependency injection?"},
        {
            "role": "assistant",
            "content": "A pattern where dependencies are passed in, not created.",
        },  # noqa: E501
    ],
)

# %% [markdown]
# ## 5. "Lost in the middle" — why position in context matters
#
# Liu et al. (2023) arXiv:2307.03172 showed that LLMs recall information from
# the beginning and end of long contexts significantly better than from the
# middle. This has practical implications:
#
# - Put the most important instructions at the start of the system prompt
# - Put the most relevant retrieved documents at the END (just before the question)
# - Avoid burying key facts in the middle of a 100k-token context
#
# The effect is strong at 128k+ tokens and nearly absent at <10k tokens.


# %%
def context_position_demo(n_documents: int = 20, target_doc_position: int = 10) -> None:
    """
    Simulate placing a key document at different positions in a long context.
    Shows token budget consumed at each position.
    """
    doc = "Document {i}: The answer is 42. " * 50  # ~50 tokens each

    positions = []
    cumulative = 0
    for i in range(n_documents):
        tokens = count_tokens(doc.format(i=i))
        cumulative += tokens
        marker = " ← KEY DOC HERE" if i == target_doc_position else ""
        positions.append((i, cumulative, marker))

    print("Cumulative tokens at each document position:")
    for i, cum, marker in positions:
        print(f"  Doc {i:>2}: {cum:>6,} tokens{marker}")


context_position_demo()
