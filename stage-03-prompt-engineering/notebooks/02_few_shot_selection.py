# ruff: noqa: F704, E402
# %% [markdown]
# # 02 — Few-Shot Prompting
#
# Teaching by example rather than by instruction.
# Sections 1–3 are offline. Section 4 requires ANTHROPIC_API_KEY.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))

from few_shot import FewShotExample, FewShotFormatter

# %% [markdown]
# ## 1. Why few-shot works
#
# Zero-shot: the model uses only its training knowledge.
# Few-shot:  examples in the context act as a distribution hint —
#            they shift the model toward the style, format, and
#            reasoning pattern you show.
#
# Key insight from the GPT-3 paper (Brown et al. 2020):
#   k=0 → baseline
#   k=1 → large jump (one example sets the expected format)
#   k=3 → further gain
#   k=10+ → diminishing returns, growing context cost

# %%
# Build a few-shot bank for sentiment classification
SENTIMENT_EXAMPLES = [
    FewShotExample(
        user="The product arrived broken and customer service was unhelpful.",
        assistant="<sentiment>negative</sentiment>",
        label="negative",
    ),
    FewShotExample(
        user="Absolutely love this! Best purchase I've made all year.",
        assistant="<sentiment>positive</sentiment>",
        label="positive",
    ),
    FewShotExample(
        user="It works as described. Nothing special.",
        assistant="<sentiment>neutral</sentiment>",
        label="neutral",
    ),
    FewShotExample(
        user="Delivery was late but the product itself is great.",
        assistant="<sentiment>neutral</sentiment>",
        label="neutral",
    ),
    FewShotExample(
        user="Outstanding quality, fast delivery, highly recommend.",
        assistant="<sentiment>positive</sentiment>",
        label="positive",
    ),
    FewShotExample(
        user="Terrible. Stopped working after one day.",
        assistant="<sentiment>negative</sentiment>",
        label="negative",
    ),
]

fmt = FewShotFormatter(SENTIMENT_EXAMPLES)

# %% [markdown]
# ## 2. Selection strategies

# %%
print("=== first (n=3) — deterministic, preserves bank order ===")
for ex in fmt.select(3, strategy="first"):
    print(f"  [{ex.label}] {ex.user[:50]}")

print("\n=== random (n=3, seed=42) — reproducible shuffle ===")
for ex in fmt.select(3, strategy="random", seed=42):
    print(f"  [{ex.label}] {ex.user[:50]}")

print("\n=== by_label (n=3) — one from each label ===")
for ex in fmt.select(3, strategy="by_label"):
    print(f"  [{ex.label}] {ex.user[:50]}")

# %% [markdown]
# ## 3. Formatting as message turns
#
# Examples are injected as prior conversation turns, not in the system prompt.
# The model reads them as evidence of how it should behave.

# %%
query = "The shipping took 3 weeks but the item is perfect."

# Build the full message list: examples + current query
messages = fmt.prepend_to_messages(
    [{"role": "user", "content": query}],
    n=3,
    strategy="by_label",
)

print(f"Total messages: {len(messages)} ({(len(messages) - 1) // 2} examples + 1 query)\n")
for msg in messages:
    role = msg["role"].upper()
    content = msg["content"][:70]
    print(f"  {role}: {content}")

# %% [markdown]
# ## 4. Live comparison: zero-shot vs few-shot (requires ANTHROPIC_API_KEY)


# %%
async def compare_few_shot(text: str, n_examples: int = 3) -> None:
    """Compare zero-shot vs few-shot sentiment classification."""
    from extractor import extract_tag
    from llm.anthropic_client import AnthropicClient
    from llm.sampling import SamplingParams
    from prompts.examples import SENTIMENT_CLASSIFIER

    client = AnthropicClient()
    system, _ = SENTIMENT_CLASSIFIER.render(text="")  # just get system

    async def classify(messages: list[dict[str, str]], label: str) -> None:
        r = await client.complete(
            messages=messages,
            system=system,
            params=SamplingParams(temperature=0.0, max_tokens=64),
        )
        sentiment = extract_tag(r.content, "sentiment") or r.content.strip()
        cost = f"${r.cost.total_usd:.5f}" if r.cost else "n/a"
        tokens = r.usage.input_tokens
        print(f"  {label:<15} → {sentiment:<12} ({tokens} input tokens, {cost})")

    print(f"Input: {text!r}\n")

    # Zero-shot
    _, user = SENTIMENT_CLASSIFIER.render(text=text)
    await classify([{"role": "user", "content": user}], "0-shot")

    # Few-shot (n examples)
    _, user = SENTIMENT_CLASSIFIER.render(text=text)
    messages = fmt.prepend_to_messages(
        [{"role": "user", "content": user}], n=n_examples, strategy="by_label"
    )
    await classify(messages, f"{n_examples}-shot (by_label)")


TEST_TEXTS = [
    "The item is fine but nothing to write home about.",
    "Worst experience ever. Will never buy again.",
    "Exactly what I needed. Quick delivery too.",
]

# Uncomment to run:
# for t in TEST_TEXTS:
#     await compare_few_shot(t)
#     print()
print("(Cell ready — uncomment to run with a real API key)")
