# ruff: noqa: F704, E402
# %% [markdown]
# # 03 — Reasoning vs Chat: When and What It Costs
#
# Reasoning models (Claude extended thinking, OpenAI o-series) think before
# answering. This makes them better at hard tasks — but significantly more
# expensive. This notebook builds intuition for the trade-off.
#
# No live API calls in the first three sections (pure arithmetic).
# Sections 4+ require ANTHROPIC_API_KEY and OPENAI_API_KEY.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))

from llm.streaming import CostBreakdown, ModelPricing, Usage

# %% [markdown]
# ## 1. Cost model: general chat vs reasoning
#
# The key insight: reasoning models spend hidden tokens thinking.
# Those tokens are billed even though you never see them.

# %%
# Pricing as of 2025 (per million tokens, USD)
CHAT_PRICING = {
    "claude-sonnet-4-6": ModelPricing(input_mtok=3.0, output_mtok=15.0),
    "gpt-4o": ModelPricing(input_mtok=2.5, output_mtok=10.0),
    "gpt-4o-mini": ModelPricing(input_mtok=0.15, output_mtok=0.60),
}

REASONING_PRICING = {
    "claude-sonnet-4-6 (thinking on)": ModelPricing(input_mtok=3.0, output_mtok=15.0),
    "o3-mini": ModelPricing(input_mtok=1.10, output_mtok=4.40),
    "o3": ModelPricing(input_mtok=10.0, output_mtok=40.0),
    "o1": ModelPricing(input_mtok=15.0, output_mtok=60.0),
}


def print_cost_table(
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    label: str = "",
) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  Input: {input_tokens:,}t  Output: {output_tokens:,}t  Thinking: {thinking_tokens:,}t")
    print(f"{'─' * 60}")
    print(f"  {'Model':<35} {'Cost':>8}")
    print(f"  {'─' * 44}")

    for name, pricing in {**CHAT_PRICING, **REASONING_PRICING}.items():
        # For reasoning models, thinking tokens are billed as output
        total_output = output_tokens + thinking_tokens
        usage = Usage(input_tokens=input_tokens, output_tokens=total_output)
        cost = CostBreakdown.from_usage(usage, pricing)
        print(f"  {name:<35} ${cost.total_usd:>7.4f}")


# Simple Q&A: ~200 input tokens, ~50 output, no thinking
print_cost_table(200, 50, 0, "Simple Q&A")

# Reasoning task: ~500 input, 200 visible output, 2000 thinking tokens
print_cost_table(500, 200, 2_000, "Reasoning task (2k thinking tokens)")

# Hard reasoning: ~1000 input, 500 visible output, 10000 thinking tokens
print_cost_table(1_000, 500, 10_000, "Hard reasoning (10k thinking tokens)")

# %% [markdown]
# ## 2. Break-even analysis
# When does paying for reasoning actually save time/money vs re-prompting?


# %%
def break_even_reasoning(
    base_model_cost: float,
    reasoning_model_cost: float,
    avg_reruns_saved: float,
) -> None:
    """
    Reasoning is worth it if:
      reasoning_cost < base_cost × avg_reruns_needed_without_reasoning

    avg_reruns_saved: how many additional base-model calls you'd need to get
                      the same quality answer without reasoning.
    """
    break_even = reasoning_model_cost / base_model_cost
    print(f"  Base model cost per call:      ${base_model_cost:.4f}")
    print(f"  Reasoning model cost per call: ${reasoning_model_cost:.4f}")
    print(f"  Cost multiplier:               {reasoning_model_cost / base_model_cost:.1f}x")
    print(f"  Break-even reruns:             {break_even:.2f}")
    if avg_reruns_saved >= break_even:
        print(f"  ✓ Reasoning is worth it (saves {avg_reruns_saved:.1f} reruns on average)")
    else:
        print(f"  ✗ Cheaper to rerun base model (only saves {avg_reruns_saved:.1f} reruns)")
    print()


print("Break-even analysis:\n")

# Scenario 1: competitive programming problem
print("Scenario: Hard coding problem")
base = CostBreakdown.from_usage(
    Usage(input_tokens=500, output_tokens=300),
    CHAT_PRICING["gpt-4o"],
).total_usd
reasoning = CostBreakdown.from_usage(
    Usage(input_tokens=500, output_tokens=300 + 5_000),
    REASONING_PRICING["o3-mini"],
).total_usd
break_even_reasoning(base, reasoning, avg_reruns_saved=3.0)

# Scenario 2: email summary
print("Scenario: Summarise email")
base = CostBreakdown.from_usage(
    Usage(input_tokens=200, output_tokens=100),
    CHAT_PRICING["gpt-4o-mini"],
).total_usd
reasoning = CostBreakdown.from_usage(
    Usage(input_tokens=200, output_tokens=100 + 2_000),
    REASONING_PRICING["o3-mini"],
).total_usd
break_even_reasoning(base, reasoning, avg_reruns_saved=1.1)

# %% [markdown]
# ## 3. Task classification: use reasoning or not?
#
# Rules of thumb (not absolute):

# %%
TASK_GUIDE = [
    # (task_type, use_reasoning, rationale)
    ("Multi-step math / proofs", True, "Chains of logic benefit from internal scratchpad"),
    ("Competitive programming", True, "Needs backtracking and verification"),
    ("Complex instruction following", True, "Reduces hallucination on ambiguous specs"),
    ("Simple Q&A / retrieval", False, "No reasoning needed; chat is 10–50x cheaper"),
    ("Summarisation", False, "Extractive; no novel reasoning required"),
    ("Code generation (simple)", False, "Template-like; fast models fine"),
    ("Translation", False, "No reasoning; even mini models work well"),
    ("JSON extraction from text", False, "Structured output, no chain-of-thought needed"),
    ("Legal / financial analysis", "Maybe", "Depends on complexity; try chat first"),
    ("Creative writing", False, "High temperature chat often better than reasoning"),
]

print(f"  {'Task':<40} {'Reasoning?':<12} {'Why'}")
print(f"  {'─' * 80}")
for task, use, why in TASK_GUIDE:
    flag = "✓ YES" if use is True else ("? MAYBE" if use == "Maybe" else "✗ NO")
    print(f"  {task:<40} {flag:<12} {why}")

# %% [markdown]
# ## 4. Live comparison: Claude extended thinking (requires API key)


# %%
async def reasoning_comparison(prompt: str) -> None:
    """Compare claude-sonnet-4-6 with and without extended thinking."""
    from llm.anthropic_client import AnthropicClient, ThinkingConfig

    client = AnthropicClient()

    # Chat mode
    chat_r = await client.complete(
        messages=[{"role": "user", "content": prompt}],
        params=__import__("llm.sampling", fromlist=["SamplingParams"]).SamplingParams(
            max_tokens=512
        ),
    )

    # Thinking mode
    think_r = await client.complete(
        messages=[{"role": "user", "content": prompt}],
        params=__import__("llm.sampling", fromlist=["SamplingParams"]).SamplingParams(
            max_tokens=512
        ),
        thinking=ThinkingConfig(budget_tokens=4_000),
    )

    print("=== Chat mode ===")
    print(f"Tokens: {chat_r.usage.input_tokens} in / {chat_r.usage.output_tokens} out")
    print(f"Cost:   ${chat_r.cost.total_usd:.4f}" if chat_r.cost else "Cost:   n/a")
    print(f"Answer: {chat_r.content[:200]}")

    print("\n=== Thinking mode ===")
    print(f"Tokens: {think_r.usage.input_tokens} in / {think_r.usage.output_tokens} out")
    print(f"Cost:   ${think_r.cost.total_usd:.4f}" if think_r.cost else "Cost:   n/a")
    if think_r.thinking:
        print(f"Thinking ({len(think_r.thinking.split())} words): {think_r.thinking[:150]}…")
    print(f"Answer: {think_r.content[:200]}")


HARD_PROBLEM = (
    "A train leaves station A at 9:00 AM travelling at 80 km/h. "
    "Another train leaves station B (320 km away) at 10:00 AM travelling at 120 km/h "
    "toward station A. At what time do they meet, and how far from station A?"
)

# Uncomment to run live:
# await reasoning_comparison(HARD_PROBLEM)
print("(Cell ready — uncomment last line to run with a real API key)")
