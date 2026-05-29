# ruff: noqa: F704, E402
# %% [markdown]
# # 03 — Chain-of-Thought Prompting
#
# Forcing the model to reason before answering.
# Sections 1–2 are offline. Sections 3–4 require ANTHROPIC_API_KEY.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))

from chain_of_thought import (
    SCRATCHPAD_SYSTEM_SNIPPET,
    build_scratchpad_system,
    extract_cot_answer,
    zero_shot_cot,
)

# %% [markdown]
# ## 1. The CoT trigger phrase
#
# Kojima et al. (2022) — "Large Language Models are Zero-Shot Reasoners"
#
# The phrase "Let's think step by step." appended to a question boosts
# accuracy on math and logic benchmarks by 20–40+ percentage points on
# GPT-3. The model has seen this phrase as a transition from problem
# statement to worked solution throughout pretraining — it activates a
# different generation mode.
#
# This is surprising because: no examples, no instruction to explain,
# just one sentence. The phrase *alone* changes behaviour.

# %%
problems = [
    "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
    "There are 23 sheep and 10 goats on a farm. How old is the farmer?",
]

print("Zero-shot CoT prompts:\n")
for p in problems:
    cot = zero_shot_cot(p)
    print(f"  Original: {p}")
    print(f"  CoT:      {cot[-50:]}")
    print()

# %% [markdown]
# ## 2. The scratchpad pattern
#
# Separate the reasoning from the answer using XML tags:
#   <thinking>…</thinking>   — internal scratchpad (can log, never show user)
#   <answer>…</answer>       — final response (show to user, parse in code)
#
# Compared to zero-shot CoT:
#   zero-shot CoT: the reasoning IS the response (visible, hard to parse)
#   scratchpad:    reasoning is a tagged section (inspectable, parseable)
#
# Compared to Claude extended thinking:
#   scratchpad: prompt-level, works on any model, always visible in response
#   ext. thinking: API parameter, billed separately, more reliable on hard tasks

# %%
print("Scratchpad system snippet:\n")
print(SCRATCHPAD_SYSTEM_SNIPPET)
print()

# Extraction demo (simulating a model response)
simulated_response = """
<thinking>
The ball costs x. The bat costs x + $1.00.
Total: x + (x + $1.00) = $1.10
2x = $0.10
x = $0.05
</thinking>
<answer>
The ball costs $0.05.
</answer>
"""

thinking, answer = extract_cot_answer(simulated_response)
print(f"Thinking: {thinking}")
print(f"Answer:   {answer}")

# %% [markdown]
# ## 3. Live: zero-shot vs scratchpad (requires ANTHROPIC_API_KEY)

# %%
HARD_PROBLEMS = [
    # Classic trick question — System 1 answer ($0.10) is wrong
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
    # Multi-step arithmetic
    (
        "A store had 150 apples. They sold 40% on Monday, then received a shipment of 30 more. "
        "On Tuesday they sold half of what remained. How many apples are left?"
    ),
]


async def compare_cot(problem: str) -> None:
    """Compare direct answer vs zero-shot CoT vs scratchpad."""
    from llm.anthropic_client import AnthropicClient
    from llm.sampling import SamplingParams

    client = AnthropicClient()
    params = SamplingParams(temperature=0.0, max_tokens=512)

    print(f"Problem: {problem[:80]}…\n")

    # Direct answer (no CoT)
    r1 = await client.complete(
        messages=[{"role": "user", "content": problem}],
        params=params,
    )
    print(f"Direct:     {r1.content[:100]}")
    print(f"  tokens:   {r1.usage.input_tokens}in / {r1.usage.output_tokens}out")

    # Zero-shot CoT
    r2 = await client.complete(
        messages=[{"role": "user", "content": zero_shot_cot(problem)}],
        params=params,
    )
    print(f"Zero-CoT:   {r2.content[:100]}")
    print(f"  tokens:   {r2.usage.input_tokens}in / {r2.usage.output_tokens}out")

    # Scratchpad
    r3 = await client.complete(
        messages=[{"role": "user", "content": problem}],
        system=build_scratchpad_system(),
        params=params,
    )
    thinking, answer = extract_cot_answer(r3.content)
    print(f"Scratchpad: answer={answer!r}")
    print(f"  thinking: {(thinking or '')[:100]}")
    print(f"  tokens:   {r3.usage.input_tokens}in / {r3.usage.output_tokens}out")
    print()


# Uncomment to run:
# for p in HARD_PROBLEMS:
#     await compare_cot(p)
print("(Cell ready — uncomment to run with a real API key)")

# %% [markdown]
# ## 4. Live: CoT vs Claude extended thinking (requires ANTHROPIC_API_KEY)
#
# Extended thinking is the API-level version of scratchpad CoT:
# - More reliable on very hard tasks (>5-step reasoning chains)
# - Thinking tokens billed separately (output rate)
# - Not visible unless you read response.thinking
#
# Rule of thumb: use scratchpad for most tasks, extended thinking for
# problems where you'd benefit from a long internal draft.

# %%
async def thinking_vs_scratchpad(problem: str) -> None:
    from llm.anthropic_client import AnthropicClient, ThinkingConfig
    from llm.sampling import SamplingParams

    client = AnthropicClient()
    params = SamplingParams(temperature=1.0, max_tokens=2048)

    # Scratchpad via prompt
    r1 = await client.complete(
        messages=[{"role": "user", "content": problem}],
        system=build_scratchpad_system(),
        params=SamplingParams(temperature=0.0, max_tokens=1024),
    )
    _, answer1 = extract_cot_answer(r1.content)

    # Extended thinking via API
    r2 = await client.complete(
        messages=[{"role": "user", "content": problem}],
        params=params,
        thinking=ThinkingConfig(budget_tokens=4_000),
    )

    print(f"Problem: {problem[:80]}\n")
    print(f"Scratchpad answer:        {answer1!r}")
    cost1 = f"${r1.cost.total_usd:.4f}" if r1.cost else "n/a"
    print(f"  out tokens: {r1.usage.output_tokens}  cost: {cost1}")

    print(f"Extended thinking answer: {r2.content[:100]!r}")
    cost2 = f"${r2.cost.total_usd:.4f}" if r2.cost else "n/a"
    print(f"  out tokens: {r2.usage.output_tokens}  cost: {cost2}")
    if r2.thinking:
        words = len(r2.thinking.split())
        print(f"  thinking:   {words} words — {r2.thinking[:120]}…")


# Uncomment to run:
# await thinking_vs_scratchpad(HARD_PROBLEMS[0])
print("(Cell ready — uncomment to run with a real API key)")
