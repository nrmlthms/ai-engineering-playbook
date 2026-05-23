# ruff: noqa: F704, E402
# %% [markdown]
# # 02 — Sampling Playground
#
# This notebook requires a real ANTHROPIC_API_KEY in your .env file.
# It makes live API calls — budget ~$0.10 for all cells.
#
# What you'll learn:
#   - How temperature changes output distribution (not creativity)
#   - How top_p nucleus sampling differs from temperature
#   - How to use stop sequences
#   - Why temperature=0 is not fully deterministic

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))

from llm.anthropic_client import AnthropicClient
from llm.sampling import SamplingParams

client = AnthropicClient()
PROMPT = [{"role": "user", "content": "Complete this sentence in exactly 5 words: The sky is"}]

# %% [markdown]
# ## 1. Temperature sweep
# Same prompt, five different temperatures, five calls each.
# Lower temperature → more repetition. Higher → more variety.


# %%
async def temperature_sweep() -> None:
    temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
    n_samples = 3

    for temp in temperatures:
        print(f"\ntemperature={temp}")
        params = SamplingParams(temperature=temp, max_tokens=20)
        for _ in range(n_samples):
            r = await client.complete(PROMPT, params=params)
            print(f"  {r.content.strip()!r}")


await temperature_sweep()

# %% [markdown]
# **What to observe:**
# - temperature=0.0: near-identical outputs across runs (greedy decoding)
# - temperature=1.5+: outputs become incoherent or unpredictable
# - The "sweet spot" for most tasks is 0.3–0.8

# %% [markdown]
# ## 2. top_p (nucleus sampling) vs temperature


# %%
async def top_p_demo() -> None:
    # Fix temperature, vary top_p
    configs = [
        {"temperature": 1.0, "top_p": 0.1},  # very narrow nucleus
        {"temperature": 1.0, "top_p": 0.5},  # moderate
        {"temperature": 1.0, "top_p": 0.95},  # nearly all tokens eligible
        {"temperature": 0.5, "top_p": 0.9},  # temperature does the heavy lifting
    ]
    for cfg in configs:
        params = SamplingParams(max_tokens=20, **cfg)  # type: ignore[arg-type]
        r = await client.complete(PROMPT, params=params)
        print(f"  temp={cfg['temperature']}, top_p={cfg['top_p']}: {r.content.strip()!r}")


print("top_p demo:")
await top_p_demo()

# %% [markdown]
# ## 3. Stop sequences
# Stop sequences let you define precise output boundaries.
# The model stops when it generates any token in the stop list.


# %%
async def stop_sequence_demo() -> None:
    list_prompt = [
        {
            "role": "user",
            "content": "List the first 10 prime numbers, one per line, then write DONE.",
        }
    ]

    # Without stop sequence — full response
    full = await client.complete(list_prompt, params=SamplingParams(max_tokens=100))
    print("Full response:")
    print(full.content)

    # With stop sequence — stops at DONE
    stopped = await client.complete(
        list_prompt,
        params=SamplingParams(max_tokens=100, stop_sequences=["DONE"]),
    )
    print("\nStopped at 'DONE':")
    print(stopped.content)


await stop_sequence_demo()

# %% [markdown]
# ## 4. Determinism at temperature=0
# temperature=0 uses greedy decoding but is NOT fully deterministic.
# Run this cell multiple times to see potential variance.


# %%
async def determinism_test() -> None:
    params = SamplingParams(temperature=0.0, max_tokens=50)
    results = set()

    for i in range(5):
        r = await client.complete(
            [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
            params=params,
        )
        results.add(r.content.strip())

    print(f"Unique outputs from 5 calls at temp=0: {len(results)}")
    for result in results:
        print(f"  {result!r}")

    if len(results) == 1:
        print("→ Consistent this time (most common outcome)")
    else:
        print("→ Variance observed — hardware/batch differences")


await determinism_test()

# %% [markdown]
# ## 5. Prompt caching — measuring cache hit savings
# This requires a system prompt long enough to be cached (>1024 tokens).
# After the first call, subsequent calls should show cache_read_tokens.


# %%
async def cache_demo() -> None:
    long_system = (
        "You are a senior Python engineer with expertise in async programming, "
        "distributed systems, and API design. " * 50
    )  # ~500 tokens — above caching threshold

    # Mark the system prompt as cacheable using the block format
    from anthropic import AsyncAnthropic

    ac = AsyncAnthropic(api_key=client._client.api_key)  # type: ignore[attr-defined]

    for call_num in range(3):
        raw = await ac.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            system=[
                {
                    "type": "text",
                    "text": long_system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": "What is async?"}],
        )
        u = raw.usage
        print(
            f"Call {call_num + 1}: input={u.input_tokens} "
            f"cache_write={getattr(u, 'cache_creation_input_tokens', 0)} "
            f"cache_read={getattr(u, 'cache_read_input_tokens', 0)}"
        )


print("Cache demo (3 consecutive calls):")
await cache_demo()
