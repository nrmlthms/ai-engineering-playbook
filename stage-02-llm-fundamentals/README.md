# Stage 02 — LLM Fundamentals

> Token-aware Anthropic SDK client with caching, streaming, and typed usage tracking.

---

## What you build

A production client layer (`src/llm/`) that wraps the Anthropic and OpenAI SDKs.
Every call records input/output/cache tokens, first-token and wall-clock latency,
model version, and cost. The types are provider-agnostic — application code depends
on `LLMResponse`, not on SDK-specific objects.

---

## Concepts

### Tokenization

LLMs don't read characters — they read **tokens**. A token is a chunk of text
ranging from a single character to a full word, determined by training.

**BPE (Byte Pair Encoding)** — the algorithm behind GPT and Claude tokenizers:

1. Start with individual bytes as the vocabulary.
2. Find the most frequent adjacent pair.
3. Merge it into a new token.
4. Repeat until vocabulary reaches the target size (e.g. 100k).

```
"unhappiness"
→ bytes: [u, n, h, a, p, p, i, n, e, s, s]
→ after BPE merges: ["un", "happ", "iness"]   — 3 tokens
```

**Vocabulary sizes** — larger = rarer subwords stay whole tokens:

| Encoding    | Vocab  | Used by             |
|-------------|--------|---------------------|
| o200k_base  | 200 k  | GPT-4o, o-series    |
| cl100k_base | 100 k  | GPT-4, Claude proxy |
| p50k_base   |  50 k  | GPT-3.5             |

**Anthropic uses a proprietary tokenizer.** `tiktoken` is a close proxy (±5 %).
For billing-accurate counts use `client.messages.count_tokens()` — a network call
that returns the exact count for your exact payload.

**Rules of thumb (English):**

| Content       | Chars/token |
|---------------|-------------|
| Prose         | ~4          |
| Code          | ~2–3        |
| Japanese/Chinese | ~1–2     |
| Arabic        | ~1–2        |

---

### Context windows and "lost in the middle"

Current context windows:

| Model                    | Context  |
|--------------------------|----------|
| Claude Sonnet 4.6        | 200 k    |
| Claude Opus 4.7          | 200 k    |
| GPT-4o                   | 128 k    |
| Gemini 1.5 Pro           | 1 M      |

A large context window does not mean the model uses all of it equally well.

**Liu et al. (2023) — "Lost in the Middle" (arXiv:2307.03172):**
Models recall information from the beginning and end of long contexts
significantly better than from the middle. At 128k+ tokens the degradation
in the middle is pronounced.

```
Recall quality across position in a 128k context:
  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████
  ^high                                                 high^
           start    ← poor recall in middle →    end
```

**Practical implications:**
- Place the most important instructions at the **start** of the system prompt.
- Place the most relevant retrieved documents just **before** the user question.
- For RAG with many chunks, put high-relevance chunks at top and bottom.

---

### Sampling parameters

The model computes a probability distribution over its vocabulary (~100–200 k tokens)
at each generation step. Sampling parameters filter or reshape that distribution.

```
Full vocabulary distribution (raw logits → softmax)
         ▲ probability
         │
  0.35  ─┤  █
  0.20  ─┤  █  █
  0.10  ─┤  █  █  █  █
  0.05  ─┤  █  █  █  █  █  █
         └──────────────────────→ tokens
```

**temperature** — divide logits by `T` before softmax:

```
T < 1  →  sharper distribution  →  more deterministic
T = 1  →  raw distribution (default)
T > 1  →  flatter distribution  →  more random
T = 0  →  greedy: always pick the highest-probability token
```

**top_p** (nucleus sampling) — keep the smallest set of tokens whose cumulative
probability ≥ p, then sample within that set:

```
top_p = 0.9 → include tokens until cumulative P ≥ 90 %
              → discards the long tail of unlikely tokens
```

**top_k** — restrict to the k highest-probability tokens. Anthropic supports this;
OpenAI does not.

**min_p** — exclude tokens with probability < `min_p × max_token_prob`. Dynamic
threshold; not supported by Anthropic or OpenAI directly (used in llama.cpp).

**Interactions:** Temperature reshapes the distribution first, THEN top_p truncates.
Use one of top_p or top_k, not both.

**Determinism caveat:** `temperature=0` is greedy but NOT fully reproducible.
Different GPU types, batch sizes, and API versions can produce different outputs
even at `temperature=0`. For regression tests: mock the HTTP call and assert on
parsed fields (tokens, stop reason), not raw model output.

**Provider validation — `SamplingParams.validate_for_model(model, provider)`**

Call this before sending a request. Different providers reject different params,
often with cryptic errors or silent ignores:

| Parameter     | Anthropic | OpenAI (standard) | OpenAI o-series        |
|---------------|-----------|-------------------|------------------------|
| `temperature` | ✓         | ✓                 | ✗ fixed internally     |
| `top_p`       | ✓         | ✓                 | ✗                      |
| `top_k`       | ✓         | ✗                 | ✗                      |
| `seed`        | ✗         | ✓ (best-effort)   | ✓ (best-effort)        |
| `min_p`       | ✗         | ✗                 | ✗                      |

o-series detection uses model name prefix: `o1`, `o3`, `o4` (e.g. `o4-mini`).

---

### Reasoning models vs general chat

Reasoning models run a chain-of-thought internally before producing a visible answer.
The "thinking" tokens are billed but not shown in the response.

**When reasoning pays off:**

| Task                         | Reasoning | Why                              |
|------------------------------|-----------|----------------------------------|
| Multi-step math / proofs     | ✓ Yes     | Internal scratchpad helps chains |
| Competitive programming      | ✓ Yes     | Backtracking + verification      |
| Complex instructions         | ✓ Yes     | Reduces hallucination            |
| Simple Q&A / retrieval       | ✗ No      | 10–50× cheaper to use chat       |
| Summarisation                | ✗ No      | No novel reasoning needed        |
| Translation                  | ✗ No      | Fast models excel here           |

**Cost structure comparison** (per 1M tokens, USD):

| Model                     | Input  | Output  | Notes                    |
|---------------------------|--------|---------|--------------------------|
| claude-sonnet-4-6         |  $3    |  $15    |                          |
| gpt-4o                    |  $2.5  |  $10    |                          |
| o3-mini                   |  $1.1  |  $4.4   | +reasoning tokens billed |
| claude-sonnet-4-6 thinking|  $3    |  $15    | +thinking tokens billed  |
| o1                        |  $15   |  $60    |                          |

**Claude extended thinking:**

```python
response = await client.complete(
    messages=[{"role": "user", "content": problem}],
    thinking=ThinkingConfig(budget_tokens=10_000),  # max thinking tokens
)
print(response.thinking)  # internal reasoning (visible in API response)
print(response.content)   # final answer
```

Requires `temperature=1.0` — the client enforces this automatically.

**OpenAI o-series:**

```python
response = await client.complete(
    messages=[{"role": "user", "content": problem}],
    model="o3-mini",
    reasoning_effort="high",  # "low" | "medium" | "high"
)
print(response.reasoning_tokens)  # billed but invisible
```

---

### Anthropic SDK

```python
from llm.anthropic_client import AnthropicClient, ThinkingConfig
from llm.sampling import SamplingParams

client = AnthropicClient()

# ── Basic completion ───────────────────────────────────────────────────────
response = await client.complete(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    params=SamplingParams(temperature=0.3, max_tokens=256),
)
print(response.content)           # "4"
print(response.usage)             # Usage(input=12, output=1, ...)
print(response.latency.wall_clock_ms)
print(response.cost.total_usd)

# ── Streaming ─────────────────────────────────────────────────────────────
async for chunk in client.complete_stream(messages):
    if not chunk.is_final:
        print(chunk.delta, end="", flush=True)
    else:
        print(f"\nUsage: {chunk.usage}")

# ── Prompt caching ─────────────────────────────────────────────────────────
# Cache long system prompts by passing content blocks with cache_control.
# TTL: 5 minutes. Min cacheable: ~1 k tokens. Write costs 25 % more.
# Read costs ~10 % of normal. Break-even after ~1.33 reads.
system_blocks = [{"type": "text", "text": long_prompt, "cache_control": {"type": "ephemeral"}}]

# ── Tool use ──────────────────────────────────────────────────────────────
from llm.anthropic_client import ToolDefinition
tools = [
    ToolDefinition(
        name="get_weather",
        description="Get current weather for a city",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
]
response = await client.complete(messages, tools=tools)
for tool_call in response.tool_use:
    print(tool_call.name, tool_call.input)  # "get_weather", {"city": "Paris"}
```

**Usage fields explained:**

| Field              | What it is                                      |
|--------------------|-------------------------------------------------|
| input_tokens       | Tokens in your messages (including cache misses)|
| output_tokens      | Tokens in the response                          |
| cache_read_tokens  | Input tokens served from cache (10 % price)    |
| cache_write_tokens | Input tokens written to cache (125 % price)    |

---

### OpenAI SDK

**Chat Completions** — stable, widely supported:

```python
from llm.openai_client import OpenAIClient
from llm.sampling import SamplingParams

client = OpenAIClient()
response = await client.complete(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o",
    system="You are a concise assistant.",
    params=SamplingParams(temperature=0.7, max_tokens=512, seed=42),
)
```

**Responses API** — newer, better for tools and structured output:

```python
response = await client.complete_responses(
    input="What is the capital of France?",
    model="gpt-4o",
    tools=[{"type": "web_search_preview"}],
)
```

**o-series (reasoning):**

```python
response = await client.complete(
    messages=[{"role": "user", "content": "Prove √2 is irrational"}],
    model="o3-mini",
    reasoning_effort="high",
)
print(f"Reasoning tokens (billed): {response.reasoning_tokens}")
```

---

### Cost math

**Asymmetric pricing:** Input and output tokens are priced differently because:
- Input: processed once through attention (expensive but bounded by sequence length)
- Output: generated auto-regressively one token at a time (more expensive per token)
- Output is typically 3–15× the price of input per million tokens

**Cost formula:**

```
cost = input_tokens × input_price_per_mtok / 1_000_000
     + output_tokens × output_price_per_mtok / 1_000_000
     + cache_write_tokens × write_price_per_mtok / 1_000_000
     + cache_read_tokens × read_price_per_mtok / 1_000_000
```

**Monthly cost at 1k RPD (requests per day):**

```python
from llm.streaming import CostBreakdown, ModelPricing, Usage

pricing = ModelPricing(input_mtok=3.0, output_mtok=15.0)
per_call = CostBreakdown.from_usage(
    Usage(input_tokens=500, output_tokens=200), pricing
).total_usd

monthly = per_call * 1_000 * 30
print(f"${monthly:.2f}/month at 1k RPD")
```

---

## Module map

```
src/
  settings.py         BaseSettings: ANTHROPIC_API_KEY, OPENAI_API_KEY (from .env)
  llm/
    streaming.py        Provider-agnostic types: LLMResponse, Usage, Latency, CostBreakdown
    tokens.py           Tokenization: count_tokens(), tokenize(), estimate_messages_tokens()
    sampling.py         SamplingParams: validation, validate_for_model(), provider serialisation
    anthropic_client.py AnthropicClient: complete(), complete_stream(), tool use, thinking
    openai_client.py    OpenAIClient: complete(), complete_stream(), complete_responses()
```

**Import pattern:** `src/` is the source root (not a package), so all imports within
it are absolute. Submodules under `llm/` import settings as:

```python
from settings import settings          # ✓ correct — src/ is on sys.path
from ..settings import settings        # ✗ fails — llm/ is a top-level package
```

---

## Running tests

```bash
# All 38 unit tests — no API key, no network (HTTP mocked with respx)
make test-stage STAGE=02

# Interactive exploration with a real API key
cd stage-02-llm-fundamentals
uv run jupyter lab notebooks/
```

Test coverage at a glance:

| File                    | What it covers                                           |
|-------------------------|----------------------------------------------------------|
| `test_sampling.py`      | Param validation, provider serialisation, validate_for_model |
| `test_tokens.py`        | count_tokens, tokenize, estimate_messages_tokens         |
| `test_anthropic_client.py` | Completion, streaming, cache tokens, cost, tool use   |
| `test_openai_client.py` | Completion, streaming, o-series max_completion_tokens    |

---

## Exercises

1. **Token budget guard** — Add a `max_input_tokens` parameter to `AnthropicClient.complete()`.
   Before sending the request, call `estimate_messages_tokens()` and raise a `ValueError`
   if the estimate exceeds the budget.

2. **Cost tracker** — Write a `CostAccumulator` class that wraps a client and tracks
   cumulative cost across all calls. Add a `budget_usd` guard that raises when exceeded.

3. **Cache effectiveness** — Run the cache demo in notebook 02 with a 5000-token system
   prompt. Calculate the actual savings vs the write overhead after 10 calls.

4. **Extend `validate_for_model()`** — The implementation handles o-series, top_k,
   seed, and min_p. Add support for Anthropic extended thinking: when a `thinking`
   flag is passed, enforce `temperature=1.0` and `max_tokens >= budget_tokens + 1`.
   Write tests for your new rules.

5. **Streaming accumulator** — Write a helper that collects all `StreamChunk` deltas
   into a final `LLMResponse`, measuring first-token and wall-clock latency from outside
   the client. This mirrors what a UI layer would do.

6. **Reasoning vs chat** — In notebook 03, run the live comparison cell. Measure whether
   extended thinking improves accuracy on the train problem. Try `budget_tokens=1000` vs
   `10000` — does more thinking budget always help?
