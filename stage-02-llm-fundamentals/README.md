# Stage 02 — LLM Fundamentals

## Concepts

- Tokenisation: BPE, vocab size, token budgets
- Context windows: KV cache, attention complexity
- Sampling: temperature, top-p, top-k
- The Anthropic SDK: messages API, streaming, tool use
- Prompt caching and cost optimisation

## Key ideas

```
Prompt tokens + completion tokens = total cost
         ↓
  Cached prefix → ~90 % cheaper on repeat calls
```

Temperature ≠ creativity — it controls the sharpness of the probability
distribution over the next token. At 0 you get greedy decoding (always the
highest-prob token); at 1 you sample from the raw distribution.

## What's in `src/`

| File | Purpose |
|------|---------|
| `client.py` | Thin wrapper around `anthropic.Anthropic` with caching |
| `tokeniser.py` | Count tokens before sending |
| `streaming.py` | Stream completions with `stream()` |

## Exercises

1. Count tokens for a system prompt and estimate monthly cost at 1 k RPD
2. Enable prompt caching and measure cache-hit rate over 10 calls
3. Compare outputs at temperature 0, 0.5, and 1.0 for a creative task
