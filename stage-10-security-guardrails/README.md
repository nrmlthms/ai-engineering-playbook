# Stage 10 — Security & Guardrails

## Concepts

- Prompt injection: direct and indirect attacks
- PII detection and redaction before sending to LLMs
- Output filtering: toxicity, off-topic, hallucination guards
- Red-teaming: adversarial inputs, jailbreak patterns
- Defence in depth: input → model → output layers

## Key ideas

```
User input
    ↓
[Input guard]  ← PII redaction, injection detection
    ↓
LLM call
    ↓
[Output guard] ← hallucination check, policy filter
    ↓
Response
```

Never trust content retrieved from the web or user documents — it may contain
injected instructions. Always process retrieved content as data, not as
instructions in the system prompt.

## What's in `src/`

| File | Purpose |
|------|---------|
| `guards/input.py` | PII redaction + injection classifier |
| `guards/output.py` | Hallucination + policy output filter |
| `redteam.py` | Automated adversarial test runner |

## Exercises

1. Build a PII redactor that strips emails, phone numbers, and SSNs before sending
2. Add an injection detector that flags `ignore previous instructions` patterns
3. Run 20 red-team prompts and measure how many your guards catch
