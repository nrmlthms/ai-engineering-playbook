# Stage 03 — Prompt Engineering

## Concepts

- System prompt design: role, context, constraints, output format
- Few-shot prompting: when examples beat instructions
- Chain-of-thought (CoT) and extended thinking
- XML tags for structured I/O
- Prompt versioning and regression testing

## Key ideas

```
System prompt
  └─ Role definition
  └─ Task description
  └─ Output format (XML / JSON schema)
  └─ Constraints + examples

User turn
  └─ Task input
  └─ Context injection (RAG hits, tool results)
```

Structured output via XML tags is more reliable than asking for JSON in free
text — the model has seen XML structure throughout pre-training.

## What's in `src/`

| File | Purpose |
|------|---------|
| `prompts/` | Versioned prompt templates |
| `extractor.py` | Parse XML tags from completions |
| `few_shot.py` | Dynamic few-shot example selection |

## Exercises

1. Write a prompt that extracts `<name>`, `<date>`, `<amount>` from invoices
2. Add 3 few-shot examples and measure accuracy on 10 test invoices
3. Enable extended thinking and compare reasoning quality
