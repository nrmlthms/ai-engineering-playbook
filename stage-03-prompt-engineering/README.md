# Stage 03 — Prompt Engineering

> Reliable structured output, few-shot prompting, and chain-of-thought — the
> three techniques that turn an LLM into a predictable production component.

---

## What you build

A reusable prompting toolkit (`src/`) with versioned templates, XML extraction,
few-shot formatting, and CoT utilities. Every piece is offline-testable — no
API key needed to run the suite.

---

## Concepts

### XML structured output

LLMs don't guarantee valid JSON in free-text completions. They may add prose,
forget closing brackets, or use trailing commas. XML tags sidestep this:

```
JSON risk                          XML tags
──────────────────────────────     ─────────────────────────────────────
Sure! Here you go:                 <vendor>Acme Corp</vendor>
                                   <date>2025-01-15</date>
```json                            <amount>1250.00</amount>
{
  "vendor": "Acme Corp",  // note: comment = invalid JSON
  "date": "Jan 15 2025"   // note: not ISO format
}
```
```

The model has seen XML throughout pre-training (HTML, DocBook, man pages, code
comments). Closing a `<tag>` is natural text completion. Closing a `}` requires
tracking stack depth.

**Extraction pattern:**

```python
from extractor import extract_tags, assert_tags_present

text = model_response.content
fields = extract_tags(text, ["vendor", "date", "amount", "invoice_number"])
assert_tags_present(fields, ["vendor", "date", "amount"])   # hard error if missing
```

---

### Prompt versioning

A prompt is executable specification. Tracking versions lets you:
- Roll back after a silent regression (output changed, no exception raised)
- A/B test two prompts with identical inputs
- Audit which prompt was in production at time T

```python
from prompts.template import PromptTemplate

SUMMARISER = PromptTemplate(
    name="summariser",
    version="2026-01-01",
    system="You are a document summarisation assistant. ...",
    user_template="Summarise this document:\n\n{document}",
)

system, user = SUMMARISER.render(document=article_text)
# → raises ValueError immediately if {document} is not passed
```

`render()` fails fast on missing variables — prompt bugs surface at call time,
not as garbled model output.

---

### Few-shot prompting

Examples demonstrate behaviour; instructions describe it. For tasks where
"correct" is hard to express as a rule (tone, edge-case handling, output
format), three good examples often beat a paragraph of prose.

```
k examples  Effect
──────────  ──────────────────────────────────────────────────────────
0           baseline — model uses training knowledge only
1           large jump — one example sets the expected output format
3           further gain
10+         diminishing returns; context cost grows linearly
```

**Message format** — examples are prior conversation turns, not system prompt text:

```python
from few_shot import FewShotExample, FewShotFormatter

fmt = FewShotFormatter(examples)
messages = fmt.prepend_to_messages(
    [{"role": "user", "content": query}],
    n=3,
    strategy="by_label",   # "first" | "random" | "by_label"
)
# → [user, assistant, user, assistant, user, assistant, user(query)]
```

**Selection strategies:**

| Strategy   | When to use                                                  |
|------------|--------------------------------------------------------------|
| `first`    | Fast prototyping; deterministic                              |
| `random`   | Reduces ordering bias; use `seed=` for reproducible runs     |
| `by_label` | Classification tasks — ensures all classes are represented   |

---

### Chain-of-thought

#### Zero-shot CoT — Kojima et al. (2022)

Appending `"Let's think step by step."` dramatically improves accuracy on
math and logic benchmarks — no examples needed. One phrase activates a
reasoning mode the model learned from worked solutions in pretraining.

```python
from chain_of_thought import zero_shot_cot

prompt = zero_shot_cot("A bat and a ball cost $1.10...")
# → "A bat and a ball cost $1.10...\n\nLet's think step by step."
```

#### Scratchpad pattern

Ask the model to reason in `<thinking>` tags, answer in `<answer>` tags:

```python
from chain_of_thought import build_scratchpad_system, extract_cot_answer

system = build_scratchpad_system("You are a math tutor.")
response = await client.complete(messages=[...], system=system)

thinking, answer = extract_cot_answer(response.content)
# thinking → inspectable, loggable, never shown to user
# answer   → parsed, returned to user
```

**Compared to Claude extended thinking (Stage 02):**

| Feature          | Scratchpad (prompt) | Extended thinking (API) |
|------------------|---------------------|-------------------------|
| Works on         | Any model           | Claude Sonnet/Opus 4+   |
| Thinking visible | Always              | Via `response.thinking` |
| Billing bucket   | Output tokens       | Separate thinking tokens |
| Reliability      | Good                | Better on hard tasks    |

---

## Module map

```
src/
  extractor.py        XML tag extraction: extract_tag(), extract_tags(), assert_tags_present()
  few_shot.py         FewShotExample, FewShotFormatter (select, to_messages, prepend_to_messages)
  chain_of_thought.py zero_shot_cot(), scratchpad pattern, extract_cot_answer()
  prompts/
    template.py       PromptTemplate: versioned, variable interpolation, render()
    examples.py       Ready-to-use templates: INVOICE_EXTRACTOR*, SENTIMENT_CLASSIFIER, STRUCTURED_SUMMARISER
```

`*` = system prompt stub — you write it in Exercise 1.

---

## Running tests

```bash
# 37 unit tests — no API key, no network
make test-stage STAGE=03
```

| File                       | What it covers                                          |
|----------------------------|---------------------------------------------------------|
| `test_extractor.py`        | Tag extraction, whitespace, missing tags, multi-tag     |
| `test_template.py`         | render(), variable detection, error on missing var      |
| `test_few_shot.py`         | All three selection strategies, message formatting      |
| `test_chain_of_thought.py` | zero_shot_cot, scratchpad build/extract                 |

---

## Exercises

1. **Write the invoice extractor system prompt** — `src/prompts/examples.py`,
   `INVOICE_EXTRACTOR.system` is a stub. Write a prompt that reliably extracts
   `<vendor>`, `<date>`, `<amount>`, `<invoice_number>` from invoice text.
   Test it in `notebooks/01_xml_structured_output.py` section 4.
   Consider: role definition, explicit tag list, handling missing fields,
   one worked example inside the system prompt.

2. **Few-shot accuracy measurement** — Run `notebooks/02_few_shot_selection.py`
   section 4. Classify 10 test reviews at 0-shot, 1-shot, and 3-shot.
   Record accuracy. At what k does performance plateau?

3. **Scratchpad vs extended thinking** — Run `notebooks/03_chain_of_thought.py`
   section 4. Compare the scratchpad pattern and Claude extended thinking on the
   bat-and-ball problem. Does extended thinking use more words? Does it get the
   right answer more reliably?

4. **Prompt version diff** — Create a `v2` of `INVOICE_EXTRACTOR` that asks the
   model to also extract `<currency>` (e.g. `GBP`, `USD`). Run both versions on
   the same invoice and compare outputs. This simulates a production prompt
   migration.

5. **Label-stratified selection deep dive** — The `by_label` strategy round-robins
   across label groups. Extend `FewShotFormatter.select()` to support
   `strategy="proportional"`: sample proportionally to label frequency in the
   bank. For example, if you have 10 negative reviews and 2 positive, proportional
   sampling preserves that 5:1 ratio. Write tests for the new strategy.
