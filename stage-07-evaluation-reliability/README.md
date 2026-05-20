# Stage 07 — Evaluation & Reliability

## Concepts

- Evaluation dimensions: correctness, faithfulness, relevance, safety
- LLM-as-judge: grading outputs with a model
- DeepEval: metrics, test cases, regression suites
- Golden datasets and prompt regression testing
- Statistical significance for A/B evals

## Key ideas

```
Eval pipeline
  Input → system under test → output
                                 ↓
                    LLM judge (scoring rubric)
                                 ↓
                    Score + reason → aggregate metrics
```

LLM-as-judge is powerful but biased toward verbosity and its own outputs.
Always calibrate against human ratings on a small gold set before trusting
automated scores.

## What's in `src/`

| File | Purpose |
|------|---------|
| `judge.py` | LLM judge with structured scoring |
| `dataset.py` | Load / manage golden test datasets |
| `runner.py` | Run evals and write reports |

## Exercises

1. Build a faithfulness judge for the Stage 04 RAG pipeline
2. Create a 20-example golden dataset and measure baseline accuracy
3. Run an A/B eval comparing two prompt versions with statistical testing
