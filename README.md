# AI Engineer Handbook

> Production-grade AI engineering — from async Python APIs to deployed, observable, secure AI systems.
> 10 stages. Real code. No fluff.

---

## Why this exists

No course covers the full stack. Books go out of date before they ship.
Tutorials stop at the demo. Asking an AI gets you answers but not structure.

This is an attempt to put everything in one place — opinionated, current, and runnable.
Every concept lives next to production-grade code you can execute, test, and extend.

---

## Who it's for

| You are… | You get… |
|----------|----------|
| A software engineer moving into AI/ML | A structured path that starts from what you already know (APIs, async, Docker) |
| An ML engineer closing the productionisation gap | Stages 06–10: agents, evals, observability, security |
| A builder who's hit the ceiling of "just prompt the model" | RAG, tool use, evals, guardrails — the full stack |
| A student who learns best from running real code | Exercises at the end of every stage |

---

## Learning path

| Stage | Topic | What you build | Key tech |
|-------|-------|----------------|----------|
| [01](stage-01-python-production-apis/) | Python Production APIs | Async REST + GraphQL service | FastAPI · Pydantic v2 · Docker |
| [02](stage-02-llm-fundamentals/) | LLM Fundamentals | Token-aware Anthropic SDK client with caching | Anthropic SDK · streaming · prompt caching |
| [03](stage-03-prompt-engineering/) | Prompt Engineering | Versioned prompt library with regression tests | CoT · few-shot · XML tags · extended thinking |
| [04](stage-04-rag-knowledge-grounding/) | RAG & Knowledge Grounding | End-to-end retrieval pipeline | Embeddings · pgvector · reranking |
| [05](stage-05-agentic-orchestration/) | Agentic Orchestration | Multi-agent tool-use loop with budget guard | Tool use · ReAct · orchestrator pattern |
| [06](stage-06-production-agents/) | Production Agents | Stateful, resumable long-running agent | State machine · episodic memory · Celery |
| [07](stage-07-evaluation-reliability/) | Evaluation & Reliability | DeepEval regression suite wired into CI | DeepEval · LLM-as-judge · golden datasets |
| [08](stage-08-multimodal-data/) | Multimodal & Data | Document ingestion pipeline (PDFs + images) | Vision · structured extraction · pypdf |
| [09](stage-09-deployment-observability/) | Deployment & Observability | Traced, cost-tracked service with Grafana dashboard | OpenTelemetry · GenAI conventions · Prometheus |
| [10](stage-10-security-guardrails/) | Security & Guardrails | Hardened agent with injection defence + PII filter | Prompt injection · guardrails · red-teaming |

---

## How to use this

Each stage is self-contained. You can start at Stage 01 and work forward, or jump to any
stage that covers a gap in your knowledge. Stages 06–10 assume familiarity with earlier concepts,
but each README is written to stand alone.

### Inside every stage

```
stage-NN-topic/
  README.md       ← concept notes, code examples, ASCII diagrams
  src/            ← production-grade Python (the thing being learned)
  tests/          ← pytest + deepeval evals
  notebooks/      ← interactive explorations and demos
  docker/         ← Dockerfile + compose (Stage 01 and 06–10)
  evals/          ← deepeval eval suites (Stage 07–10)
```

The README for each stage is the primary learning document — it is more than a table of
contents. Read it, run the code, do the exercises at the end.

---

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — used for all package management
- Docker (for Stage 01 and Stage 06+)
- An Anthropic API key (for Stage 02+)

```bash
# Clone
git clone https://github.com/nrmlthms/ai-engineering-playbook
cd ai-engineering-playbook

# Copy env template
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Install all dependencies
make install
```

### Running a stage

```bash
# Lint
make lint

# Type check
make typecheck

# Run tests for a specific stage
make test-stage STAGE=01

# Run all tests
make test
```

### Stage-specific notebooks

```bash
cd stage-02-llm-fundamentals
uv run jupyter lab notebooks/
```

---

## Conventions

This project enforces a consistent set of practices across all stages.
See [CLAUDE.md](CLAUDE.md) for the full spec. The short version:

| Concern | Choice | Why |
|---------|--------|-----|
| Package manager | `uv` | Fast, deterministic, modern |
| Linting + formatting | `ruff` | Single tool, very fast |
| Type checking | `mypy --strict` | Catches real bugs before runtime |
| Logging | `structlog` | Machine-readable, context-carrying |
| Config / secrets | `pydantic-settings` | Validated at startup, not scattered `os.environ[]` calls |
| Testing | `pytest` + `pytest-asyncio` | Standard, composable |
| Observability | OpenTelemetry (Stage 09+) | GenAI semantic conventions |

---

## Stage dependency map

```
01 ──► 02 ──► 03 ──► 04 ──► 05
                              │
                              ▼
                        06 ──► 07 ──► 08
                              │
                              ▼
                        09 ──► 10
```

Stages 01–05 form the core path. Stage 06+ builds on all of them.
You can read Stage 08 (multimodal) independently if you already know RAG.

---

## Contributing

Contributions are welcome — bug fixes, new exercises, improved explanations, additional
code examples. A few ground rules to keep the repo coherent:

1. **Follow the conventions** in [CLAUDE.md](CLAUDE.md) — `uv`, `ruff`, `mypy --strict`,
   `structlog`, `pydantic-settings`. PRs that use `print()` or `os.environ[]` will be
   sent back.

2. **One stage, one PR** — keep changes focused. A PR that touches Stage 04 and Stage 07
   is hard to review.

3. **Exercises must have solutions** — if you add an exercise, add a reference solution
   in the stage's `src/` or `tests/`. Unsolvable exercises are worse than no exercises.

4. **No OTel before Stage 09** — observability is introduced at the right time.
   Adding spans to Stage 03 obscures what's being taught.

5. **Run `make lint && make typecheck && make test-stage STAGE=NN`** before opening a PR.

---

## Roadmap

- [ ] Stage 01 — Python Production APIs ✅
- [ ] Stage 02 — LLM Fundamentals (src + tests in progress)
- [ ] Stage 03 — Prompt Engineering (src + tests in progress)
- [ ] Stage 04 — RAG & Knowledge Grounding
- [ ] Stage 05 — Agentic Orchestration
- [ ] Stage 06 — Production Agents
- [ ] Stage 07 — Evaluation & Reliability
- [ ] Stage 08 — Multimodal & Data
- [ ] Stage 09 — Deployment & Observability
- [ ] Stage 10 — Security & Guardrails

---

## License

MIT
