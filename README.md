# ai-engineer-handbook

> Production-grade AI engineering — concepts + code, Stage 1 to 10

## Roadmap

| Stage | Topic | Focus |
|-------|-------|-------|
| [01](stage-01-python-production-apis/) | Python Production APIs | FastAPI, async, Pydantic, Docker |
| [02](stage-02-llm-fundamentals/) | LLM Fundamentals | Tokens, context, sampling, the Anthropic SDK |
| [03](stage-03-prompt-engineering/) | Prompt Engineering | System prompts, few-shot, chain-of-thought |
| [04](stage-04-rag-knowledge-grounding/) | RAG & Knowledge Grounding | Embeddings, vector DBs, retrieval pipelines |
| [05](stage-05-agentic-orchestration/) | Agentic Orchestration | Tool use, planning loops, multi-agent patterns |
| [06](stage-06-production-agents/) | Production Agents | State management, memory, long-running tasks |
| [07](stage-07-evaluation-reliability/) | Evaluation & Reliability | DeepEval, LLM-as-judge, regression suites |
| [08](stage-08-multimodal-data/) | Multimodal & Data | Vision, audio, structured extraction |
| [09](stage-09-deployment-observability/) | Deployment & Observability | Logging, tracing, cost tracking, latency |
| [10](stage-10-security-guardrails/) | Security & Guardrails | Prompt injection, PII filtering, red-teaming |

## Stage anatomy

Every stage shares the same layout:

```
stage-NN-topic/
  README.md       # concept notes + diagrams
  src/            # production-grade Python
  tests/          # pytest + deepeval evals
  notebooks/      # explorations + demos
```

## Quickstart

```bash
make install    # install all deps
make lint       # ruff check + format check
make typecheck  # mypy
make test       # run tests for a stage: make test-stage STAGE=01
```
