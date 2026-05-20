# Stage 06 — Production Agents

## Concepts

- Agent state: in-memory vs persistent (Redis, DB)
- Long-running tasks: async queues, webhooks, polling
- Memory systems: episodic, semantic, procedural
- Checkpointing and resumability
- Observability for agents: tracing multi-step runs

## Key ideas

```
Agent run
  ├─ Working memory  (current context window)
  ├─ Episodic memory (past runs, stored externally)
  └─ Semantic memory (facts / embeddings in vector DB)
```

Production agents fail mid-run. Checkpointing every tool result to durable
storage means you can resume from the last successful step instead of
restarting from scratch (and paying for all those tokens again).

## What's in `src/`

| File | Purpose |
|------|---------|
| `state.py` | Checkpoint + restore agent state |
| `memory/` | Episodic and semantic memory backends |
| `queue.py` | Async task queue with status polling |

## Exercises

1. Add Redis-backed checkpointing to the Stage 05 agent loop
2. Implement an episodic memory that summarises past runs
3. Expose a `/status/{run_id}` endpoint for long-running tasks
