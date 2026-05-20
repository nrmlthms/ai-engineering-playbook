# Stage 09 — Deployment & Observability

## Concepts

- Structured logging: request IDs, latency, token counts
- Distributed tracing: OpenTelemetry spans across agent steps
- Metrics: p50/p95 latency, error rate, cost per request
- Alerting: cost anomalies, latency regressions
- Deployment: Docker, environment config, secrets management

## Key ideas

```
Every LLM call should emit:
  ├─ trace_id      (tie all steps of one user request together)
  ├─ model         (know which model version answered)
  ├─ input_tokens  (cost numerator)
  ├─ output_tokens (cost numerator)
  ├─ latency_ms    (p95 drives SLA)
  └─ cache_hit     (measure caching ROI)
```

Instrument first, optimise second. You cannot reduce costs you cannot see.

## What's in `src/`

| File | Purpose |
|------|---------|
| `telemetry.py` | OTel setup: tracer, meter, logger |
| `middleware.py` | FastAPI middleware that wraps every call |
| `cost.py` | Real-time cost calculator per model |

## Exercises

1. Add OTel spans to the Stage 05 agentic loop
2. Build a Prometheus dashboard showing cost per minute
3. Set up an alert that fires when p95 latency exceeds 5 s
