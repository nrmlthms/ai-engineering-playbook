# Stage 05 — Agentic Orchestration

## Concepts

- Tool use: defining tools, parsing tool calls, returning results
- Planning loops: ReAct, plan-and-execute, reflection
- Multi-agent patterns: orchestrator + subagents
- Interruption and human-in-the-loop
- Token budgets across multi-step traces

## Key ideas

```
while not done:
    response = llm(messages + tools)
    if response.stop_reason == "tool_use":
        result = execute_tool(response.tool_call)
        messages.append(tool_result(result))
    else:
        done = True
```

The agentic loop is simple — the complexity is in tool design (idempotency,
error messages the model can act on) and knowing when to stop.

## What's in `src/`

| File | Purpose |
|------|---------|
| `tools/` | Tool definitions + implementations |
| `loop.py` | Core agentic loop with budget guard |
| `orchestrator.py` | Spawn and coordinate subagents |

## Exercises

1. Build a research agent with `web_search` + `read_url` tools
2. Add a budget guard that stops after N tool calls
3. Implement human-in-the-loop approval for destructive tool calls
