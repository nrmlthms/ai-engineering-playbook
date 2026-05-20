# AI Engineer Handbook — Project Conventions

## Language & runtime

- **Python 3.12** minimum
- **uv** for all package management (`uv sync`, `uv run`, `uv lock`)
- Never use `pip` directly — always go through `uv`

## Code quality

| Tool | Config | Command |
|------|--------|---------|
| **ruff** | `pyproject.toml` → `[tool.ruff]` | `make lint` / `make fmt` |
| **mypy --strict** | `pyproject.toml` → `[tool.mypy]` | `make typecheck` |

All code must pass both before committing.

## Logging

Use **structlog** everywhere. Never use `print()` or the stdlib `logging` module directly.

```python
import structlog
log = structlog.get_logger()

log.info("item_created", item_id=item.id, price=item.price)
log.warning("retry", attempt=2, delay_s=1.2, url=url)
log.error("payment_failed", order_id=order_id, error=str(exc))
```

- Log events as snake_case verbs: `item_created`, `payment_failed`, `retry`
- Use keyword args for structured fields — never f-strings in the message
- In FastAPI: configure structlog in `lifespan`, not at module level

## Configuration & secrets

Use **pydantic-settings** `BaseSettings`. Never hardcode secrets or use `os.environ[]` directly.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    anthropic_api_key: str
    database_url: str = "sqlite+aiosqlite:///./dev.db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
```

## Testing

- **pytest** + **pytest-asyncio** (`asyncio_mode = "auto"` in `pyproject.toml`)
- Each stage has a `conftest.py` that adds its `src/` to `sys.path`
- Tests import directly from `src/` — no package installation required
- Run one stage: `make test-stage STAGE=01`

## Observability (Stage 09+ only)

**OpenTelemetry** with GenAI semantic conventions is introduced in Stage 09.
Do **not** add OTel spans or metrics to Stages 01–08 — it obscures the concepts being taught.

When OTel is used (Stage 09+), follow the GenAI semantic conventions:
- `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, etc.

## Stage directory layout

```
stage-NN-<topic>/
  README.md          # concept notes + ASCII diagrams + exercises
  conftest.py        # adds src/ to sys.path for pytest
  src/               # production-grade Python (the thing being learned)
  tests/             # pytest unit tests
  notebooks/         # Jupyter explorations and demos
  docker/            # Dockerfile + compose (Stage 06+ only)
  evals/             # deepeval eval suites (Stage 07+ only)
```

### Which stages get which extras

| Directory | Stages |
|-----------|--------|
| `src/` `tests/` `notebooks/` | All (01–10) |
| `docker/` | 01 (Docker is a named Stage 01 topic), 06–10 |
| `evals/` | 07–10 |

## What NOT to add

- `examples/` — notebooks cover this; two folders for the same purpose adds confusion
- OTel in Stages 01–08 — introduce it at the right time (Stage 09)
- `print()` or raw `logging` calls — use structlog
- Hardcoded secrets or `os.environ[]` calls — use pydantic-settings
