.PHONY: install sync test test-stage lint fmt typecheck clean notebook

# ── Setup (uv) ────────────────────────────────────────────────────────────────
# uv is the project's package manager. Install it once with: curl -LsSf https://astral.sh/uv/install.sh | sh

install:
	uv sync --all-extras

# Re-lock dependencies after editing pyproject.toml
lock:
	uv lock

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy .

check: lint typecheck   ## Run all static checks

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	uv run pytest -v --tb=short

# Run a single stage: make test-stage STAGE=01
test-stage:
	uv run pytest -v --tb=short stage-$(STAGE)-*/tests/

# ── Evals (Stage 07+) ─────────────────────────────────────────────────────────

eval:
	uv run deepeval test run stage-07-*/evals/ stage-08-*/evals/ stage-09-*/evals/ stage-10-*/evals/

# ── Notebooks ─────────────────────────────────────────────────────────────────

notebook:
	uv run jupyter lab

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
