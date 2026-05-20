# Multi-stage build targeting < 250 MB final image
# Stage 1: builder  — installs dependencies with uv
# Stage 2: final    — distroless Python, non-root user, no shell

# ── Builder ───────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv (fast pip replacement)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (layer-cached until pyproject.toml changes)
COPY pyproject.toml .
COPY uv.lock* .

# Install production deps only into /app/.venv
RUN uv sync --no-dev --frozen

# Copy application source
COPY src/ src/


# ── Final (distroless) ────────────────────────────────────────────────────────
# gcr.io/distroless/python3-debian12:
#   - No shell, no package manager → smaller attack surface
#   - ~50 MB vs ~120 MB for python:3.12-slim
#   - Use the debug variant (has busybox shell) during development
FROM gcr.io/distroless/python3-debian12 AS final

WORKDIR /app

# Non-root: distroless has user 65532 (nonroot) built in
USER nonroot

# Copy installed packages and app from builder
COPY --from=builder --chown=nonroot:nonroot /app/.venv /app/.venv
COPY --from=builder --chown=nonroot:nonroot /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Granian: Rust-native ASGI server — faster cold start than uvicorn+gunicorn
# Scale horizontally (more replicas), not vertically (workers per container)
CMD ["granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "api.main:app"]

# HEALTHCHECK used by Docker (and ECS, Fly.io, etc.) to know when the
# container is ready. start_period gives the app time to bind the port.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", \
         "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

EXPOSE 8000
