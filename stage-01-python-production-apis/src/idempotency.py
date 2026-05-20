"""
Idempotency key middleware.

The Idempotency-Key header lets clients safely retry POST/PATCH requests.
If the server already processed a request with this key, it returns the
cached response instead of processing again.

This is critical for payment APIs, order creation, or any mutation where
a network timeout could cause the client to retry an already-completed action.

Flow:
  Client sends POST /v1/orders with Idempotency-Key: <uuid>
    → First call:  process normally, cache (key, request_hash, response), return
    → Retry call:  return cached response immediately (no processing)
    → Different body, same key: return 422 IdempotencyConflict

Storage: in-memory dict here. In production use Redis with TTL.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from .errors import IdempotencyConflict

log = structlog.get_logger()

IDEMPOTENCY_HEADER = "Idempotency-Key"
# Only cache these methods — GET/HEAD are already idempotent
IDEMPOTENT_METHODS = {"POST", "PATCH"}


@dataclass
class CachedResponse:
    status_code: int
    body: bytes
    headers: dict[str, str]
    request_hash: str
    created_at: float = field(default_factory=time.monotonic)


# In-memory store. Replace with Redis in production:
#   await redis.setex(f"idempotency:{key}", ttl_seconds, serialised_response)
_cache: dict[str, CachedResponse] = {}
_TTL_SECONDS = 86_400  # 24 hours


def _hash_body(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _is_expired(entry: CachedResponse) -> bool:
    return (time.monotonic() - entry.created_at) > _TTL_SECONDS


class IdempotencyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method not in IDEMPOTENT_METHODS:
            return await call_next(request)

        key = request.headers.get(IDEMPOTENCY_HEADER)
        if not key:
            return await call_next(request)

        # Read body early — BaseHTTPMiddleware only allows one read
        body = await request.body()
        request_hash = _hash_body(body)

        cached = _cache.get(key)

        if cached and not _is_expired(cached):
            if cached.request_hash != request_hash:
                # Same key, different body — client error
                raise IdempotencyConflict()

            log.info("idempotency_cache_hit", key=key)
            return JSONResponse(
                status_code=cached.status_code,
                content=json.loads(cached.body),
                headers={**cached.headers, "Idempotency-Replayed": "true"},
            )

        # First time — process and cache
        response = await call_next(request)

        # Only cache successful responses (2xx)
        if 200 <= response.status_code < 300:
            response_body = b""
            async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                response_body += chunk

            _cache[key] = CachedResponse(
                status_code=response.status_code,
                body=response_body,
                headers=dict(response.headers),
                request_hash=request_hash,
            )

            return JSONResponse(
                status_code=response.status_code,
                content=json.loads(response_body),
                headers=dict(response.headers),
            )

        return response
