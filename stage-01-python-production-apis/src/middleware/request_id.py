"""
Request ID middleware.

Attaches a unique ID to every request. If the caller sends X-Request-Id,
we echo it back (useful for client-side correlation). Otherwise we generate one.

Every log line inside the request handler should include this ID so you can
grep all logs for a single request across distributed services.
"""

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


class RequestIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, header: str = "X-Request-Id") -> None:
        super().__init__(app)
        self.header = header

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        req_id = request.headers.get(self.header) or str(uuid.uuid4())

        # Attach to request.state so handlers can access it
        request.state.request_id = req_id

        # Bind to structlog context — all log calls within this request will
        # automatically include request_id without passing it manually
        structlog.contextvars.bind_contextvars(request_id=req_id)

        response = await call_next(request)
        response.headers[self.header] = req_id

        # Clear bound vars so the next request on this worker starts clean
        structlog.contextvars.clear_contextvars()

        return response
