"""
Correlation ID middleware.

X-Request-Id is local to one service. X-Correlation-Id flows across the
entire call chain — Service A sets it, passes it to Service B, which passes
it to Service C. Every service logs it so you can trace a single user action
across all services in a distributed system.

If no correlation ID arrives from upstream, we generate one (this service
is the root of the call chain).
"""

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    HEADER = "X-Correlation-Id"

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = request.headers.get(self.HEADER) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        response = await call_next(request)

        # Propagate downstream so callers can correlate their logs
        response.headers[self.HEADER] = correlation_id
        return response
