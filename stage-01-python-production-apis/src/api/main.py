"""
FastAPI application factory.

Registers middleware in order (outermost first — last added runs first):
  CorrelationId → RequestId → Timing → Idempotency → routes

Sub-apps:
  /v1/* → REST API (items, webhooks)
  /graphql → GraphQL (optional, requires strawberry-graphql)
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
import structlog.contextvars
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..clients.llm_http import close_http_client
from ..errors import register_handlers
from ..health import router as health_router
from ..idempotency import IdempotencyMiddleware
from ..middleware.correlation import CorrelationIdMiddleware
from ..middleware.request_id import RequestIdMiddleware
from ..middleware.timing import TimingMiddleware
from ..routes.items import router as items_router
from ..routes.webhooks import router as webhooks_router
from ..settings import settings


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            10 if settings.debug else 20  # DEBUG in dev, INFO in prod
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _configure_logging()
    log = structlog.get_logger()
    log.info("startup", version=settings.app_version, debug=settings.debug)
    yield
    await close_http_client()
    log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Engineer Handbook — Stage 01",
        description="Python Production APIs",
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        # Disable the default 422 detail in OpenAPI (we use RFC 7807 shape)
        responses={422: {"description": "Validation Error"}},
    )

    # ── Middleware (outermost first) ──────────────────────────────────────────
    # Note: FastAPI/Starlette executes middleware in reverse registration order.
    # Last added = outermost = runs first on request, last on response.
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(TimingMiddleware, slow_threshold_ms=500)
    app.add_middleware(IdempotencyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────────────
    register_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(items_router, prefix="/v1")
    app.include_router(webhooks_router, prefix="/v1")

    # Optional GraphQL — only mounted if strawberry is installed
    try:
        from ..routes.graphql import graphql_router

        app.include_router(graphql_router, prefix="/graphql")
    except ImportError:
        pass

    return app


# Entry points
# uvicorn:  uvicorn src.api.main:app --reload
# granian:  granian --interface asgi src.api.main:app
app = create_app()
