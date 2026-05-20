"""
RFC 7807 Problem Details for HTTP APIs.
https://www.rfc-editor.org/rfc/rfc7807

Standard error envelope — every error, from 400 to 500, returns the same shape.
Clients only need one error-handling branch.

Content-Type: application/problem+json

{
  "type":     "https://errors.example.com/item-not-found",
  "title":    "Item Not Found",
  "status":   404,
  "detail":   "Item 42 does not exist in this catalogue.",
  "instance": "/v1/items/42"
}
"""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

PROBLEM_CONTENT_TYPE = "application/problem+json"


# ── RFC 7807 envelope ─────────────────────────────────────────────────────────


class ProblemDetail(BaseModel):
    type: str  # URI identifying the problem type
    title: str  # Short, human-readable summary (stable per type)
    status: int  # HTTP status code
    detail: str | None = None  # Human-readable explanation for this occurrence
    instance: str | None = None  # URI of the specific occurrence (e.g. request path)
    # Extensions: any extra fields are allowed per RFC 7807 §3.2
    extensions: dict | None = None


def problem_response(
    status: int,
    type_slug: str,
    title: str,
    detail: str | None = None,
    instance: str | None = None,
    **extensions,
) -> JSONResponse:
    body = ProblemDetail(
        type=f"https://errors.example.com/{type_slug}",
        title=title,
        status=status,
        detail=detail,
        instance=instance,
        extensions=extensions or None,
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(exclude_none=True),
        media_type=PROBLEM_CONTENT_TYPE,
    )


# ── Domain exceptions — no HTTP knowledge ────────────────────────────────────


class ItemNotFound(Exception):
    def __init__(self, item_id: int) -> None:
        self.item_id = item_id


class InsufficientStock(Exception):
    def __init__(self, item_id: int, requested: int, available: int) -> None:
        self.item_id = item_id
        self.requested = requested
        self.available = available


class WebhookSignatureInvalid(Exception):
    pass


class IdempotencyConflict(Exception):
    """Same idempotency key, different request body."""

    pass


# ── Handlers ──────────────────────────────────────────────────────────────────


async def item_not_found_handler(request: Request, exc: ItemNotFound) -> JSONResponse:
    return problem_response(
        404,
        "item-not-found",
        "Item Not Found",
        detail=f"Item {exc.item_id} does not exist.",
        instance=str(request.url.path),
    )


async def insufficient_stock_handler(request: Request, exc: InsufficientStock) -> JSONResponse:
    return problem_response(
        409,
        "insufficient-stock",
        "Insufficient Stock",
        detail=f"Requested {exc.requested}, only {exc.available} available.",
        instance=str(request.url.path),
        item_id=exc.item_id,
    )


async def webhook_signature_handler(request: Request, exc: WebhookSignatureInvalid) -> JSONResponse:
    return problem_response(400, "invalid-signature", "Invalid Webhook Signature")


async def idempotency_conflict_handler(request: Request, exc: IdempotencyConflict) -> JSONResponse:
    return problem_response(
        422,
        "idempotency-conflict",
        "Idempotency Key Reused With Different Body",
        detail="The same Idempotency-Key was used with a different request body.",
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return problem_response(
        422,
        "validation-error",
        "Request Validation Failed",
        detail="One or more fields failed validation.",
        instance=str(request.url.path),
        errors=exc.errors(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    import structlog

    log = structlog.get_logger()
    log.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
    return problem_response(500, "internal-error", "Internal Server Error")


def register_handlers(app) -> None:  # type: ignore[type-arg]
    app.add_exception_handler(ItemNotFound, item_not_found_handler)
    app.add_exception_handler(InsufficientStock, insufficient_stock_handler)
    app.add_exception_handler(WebhookSignatureInvalid, webhook_signature_handler)
    app.add_exception_handler(IdempotencyConflict, idempotency_conflict_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
