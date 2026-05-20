"""
Pydantic v2 patterns for production APIs.

Covers:
  - Separate input/output models
  - field_validator and model_validator
  - Discriminated unions (tagged payloads)
  - TypeAdapter for validating arbitrary structures
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

# ── Basic CRUD models ─────────────────────────────────────────────────────────


class ItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    price: float = Field(gt=0, description="Price in USD")
    tags: list[str] = []

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, tags: list[str]) -> list[str]:
        return [t.lower().strip() for t in tags if t.strip()]


class ItemUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    price: float | None = Field(default=None, gt=0)
    tags: list[str] | None = None


class ItemResponse(BaseModel):
    id: int
    name: str
    price: float
    tags: list[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedItems(BaseModel):
    data: list[ItemResponse]
    next_cursor: str | None = None
    has_more: bool


# ── model_validator — cross-field validation ──────────────────────────────────
# model_validator runs after all individual field validators succeed.
# Use it when the validity of one field depends on another.


class DateRange(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def end_after_start(self) -> "DateRange":
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self

    @model_validator(mode="after")
    def max_90_days(self) -> "DateRange":
        delta = self.end - self.start
        if delta.days > 90:
            raise ValueError("date range cannot exceed 90 days")
        return self


# ── Discriminated unions — typed event payloads ───────────────────────────────
# A discriminated union uses a literal "type" field to select the right model.
# FastAPI will validate the correct shape based on the `event_type` field.
# Much more reliable than `Union[A, B]` which tries each model in order.


class ItemCreatedEvent(BaseModel):
    event_type: Literal["item_created"]
    item_id: int
    name: str
    price: float


class ItemDeletedEvent(BaseModel):
    event_type: Literal["item_deleted"]
    item_id: int


class PaymentSucceededEvent(BaseModel):
    event_type: Literal["payment_succeeded"]
    order_id: str
    amount_cents: int
    currency: str = "usd"


# Annotated with the discriminator field — Pydantic picks the right model
# based on the value of `event_type` before any other validation.
WebhookEvent = Annotated[
    ItemCreatedEvent | ItemDeletedEvent | PaymentSucceededEvent,
    Field(discriminator="event_type"),
]

# Usage in a route:
#   async def handle_event(event: WebhookEvent) -> None:
#       match event.event_type:
#           case "item_created": ...


# ── TypeAdapter — validate outside a BaseModel ────────────────────────────────
# Useful for validating plain lists, dicts, or primitives that aren't
# wrapped in a BaseModel (e.g. validating raw JSON from an external source).

TagList = TypeAdapter(list[str])
PositiveFloat = TypeAdapter(Annotated[float, Field(gt=0)])

# Examples:
#   TagList.validate_python(["a", "b"])        # ✓
#   TagList.validate_python("not-a-list")      # raises ValidationError
#   PositiveFloat.validate_python(3.14)        # ✓
#   PositiveFloat.validate_python(-1)          # raises ValidationError


# ── Health / meta ─────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_s: float


class ReadinessResponse(BaseModel):
    status: str  # "ready" | "degraded"
    checks: dict[str, str]  # e.g. {"db": "ok", "cache": "timeout"}
