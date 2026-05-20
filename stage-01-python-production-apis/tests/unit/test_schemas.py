"""Unit tests for Pydantic v2 models."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from schemas import (
    DateRange,
    ItemCreate,
    ItemUpdate,
    TagList,
    WebhookEvent,
)

# ── ItemCreate ────────────────────────────────────────────────────────────────


def test_item_create_strips_name():
    item = ItemCreate(name="  Widget  ", price=9.99)
    assert item.name == "Widget"


def test_item_create_normalises_tags():
    item = ItemCreate(name="X", price=1.0, tags=["  Python  ", "API", ""])
    assert item.tags == ["python", "api"]


def test_item_create_rejects_empty_name():
    with pytest.raises(ValidationError, match="min_length"):
        ItemCreate(name="", price=9.99)


def test_item_create_rejects_negative_price():
    with pytest.raises(ValidationError):
        ItemCreate(name="X", price=-1.0)


def test_item_create_rejects_zero_price():
    with pytest.raises(ValidationError):
        ItemCreate(name="X", price=0)


# ── ItemUpdate ────────────────────────────────────────────────────────────────


def test_item_update_all_optional():
    # All fields optional — empty update is valid (even if no-op)
    update = ItemUpdate()
    assert update.name is None
    assert update.price is None


def test_item_update_partial():
    update = ItemUpdate(price=5.0)
    dumped = update.model_dump(exclude_none=True)
    assert dumped == {"price": 5.0}
    assert "name" not in dumped


# ── DateRange model_validator ─────────────────────────────────────────────────


def _dt(days_offset: int = 0) -> datetime:
    return datetime.now(UTC) + timedelta(days=days_offset)


def test_date_range_valid():
    dr = DateRange(start=_dt(0), end=_dt(10))
    assert dr.end > dr.start


def test_date_range_end_before_start():
    with pytest.raises(ValidationError, match="end must be after start"):
        DateRange(start=_dt(10), end=_dt(0))


def test_date_range_too_long():
    with pytest.raises(ValidationError, match="90 days"):
        DateRange(start=_dt(0), end=_dt(91))


# ── Discriminated union ───────────────────────────────────────────────────────


def test_discriminated_union_item_created():
    event = WebhookEvent.validate_python(  # type: ignore[attr-defined]
        {"event_type": "item_created", "item_id": 1, "name": "Widget", "price": 9.99}
    )
    assert event.event_type == "item_created"
    assert event.item_id == 1


def test_discriminated_union_payment():
    from pydantic import TypeAdapter
    from schemas import WebhookEvent

    ta = TypeAdapter(WebhookEvent)
    event = ta.validate_python(
        {"event_type": "payment_succeeded", "order_id": "ord_001", "amount_cents": 999}
    )
    assert event.amount_cents == 999


def test_discriminated_union_unknown_type():
    from pydantic import TypeAdapter, ValidationError
    from schemas import WebhookEvent

    ta = TypeAdapter(WebhookEvent)
    with pytest.raises(ValidationError):
        ta.validate_python({"event_type": "unknown"})


# ── TypeAdapter ───────────────────────────────────────────────────────────────


def test_tag_list_valid():
    assert TagList.validate_python(["a", "b"]) == ["a", "b"]


def test_tag_list_rejects_non_list():
    with pytest.raises(ValidationError):
        TagList.validate_python("not-a-list")
