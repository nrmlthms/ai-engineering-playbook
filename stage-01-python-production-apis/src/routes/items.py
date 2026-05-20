"""
REST CRUD for Items.

Demonstrates:
  - Proper HTTP methods + status codes
  - Cursor-based pagination (stable under concurrent inserts)
  - Dependency injection for DB session
  - Raising domain exceptions (handled globally in errors.py)
"""

import base64
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, status

from ..errors import ItemNotFound
from ..schemas import ItemCreate, ItemResponse, ItemUpdate, PaginatedItems

router = APIRouter(prefix="/items", tags=["items"])


# ── In-memory store (replace with real DB in production) ─────────────────────

_store: dict[int, dict[str, Any]] = {}
_next_id: int = 1


def _now() -> datetime:
    return datetime.now(UTC)


def _encode_cursor(item_id: int) -> str:
    return base64.urlsafe_b64encode(str(item_id).encode()).decode()


def _decode_cursor(cursor: str) -> int:
    return int(base64.urlsafe_b64decode(cursor).decode())


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/", response_model=PaginatedItems)
async def list_items(
    cursor: str | None = Query(default=None, description="Cursor from previous response"),
    limit: int = Query(default=20, ge=1, le=100),
) -> PaginatedItems:
    """
    List items with cursor-based pagination.

    Cursor pagination is stable when rows are inserted between pages.
    Offset pagination (`?page=5`) can skip or duplicate rows during inserts.
    """
    all_items = sorted(_store.values(), key=lambda x: x["id"])

    if cursor:
        after_id = _decode_cursor(cursor)
        all_items = [i for i in all_items if i["id"] > after_id]

    # Fetch one extra to detect if there's a next page
    page = all_items[: limit + 1]
    has_more = len(page) > limit
    data = page[:limit]

    return PaginatedItems(
        data=[ItemResponse(**i) for i in data],
        next_cursor=_encode_cursor(data[-1]["id"]) if has_more and data else None,
        has_more=has_more,
    )


@router.post("/", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
async def create_item(body: ItemCreate) -> ItemResponse:
    """
    Create a new item. Returns 201 (not 200) — the resource was created.
    The Location header would point to the new resource in a real API.
    """
    global _next_id
    now = _now()
    item = {
        "id": _next_id,
        "name": body.name,
        "price": body.price,
        "tags": body.tags,
        "created_at": now,
        "updated_at": now,
    }
    _store[_next_id] = item
    _next_id += 1
    return ItemResponse(**item)


@router.get("/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int) -> ItemResponse:
    """
    Get a single item. Raises ItemNotFound (→ 404) if it doesn't exist.

    The exception is a domain exception — no HTTP knowledge inside the route.
    The global handler in errors.py converts it to a JSON 404 response.
    """
    if item_id not in _store:
        raise ItemNotFound(item_id)
    return ItemResponse(**_store[item_id])


@router.patch("/{item_id}", response_model=ItemResponse)
async def update_item(item_id: int, body: ItemUpdate) -> ItemResponse:
    """
    Partial update (PATCH). Only provided fields are changed.

    PATCH vs PUT:
      PUT  = replace the entire resource (all fields required)
      PATCH = update only the provided fields (all fields optional)
    """
    if item_id not in _store:
        raise ItemNotFound(item_id)

    item = _store[item_id]
    updates = body.model_dump(exclude_none=True)  # skip fields that were None
    item.update(updates)
    item["updated_at"] = _now()
    return ItemResponse(**item)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int) -> None:
    """
    Delete an item. Returns 204 No Content — success with no body.
    """
    if item_id not in _store:
        raise ItemNotFound(item_id)
    del _store[item_id]
