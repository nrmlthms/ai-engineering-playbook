"""
Integration tests using ASGITransport.

ASGITransport lets httpx talk directly to the ASGI app without binding
a real port. Requests go through the full middleware stack — request_id,
correlation_id, timing, idempotency — just like in production.
"""

import pytest
import httpx
from httpx import AsyncClient, ASGITransport

from api.main import create_app


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────

async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "version" in r.json()


async def test_response_time_header(client):
    r = await client.get("/health")
    assert "X-Response-Time" in r.headers


async def test_request_id_echoed(client):
    r = await client.get("/health", headers={"X-Request-Id": "test-123"})
    assert r.headers.get("X-Request-Id") == "test-123"


async def test_request_id_generated(client):
    r = await client.get("/health")
    assert "X-Request-Id" in r.headers


# ── Items CRUD ────────────────────────────────────────────────────────────────

async def test_create_and_get_item(client):
    create = await client.post("/v1/items/", json={"name": "Widget", "price": 9.99})
    assert create.status_code == 201
    item_id = create.json()["id"]

    get = await client.get(f"/v1/items/{item_id}")
    assert get.status_code == 200
    assert get.json()["name"] == "Widget"


async def test_not_found_returns_rfc7807(client):
    r = await client.get("/v1/items/9999")
    assert r.status_code == 404
    assert r.headers["content-type"] == "application/problem+json"
    body = r.json()
    assert "type" in body
    assert "title" in body
    assert body["status"] == 404


async def test_validation_error_returns_rfc7807(client):
    r = await client.post("/v1/items/", json={"name": "", "price": -1})
    assert r.status_code == 422
    assert "application/problem+json" in r.headers["content-type"]


# ── Idempotency ───────────────────────────────────────────────────────────────

async def test_idempotent_create(client):
    payload = {"name": "IdempotentWidget", "price": 1.0}
    headers = {"Idempotency-Key": "unique-key-001"}

    r1 = await client.post("/v1/items/", json=payload, headers=headers)
    r2 = await client.post("/v1/items/", json=payload, headers=headers)

    assert r1.status_code == 201
    assert r2.status_code == 201
    # Same item returned, not a duplicate
    assert r1.json()["id"] == r2.json()["id"]
    assert r2.headers.get("Idempotency-Replayed") == "true"


async def test_idempotency_key_body_mismatch(client):
    headers = {"Idempotency-Key": "unique-key-002"}
    await client.post("/v1/items/", json={"name": "A", "price": 1.0}, headers=headers)
    r = await client.post("/v1/items/", json={"name": "B", "price": 2.0}, headers=headers)
    assert r.status_code == 422
    assert "idempotency" in r.json()["type"]
