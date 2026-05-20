"""
Tests for the Items REST API.

Uses FastAPI's TestClient (sync) and AsyncClient (async) — both work
without running a real server.
"""

import pytest
from fastapi.testclient import TestClient

# Import the factory, not the module-level `app`, so each test gets
# a fresh in-memory store.
import importlib
import sys


@pytest.fixture()
def client():
    # Re-import routes module to reset the in-memory store between tests
    for mod in list(sys.modules):
        if "stage_01" in mod or "stage-01" in mod:
            del sys.modules[mod]

    from stage_01.src.app import create_app  # type: ignore[import]
    return TestClient(create_app())


# ── CRUD happy paths ──────────────────────────────────────────────────────────

def test_create_item(client):
    response = client.post("/v1/items/", json={"name": "Widget", "price": 9.99})
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Widget"
    assert data["price"] == 9.99
    assert "id" in data


def test_get_item(client):
    created = client.post("/v1/items/", json={"name": "Gadget", "price": 4.99}).json()
    response = client.get(f"/v1/items/{created['id']}")
    assert response.status_code == 200
    assert response.json()["name"] == "Gadget"


def test_get_item_not_found(client):
    response = client.get("/v1/items/9999")
    assert response.status_code == 404
    assert response.json()["error"] == "item_not_found"


def test_patch_item(client):
    created = client.post("/v1/items/", json={"name": "Old", "price": 1.0}).json()
    response = client.patch(f"/v1/items/{created['id']}", json={"price": 2.5})
    assert response.status_code == 200
    assert response.json()["price"] == 2.5
    assert response.json()["name"] == "Old"  # unchanged


def test_delete_item(client):
    created = client.post("/v1/items/", json={"name": "Temp", "price": 1.0}).json()
    delete_resp = client.delete(f"/v1/items/{created['id']}")
    assert delete_resp.status_code == 204

    get_resp = client.get(f"/v1/items/{created['id']}")
    assert get_resp.status_code == 404


# ── Validation ────────────────────────────────────────────────────────────────

def test_create_item_negative_price(client):
    response = client.post("/v1/items/", json={"name": "Bad", "price": -1})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_create_item_empty_name(client):
    response = client.post("/v1/items/", json={"name": "", "price": 1.0})
    assert response.status_code == 422


# ── Pagination ────────────────────────────────────────────────────────────────

def test_pagination(client):
    for i in range(5):
        client.post("/v1/items/", json={"name": f"Item {i}", "price": float(i + 1)})

    page1 = client.get("/v1/items/?limit=3").json()
    assert len(page1["data"]) == 3
    assert page1["has_more"] is True
    assert page1["next_cursor"] is not None

    page2 = client.get(f"/v1/items/?limit=3&cursor={page1['next_cursor']}").json()
    assert len(page2["data"]) == 2
    assert page2["has_more"] is False

    # All IDs should be unique across pages
    ids_p1 = {i["id"] for i in page1["data"]}
    ids_p2 = {i["id"] for i in page2["data"]}
    assert ids_p1.isdisjoint(ids_p2)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_s" in data
    assert "version" in data
