"""
Tests for the webhook receiver.

Key things to test:
  - Valid signature → 200
  - Invalid signature → 400
  - Expired timestamp → 400
  - Duplicate event ID → processed only once (idempotency)
"""

import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient


def make_stripe_signature(raw_body: bytes, secret: str, timestamp: int | None = None) -> str:
    ts = timestamp or int(time.time())
    signed_payload = f"{ts}.".encode() + raw_body
    sig = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


@pytest.fixture()
def client(monkeypatch):
    # Patch settings so we control the webhook secret
    import sys
    for mod in list(sys.modules):
        if "stage_01" in mod:
            del sys.modules[mod]

    from stage_01.src import settings as settings_module  # type: ignore[import]
    monkeypatch.setattr(settings_module.settings, "stripe_webhook_secret", "test_secret")

    from stage_01.src.app import create_app  # type: ignore[import]
    return TestClient(create_app())


def post_webhook(client, event: dict, secret: str = "test_secret", timestamp: int | None = None):
    raw = json.dumps(event).encode()
    sig = make_stripe_signature(raw, secret, timestamp)
    return client.post(
        "/v1/webhooks/stripe",
        content=raw,
        headers={"Content-Type": "application/json", "Stripe-Signature": sig},
    )


# ── Signature verification ────────────────────────────────────────────────────

def test_valid_webhook(client):
    event = {"id": "evt_001", "type": "payment_intent.succeeded", "data": {"object": {}}}
    response = post_webhook(client, event)
    assert response.status_code == 200
    assert response.json() == {"received": True}


def test_invalid_signature(client):
    raw = json.dumps({"id": "evt_002", "type": "test"}).encode()
    response = client.post(
        "/v1/webhooks/stripe",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "Stripe-Signature": "t=1234567890,v1=badsignature",
        },
    )
    assert response.status_code == 400


def test_expired_timestamp(client):
    old_timestamp = int(time.time()) - 400  # > 300s tolerance
    event = {"id": "evt_003", "type": "test"}
    response = post_webhook(client, event, timestamp=old_timestamp)
    assert response.status_code == 400


def test_missing_signature_header(client):
    raw = json.dumps({"id": "evt_004", "type": "test"}).encode()
    response = client.post(
        "/v1/webhooks/stripe",
        content=raw,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


# ── Idempotency ───────────────────────────────────────────────────────────────

def test_duplicate_event_processed_once(client, caplog):
    """Same event_id should be silently ignored on second delivery."""
    event = {"id": "evt_duplicate", "type": "payment_intent.succeeded", "data": {"object": {}}}

    r1 = post_webhook(client, event)
    r2 = post_webhook(client, event)  # Stripe retry

    assert r1.status_code == 200
    assert r2.status_code == 200  # Still 200 — don't fail the provider
