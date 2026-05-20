"""Unit tests for idempotency key logic."""

import pytest
from idempotency import _hash_body, _is_expired, CachedResponse
import time


def test_hash_body_deterministic():
    body = b'{"name": "Widget"}'
    assert _hash_body(body) == _hash_body(body)

def test_hash_body_different_bodies():
    assert _hash_body(b"a") != _hash_body(b"b")

def test_cached_response_not_expired_immediately():
    entry = CachedResponse(
        status_code=200,
        body=b"{}",
        headers={},
        request_hash="abc",
    )
    assert not _is_expired(entry)

def test_cached_response_expired_after_ttl(monkeypatch):
    from idempotency import _TTL_SECONDS
    entry = CachedResponse(
        status_code=200,
        body=b"{}",
        headers={},
        request_hash="abc",
        created_at=time.monotonic() - _TTL_SECONDS - 1,
    )
    assert _is_expired(entry)
