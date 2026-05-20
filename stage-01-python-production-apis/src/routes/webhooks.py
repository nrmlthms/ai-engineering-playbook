"""
Webhook receiver.

Demonstrates:
  - Reading raw bytes before JSON parsing (signature is over raw bytes)
  - HMAC-SHA256 signature verification
  - Returning 200 immediately + processing in BackgroundTasks
  - Idempotency guard (prevents double-processing on retries)
"""

import hashlib
import hmac
import json
import logging
import time

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from ..errors import WebhookSignatureInvalid
from ..settings import settings

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)

# In production this would be a DB table. Here it's an in-memory set.
_processed_event_ids: set[str] = set()


# ── Signature verification ────────────────────────────────────────────────────

def verify_stripe_signature(
    raw_body: bytes,
    signature_header: str,
    secret: str,
    tolerance_seconds: int = 300,
) -> None:
    """
    Stripe signs webhooks with HMAC-SHA256 over "<timestamp>.<raw_body>".

    The tolerance rejects replayed events — a provider could theoretically
    capture a valid event and re-POST it later. Rejecting anything older
    than 5 minutes makes replays useless.
    """
    try:
        parts = dict(part.split("=", 1) for part in signature_header.split(","))
        timestamp = int(parts["t"])
        received_sig = parts["v1"]
    except (ValueError, KeyError):
        raise WebhookSignatureInvalid("malformed Stripe-Signature header")

    if abs(time.time() - timestamp) > tolerance_seconds:
        raise WebhookSignatureInvalid("webhook timestamp outside tolerance window")

    signed_payload = f"{timestamp}.".encode() + raw_body
    expected_sig = hmac.new(
        secret.encode(), signed_payload, hashlib.sha256
    ).hexdigest()

    # compare_digest is constant-time — prevents timing attacks
    if not hmac.compare_digest(expected_sig, received_sig):
        raise WebhookSignatureInvalid("signature mismatch")


# ── Event processing (runs in background after 200 is returned) ───────────────

async def process_stripe_event(event: dict) -> None:
    event_id: str = event.get("id", "")

    # Idempotency guard — Stripe retries on network failure
    if event_id in _processed_event_ids:
        logger.info("duplicate stripe event, skipping", event_id=event_id)
        return

    _processed_event_ids.add(event_id)

    match event.get("type"):
        case "payment_intent.succeeded":
            logger.info("payment succeeded", event_id=event_id)
            # await handle_payment_success(event["data"]["object"])
        case "customer.subscription.deleted":
            logger.info("subscription cancelled", event_id=event_id)
            # await handle_subscription_cancelled(event["data"]["object"])
        case _:
            logger.info("unhandled stripe event type", type=event.get("type"))


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/stripe", status_code=200)
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Receive a Stripe webhook.

    Critical points:
    1. Read raw_body BEFORE JSON parsing — signature covers the raw bytes.
    2. Verify signature before touching the payload.
    3. Return 200 immediately — do work in background.
       If you do work synchronously and it takes >30s, Stripe marks the
       delivery as failed and retries, causing double-processing.
    """
    raw_body = await request.body()
    signature = request.headers.get("Stripe-Signature", "")

    verify_stripe_signature(raw_body, signature, settings.stripe_webhook_secret)

    try:
        event = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "invalid_json"})

    background_tasks.add_task(process_stripe_event, event)

    return JSONResponse(content={"received": True})
