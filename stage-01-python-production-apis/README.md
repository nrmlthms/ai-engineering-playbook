# Stage 01 — Python Production APIs

> FastAPI · async/await · error handling · webhooks · REST · GraphQL · third-party SDKs

---

## Table of contents

1. [FastAPI fundamentals](#1-fastapi-fundamentals)
2. [Async Python](#2-async-python)
3. [Error handling](#3-error-handling)
4. [Webhooks](#4-webhooks)
5. [REST design](#5-rest-design)
6. [GraphQL with Strawberry](#6-graphql-with-strawberry)
7. [Third-party SDKs](#7-third-party-sdks)
8. [Deliberately left out](#8-deliberately-left-out)
9. [Exercises](#9-exercises)

---

## 1. FastAPI fundamentals

### The application factory pattern

Never instantiate `FastAPI()` at module level in production — use a factory
function so you can control startup order, inject config, and instantiate
multiple apps in tests without side effects.

```python
# src/app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    await db.connect()          # open DB pool once
    yield
    # --- shutdown ---
    await db.disconnect()       # drain connections cleanly

def create_app() -> FastAPI:
    app = FastAPI(title="My API", lifespan=lifespan)
    app.include_router(items_router, prefix="/items")
    return app
```

`lifespan` replaces the old `@app.on_event("startup")` pattern (deprecated).
It uses an async context manager so startup and shutdown live together.

### Dependency injection

FastAPI's `Depends()` wires shared resources (DB session, current user, config)
into handlers without globals.

```python
from fastapi import Depends

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session           # FastAPI closes it after the response

@router.get("/{item_id}")
async def read_item(
    item_id: int,
    db: AsyncSession = Depends(get_db),        # injected
    user: User = Depends(get_current_user),    # injected
):
    ...
```

Dependencies can depend on other dependencies — FastAPI resolves the graph.
The same dependency is only called once per request even if used in multiple
places (cached by default).

### Pydantic v2 models

```python
from pydantic import BaseModel, Field, field_validator

class ItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    price: float = Field(gt=0)
    tags: list[str] = []

    @field_validator("name")
    @classmethod
    def no_emoji(cls, v: str) -> str:
        if any(ord(c) > 127 for c in v):
            raise ValueError("name must be ASCII")
        return v.strip()

class ItemResponse(ItemCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)  # ORM mode
```

`ItemCreate` validates incoming JSON. `ItemResponse` serialises the ORM object
back to JSON. One model per direction — never reuse the same model for input
and output (their constraints differ).

### Settings via BaseSettings

```python
# src/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    debug: bool = False
    workers: int = 4

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()  # reads from env / .env at import time
```

`BaseSettings` validates env vars with the same Pydantic rules as request
models. A missing required var raises at startup, not at the first request.

---

## 2. Async Python

### The event loop mental model

```
Thread 1 (event loop)
  │
  ├─ handle request A → await db.query()  ─┐  (suspended, no thread blocked)
  │                                         │
  ├─ handle request B → await httpx.get() ─┤  (runs while A waits)
  │                                         │
  └─ resume request A ←────────────────────┘  (DB replied)
```

`async def` functions are coroutines — they don't run until awaited.
`await` yields control back to the event loop, which can run other coroutines.
No OS threads are created; one thread handles thousands of concurrent I/O waits.

### asyncio.gather — parallel I/O

```python
import asyncio

async def fetch_dashboard(user_id: int) -> dict:
    # Run all three DB queries in parallel — total time = slowest query
    profile, orders, notifications = await asyncio.gather(
        db.get_profile(user_id),
        db.get_orders(user_id),
        db.get_notifications(user_id),
    )
    return {"profile": profile, "orders": orders, "notifications": notifications}
```

Sequential version would take `t1 + t2 + t3`. Parallel takes `max(t1, t2, t3)`.

### asyncio.gather with error handling

```python
results = await asyncio.gather(
    fetch_a(),
    fetch_b(),
    return_exceptions=True,   # don't cancel siblings on first failure
)

for r in results:
    if isinstance(r, Exception):
        logger.warning("partial failure", error=str(r))
```

### Timeouts

```python
import asyncio

async def call_with_timeout(coro, timeout: float = 5.0):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(504, "upstream timeout")
```

### Semaphores — rate-limiting concurrency

```python
sem = asyncio.Semaphore(10)   # max 10 concurrent external calls

async def safe_fetch(url: str) -> str:
    async with sem:            # blocks if 10 calls are already running
        return await http_client.get(url)
```

### Sync code in async context

CPU-bound or blocking-I/O code blocks the event loop. Offload it:

```python
import asyncio
from functools import partial

def cpu_heavy(data: bytes) -> str:
    ...  # blocking

async def handler():
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, partial(cpu_heavy, data))
```

---

## 3. Error handling

### HTTPException

```python
from fastapi import HTTPException

@router.get("/{item_id}")
async def get_item(item_id: int, db: AsyncSession = Depends(get_db)):
    item = await db.get(Item, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return item
```

### Custom exception classes + global handlers

Define domain exceptions separately from HTTP concerns:

```python
# src/errors.py
class ItemNotFound(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id

class InsufficientStock(Exception):
    def __init__(self, item_id: int, requested: int, available: int):
        self.item_id = item_id
        self.requested = requested
        self.available = available
```

Register handlers on the app — handlers return `JSONResponse`, not `raise`:

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ItemNotFound)
async def item_not_found_handler(request: Request, exc: ItemNotFound):
    return JSONResponse(
        status_code=404,
        content={"error": "item_not_found", "item_id": exc.item_id},
    )

@app.exception_handler(InsufficientStock)
async def insufficient_stock_handler(request: Request, exc: InsufficientStock):
    return JSONResponse(
        status_code=409,
        content={
            "error": "insufficient_stock",
            "item_id": exc.item_id,
            "requested": exc.requested,
            "available": exc.available,
        },
    )
```

### Validation error shape

FastAPI returns 422 for Pydantic validation failures. Override the shape to
match your error envelope:

```python
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),   # list of {loc, msg, type}
        },
    )
```

### Error envelope pattern

All errors share the same shape — clients only need to handle one format:

```json
{
  "error": "item_not_found",
  "message": "Item 42 does not exist",
  "request_id": "req_01j..."
}
```

---

## 4. Webhooks

A webhook is an inbound HTTP POST your server receives when something happens
in an external system (Stripe payment, GitHub push, etc).

### Anatomy of a webhook handler

```
External system
      │
      │  POST /webhooks/stripe
      │  Headers: Stripe-Signature: t=...,v1=...
      │  Body: {"type": "payment_intent.succeeded", ...}
      ↓
Your server
  1. Read raw body (before JSON parsing — signature is over raw bytes)
  2. Verify HMAC signature
  3. Return 200 immediately
  4. Process event in background task
```

Always return 200 quickly. If you do work synchronously and timeout, the
provider will retry — leading to duplicate processing.

### Signature verification (Stripe example)

```python
import hashlib, hmac, time

def verify_stripe_signature(
    raw_body: bytes,
    signature_header: str,
    secret: str,
    tolerance: int = 300,   # reject events older than 5 min
) -> None:
    parts = dict(p.split("=", 1) for p in signature_header.split(","))
    timestamp = int(parts["t"])

    if abs(time.time() - timestamp) > tolerance:
        raise HTTPException(400, "webhook timestamp out of tolerance")

    signed_payload = f"{timestamp}.".encode() + raw_body
    expected = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, parts["v1"]):
        raise HTTPException(400, "invalid webhook signature")
```

### Full webhook endpoint

```python
from fastapi import BackgroundTasks, Request

@router.post("/webhooks/stripe", status_code=200)
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    raw_body = await request.body()          # must read before any parsing
    sig = request.headers.get("Stripe-Signature", "")
    verify_stripe_signature(raw_body, sig, settings.stripe_webhook_secret)

    event = json.loads(raw_body)
    background_tasks.add_task(process_stripe_event, event)  # async, after 200

    return {"received": True}

async def process_stripe_event(event: dict) -> None:
    match event["type"]:
        case "payment_intent.succeeded":
            await handle_payment_success(event["data"]["object"])
        case "customer.subscription.deleted":
            await handle_subscription_cancelled(event["data"]["object"])
        case _:
            logger.info("unhandled stripe event", type=event["type"])
```

### Idempotency

Providers retry on network failure — your handler may receive the same event
twice. Guard with a processed-events table:

```python
async def process_stripe_event(event: dict) -> None:
    event_id = event["id"]
    if await db.event_already_processed(event_id):
        return                                  # idempotent no-op
    await db.mark_event_processing(event_id)
    # ... handle event ...
    await db.mark_event_done(event_id)
```

---

## 5. REST design

### URL and method conventions

| Action | Method | URL | Status |
|--------|--------|-----|--------|
| List | GET | `/items` | 200 |
| Create | POST | `/items` | 201 |
| Read | GET | `/items/{id}` | 200 / 404 |
| Replace | PUT | `/items/{id}` | 200 / 404 |
| Patch | PATCH | `/items/{id}` | 200 / 404 |
| Delete | DELETE | `/items/{id}` | 204 / 404 |

### Cursor-based pagination

Offset pagination (`?page=5`) breaks when rows are inserted mid-query.
Cursor pagination is stable under inserts:

```python
class PaginatedItems(BaseModel):
    data: list[ItemResponse]
    next_cursor: str | None    # base64-encoded last item's id
    has_more: bool

@router.get("/", response_model=PaginatedItems)
async def list_items(
    cursor: str | None = None,
    limit: int = Query(default=20, le=100),
    db: AsyncSession = Depends(get_db),
):
    after_id = decode_cursor(cursor) if cursor else None
    items = await db.get_items_after(after_id, limit + 1)  # fetch one extra
    has_more = len(items) > limit
    return PaginatedItems(
        data=items[:limit],
        next_cursor=encode_cursor(items[limit - 1].id) if has_more else None,
        has_more=has_more,
    )
```

### Versioning

Prefix routes with `/v1/`, `/v2/` etc. Run multiple versions simultaneously
using separate routers:

```python
app.include_router(v1_router, prefix="/v1")
app.include_router(v2_router, prefix="/v2")
```

---

## 6. GraphQL with Strawberry

REST returns fixed shapes. GraphQL lets the client specify exactly which fields
it needs — useful when you have many client types (web, mobile, partners) with
different data needs.

### Schema-first with Strawberry

```python
# pip install strawberry-graphql[fastapi]
import strawberry
from strawberry.fastapi import GraphQLRouter

@strawberry.type
class Item:
    id: int
    name: str
    price: float

@strawberry.type
class Query:
    @strawberry.field
    async def item(self, id: int, info: strawberry.types.Info) -> Item | None:
        db = info.context["db"]
        return await db.get(Item, id)

    @strawberry.field
    async def items(self, info: strawberry.types.Info) -> list[Item]:
        db = info.context["db"]
        return await db.get_all_items()

schema = strawberry.Schema(query=Query)
graphql_router = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_router, prefix="/graphql")
```

### When REST vs GraphQL

| Situation | Choose |
|-----------|--------|
| Simple CRUD, internal service | REST |
| Many client types with different field needs | GraphQL |
| File upload, streaming | REST |
| Complex nested data in one round-trip | GraphQL |
| Public API with clear versioning | REST |

---

## 7. Third-party SDKs

### httpx async client — the foundation

Always reuse a single `httpx.AsyncClient` across requests (connection pooling):

```python
# src/sdk_client.py
import httpx

_client: httpx.AsyncClient | None = None

def get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=10.0, write=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _client
```

### Retry with exponential backoff

```python
import asyncio, random

async def fetch_with_retry(
    url: str,
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
) -> dict:
    client = get_http_client()
    for attempt in range(max_attempts):
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
            await asyncio.sleep(delay)
```

Jitter (`random.uniform`) prevents thundering herd when many instances retry
simultaneously after an outage.

### Wrapping an SDK in a typed client

Never scatter raw SDK calls through your codebase. Wrap them in a typed class:

```python
class StripeClient:
    def __init__(self, api_key: str) -> None:
        self._client = stripe.AsyncStripe(api_key)

    async def create_payment_intent(
        self, amount_cents: int, currency: str = "usd"
    ) -> str:
        """Returns the payment intent client secret."""
        intent = await self._client.payment_intents.create(
            amount=amount_cents,
            currency=currency,
        )
        return intent.client_secret

    async def get_customer(self, customer_id: str) -> CustomerResponse:
        raw = await self._client.customers.retrieve(customer_id)
        return CustomerResponse.model_validate(raw)   # Pydantic parse + validate
```

Benefits:
- One import per service, not `import stripe` scattered everywhere
- Easy to mock in tests (`mock.patch` one class, not the whole SDK)
- Pydantic validation on the boundary — you own the schema

---

## 8. What's in `src/`

| File | What it shows |
|------|--------------|
| `api/main.py` | Application factory, lifespan, middleware + router wiring |
| `settings.py` | 12-factor config with `BaseSettings` |
| `schemas.py` | Pydantic v2: discriminated unions, `model_validator`, `TypeAdapter` |
| `errors.py` | RFC 7807 Problem Details |
| `auth.py` | JWT, OAuth 2.1 PKCE, mTLS notes |
| `health.py` | `/health` (liveness) and `/ready` (readiness) |
| `idempotency.py` | Idempotency-Key middleware |
| `middleware/request_id.py` | X-Request-Id — per-request unique ID |
| `middleware/correlation.py` | X-Correlation-Id — flows across service calls |
| `middleware/timing.py` | Response time logging + X-Response-Time header |
| `clients/llm_http.py` | httpx + tenacity + semaphore + circuit breaker |
| `routes/items.py` | REST CRUD with cursor pagination |
| `routes/webhooks.py` | Webhook receiver with HMAC signature verification |
| `routes/graphql.py` | Strawberry schema, Query + Mutation |

---

## 8. Deliberately left out

### Inbound rate limiting

This stage does not include a per-client rate limiter in the application layer,
and that is intentional.

**What we do have** (outbound concurrency limits in `clients/llm_http.py`):
- `asyncio.Semaphore` — caps simultaneous outbound calls to external APIs
- Circuit breaker — stops sending when the upstream is unhealthy

**Why inbound rate limiting belongs at the infrastructure layer:**

```
Client → [Nginx / Cloudflare / API Gateway] → your FastAPI app
              ↑
     rate limiting lives here
```

Implementing a rate limiter inside FastAPI with an in-process counter breaks
as soon as you run more than one worker or replica — each process has its own
counter, so the effective limit is `limit × num_workers`. A Redis-backed sliding
window fixes this, but now every request pays a Redis round-trip.

In production, use one of:

| Option | When |
|--------|------|
| Nginx `limit_req` | Single-server deployments |
| Cloudflare Rate Limiting | Edge, before traffic reaches your infra |
| AWS API Gateway / Kong | Managed API gateway |
| Redis-backed middleware | If you need per-user limits inside the app (see Stage 06) |

The `RateLimiter` class in `notebooks/03_fastapi_di.py` demonstrates the
class-based dependency pattern — it is not production rate limiting.

---

## 9. Exercises

1. **Circuit breaker** — implement `_should_attempt()` and `_record_outcome()` in `clients/llm_http.py`. Test it by mocking a flaky upstream that fails 6 times then recovers.
2. **Readiness probe** — add a real DB ping to `health.py`'s `/ready` endpoint. Return `503` when the DB is unreachable.
3. **Webhook replay** — add `POST /v1/webhooks/replay/{event_id}` that re-processes a stored event (tests your idempotency guard).
4. **PKCE flow** — wire `generate_pkce_pair()` and `verify_pkce()` from `auth.py` into a `/auth/token` exchange endpoint.
5. **Pagination** — the `RateLimiter` in `notebooks/03_fastapi_di.py` resets across restarts. Rewrite it backed by a `collections.deque` sliding window that is accurate under concurrent requests.
