# %% [markdown]
# # 03 — FastAPI Dependency Injection
#
# FastAPI's `Depends()` system is its most powerful feature.
# It wires shared resources (DB, current user, feature flags, rate limiters)
# into handlers without globals or manual plumbing.
#
# Key properties:
# - A dependency is any callable (function, class, generator)
# - Dependencies can depend on other dependencies (graph, not just chain)
# - Same dependency is called only **once per request** (cached by default)
# - Yields-based dependencies clean up after the response

# %% [markdown]
# ## 1. Basic dependency

# %%
from fastapi import FastAPI, Depends, HTTPException
from fastapi.testclient import TestClient

app = FastAPI()

def get_db():
    """Simulates opening a DB session."""
    db = {"connected": True, "items": {1: "Widget", 2: "Gadget"}}
    return db

@app.get("/items/{item_id}")
def read_item(item_id: int, db: dict = Depends(get_db)):
    item = db["items"].get(item_id)
    if item is None:
        raise HTTPException(404, "Not found")
    return {"id": item_id, "name": item}

with TestClient(app) as client:
    print(client.get("/items/1").json())   # {"id": 1, "name": "Widget"}
    print(client.get("/items/9").json())   # 404

# %% [markdown]
# ## 2. Generator dependencies (lifespan scoped)
# Use `yield` to ensure cleanup runs after the response is sent.

# %%
from contextlib import asynccontextmanager

class FakeSession:
    def __init__(self): self.closed = False
    def close(self): self.closed = True

async def get_session():
    session = FakeSession()
    try:
        yield session      # handler receives the session
    finally:
        session.close()    # cleanup — runs even if handler raises

app2 = FastAPI()

@app2.get("/ping")
async def ping(session = Depends(get_session)):
    return {"session_open": not session.closed}

with TestClient(app2) as c:
    print(c.get("/ping").json())

# %% [markdown]
# ## 3. Dependency chaining

# %%
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

bearer = HTTPBearer()

def get_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
    return credentials.credentials

def get_current_user(token: str = Depends(get_token)) -> dict:
    # In production: decode JWT here
    if token != "valid-token":
        raise HTTPException(401, "Unauthorized")
    return {"user_id": 42, "role": "admin"}

def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(403, "Forbidden")
    return user

app3 = FastAPI()

@app3.get("/admin/stats")
def admin_stats(user: dict = Depends(require_admin)):
    return {"user": user, "stats": "..."}

with TestClient(app3) as c:
    r = c.get("/admin/stats", headers={"Authorization": "Bearer valid-token"})
    print(r.json())
    r2 = c.get("/admin/stats", headers={"Authorization": "Bearer wrong"})
    print(r2.status_code)  # 401

# %% [markdown]
# ## 4. Class-based dependencies (stateful)
# Classes with `__call__` can hold state (rate limiters, feature flags, etc.)

# %%
class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: list[float] = []

    def __call__(self):
        import time
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < self.window]
        if len(self._calls) >= self.max_calls:
            raise HTTPException(429, "Rate limit exceeded")
        self._calls.append(now)

limiter = RateLimiter(max_calls=3, window_seconds=1.0)

app4 = FastAPI()

@app4.get("/limited")
def limited_endpoint(_: None = Depends(limiter)):
    return {"ok": True}

with TestClient(app4) as c:
    for i in range(5):
        r = c.get("/limited")
        print(f"call {i+1}: {r.status_code}")

# %% [markdown]
# ## 5. Sub-applications (mount)
# Split a large API into independently routable sub-apps.

# %%
from fastapi import APIRouter

v1 = APIRouter(prefix="/v1", tags=["v1"])
v2 = APIRouter(prefix="/v2", tags=["v2"])

@v1.get("/items")
def v1_items(): return {"version": 1, "items": []}

@v2.get("/items")
def v2_items(): return {"version": 2, "items": [], "cursor": None}

main_app = FastAPI()
main_app.include_router(v1)
main_app.include_router(v2)

with TestClient(main_app) as c:
    print(c.get("/v1/items").json())
    print(c.get("/v2/items").json())

# %% [markdown]
# ## Exercise
#
# Build a `PaginationParams` dependency class that:
# 1. Accepts `limit: int = 20` and `cursor: str | None = None` as query params
# 2. Rejects `limit > 100` with a 422 error
# 3. Decodes the cursor (base64) if provided and returns the decoded `after_id`
# 4. Use it in a `GET /items` endpoint
