"""
GraphQL with Strawberry.

Install: pip install 'strawberry-graphql[fastapi]'

Demonstrates:
  - Schema-first type definitions with @strawberry.type
  - Query and Mutation resolvers
  - Sharing context (e.g. DB session) via info.context
  - Mounting the GraphQL router alongside REST routes

When to use GraphQL vs REST:
  - Many client types (web, mobile, partners) needing different field shapes → GraphQL
  - Simple CRUD, file uploads, streaming → REST
  - Public API needing clear versioning → REST
  - Complex nested data in one round-trip → GraphQL
"""

from datetime import datetime
from typing import Any

# NOTE: requires `pip install 'strawberry-graphql[fastapi]'`
try:
    import strawberry
    from strawberry.fastapi import GraphQLRouter
    from strawberry.types import Info

    STRAWBERRY_AVAILABLE = True
except ImportError:
    STRAWBERRY_AVAILABLE = False


# ── Types ─────────────────────────────────────────────────────────────────────

if STRAWBERRY_AVAILABLE:

    @strawberry.type
    class Item:
        id: int
        name: str
        price: float
        tags: list[str]
        created_at: datetime

    @strawberry.input
    class CreateItemInput:
        name: str
        price: float
        tags: list[str] = strawberry.field(default_factory=list)

    # ── Resolvers ─────────────────────────────────────────────────────────────
    # info.context lets resolvers access request-scoped state (DB, current user)
    # without global variables.

    @strawberry.type
    class Query:
        @strawberry.field  # type: ignore[untyped-decorator]
        async def item(self, id: int, info: Info) -> Item | None:
            store: dict[int, Any] = info.context["item_store"]
            raw = store.get(id)
            if raw is None:
                return None
            return Item(**raw)

        @strawberry.field  # type: ignore[untyped-decorator]
        async def items(self, info: Info) -> list[Item]:
            store: dict[int, Any] = info.context["item_store"]
            return [Item(**v) for v in store.values()]

    @strawberry.type
    class Mutation:
        @strawberry.mutation  # type: ignore[untyped-decorator]
        async def create_item(self, input: CreateItemInput, info: Info) -> Item:
            store: dict[int, Any] = info.context["item_store"]
            next_id = max(store.keys(), default=0) + 1
            now = datetime.utcnow()
            raw = {
                "id": next_id,
                "name": input.name,
                "price": input.price,
                "tags": input.tags,
                "created_at": now,
            }
            store[next_id] = raw
            return Item(**raw)

    # ── Schema + router ───────────────────────────────────────────────────────

    # Import this in-memory store from routes/items.py in a real app
    _item_store: dict[int, Any] = {}

    async def get_context() -> dict[str, Any]:
        """
        Called per-request. Returns a dict available as info.context
        in every resolver. Inject DB sessions, current user, etc. here.
        """
        return {"item_store": _item_store}

    schema = strawberry.Schema(query=Query, mutation=Mutation)
    graphql_router = GraphQLRouter(schema, context_getter=get_context)

else:
    # Fallback when strawberry isn't installed — avoids ImportError at startup
    from fastapi import APIRouter

    graphql_router = APIRouter()

    @graphql_router.get("/graphql")
    async def graphql_unavailable() -> dict[str, str]:
        return {"error": "Install strawberry-graphql[fastapi] to enable GraphQL"}
