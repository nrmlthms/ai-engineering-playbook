"""
Health and readiness endpoints.

/health  → liveness  — is the process alive and not deadlocked?
           Used by the container orchestrator to know when to restart the pod.

/ready   → readiness — can the process serve traffic right now?
           Used by the load balancer to stop routing to a pod during startup
           or when a dependency (DB, cache) is unavailable.

Key distinction: a pod can be alive but not ready (e.g. DB is down).
Kubernetes restarts on failed liveness; stops routing on failed readiness.
"""

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .schemas import HealthResponse, ReadinessResponse
from .settings import settings

router = APIRouter(tags=["meta"])
_started_at: float = time.monotonic()


@router.get("/health", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Always returns 200 unless the process is completely broken."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        uptime_s=round(time.monotonic() - _started_at, 2),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness() -> JSONResponse:
    """
    Checks actual dependencies. Returns 200 if ready, 503 if degraded.

    Extend checks dict with real probes: DB ping, cache ping, etc.
    """
    checks: dict[str, str] = {}
    all_ok = True

    # Example: DB check (replace with real async ping)
    try:
        # await db.execute("SELECT 1")
        checks["db"] = "ok"
    except Exception:
        checks["db"] = "unavailable"
        all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content=ReadinessResponse(
            status="ready" if all_ok else "degraded",
            checks=checks,
        ).model_dump(),
    )
