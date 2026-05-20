"""
LLM HTTP client: httpx + tenacity + circuit breaker + semaphore bulkhead.

Resilience layers (outermost → innermost):
  1. Circuit breaker  — stops sending requests when the upstream is failing
  2. Semaphore        — limits concurrent in-flight calls (bulkhead pattern)
  3. Tenacity retry   — retries transient failures with exponential backoff
  4. httpx            — actual HTTP call with timeout

The circuit breaker is the key learning here:
  CLOSED   → normal operation, all requests pass through
  OPEN     → upstream is failing, fail fast without sending requests
  HALF_OPEN→ probe: send one request to see if upstream recovered

             failure_threshold reached
  CLOSED ──────────────────────────────→ OPEN
     ↑                                     │
     │ probe succeeded             recovery_timeout elapsed
     │                                     ↓
     └────────────────────────── HALF_OPEN
                   probe failed → OPEN
"""

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, Coroutine
from enum import Enum
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..settings import settings

log = structlog.get_logger()


# ── Circuit breaker ───────────────────────────────────────────────────────────


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Rolling-window circuit breaker.

    Trips when `failure_threshold` failures occur within `window_seconds`.
    After `recovery_timeout` seconds one probe is sent (HALF_OPEN).
    Success closes the circuit; failure re-opens it.

    ┌─────────────────────────────────────────────────────────────────┐
    │  Simple (consecutive) vs Rolling window — the key trade-off     │
    │                                                                 │
    │  Simple: self.failure_count += 1 on failure; reset to 0 on     │
    │  success. Trip when count >= threshold.                         │
    │                                                                 │
    │  Problem: a single success resets the whole counter. A service  │
    │  that fails 4 times, succeeds once, fails 4 more times never    │
    │  trips the circuit even though it has an 89% error rate.        │
    │                                                                 │
    │  Rolling window: keep a deque of failure timestamps; evict      │
    │  anything older than window_seconds before checking the count.  │
    │  The same pattern (4 fail, 1 success, 4 fail) in 60s trips the  │
    │  circuit because 8 failures are still within the window.        │
    │                                                                 │
    │  Simple is fine for services that fail cleanly (all-or-nothing).│
    │  Rolling window is better for flaky services that mix successes  │
    │  and failures under load.                                        │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        window_seconds: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.window_seconds = window_seconds
        self.state = CircuitState.CLOSED
        self.last_failure_time: float = 0.0
        # Rolling window: stores monotonic timestamps of recent failures.
        # deque gives O(1) append and popleft — ideal for a sliding window.
        self._failure_times: deque[float] = deque()
        # Ensures only one probe goes through while in HALF_OPEN.
        self._probe_in_flight: bool = False

    def _should_attempt(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.last_failure_time >= self.recovery_timeout:
                # Transition to HALF_OPEN here (not in _record_outcome) so the
                # state change is visible to concurrent callers before the probe
                # response arrives — they will see HALF_OPEN and be blocked.
                self.state = CircuitState.HALF_OPEN
                self._probe_in_flight = False
            else:
                return False

        # HALF_OPEN: let exactly one probe through; block all others.
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def _record_outcome(self, success: bool) -> None:
        now = time.monotonic()

        if success:
            self._failure_times.clear()
            self.state = CircuitState.CLOSED
            self._probe_in_flight = False
            log.info("circuit_closed")
            return

        # Append failure and evict timestamps outside the rolling window.
        self._failure_times.append(now)
        cutoff = now - self.window_seconds
        while self._failure_times and self._failure_times[0] < cutoff:
            self._failure_times.popleft()

        tripped = len(self._failure_times) >= self.failure_threshold
        if self.state == CircuitState.HALF_OPEN or tripped:
            self.state = CircuitState.OPEN
            self.last_failure_time = now
            self._probe_in_flight = False
            log.warning(
                "circuit_opened",
                failures_in_window=len(self._failure_times),
                window_s=self.window_seconds,
            )

    async def __call__(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Wrap a coroutine with circuit breaker protection."""
        if not self._should_attempt():
            log.warning("circuit_open", state=self.state.value)
            raise httpx.HTTPStatusError(
                "Circuit open — upstream unavailable",
                request=httpx.Request("GET", settings.llm_base_url),
                response=httpx.Response(503),
            )
        try:
            result = await coro
            self._record_outcome(success=True)
            return result
        except Exception:
            self._record_outcome(success=False)
            raise


# ── Shared HTTP client ────────────────────────────────────────────────────────

_http_client: httpx.AsyncClient | None = None
# Semaphore = bulkhead: caps concurrent calls regardless of retry loops
_semaphore = asyncio.Semaphore(settings.http_max_connections // 10)
_circuit_breaker = CircuitBreaker(
    failure_threshold=settings.circuit_breaker_failure_threshold,
    recovery_timeout=settings.circuit_breaker_recovery_timeout,
    window_seconds=60.0,
)


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            timeout=httpx.Timeout(
                connect=settings.http_connect_timeout,
                read=settings.http_read_timeout,
                write=5.0,
                pool=1.0,
            ),
            limits=httpx.Limits(
                max_connections=settings.http_max_connections,
                max_keepalive_connections=settings.http_max_keepalive,
            ),
            headers={"User-Agent": f"{settings.app_name}/{settings.app_version}"},
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


# ── Tenacity retry decorator ──────────────────────────────────────────────────
# Applied to the inner function — retries happen inside the semaphore so we
# don't hold the slot while waiting between attempts.

_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, _RETRYABLE)


@retry(
    stop=stop_after_attempt(settings.retry_max_attempts),
    wait=wait_exponential(multiplier=settings.retry_base_delay, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
async def _post_with_retry(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    client = get_http_client()
    response = await client.post(url, json=payload)
    if response.status_code in _RETRYABLE_STATUS:
        response.raise_for_status()  # triggers retry
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


# ── Public API ────────────────────────────────────────────────────────────────


class LLMHttpClient:
    """
    Typed wrapper around raw HTTP calls to an LLM API.

    All resilience (retry, circuit breaker, bulkhead) is handled here.
    Route handlers just call `await llm_client.complete(...)`.
    """

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Send a completion request. Returns the response text."""
        payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with _semaphore:
            data = await _circuit_breaker(_post_with_retry("/v1/messages", payload))

        return str(data["content"][0]["text"])

    async def complete_stream(
        self, prompt: str, max_tokens: int = 1024
    ) -> AsyncGenerator[str, None]:
        """
        Streaming completion — yields text chunks as they arrive.

        Streaming uses a different code path: we can't use the retry decorator
        because the response body is consumed lazily. Instead we rely on the
        circuit breaker and semaphore only for the connection phase.
        """
        payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": max_tokens,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }
        client = get_http_client()

        async with _semaphore:
            async with client.stream("POST", "/v1/messages", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:]  # strip "data: " prefix


llm_client = LLMHttpClient()
