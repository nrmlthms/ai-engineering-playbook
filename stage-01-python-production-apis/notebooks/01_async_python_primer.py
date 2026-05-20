# ruff: noqa: F704, E402
# %% [markdown]
# # 01 — Async Python Primer
#
# **Goal:** Build an intuition for the async event loop before touching FastAPI.
#
# | Concept | Key takeaway |
# |---------|-------------|
# | `async def` | Defines a coroutine — doesn't run until awaited |
# | `await` | Suspends the current coroutine, yields to the event loop |
# | `asyncio.gather` | Runs coroutines concurrently in the same thread |
# | `asyncio.TaskGroup` | Python 3.11+ — cleaner gather with structured cancellation |
# | `anyio` | Abstracts asyncio/trio — use for library code |

# %%
import asyncio
import time

# ── 1. Sequential vs concurrent ──────────────────────────────────────────────
# The event loop can overlap I/O waits. asyncio.sleep() simulates network I/O.


async def fetch(name: str, delay: float) -> str:
    await asyncio.sleep(delay)  # yields control — event loop runs other tasks
    return f"{name} done"


async def sequential():
    start = time.perf_counter()
    a = await fetch("A", 0.3)
    b = await fetch("B", 0.3)
    c = await fetch("C", 0.3)
    print(f"sequential: {time.perf_counter() - start:.2f}s")  # ~0.9s
    return [a, b, c]


async def concurrent():
    start = time.perf_counter()
    results = await asyncio.gather(fetch("A", 0.3), fetch("B", 0.3), fetch("C", 0.3))
    print(f"concurrent: {time.perf_counter() - start:.2f}s")  # ~0.3s
    return results


await sequential()
await concurrent()

# %%
# ── 2. TaskGroup (Python 3.11+) ───────────────────────────────────────────────
# TaskGroup cancels all remaining tasks if one raises — cleaner than gather().


async def with_task_group():
    results = {}
    async with asyncio.TaskGroup() as tg:
        a_task = tg.create_task(fetch("A", 0.2))
        b_task = tg.create_task(fetch("B", 0.1))
    results["a"] = a_task.result()
    results["b"] = b_task.result()
    return results


print(await with_task_group())

# %%
# ── 3. Cancellation and timeouts ──────────────────────────────────────────────


async def slow_operation():
    await asyncio.sleep(5)
    return "done"


async def with_timeout():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=1.0)
    except TimeoutError:
        print("timed out after 1s")
        result = None
    return result


await with_timeout()

# %%
# ── 4. Semaphore as concurrency limiter ───────────────────────────────────────
# Without limits, 1000 concurrent gather() calls could exhaust file descriptors
# or overwhelm a downstream service.

sem = asyncio.Semaphore(3)  # max 3 concurrent calls


async def rate_limited_fetch(name: str) -> str:
    async with sem:
        print(f"  {name} running (slot acquired)")
        await asyncio.sleep(0.1)
        return name


names = [f"task_{i}" for i in range(10)]
results = await asyncio.gather(*[rate_limited_fetch(n) for n in names])
print(f"completed: {len(results)} tasks")

# %%
# ── 5. anyio — write portable async code ──────────────────────────────────────
# anyio runs on asyncio (default) or trio. Prefer it for library code.

import anyio


async def anyio_example():
    async with anyio.create_task_group() as tg:
        tg.start_soon(fetch, "X", 0.1)
        tg.start_soon(fetch, "Y", 0.2)
    print("both done")


await anyio_example()

# %%
# ── Exercise ──────────────────────────────────────────────────────────────────
# Implement fetch_all(urls: list[str]) that:
#   1. Fetches all URLs concurrently using httpx.AsyncClient
#   2. Limits concurrency to 5 simultaneous requests with a semaphore
#   3. Returns a list of (url, status_code) tuples
#   4. Logs a warning for any request that takes > 2 seconds


async def fetch_all(urls: list[str]) -> list[tuple[str, int]]:
    # YOUR CODE HERE
    ...
