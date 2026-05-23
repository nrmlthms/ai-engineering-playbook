"""
Anthropic client unit tests.

All HTTP calls are intercepted by respx — no real API key required.
respx patches httpx.AsyncClient at the transport level, which is what the
Anthropic SDK uses under the hood.

Determinism note
────────────────
These tests assert on PARSED response fields (token counts, model name, stop
reason), NOT on raw model output. Even at temperature=0, LLM output is not
fully deterministic across API versions and hardware. Mocking the HTTP layer
gives us true reproducibility.
"""

import httpx
import pytest
import respx
from llm.anthropic_client import AnthropicClient
from llm.sampling import SamplingParams
from llm.streaming import StopReason

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _mock_response(
    content: str = "Hello, world!",
    input_tokens: int = 12,
    output_tokens: int = 5,
    cache_read: int = 0,
    cache_write: int = 0,
    model: str = "claude-sonnet-4-6",
    stop_reason: str = "end_turn",
) -> dict:
    return {
        "id": "msg_test01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        },
    }


@pytest.fixture
def client() -> AnthropicClient:
    return AnthropicClient(api_key="test-key-not-real")


# ── complete() ────────────────────────────────────────────────────────────────


async def test_complete_returns_content(client: AnthropicClient) -> None:
    with respx.mock:
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=_mock_response("42")))
        response = await client.complete(
            messages=[{"role": "user", "content": "What is 6 × 7?"}],
        )

    assert response.content == "42"
    assert response.provider == "anthropic"


async def test_complete_usage_parsed(client: AnthropicClient) -> None:
    with respx.mock:
        respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(
                200, json=_mock_response(input_tokens=100, output_tokens=20)
            )
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 20
    assert response.usage.total_tokens == 120


async def test_complete_cache_tokens_parsed(client: AnthropicClient) -> None:
    with respx.mock:
        respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json=_mock_response(cache_read=500, cache_write=50))
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert response.usage.cache_read_tokens == 500
    assert response.usage.cache_write_tokens == 50


async def test_complete_stop_reason_end_turn(client: AnthropicClient) -> None:
    with respx.mock:
        respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json=_mock_response(stop_reason="end_turn"))
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert response.stop_reason == StopReason.END_TURN


async def test_complete_stop_reason_max_tokens(client: AnthropicClient) -> None:
    with respx.mock:
        respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json=_mock_response(stop_reason="max_tokens"))
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Tell me everything"}],
            params=SamplingParams(max_tokens=5),
        )

    assert response.stop_reason == StopReason.MAX_TOKENS


async def test_complete_latency_recorded(client: AnthropicClient) -> None:
    with respx.mock:
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=_mock_response()))
        response = await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert response.latency.wall_clock_ms > 0
    # Non-streaming: first_token_ms is None
    assert response.latency.first_token_ms is None


async def test_complete_model_version_from_api(client: AnthropicClient) -> None:
    # The client must use the model version RETURNED BY THE API, not what was requested.
    # APIs sometimes return a pinned version (e.g. "claude-sonnet-4-6-20260301") even
    # when you requested the alias ("claude-sonnet-4-6").
    pinned = "claude-sonnet-4-6-20260301"
    with respx.mock:
        respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json=_mock_response(model=pinned))
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert response.model == pinned


async def test_complete_cost_calculated(client: AnthropicClient) -> None:
    with respx.mock:
        respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_response(
                    input_tokens=1_000_000, output_tokens=1_000_000, model="claude-sonnet-4-6"
                ),
            )
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-sonnet-4-6",
        )

    # 1M input @ $3/MTok + 1M output @ $15/MTok = $18
    assert response.cost is not None
    assert abs(response.cost.total_usd - 18.0) < 0.01


async def test_complete_unknown_model_no_cost(client: AnthropicClient) -> None:
    with respx.mock:
        respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json=_mock_response(model="claude-future-model-99"))
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-future-model-99",
        )

    # No pricing entry → cost is None (don't crash)
    assert response.cost is None


# ── sampling params forwarded ─────────────────────────────────────────────────


async def test_temperature_forwarded_in_request(client: AnthropicClient) -> None:
    captured: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_mock_response())

    with respx.mock:
        respx.post(ANTHROPIC_URL).mock(side_effect=capture)
        await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
            params=SamplingParams(temperature=0.2, max_tokens=256),
        )

    assert captured[0]["temperature"] == 0.2
    assert captured[0]["max_tokens"] == 256
