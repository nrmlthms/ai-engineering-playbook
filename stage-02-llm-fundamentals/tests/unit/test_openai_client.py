"""
OpenAI client unit tests — respx mocks, no real API key.
"""

import httpx
import pytest
import respx
from llm.openai_client import OpenAIClient
from llm.sampling import SamplingParams
from llm.streaming import StopReason

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _mock_completion(
    content: str = "4",
    prompt_tokens: int = 10,
    completion_tokens: int = 1,
    model: str = "gpt-4o-2024-08-06",
    finish_reason: str = "stop",
    reasoning_tokens: int = 0,
) -> dict:
    return {
        "id": "chatcmpl-test01",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


@pytest.fixture
def client() -> OpenAIClient:
    return OpenAIClient(api_key="test-key-not-real")


# ── complete() ────────────────────────────────────────────────────────────────


async def test_complete_returns_content(client: OpenAIClient) -> None:
    with respx.mock:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=_mock_completion("4")))
        response = await client.complete(
            messages=[{"role": "user", "content": "2+2?"}],
        )

    assert response.content == "4"
    assert response.provider == "openai"


async def test_complete_usage_parsed(client: OpenAIClient) -> None:
    with respx.mock:
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(
                200, json=_mock_completion(prompt_tokens=50, completion_tokens=10)
            )
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert response.usage.input_tokens == 50
    assert response.usage.output_tokens == 10


async def test_complete_stop_reason_end_turn(client: OpenAIClient) -> None:
    with respx.mock:
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=_mock_completion(finish_reason="stop"))
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert response.stop_reason == StopReason.END_TURN


async def test_complete_stop_reason_max_tokens(client: OpenAIClient) -> None:
    with respx.mock:
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=_mock_completion(finish_reason="length"))
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Long answer"}],
            params=SamplingParams(max_tokens=5),
        )

    assert response.stop_reason == StopReason.MAX_TOKENS


async def test_complete_latency_recorded(client: OpenAIClient) -> None:
    with respx.mock:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=_mock_completion()))
        response = await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert response.latency.wall_clock_ms > 0
    assert response.latency.first_token_ms is None


async def test_complete_cost_calculated(client: OpenAIClient) -> None:
    with respx.mock:
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_completion(
                    prompt_tokens=1_000_000,
                    completion_tokens=1_000_000,
                    model="gpt-4o",
                ),
            )
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
        )

    # 1M input @ $2.5/MTok + 1M output @ $10/MTok = $12.5
    assert response.cost is not None
    assert abs(response.cost.total_usd - 12.5) < 0.01


async def test_complete_reasoning_tokens_parsed(client: OpenAIClient) -> None:
    with respx.mock:
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_completion(
                    model="o3-mini",
                    completion_tokens=200,
                    reasoning_tokens=150,
                ),
            )
        )
        response = await client.complete(
            messages=[{"role": "user", "content": "Prove √2 is irrational"}],
            model="o3-mini",
        )

    assert response.reasoning_tokens == 150


async def test_system_message_prepended(client: OpenAIClient) -> None:
    captured: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_mock_completion())

    with respx.mock:
        respx.post(OPENAI_URL).mock(side_effect=capture)
        await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
            system="You are a pirate.",
        )

    messages = captured[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "pirate" in messages[0]["content"]
    assert messages[1]["role"] == "user"


async def test_o_series_uses_max_completion_tokens(client: OpenAIClient) -> None:
    captured: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_mock_completion(model="o3-mini"))

    with respx.mock:
        respx.post(OPENAI_URL).mock(side_effect=capture)
        await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="o3-mini",
            params=SamplingParams(max_tokens=512),
        )

    # o-series must use max_completion_tokens, not max_tokens
    assert "max_completion_tokens" in captured[0]
    assert captured[0]["max_completion_tokens"] == 512
    # temperature must NOT be forwarded for o-series
    assert "temperature" not in captured[0]
