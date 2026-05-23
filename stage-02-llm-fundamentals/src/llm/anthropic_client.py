"""
Anthropic Messages API — production-grade client.

Every call records
  - input / output / cache_read / cache_write tokens
  - wall-clock latency + first-token latency (streaming only)
  - model version (exact string returned by the API, not what you requested)
  - cost breakdown in USD

Prompt caching
──────────────
Add cache_control to a message block to make it cacheable:

    {"role": "user", "content": [
        {"type": "text", "text": long_document, "cache_control": {"type": "ephemeral"}}
    ]}

Cache TTL is 5 minutes. Best candidates for caching:
  - System prompts longer than ~1k tokens
  - Few-shot examples included in every call
  - Large documents you query multiple times

Economics: cache write costs 25 % more than normal input.
           cache read  costs  10 % of normal input.
Break-even: after ~1.33 reads from the same cache entry you save money.

Extended thinking
─────────────────
Pass ThinkingConfig to unlock chain-of-thought reasoning. Requires temperature=1
(the client enforces this automatically). Thinking tokens are billed as output tokens
but appear in a separate block, not in content.

  Best for: multi-step math, complex reasoning, ambiguous instructions
  Avoid for: simple tasks, latency-sensitive paths (thinking adds significant tokens)
"""

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import anthropic
import structlog

from settings import settings
from .sampling import SamplingParams
from .streaming import (
    CostBreakdown,
    Latency,
    LLMResponse,
    ModelPricing,
    StopReason,
    StreamChunk,
    Usage,
)

log = structlog.get_logger()

DEFAULT_MODEL = "claude-sonnet-4-6"

# Pricing per million tokens (USD). Source: https://www.anthropic.com/pricing
# Update when Anthropic changes rates.
_PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-7": ModelPricing(
        input_mtok=15.0, output_mtok=75.0, cache_write_mtok=18.75, cache_read_mtok=1.50
    ),
    "claude-sonnet-4-6": ModelPricing(
        input_mtok=3.0, output_mtok=15.0, cache_write_mtok=3.75, cache_read_mtok=0.30
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        input_mtok=0.80, output_mtok=4.0, cache_write_mtok=1.0, cache_read_mtok=0.08
    ),
}


@dataclass
class ThinkingConfig:
    """Budget (max) tokens for extended thinking. Minimum is 1024."""

    budget_tokens: int = 10_000


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AnthropicResponse(LLMResponse):
    thinking: str | None = None  # populated when ThinkingConfig is passed
    tool_use: list[ToolUseBlock] = field(default_factory=list)


class AnthropicClient:
    """
    Typed wrapper around the Anthropic Messages API.

    All calls log to structlog with full token + latency data so you can
    derive cost and performance metrics without extra instrumentation.

    Example — basic completion:
        client = AnthropicClient()
        response = await client.complete(
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        print(response.content)           # "4"
        print(response.usage.input_tokens)
        print(response.cost.total_usd)

    Example — streaming:
        async for chunk in client.complete_stream(messages, model=model):
            if not chunk.is_final:
                print(chunk.delta, end="", flush=True)
            else:
                print(f"\\n{chunk.usage}")
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.anthropic_api_key,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        params: SamplingParams | None = None,
        thinking: ThinkingConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AnthropicResponse:
        """Non-streaming completion. Use complete_stream() for lower TTFT."""
        params = params or SamplingParams()
        kwargs = self._build_kwargs(messages, model, system, params, thinking, tools)

        t0 = time.perf_counter()
        raw = await self._client.messages.create(**kwargs)
        wall_ms = (time.perf_counter() - t0) * 1000

        response = self._parse_message(raw, wall_ms=wall_ms)
        self._emit_log(response, streaming=False)
        return response

    async def complete_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        params: SamplingParams | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Streaming completion.

        Yields StreamChunk for each text delta, then a final chunk with
        is_final=True that carries usage and cost metadata.
        """
        params = params or SamplingParams()
        kwargs = self._build_kwargs(messages, model, system, params, thinking=None, tools=None)
        # Remove max_tokens from kwargs for stream() — it uses a separate param name
        # in some SDK versions; keep it consistent via the stream API's own handling.

        t0 = time.perf_counter()
        first_token_t: float | None = None
        accumulated = ""

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                accumulated += text
                yield StreamChunk(delta=text, model=model)

            final = await stream.get_final_message()

        wall_ms = (time.perf_counter() - t0) * 1000
        first_ms = (first_token_t - t0) * 1000 if first_token_t else None

        usage = _parse_usage(final.usage)
        pricing = _PRICING.get(final.model)
        cost = CostBreakdown.from_usage(usage, pricing) if pricing else None

        response = AnthropicResponse(
            content=accumulated,
            model=final.model,
            usage=usage,
            latency=Latency(wall_clock_ms=wall_ms, first_token_ms=first_ms),
            stop_reason=_parse_stop_reason(final.stop_reason),
            provider="anthropic",
            cost=cost,
        )
        self._emit_log(response, streaming=True)

        yield StreamChunk(delta="", model=final.model, usage=usage, is_final=True)

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        model: str,
        system: str | None,
        params: SamplingParams,
        thinking: ThinkingConfig | None,
        tools: list[ToolDefinition] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **params.to_anthropic_kwargs(),
        }
        if system:
            kwargs["system"] = system
        if thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking.budget_tokens,
            }
            kwargs["temperature"] = 1.0  # required by API when thinking is enabled
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        return kwargs

    def _parse_message(self, raw: Any, wall_ms: float) -> AnthropicResponse:
        content = ""
        thinking_text: str | None = None
        tool_blocks: list[ToolUseBlock] = []

        for block in raw.content:
            if block.type == "text":
                content += block.text
            elif block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "tool_use":
                tool_blocks.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))

        usage = _parse_usage(raw.usage)
        pricing = _PRICING.get(raw.model)
        cost = CostBreakdown.from_usage(usage, pricing) if pricing else None

        return AnthropicResponse(
            content=content,
            model=raw.model,
            usage=usage,
            latency=Latency(wall_clock_ms=wall_ms),
            stop_reason=_parse_stop_reason(raw.stop_reason),
            provider="anthropic",
            cost=cost,
            thinking=thinking_text,
            tool_use=tool_blocks,
        )

    @staticmethod
    def _emit_log(response: AnthropicResponse, streaming: bool) -> None:
        log.info(
            "llm_call",
            provider="anthropic",
            model=response.model,
            streaming=streaming,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_tokens,
            cache_write_tokens=response.usage.cache_write_tokens,
            wall_clock_ms=round(response.latency.wall_clock_ms, 1),
            first_token_ms=(
                round(response.latency.first_token_ms, 1)
                if response.latency.first_token_ms is not None
                else None
            ),
            stop_reason=response.stop_reason.value,
            cost_usd=round(response.cost.total_usd, 6) if response.cost else None,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_usage(raw: Any) -> Usage:
    return Usage(
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        cache_read_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
    )


def _parse_stop_reason(reason: str | None) -> StopReason:
    return {
        "end_turn": StopReason.END_TURN,
        "max_tokens": StopReason.MAX_TOKENS,
        "stop_sequence": StopReason.STOP_SEQUENCE,
        "tool_use": StopReason.TOOL_USE,
    }.get(reason or "end_turn", StopReason.END_TURN)
