"""
OpenAI client — Chat Completions and Responses API.

Chat Completions vs Responses API
───────────────────────────────────
Chat Completions  Stable, widely supported. gpt-4o, gpt-3.5, o-series.
                  Uses `messages` list. Streaming with `stream=True`.

Responses API     Newer (2025). Unified interface for tools, structured output,
                  multi-turn state. Uses `input` (str or list). Built-in tools
                  (code_interpreter, web_search_preview, file_search).
                  Better for complex agentic workflows.

When to use which
  - Standard chat / completion tasks → Chat Completions
  - Structured output (JSON schema) → either (both support response_format)
  - Built-in tools (search, code) → Responses API
  - o-series reasoning models → both work, Responses API preferred for new code

o-series specifics
───────────────────
o1, o3, o4-mini think before responding. Their usage reports both:
  completion_tokens         visible output tokens (billed)
  reasoning_tokens          internal thinking tokens (billed, not visible)

  total_billed ≈ input_tokens + completion_tokens + reasoning_tokens

Control reasoning depth with reasoning_effort: "low" | "medium" | "high"
  "low"    faster, cheaper, good for simple tasks
  "high"   thorough, expensive, use for hard problems

o-series do NOT support temperature or top_p (the parameters are ignored or error).
"""

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal

import openai
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

DEFAULT_MODEL = "gpt-4o"

ReasoningEffort = Literal["low", "medium", "high"]

_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(input_mtok=2.5, output_mtok=10.0),
    "gpt-4o-mini": ModelPricing(input_mtok=0.15, output_mtok=0.60),
    "o1": ModelPricing(input_mtok=15.0, output_mtok=60.0),
    "o3": ModelPricing(input_mtok=10.0, output_mtok=40.0),
    "o3-mini": ModelPricing(input_mtok=1.10, output_mtok=4.40),
    "o4-mini": ModelPricing(input_mtok=1.10, output_mtok=4.40),
}

_O_SERIES_PREFIXES = ("o1", "o3", "o4")


def _is_o_series(model: str) -> bool:
    return any(model.startswith(p) for p in _O_SERIES_PREFIXES)


@dataclass
class OpenAIResponse(LLMResponse):
    reasoning_tokens: int = 0  # o-series thinking tokens (billed but not shown)


class OpenAIClient:
    """
    Typed wrapper over the OpenAI Chat Completions and Responses APIs.

    Example — Chat Completions:
        client = OpenAIClient()
        response = await client.complete(
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        print(response.content, response.cost.total_usd)

    Example — o-series with reasoning:
        response = await client.complete(
            messages=[{"role": "user", "content": "Prove √2 is irrational"}],
            model="o3-mini",
            reasoning_effort="high",
        )
        print(f"Reasoning tokens: {response.reasoning_tokens}")
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        params: SamplingParams | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> OpenAIResponse:
        """
        Chat Completions — non-streaming.

        Pass response_format={"type": "json_schema", "json_schema": {...}} for
        structured output. Pass reasoning_effort for o-series models.
        """
        params = params or SamplingParams()
        all_messages = self._prepend_system(messages, system)
        kwargs = self._build_kwargs(all_messages, model, params, response_format, reasoning_effort)

        t0 = time.perf_counter()
        raw = await self._client.chat.completions.create(**kwargs)
        wall_ms = (time.perf_counter() - t0) * 1000

        response = self._parse_completion(raw, wall_ms=wall_ms)
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
        Streaming Chat Completions.

        Yields StreamChunk per delta, then a final chunk with usage.
        Note: usage in streaming requires stream_options={"include_usage": True}.
        """
        params = params or SamplingParams()
        all_messages = self._prepend_system(messages, system)
        kwargs = self._build_kwargs(all_messages, model, params, None, None)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        t0 = time.perf_counter()
        first_token_t: float | None = None
        accumulated = ""
        final_usage: openai.types.CompletionUsage | None = None

        async with await self._client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                if chunk.choices:
                    delta_text = chunk.choices[0].delta.content or ""
                    if delta_text:
                        if first_token_t is None:
                            first_token_t = time.perf_counter()
                        accumulated += delta_text
                        yield StreamChunk(delta=delta_text, model=model)
                if chunk.usage:
                    final_usage = chunk.usage

        wall_ms = (time.perf_counter() - t0) * 1000
        first_ms = (first_token_t - t0) * 1000 if first_token_t else None

        usage = _parse_usage(final_usage) if final_usage else Usage(0, 0)
        pricing = _get_pricing(model)
        cost = CostBreakdown.from_usage(usage, pricing) if pricing else None

        response = OpenAIResponse(
            content=accumulated,
            model=model,
            usage=usage,
            latency=Latency(wall_clock_ms=wall_ms, first_token_ms=first_ms),
            stop_reason=StopReason.END_TURN,
            provider="openai",
            cost=cost,
            reasoning_tokens=_reasoning_tokens(final_usage),
        )
        self._emit_log(response, streaming=True)
        yield StreamChunk(delta="", model=model, usage=usage, is_final=True)

    async def complete_responses(
        self,
        input: str | list[dict[str, Any]],
        *,
        model: str = DEFAULT_MODEL,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> OpenAIResponse:
        """
        OpenAI Responses API — newer interface for tools and structured output.

        Prefer this over Chat Completions for new code targeting o-series models
        or when using built-in tools (code_interpreter, web_search_preview).

        The Responses API uses `input` instead of `messages` and `instructions`
        instead of a system message in the messages list.
        """
        kwargs: dict[str, Any] = {"model": model, "input": input}
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["text"] = {"format": response_format}
        if reasoning_effort and _is_o_series(model):
            kwargs["reasoning"] = {"effort": reasoning_effort}

        t0 = time.perf_counter()
        raw = await self._client.responses.create(**kwargs)
        wall_ms = (time.perf_counter() - t0) * 1000

        content = raw.output_text if hasattr(raw, "output_text") else ""
        usage_obj = raw.usage if hasattr(raw, "usage") else None
        usage = _parse_responses_usage(usage_obj) if usage_obj else Usage(0, 0)
        pricing = _get_pricing(model)
        cost = CostBreakdown.from_usage(usage, pricing) if pricing else None

        response = OpenAIResponse(
            content=content,
            model=model,
            usage=usage,
            latency=Latency(wall_clock_ms=wall_ms),
            provider="openai",
            cost=cost,
            reasoning_tokens=getattr(
                getattr(usage_obj, "output_tokens_details", None), "reasoning_tokens", 0
            )
            or 0,
        )
        self._emit_log(response, streaming=False)
        return response

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        model: str,
        params: SamplingParams,
        response_format: dict[str, Any] | None,
        reasoning_effort: ReasoningEffort | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}

        if _is_o_series(model):
            # o-series: temperature/top_p are not user-controllable
            kwargs["max_completion_tokens"] = params.max_tokens
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs.update(params.to_openai_kwargs())

        if response_format:
            kwargs["response_format"] = response_format

        return kwargs

    @staticmethod
    def _prepend_system(messages: list[dict[str, Any]], system: str | None) -> list[dict[str, Any]]:
        if not system:
            return messages
        return [{"role": "system", "content": system}, *messages]

    def _parse_completion(self, raw: Any, wall_ms: float) -> OpenAIResponse:
        choice = raw.choices[0]
        content = choice.message.content or ""
        stop = StopReason.MAX_TOKENS if choice.finish_reason == "length" else StopReason.END_TURN
        usage = _parse_usage(raw.usage) if raw.usage else Usage(0, 0)
        pricing = _get_pricing(raw.model)
        cost = CostBreakdown.from_usage(usage, pricing) if pricing else None

        return OpenAIResponse(
            content=content,
            model=raw.model,
            usage=usage,
            latency=Latency(wall_clock_ms=wall_ms),
            stop_reason=stop,
            provider="openai",
            cost=cost,
            reasoning_tokens=_reasoning_tokens(raw.usage),
        )

    @staticmethod
    def _emit_log(response: OpenAIResponse, streaming: bool) -> None:
        log.info(
            "llm_call",
            provider="openai",
            model=response.model,
            streaming=streaming,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning_tokens=response.reasoning_tokens,
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
        input_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        output_tokens=getattr(raw, "completion_tokens", 0) or 0,
    )


def _parse_responses_usage(raw: Any) -> Usage:
    return Usage(
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
    )


def _reasoning_tokens(raw_usage: Any) -> int:
    if raw_usage is None:
        return 0
    details = getattr(raw_usage, "completion_tokens_details", None)
    if details is None:
        return 0
    return getattr(details, "reasoning_tokens", 0) or 0


def _get_pricing(model: str) -> ModelPricing | None:
    if model in _PRICING:
        return _PRICING[model]
    # Match on prefix (e.g. "gpt-4o-2024-08-06" → "gpt-4o")
    for key in _PRICING:
        if model.startswith(key):
            return _PRICING[key]
    return None
