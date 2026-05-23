"""
Provider-agnostic response types.

Every LLM call in this codebase returns an LLMResponse (or a subclass).
Application code depends only on these types, not on provider-specific objects.

Design notes
────────────
Usage.cache_read_tokens / cache_write_tokens
  Anthropic charges differently for cache writes (+25 % vs normal input) and
  cache reads (~10 % of normal input). Tracking them separately lets you see
  the savings vs the write overhead in your cost breakdown.

Latency.first_token_ms
  Set only for streaming calls. Time-to-first-token (TTFT) is the metric users
  feel most strongly — it's what makes an interface feel "responsive". Wall-clock
  measures total round-trip cost. Both matter for different reasons.

CostBreakdown.from_usage()
  Convenience factory that converts a Usage + ModelPricing into dollar amounts.
  Keep the pricing tables in each client module (they change regularly).
"""

from dataclasses import dataclass
from enum import Enum


class StopReason(Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def effective_input_tokens(self) -> int:
        """Tokens that count toward input billing (excludes cache reads)."""
        return self.input_tokens - self.cache_read_tokens + self.cache_write_tokens


@dataclass
class Latency:
    wall_clock_ms: float
    first_token_ms: float | None = None  # None for non-streaming


@dataclass
class ModelPricing:
    """Per-million-token pricing in USD. Update as providers change rates."""

    input_mtok: float
    output_mtok: float
    cache_write_mtok: float = 0.0
    cache_read_mtok: float = 0.0


@dataclass
class CostBreakdown:
    input_usd: float
    output_usd: float
    cache_write_usd: float = 0.0
    cache_read_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.output_usd + self.cache_write_usd + self.cache_read_usd

    @classmethod
    def from_usage(cls, usage: "Usage", pricing: "ModelPricing") -> "CostBreakdown":
        return cls(
            input_usd=usage.input_tokens * pricing.input_mtok / 1_000_000,
            output_usd=usage.output_tokens * pricing.output_mtok / 1_000_000,
            cache_write_usd=usage.cache_write_tokens * pricing.cache_write_mtok / 1_000_000,
            cache_read_usd=usage.cache_read_tokens * pricing.cache_read_mtok / 1_000_000,
        )


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Usage
    latency: Latency
    stop_reason: StopReason = StopReason.END_TURN
    provider: str = ""
    cost: CostBreakdown | None = None


@dataclass
class StreamChunk:
    """One chunk from a streaming response."""

    delta: str  # text appended so far in this chunk
    model: str = ""
    usage: Usage | None = None  # populated only in the final chunk
    is_final: bool = False
