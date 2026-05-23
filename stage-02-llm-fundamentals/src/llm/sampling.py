"""
Sampling parameter types.

How sampling works (simplified)
─────────────────────────────────
The model produces a probability distribution over its vocabulary (~100 k tokens).
Each sampling parameter filters or reshapes that distribution before drawing:

  temperature  Divide logits by T before softmax.
               T < 1 → sharper (more deterministic).
               T > 1 → flatter (more random).
               T = 0 → greedy (always pick max-prob token).

  top_p        Keep only the smallest set of tokens whose cumulative probability
               ≥ p, then sample uniformly within that set.
               top_p=0.9 discards the long tail of unlikely tokens.

  top_k        Keep only the k highest-probability tokens.
               Anthropic supports this; OpenAI does not.

  min_p        Exclude tokens with probability < min_p × max_token_prob.
               A dynamic threshold that scales with confidence.
               Not supported by Anthropic or OpenAI directly.

Parameter interactions
  temperature + top_p: temperature reshapes the distribution first, THEN top_p
  truncates it. At temperature=0, top_p/top_k are irrelevant (greedy).
  Use one of top_p or top_k, not both.

Determinism caveat
  temperature=0 is "greedy" but NOT fully deterministic in practice:
    - Different hardware (GPU type, batch size) can produce different results
    - Cloud providers run on heterogeneous fleets
    - Streaming vs non-streaming can give different results on some backends
  For regression tests: mock the HTTP call and assert on the parsed response,
  not on the raw model output.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 1024
    top_p: float | None = None
    top_k: int | None = None  # Anthropic only
    min_p: float | None = None  # not supported by major providers
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None  # OpenAI: best-effort determinism; Anthropic: not supported

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.min_p is not None and not 0.0 <= self.min_p < 1.0:
            raise ValueError(f"min_p must be in [0, 1), got {self.min_p}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    def validate_for_model(self, model: str, provider: str) -> None:
        """
        Check that all set parameters are supported by the given model/provider.

        Raises ValueError for unsupported combinations. Call this before sending
        the request — the APIs return confusing errors when you pass unsupported
        params (some silently ignore them, some error, some produce bad output).

        Rules to implement
        ──────────────────
        OpenAI o-series (o1, o1-mini, o3, o3-mini, o4-mini):
          - temperature is not user-controllable (fixed at 1 internally)
          - top_p is not supported
          - Raise if either is set to a non-default value

        Anthropic extended thinking (handled in AnthropicClient but worth
        documenting here):
          - temperature must be exactly 1.0 when thinking is enabled
          - Not something SamplingParams can detect alone, but document it

        top_k:
          - Anthropic only. Raise if provider != "anthropic" and top_k is set.

        seed:
          - OpenAI only (best-effort). Raise if provider != "openai" and seed is set.

        min_p:
          - Not supported by anthropic or openai. Always raise if set.
        """
        _O_PREFIXES = ("o1", "o3", "o4")
        is_o_series = provider == "openai" and any(model.startswith(p) for p in _O_PREFIXES)

        if is_o_series:
            if self.temperature != 1.0:
                raise ValueError(
                    f"o-series models do not support temperature (got {self.temperature}); "
                    "remove it or set temperature=1.0."
                )
            if self.top_p is not None:
                raise ValueError(
                    f"o-series models do not support top_p (got {self.top_p})."
                )

        if provider != "anthropic" and self.top_k is not None:
            raise ValueError(f"top_k is Anthropic-only; not supported by {provider!r}.")

        if provider != "openai" and self.seed is not None:
            raise ValueError(f"seed is OpenAI-only; not supported by {provider!r}.")

        if self.min_p is not None:
            raise ValueError(
                f"min_p is not supported by {provider!r}. "
                "Use top_p for nucleus sampling instead."
            )

    def to_anthropic_kwargs(self) -> dict[str, Any]:
        """Serialise to keyword args for anthropic.messages.create()."""
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences
        return kwargs

    def to_openai_kwargs(self) -> dict[str, Any]:
        """Serialise to keyword args for openai.chat.completions.create()."""
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.stop_sequences:
            kwargs["stop"] = self.stop_sequences
        return kwargs
