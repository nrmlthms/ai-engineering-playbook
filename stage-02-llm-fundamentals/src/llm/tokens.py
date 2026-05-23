"""
Tokenization utilities.

BPE (Byte Pair Encoding) primer
────────────────────────────────
Training starts with individual bytes, then iteratively merges the most frequent
adjacent pair into a new token. After enough merges, common English words become
single tokens; rare words are split into subwords.

  "unhappiness" → ["un", "happ", "iness"]  (3 tokens)
  "hello"       → ["hello"]                (1 token)
  "πάρε"        → ["π", "ά", "ρ", "ε"]    (4 tokens — Greek less frequent)

Vocabulary sizes
  o200k_base  (GPT-4o)       200 k tokens
  cl100k_base (GPT-4, Claude proxy) 100 k tokens
  p50k_base   (GPT-3.5)       50 k tokens

Anthropic uses a proprietary tokenizer. tiktoken's cl100k_base is a reasonable
proxy (counts typically within ±5 %). For exact billing-grade counts, use the
SDK's `count_tokens()` API (network call, adds latency).

Practical rules of thumb (English prose)
  1 token ≈ 4 characters
  1 token ≈ 0.75 words
  1 page  ≈ 750 tokens

Code is denser: 1 token ≈ 2–3 characters.
Non-Latin scripts (Chinese, Japanese): 1–2 characters per token.
"""

from dataclasses import dataclass

import tiktoken

# Map model names to their tiktoken encoding.
# Anthropic models are marked with (proxy) — see module docstring.
_ENCODING_FOR_MODEL: dict[str, str] = {
    # OpenAI — exact
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o4-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    # Anthropic — cl100k_base is a proxy, not exact
    "claude-opus-4-7": "cl100k_base",
    "claude-sonnet-4-6": "cl100k_base",
    "claude-haiku-4-5-20251001": "cl100k_base",
}

_ANTHROPIC_MODEL_PREFIXES = ("claude-",)


@dataclass
class TokenSpan:
    """A single token and its position in the original string."""

    text: str  # decoded bytes of this token (may differ from original bytes in non-BMP text)
    token_id: int
    index: int  # 0-based token index in the sequence


def get_encoding(model: str) -> tiktoken.Encoding:
    encoding_name = _ENCODING_FOR_MODEL.get(model)
    if encoding_name is None:
        # Fall back to cl100k_base for unknown models rather than raising
        encoding_name = "o200k_base" if model.startswith("gpt-4o") else "cl100k_base"
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in a plain-text string.

    For Anthropic models this is an approximation. For exact counts pass
    the text through the SDK's count_tokens() endpoint (requires a network call).
    """
    return len(get_encoding(model).encode(text))


def tokenize(text: str, model: str = "gpt-4o") -> list[TokenSpan]:
    """
    Return one TokenSpan per token, preserving order.

    Useful for visualizing how a model sees your text before sending it.
    Non-obvious splits often explain surprising model behaviour.
    """
    enc = get_encoding(model)
    token_ids = enc.encode(text)
    return [
        TokenSpan(text=enc.decode([tid]), token_id=tid, index=i) for i, tid in enumerate(token_ids)
    ]


def estimate_messages_tokens(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
    system: str | None = None,
) -> int:
    """
    Estimate token count for a chat messages list (OpenAI format).

    Based on OpenAI's official counting guide — each message carries 4 overhead
    tokens (role + content delimiters) plus 3 tokens for the reply primer.
    System prompts add 3 more. This is an approximation; actual counts can differ
    by a few tokens due to special tokens added by the model's chat template.

    Reference: https://platform.openai.com/docs/guides/text-generation/managing-tokens
    """
    enc = get_encoding(model)
    total = 3  # reply primer tokens

    if system:
        total += 3 + len(enc.encode(system))

    for msg in messages:
        total += 4  # per-message overhead (role + delimiters)
        for value in msg.values():
            total += len(enc.encode(value))

    return total


def is_anthropic_model(model: str) -> bool:
    return any(model.startswith(p) for p in _ANTHROPIC_MODEL_PREFIXES)
