"""
Token counting tests.

These run offline — no API key required. tiktoken downloads encoding files
on first use; subsequent runs are instant (files are cached in ~/.tiktoken).

Snapshot counts: the expected values below ARE the snapshots. If tiktoken
changes its encoding (very rare), these tests will fail and you should update
the values after verifying the new counts are correct.
"""

import pytest
from llm.tokens import count_tokens, estimate_messages_tokens, is_anthropic_model, tokenize

# ── count_tokens ──────────────────────────────────────────────────────────────


def test_count_tokens_simple_english() -> None:
    # "Hello, world!" tokenises to ["Hello", ",", " world", "!"] in cl100k_base
    assert count_tokens("Hello, world!", model="gpt-4o") == 4


def test_count_tokens_empty_string() -> None:
    assert count_tokens("", model="gpt-4o") == 0


def test_count_tokens_single_word() -> None:
    # "Hello" is one token in both o200k_base and cl100k_base
    assert count_tokens("Hello", model="gpt-4o") == 1


def test_count_tokens_code_denser_than_prose() -> None:
    prose = "This is a normal sentence with common English words."
    code = "def f(x): return x**2+3*x-7"
    prose_tokens = count_tokens(prose, model="gpt-4o")
    code_tokens = count_tokens(code, model="gpt-4o")
    # Code should have more tokens per character than prose
    prose_chars_per_token = len(prose) / prose_tokens
    code_chars_per_token = len(code) / code_tokens
    assert code_chars_per_token < prose_chars_per_token


def test_count_tokens_anthropic_model_uses_proxy() -> None:
    # Anthropic models use cl100k_base as a proxy — just check it runs without error
    n = count_tokens("Hello, world!", model="claude-sonnet-4-6")
    assert n > 0


def test_count_tokens_unknown_model_falls_back() -> None:
    # Unknown model names should not raise
    n = count_tokens("test string", model="some-future-model-v99")
    assert n > 0


# ── tokenize ──────────────────────────────────────────────────────────────────


def test_tokenize_returns_correct_count() -> None:
    spans = tokenize("Hello, world!", model="gpt-4o")
    assert len(spans) == count_tokens("Hello, world!", model="gpt-4o")


def test_tokenize_span_indices_sequential() -> None:
    spans = tokenize("Hello, world!", model="gpt-4o")
    for i, span in enumerate(spans):
        assert span.index == i


def test_tokenize_spans_cover_full_text() -> None:
    text = "Quick brown fox"
    spans = tokenize(text, model="gpt-4o")
    reconstructed = "".join(s.text for s in spans)
    assert reconstructed == text


def test_tokenize_unique_token_ids() -> None:
    # "aardvark" is likely one token; just check we get token IDs back
    spans = tokenize("aardvark", model="gpt-4o")
    assert all(isinstance(s.token_id, int) for s in spans)


# ── estimate_messages_tokens ──────────────────────────────────────────────────


def test_estimate_messages_tokens_basic() -> None:
    messages = [{"role": "user", "content": "Hello"}]
    estimate = estimate_messages_tokens(messages)
    # Must be positive and in a reasonable range
    assert 5 <= estimate <= 20


def test_estimate_messages_tokens_with_system() -> None:
    messages = [{"role": "user", "content": "Hello"}]
    without_system = estimate_messages_tokens(messages)
    with_system = estimate_messages_tokens(messages, system="You are a helpful assistant.")
    assert with_system > without_system


def test_estimate_messages_tokens_longer_content_costs_more() -> None:
    short = [{"role": "user", "content": "Hi"}]
    long = [{"role": "user", "content": "Hi " * 100}]
    assert estimate_messages_tokens(long) > estimate_messages_tokens(short)


def test_estimate_messages_tokens_multi_turn() -> None:
    one_turn = [{"role": "user", "content": "Hello"}]
    two_turns = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]
    assert estimate_messages_tokens(two_turns) > estimate_messages_tokens(one_turn)


# ── is_anthropic_model ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-7", True),
        ("claude-haiku-4-5-20251001", True),
        ("gpt-4o", False),
        ("o3-mini", False),
    ],
)
def test_is_anthropic_model(model: str, expected: bool) -> None:
    assert is_anthropic_model(model) == expected
