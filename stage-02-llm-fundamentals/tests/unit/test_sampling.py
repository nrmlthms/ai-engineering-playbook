"""
SamplingParams tests — no network, no API key.
"""

import pytest
from llm.sampling import SamplingParams

# ── Validation ────────────────────────────────────────────────────────────────


def test_default_params_valid() -> None:
    p = SamplingParams()
    assert p.temperature == 1.0
    assert p.max_tokens == 1024


@pytest.mark.parametrize("temp", [-0.01, 2.01, -1, 3])
def test_temperature_out_of_range_raises(temp: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=temp)


@pytest.mark.parametrize("temp", [0.0, 0.5, 1.0, 2.0])
def test_temperature_valid_range(temp: float) -> None:
    SamplingParams(temperature=temp)  # should not raise


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1])
def test_top_p_out_of_range_raises(top_p: float) -> None:
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=top_p)


def test_top_p_valid() -> None:
    SamplingParams(top_p=0.9)  # should not raise


def test_top_k_zero_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(top_k=0)


def test_top_k_valid() -> None:
    SamplingParams(top_k=40)  # should not raise


@pytest.mark.parametrize("min_p", [-0.1, 1.0, 1.5])
def test_min_p_out_of_range_raises(min_p: float) -> None:
    with pytest.raises(ValueError, match="min_p"):
        SamplingParams(min_p=min_p)


def test_max_tokens_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=0)


# ── Serialisation ─────────────────────────────────────────────────────────────


def test_to_anthropic_kwargs_defaults() -> None:
    kwargs = SamplingParams().to_anthropic_kwargs()
    assert kwargs["temperature"] == 1.0
    assert kwargs["max_tokens"] == 1024
    assert "top_k" not in kwargs
    assert "top_p" not in kwargs
    assert "stop_sequences" not in kwargs


def test_to_anthropic_kwargs_optional_fields() -> None:
    p = SamplingParams(
        temperature=0.5,
        max_tokens=512,
        top_p=0.9,
        top_k=40,
        stop_sequences=["###"],
    )
    kwargs = p.to_anthropic_kwargs()
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 40
    assert kwargs["stop_sequences"] == ["###"]


def test_to_anthropic_kwargs_no_seed() -> None:
    # seed is OpenAI-only; must not appear in Anthropic kwargs
    p = SamplingParams(seed=42)
    assert "seed" not in p.to_anthropic_kwargs()


def test_to_openai_kwargs_defaults() -> None:
    kwargs = SamplingParams().to_openai_kwargs()
    assert kwargs["temperature"] == 1.0
    assert kwargs["max_tokens"] == 1024
    assert "top_k" not in kwargs


def test_to_openai_kwargs_seed() -> None:
    p = SamplingParams(seed=99)
    assert p.to_openai_kwargs()["seed"] == 99


def test_to_openai_kwargs_no_top_k() -> None:
    # top_k is Anthropic-only; must not appear in OpenAI kwargs even if set
    p = SamplingParams(top_k=40)
    assert "top_k" not in p.to_openai_kwargs()


def test_to_openai_kwargs_stop() -> None:
    p = SamplingParams(stop_sequences=["STOP", "END"])
    kwargs = p.to_openai_kwargs()
    assert kwargs["stop"] == ["STOP", "END"]


# ── validate_for_model (post-user implementation) ─────────────────────────────
# These tests are placeholders — they currently pass because validate_for_model
# is a no-op stub. Once you implement it, uncomment the pytest.raises blocks.


def test_validate_for_model_stub_does_not_raise() -> None:
    # The stub passes without error — tests below test the full implementation
    SamplingParams(temperature=0.5).validate_for_model("gpt-4o", "openai")


def test_o_series_rejects_non_default_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=0.5).validate_for_model("o3-mini", "openai")


def test_o_series_rejects_top_p() -> None:
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=0.9).validate_for_model("o1", "openai")


def test_o_series_default_temperature_passes() -> None:
    SamplingParams(temperature=1.0).validate_for_model("o3-mini", "openai")


def test_openai_rejects_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(top_k=40).validate_for_model("gpt-4o", "openai")


def test_anthropic_accepts_top_k() -> None:
    SamplingParams(top_k=40).validate_for_model("claude-sonnet-4-6", "anthropic")


def test_anthropic_rejects_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        SamplingParams(seed=42).validate_for_model("claude-sonnet-4-6", "anthropic")


def test_openai_accepts_seed() -> None:
    SamplingParams(seed=42).validate_for_model("gpt-4o", "openai")


def test_min_p_always_raises_openai() -> None:
    with pytest.raises(ValueError, match="min_p"):
        SamplingParams(min_p=0.05).validate_for_model("gpt-4o", "openai")


def test_min_p_always_raises_anthropic() -> None:
    with pytest.raises(ValueError, match="min_p"):
        SamplingParams(min_p=0.05).validate_for_model("claude-sonnet-4-6", "anthropic")


def test_o4_series_detected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=0.3).validate_for_model("o4-mini", "openai")


def test_non_o_series_openai_accepts_temperature() -> None:
    SamplingParams(temperature=0.5).validate_for_model("gpt-4o", "openai")
