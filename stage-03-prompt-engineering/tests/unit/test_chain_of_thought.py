"""
Chain-of-thought utility tests — no network.
"""

from chain_of_thought import (
    ZERO_SHOT_COT_SUFFIX,
    build_scratchpad_system,
    extract_cot_answer,
    zero_shot_cot,
)


def test_zero_shot_cot_contains_question() -> None:
    q = "What is 2+2?"
    assert q in zero_shot_cot(q)


def test_zero_shot_cot_contains_suffix() -> None:
    assert ZERO_SHOT_COT_SUFFIX in zero_shot_cot("any question")


def test_zero_shot_cot_question_comes_first() -> None:
    q = "If a train travels 80 km/h for 2 hours, how far?"
    result = zero_shot_cot(q)
    assert result.startswith(q)


def test_extract_cot_answer_both_tags() -> None:
    text = "<thinking>2 + 2 = 4</thinking><answer>4</answer>"
    thinking, answer = extract_cot_answer(text)
    assert thinking == "2 + 2 = 4"
    assert answer == "4"


def test_extract_cot_answer_multiline_thinking() -> None:
    text = "<thinking>\nStep 1: identify operands\nStep 2: add\n</thinking><answer>42</answer>"
    thinking, answer = extract_cot_answer(text)
    assert thinking is not None
    assert "Step 1" in thinking
    assert answer == "42"


def test_extract_cot_answer_missing_thinking() -> None:
    thinking, answer = extract_cot_answer("The answer is <answer>42</answer>")
    assert thinking is None
    assert answer == "42"


def test_extract_cot_answer_neither_tag() -> None:
    thinking, answer = extract_cot_answer("Just some plain text.")
    assert thinking is None
    assert answer is None


def test_build_scratchpad_system_no_base() -> None:
    result = build_scratchpad_system()
    assert "<thinking>" in result
    assert "<answer>" in result


def test_build_scratchpad_system_with_base() -> None:
    base = "You are a math tutor."
    result = build_scratchpad_system(base)
    assert base in result
    assert "<thinking>" in result


def test_build_scratchpad_system_base_comes_first() -> None:
    base = "You are a helpful assistant."
    result = build_scratchpad_system(base)
    assert result.index(base) < result.index("<thinking>")


def test_build_scratchpad_system_empty_base_equals_snippet() -> None:
    from chain_of_thought import SCRATCHPAD_SYSTEM_SNIPPET

    assert build_scratchpad_system() == SCRATCHPAD_SYSTEM_SNIPPET
