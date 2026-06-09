"""
FewShotExample and FewShotFormatter tests — no network.
"""

import pytest
from few_shot import FewShotExample, FewShotFormatter

EXAMPLES = [
    FewShotExample(user="What is 2+2?", assistant="4", label="math"),
    FewShotExample(user="Capital of France?", assistant="Paris", label="geography"),
    FewShotExample(user="What is 10×5?", assistant="50", label="math"),
    FewShotExample(user="Largest ocean?", assistant="Pacific", label="geography"),
]


# ── to_messages ───────────────────────────────────────────────────────────────


def test_to_messages_alternates_roles() -> None:
    fmt = FewShotFormatter(EXAMPLES[:2])
    msgs = fmt.to_messages(EXAMPLES[:2])
    assert msgs[0] == {"role": "user", "content": "What is 2+2?"}
    assert msgs[1] == {"role": "assistant", "content": "4"}
    assert msgs[2] == {"role": "user", "content": "Capital of France?"}
    assert msgs[3] == {"role": "assistant", "content": "Paris"}


def test_to_messages_length() -> None:
    msgs = FewShotFormatter(EXAMPLES).to_messages(EXAMPLES)
    assert len(msgs) == len(EXAMPLES) * 2


def test_to_messages_empty() -> None:
    assert FewShotFormatter([]).to_messages([]) == []


# ── select: first ─────────────────────────────────────────────────────────────


def test_select_first_returns_n() -> None:
    fmt = FewShotFormatter(EXAMPLES)
    assert fmt.select(2, strategy="first") == EXAMPLES[:2]


def test_select_first_capped_at_available() -> None:
    fmt = FewShotFormatter(EXAMPLES[:2])
    assert len(fmt.select(10, strategy="first")) == 2


def test_select_zero_returns_empty() -> None:
    assert FewShotFormatter(EXAMPLES).select(0) == []


# ── select: random ────────────────────────────────────────────────────────────


def test_select_random_returns_correct_count() -> None:
    selected = FewShotFormatter(EXAMPLES).select(2, strategy="random")
    assert len(selected) == 2


def test_select_random_all_from_pool() -> None:
    selected = FewShotFormatter(EXAMPLES).select(3, strategy="random")
    assert all(e in EXAMPLES for e in selected)


def test_select_random_seed_is_deterministic() -> None:
    fmt = FewShotFormatter(EXAMPLES)
    a = fmt.select(3, strategy="random", seed=42)
    b = fmt.select(3, strategy="random", seed=42)
    assert a == b


# ── select: by_label ──────────────────────────────────────────────────────────


def test_select_by_label_covers_labels() -> None:
    selected = FewShotFormatter(EXAMPLES).select(2, strategy="by_label")
    labels = {e.label for e in selected}
    assert len(labels) == 2  # both "math" and "geography"


def test_select_by_label_count() -> None:
    selected = FewShotFormatter(EXAMPLES).select(3, strategy="by_label")
    assert len(selected) == 3


def test_select_by_label_more_than_pool() -> None:
    fmt = FewShotFormatter(EXAMPLES[:2])
    selected = fmt.select(10, strategy="by_label")
    assert len(selected) == 2  # capped at pool size


# ── select: invalid strategy ──────────────────────────────────────────────────


def test_select_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError, match="strategy"):
        FewShotFormatter(EXAMPLES).select(2, strategy="unsupported")  # type: ignore[arg-type]


# ── prepend_to_messages ───────────────────────────────────────────────────────


def test_prepend_appends_question_last() -> None:
    fmt = FewShotFormatter(EXAMPLES)
    question: dict[str, str] = {"role": "user", "content": "What is 3+3?"}
    msgs = fmt.prepend_to_messages([question], n=1, strategy="first")
    assert msgs[-1] == question


def test_prepend_total_length() -> None:
    fmt = FewShotFormatter(EXAMPLES)
    question: dict[str, str] = {"role": "user", "content": "q"}
    msgs = fmt.prepend_to_messages([question], n=2, strategy="first")
    assert len(msgs) == 2 * 2 + 1  # 2 examples × 2 turns + 1 question
