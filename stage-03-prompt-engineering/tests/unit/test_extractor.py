"""
XML extractor tests — no network, no LLM calls.
"""

import pytest
from extractor import assert_tags_present, extract_all_tags, extract_tag, extract_tags


def test_extract_simple() -> None:
    assert extract_tag("<answer>42</answer>", "answer") == "42"


def test_extract_strips_whitespace() -> None:
    assert extract_tag("<answer>\n  42  \n</answer>", "answer") == "42"


def test_extract_missing_returns_none() -> None:
    assert extract_tag("no tags here", "answer") is None


def test_extract_multiline_content() -> None:
    text = "<answer>\nline one\nline two\n</answer>"
    assert extract_tag(text, "answer") == "line one\nline two"


def test_extract_first_of_multiple() -> None:
    text = "<answer>first</answer> then <answer>second</answer>"
    assert extract_tag(text, "answer") == "first"


def test_extract_with_surrounding_prose() -> None:
    text = "Here is my answer: <answer>Paris</answer> Hope that helps!"
    assert extract_tag(text, "answer") == "Paris"


def test_extract_inner_xml_preserved() -> None:
    text = "<output><item>a</item><item>b</item></output>"
    inner = extract_tag(text, "output")
    assert inner is not None
    assert "<item>a</item>" in inner


def test_extract_all_tags_ordered() -> None:
    text = "<step>one</step><step>two</step><step>three</step>"
    assert extract_all_tags(text, "step") == ["one", "two", "three"]


def test_extract_all_tags_empty() -> None:
    assert extract_all_tags("no steps here", "step") == []


def test_extract_all_tags_strips_whitespace() -> None:
    text = "<step>  alpha  </step><step>  beta  </step>"
    assert extract_all_tags(text, "step") == ["alpha", "beta"]


def test_extract_tags_multiple_present() -> None:
    text = "<name>Alice</name><date>2025-01-15</date><amount>100</amount>"
    result = extract_tags(text, ["name", "date", "amount"])
    assert result == {"name": "Alice", "date": "2025-01-15", "amount": "100"}


def test_extract_tags_missing_maps_to_none() -> None:
    result = extract_tags("<name>Alice</name>", ["name", "date"])
    assert result["name"] == "Alice"
    assert result["date"] is None


def test_assert_tags_present_raises_on_missing() -> None:
    result = {"name": "Alice", "date": None}
    with pytest.raises(ValueError, match="date"):
        assert_tags_present(result, ["name", "date"])


def test_assert_tags_present_passes_when_all_present() -> None:
    result = {"name": "Alice", "date": "2025-01-15"}
    assert_tags_present(result, ["name", "date"])  # must not raise


def test_assert_tags_present_reports_all_missing() -> None:
    result: dict[str, str | None] = {"a": None, "b": None, "c": "present"}
    with pytest.raises(ValueError) as exc_info:
        assert_tags_present(result, ["a", "b", "c"])
    msg = str(exc_info.value)
    assert "a" in msg
    assert "b" in msg
