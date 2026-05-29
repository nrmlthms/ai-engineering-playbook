"""
PromptTemplate tests — no network.
"""

import pytest

from prompts.template import PromptTemplate


def test_render_single_variable() -> None:
    t = PromptTemplate(
        name="test",
        version="2026-01-01",
        system="You are helpful.",
        user_template="Summarise: {text}",
    )
    system, user = t.render(text="Hello world")
    assert system == "You are helpful."
    assert user == "Summarise: Hello world"


def test_render_multiple_variables() -> None:
    t = PromptTemplate(
        name="test",
        version="1.0",
        system="sys",
        user_template="Translate {text} from {source} to {target}.",
    )
    _, user = t.render(text="hello", source="English", target="French")
    assert user == "Translate hello from English to French."


def test_render_no_variables() -> None:
    t = PromptTemplate(
        name="test",
        version="1.0",
        system="You are a poet.",
        user_template="Write me a haiku.",
    )
    system, user = t.render()
    assert system == "You are a poet."
    assert user == "Write me a haiku."


def test_render_missing_variable_raises() -> None:
    t = PromptTemplate(
        name="test",
        version="1.0",
        system="sys",
        user_template="Hello {name}, you have {count} messages.",
    )
    with pytest.raises(ValueError, match="count"):
        t.render(name="Alice")  # missing 'count'


def test_render_extra_variables_are_ignored() -> None:
    t = PromptTemplate(
        name="test",
        version="1.0",
        system="sys",
        user_template="Hello {name}.",
    )
    _, user = t.render(name="Alice", extra="ignored")
    assert user == "Hello Alice."


def test_variable_names_detected() -> None:
    t = PromptTemplate(
        name="test",
        version="1.0",
        system="no vars here",
        user_template="Hello {name}, items: {items}",
    )
    assert t.variable_names == frozenset({"name", "items"})


def test_variable_names_empty_when_no_placeholders() -> None:
    t = PromptTemplate(
        name="test",
        version="1.0",
        system="sys",
        user_template="No variables here.",
    )
    assert t.variable_names == frozenset()


def test_str_includes_name_and_version() -> None:
    t = PromptTemplate(
        name="invoice-extractor",
        version="2026-01",
        system="",
        user_template="",
    )
    s = str(t)
    assert "invoice-extractor" in s
    assert "2026-01" in s


def test_render_preserves_system_verbatim() -> None:
    system_prompt = "You are an assistant.\n\nAlways respond in JSON."
    t = PromptTemplate(
        name="test",
        version="1.0",
        system=system_prompt,
        user_template="{query}",
    )
    system, _ = t.render(query="hi")
    assert system == system_prompt
