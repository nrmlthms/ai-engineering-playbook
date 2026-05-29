"""
XML tag extraction for structured LLM output.

Why XML over JSON?
──────────────────
JSON parsing is fragile in free-text completions: a model might add prose before
the JSON, forget to close brackets, or use trailing commas (not valid JSON).
XML tags are more forgiving — the model encountered XML-like structure throughout
pre-training (HTML, DocBook, man pages, code comments), so tag-wrapped content is
highly reliable.

Claude's own system prompt uses XML internally; using it in your prompts aligns
with how the model was trained to structure text.

Pattern: ask for a specific tag, extract it, parse its contents.
  "Respond with <answer>your answer</answer>"
  → extract_tag(text, "answer") → "42"

  "List each step in <step> tags"
  → extract_all_tags(text, "step") → ["step one", "step two", ...]

  "Use <name>, <date>, <amount> tags"
  → extract_tags(text, ["name", "date", "amount"]) → dict
"""

import re


def extract_tag(text: str, tag: str) -> str | None:
    """
    Extract the first occurrence of <tag>…</tag> from text.
    Returns stripped content, or None if the tag is not present.
    """
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_all_tags(text: str, tag: str) -> list[str]:
    """Extract all occurrences of <tag>…</tag>, returning a list in order."""
    return [m.strip() for m in re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)]


def extract_tags(text: str, tags: list[str]) -> dict[str, str | None]:
    """Extract multiple named tags in one pass. Missing tags map to None."""
    return {tag: extract_tag(text, tag) for tag in tags}


def assert_tags_present(result: dict[str, str | None], required: list[str]) -> None:
    """
    Raise ValueError if any required tags are missing from an extract_tags() result.
    Call this at the boundary where a missing tag is a hard error, not just unknown.
    """
    missing = [tag for tag in required if result.get(tag) is None]
    if missing:
        raise ValueError(f"Required tags missing from completion: {missing}")
