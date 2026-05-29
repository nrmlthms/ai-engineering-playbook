"""
Chain-of-thought (CoT) prompting patterns.

Zero-shot CoT — Kojima et al. (2022)
  Appending "Let's think step by step." to a question dramatically improves
  accuracy on math, logic, and symbolic reasoning tasks — no examples needed.
  The paper calls this zero-shot-CoT: a single phrase unlocks latent reasoning.

Few-shot CoT — Wei et al. (2022)
  Including worked examples of reasoning (question + step-by-step answer) in
  the context further improves accuracy. Cost: longer prompts, and examples
  must be carefully chosen (bad examples hurt more than they help).

Scratchpad pattern
  Ask the model to reason inside <thinking> tags, then answer in <answer> tags.
  Benefits:
    - Separates reasoning from the final answer (parse with extract_tag)
    - Makes reasoning inspectable/loggable without exposing it to users
    - Gives the model "space" to work through multi-step problems

  Compared to Claude extended thinking (Stage 02): the scratchpad pattern is
  prompt-level, works on any model, and is always visible in the response.
  Extended thinking uses a dedicated API parameter and billing bucket.
"""

from extractor import extract_tag

ZERO_SHOT_COT_SUFFIX = "\n\nLet's think step by step."

SCRATCHPAD_SYSTEM_SNIPPET = """\
Before giving your final answer, reason through the problem inside <thinking> tags.
Then provide your answer inside <answer> tags.

Example format:
<thinking>
[Your step-by-step reasoning here]
</thinking>
<answer>
[Your final answer here]
</answer>"""


def zero_shot_cot(question: str) -> str:
    """Append the CoT trigger phrase to a question."""
    return f"{question}{ZERO_SHOT_COT_SUFFIX}"


def extract_cot_answer(text: str) -> tuple[str | None, str | None]:
    """
    Parse scratchpad-style output into (thinking, answer).
    Either field may be None if the model didn't use the tags.
    """
    return extract_tag(text, "thinking"), extract_tag(text, "answer")


def build_scratchpad_system(base_system: str = "") -> str:
    """
    Build a system prompt that instructs the model to use the scratchpad pattern.
    If base_system is provided it is prepended so role/context comes first.
    """
    if base_system:
        return f"{base_system.rstrip()}\n\n{SCRATCHPAD_SYSTEM_SNIPPET}"
    return SCRATCHPAD_SYSTEM_SNIPPET
