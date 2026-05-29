"""
Few-shot example selection and formatting.

Why few-shot prompting?
───────────────────────
Instructions describe behaviour in the abstract; examples demonstrate it concretely.
For tasks where "correct" is hard to express as a rule (tone, formatting conventions,
edge-case handling), showing three good examples often beats a paragraph of prose.

Research (Brown et al. 2020 — GPT-3 paper) shows that:
  0-shot: "Translate to French: {text}"
  1-shot: same prompt + 1 example                  → big jump
  3-shot: same prompt + 3 examples                 → further gain
  10-shot: diminishing returns; context cost grows

Selection strategies
────────────────────
  first    Deterministic. Good for fast prototyping.
  random   Reduces ordering bias. Seed for reproducible experiments.
  by_label Round-robin across label groups. Ensures label coverage in the
           context window — important for classification tasks.

Message format
──────────────
Few-shot examples are injected as prior user/assistant turns, not in the system
prompt. The model treats them as evidence of expected behaviour.

  messages = formatter.prepend_to_messages(
      [{"role": "user", "content": query}], n=3
  )
"""

import random as _random
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal


@dataclass
class FewShotExample:
    user: str
    assistant: str
    label: str | None = None


class FewShotFormatter:
    def __init__(self, examples: list[FewShotExample]) -> None:
        self.examples = examples

    def select(
        self,
        n: int,
        *,
        strategy: Literal["first", "random", "by_label"] = "first",
        seed: int | None = None,
    ) -> list[FewShotExample]:
        """Return up to n examples using the chosen strategy."""
        if n <= 0:
            return []

        if strategy == "first":
            return self.examples[:n]

        elif strategy == "random":
            rng = _random.Random(seed)
            pool = list(self.examples)
            rng.shuffle(pool)
            return pool[:n]

        elif strategy == "by_label":
            # Group by label, then round-robin so every label gets representation.
            groups: dict[str | None, list[FewShotExample]] = defaultdict(list)
            for ex in self.examples:
                groups[ex.label].append(ex)

            label_lists = list(groups.values())
            positions = [0] * len(label_lists)
            result: list[FewShotExample] = []

            while len(result) < n:
                progress = False
                for i, label_list in enumerate(label_lists):
                    if len(result) >= n:
                        break
                    if positions[i] < len(label_list):
                        result.append(label_list[positions[i]])
                        positions[i] += 1
                        progress = True
                if not progress:
                    break

            return result

        else:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose 'first', 'random', or 'by_label'."
            )

    def to_messages(self, examples: list[FewShotExample]) -> list[dict[str, str]]:
        """Format examples as alternating user/assistant message turns."""
        msgs: list[dict[str, str]] = []
        for ex in examples:
            msgs.append({"role": "user", "content": ex.user})
            msgs.append({"role": "assistant", "content": ex.assistant})
        return msgs

    def prepend_to_messages(
        self,
        messages: list[dict[str, str]],
        n: int = 3,
        *,
        strategy: Literal["first", "random", "by_label"] = "first",
        seed: int | None = None,
    ) -> list[dict[str, str]]:
        """Select n examples and prepend them as context turns before messages."""
        selected = self.select(n, strategy=strategy, seed=seed)
        return self.to_messages(selected) + messages
