"""
Versioned prompt templates with variable interpolation.

Why version prompts?
────────────────────
A prompt is executable specification. When you change it:
  - Model outputs can change silently (no Python exception, no test failure)
  - You lose the ability to roll back after a production regression
  - A/B testing requires knowing exactly which prompt was in use

Version strings follow calver (YYYY-MM-DD) or semver — the key property is
ordering: you can always tell which version came before another.

Variable syntax uses Python's str.format_map(), which supports:
  {name}        Simple substitution
  {count:,}     Format spec (comma thousands separator)

render() returns (system_prompt, user_message) so the caller can pass them
directly to messages.create(system=..., messages=[{"role": "user", ...}]).
"""

from dataclasses import dataclass
from string import Formatter


@dataclass
class PromptTemplate:
    name: str
    version: str
    system: str
    user_template: str
    description: str = ""

    @property
    def variable_names(self) -> frozenset[str]:
        """All named placeholders in user_template (parsed, not regex)."""
        return frozenset(
            fname
            for _, fname, _, _ in Formatter().parse(self.user_template)
            if fname is not None
        )

    def render(self, **variables: str) -> tuple[str, str]:
        """
        Interpolate variables and return (system_prompt, user_message).

        Raises ValueError on missing variables so prompt bugs surface at call
        time, not as garbled model output.
        """
        missing = self.variable_names - set(variables)
        if missing:
            raise ValueError(
                f"Template '{self.name}' missing variables: {sorted(missing)}"
            )
        return (self.system, self.user_template.format_map(variables))

    def __str__(self) -> str:
        return f"PromptTemplate(name={self.name!r}, version={self.version!r})"
