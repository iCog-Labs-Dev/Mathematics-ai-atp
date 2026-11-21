"""Lean command rendering shared by policy identities and execution."""

from typing import Protocol, Sequence


class TacticInput(Protocol):
    tactic_name: str
    arguments: Sequence[str]


def _is_lean_identifier(value: str) -> bool:
    parts = value.split(".")
    return bool(parts) and all(
        bool(core := part.rstrip("'!?")) and core.isidentifier()
        for part in parts
    )


_BRACKET_REQUIRED_TACTICS = frozenset({
    "rw", "rwa", "rewrite", "simp_rw", "simp only", "erw",
})
_BRACKET_OPTIONAL_TACTICS = frozenset({
    "simp", "simpa", "field_simp", "norm_num", "linarith", "nlinarith", "aesop",
})


def render_tactic_command(tactic: TacticInput) -> str | None:
    """Return the execution command, or None when it cannot be rendered."""
    name = tactic.tactic_name.strip()
    if not name:
        return None
    arguments = [arg.rstrip(":").strip() for arg in tactic.arguments]
    arguments = [arg for arg in arguments if arg]
    if name in _BRACKET_REQUIRED_TACTICS or name in _BRACKET_OPTIONAL_TACTICS:
        # Every selected argument is part of the action: one that is not a rule
        # name rejects the whole action rather than leaving a different one.
        if any(not _is_lean_identifier(arg) for arg in arguments):
            return None
        rules = arguments
        if rules:
            return f"{name} [{', '.join(rules)}]"
        if name in _BRACKET_REQUIRED_TACTICS:
            return None
        return name
    if not arguments:
        return name
    return " ".join([name, *arguments])
