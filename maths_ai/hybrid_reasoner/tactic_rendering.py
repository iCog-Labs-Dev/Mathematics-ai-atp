"""Render a tactic candidate as Lean surface syntax.

Search and replay render with this one function, so a replayed step runs the
same command shape as the explored one.
"""

from __future__ import annotations

from typing import Optional

from maths_ai.data_models.proof_components import TacticCandidate

# A Lean identifier usable as a rewrite rule or simp lemma: hypothesis names and
# dotted lemma names, but not terms like `↑13` or `?m.2235`.


def _is_lean_identifier(value: str) -> bool:
    parts = value.split(".")
    return bool(parts) and all(
        bool(core := part.rstrip("'!?")) and core.isidentifier() for part in parts
    )


# Tactics whose arguments must be a bracketed list. `rw h` is a parse error at the
# column where `[` was expected; the rule list is `rw [h]`. With no arguments these
# tactics have nothing to rewrite with and are unplayable.
_BRACKET_REQUIRED_TACTICS = frozenset(
    {
        "rw",
        "rwa",
        "rewrite",
        "simp_rw",
        "simp only",
        "erw",
    }
)

# Tactics that run bare but take a bracketed list when given lemmas.
_BRACKET_OPTIONAL_TACTICS = frozenset(
    {
        "simp",
        "simpa",
        "field_simp",
        "norm_num",
        "linarith",
        "nlinarith",
        "aesop",
    }
)



def render_tactic_command(tactic: TacticCandidate) -> Optional[str]:
    """Render a tactic and its arguments as Lean surface syntax.

    Argument shape is per tactic, not uniform: ``rw`` needs ``rw [h]`` while
    ``apply`` needs ``apply h``. Rendering everything as space-separated arguments
    makes every ``rw`` draw a parse error before any rewriting is attempted.

    Returns ``None`` when the tactic cannot be rendered into a playable command —
    a bracket-requiring tactic with no usable rule names — so the caller can drop
    the candidate instead of sending Lean something it will certainly reject.
    """
    name = tactic.tactic_name.strip()
    arguments = [arg.rstrip(":").strip() for arg in tactic.arguments]
    arguments = [arg for arg in arguments if arg]

    if name in _BRACKET_REQUIRED_TACTICS or name in _BRACKET_OPTIONAL_TACTICS:
        # A rewrite rule must be a name Lean can look up. The policy samples nodes
        # from the goal's own graph, so a draw can land on a term like `↑13`, which
        # is a valid expression but not a rule.
        rules = [arg for arg in arguments if _is_lean_identifier(arg)]
        if rules:
            return f"{name} [{', '.join(rules)}]"
        if name in _BRACKET_REQUIRED_TACTICS:
            return None
        return name

    if not arguments:
        return name
    return " ".join([name, *arguments])
