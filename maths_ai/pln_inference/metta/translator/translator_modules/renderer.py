"""
PeTTaChainer formula rendering and MeTTa query building.

Converts the translator's intermediate representation (nested lists/strings)
into the MeTTa text format expected by PeTTaChainer.
"""

from __future__ import annotations

from typing import Any


def render_term(obj: Any) -> str:
    if isinstance(obj, list):
        return "(" + " ".join(render_term(x) for x in obj) + ")"
    return str(obj)


def render_formula(obj: Any) -> str:
    """Render formula in the same format emitted by the updated parser."""
    if isinstance(obj, list):
        if not obj:
            return "()"
        head = obj[0]
        if head == "Implication" and len(obj) == 3:
            premise = render_formula(obj[1])
            conclusion = render_formula(obj[2])
            return (
                "(Implication\n"
                f"   (Premises {premise})\n"
                f"   (Conclusions {conclusion})\n"
                ")"
            )
        if head == "Not" and len(obj) == 2:
            return f"(Not {render_formula(obj[1])})"
        if head in {"∧", "∨", "↔"} and len(obj) == 3:
            return f"({head} {render_formula(obj[1])} {render_formula(obj[2])})"
        return "(" + " ".join(render_term(x) for x in obj) + ")"
    return f"({obj})"


def validate_formula_shape(expr: Any) -> None:
    forbidden = {"Provable"}

    def walk(x: Any) -> None:
        if isinstance(x, list):
            if x and x[0] in forbidden:
                raise ValueError("The updated parser no longer wraps formulas in (Provable ...).")
            for y in x:
                walk(y)

    walk(expr)


def build_query(formula: Any, *, kb: str = "kb", depth: int = 10) -> str:
    validate_formula_shape(formula)
    return f"!(query {depth} {kb} (: $prf {render_formula(formula)} $tv))"
