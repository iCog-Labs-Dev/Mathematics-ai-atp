"""
Hypothesis/target extraction and subgoal processing.

Bridges the parsing and rendering layers: takes Lean proof states (from
Pantograph) and produces structured data ready to be written as .metta files.

Also contains proof_state_after_tactics() for driving Pantograph.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable

from .constants import TYPE_ATOMS, map_const
from .normalizer import VariableNormalizer
from .parser import (
    expr_from_field,
    normalize_logic,
    parse_simple_pp,
    to_plain,
)
from .renderer import build_query, render_formula, validate_formula_shape


def is_nonlogical_type_expr(expr: Any) -> bool:
    return isinstance(expr, str) and expr in TYPE_ATOMS


def is_has_type_expr(expr: Any) -> bool:
    return isinstance(expr, list) and len(expr) >= 1 and expr[0] == "HasType"


def extract_hypotheses(goal_data: Any, normalizer: VariableNormalizer) -> list[Any]:
    g = to_plain(goal_data)
    if not isinstance(g, dict):
        return []
    hyps: list[Any] = []

    for h in g.get("hypotheses", []):
        h_plain = to_plain(h)
        h_type_source = h_plain.get("type") or h_plain.get("t") or h_plain.get("target") or h_plain
        h_expr = normalize_logic(expr_from_field(h_type_source, normalizer))
        if not is_nonlogical_type_expr(h_expr) and not is_has_type_expr(h_expr):
            hyps.append(h_expr)

    for v in g.get("variables", []):
        v_plain = to_plain(v)
        t_source = v_plain.get("type") or v_plain.get("t")
        if t_source is None:
            continue
        t_expr = normalize_logic(expr_from_field(t_source, normalizer))
        if is_nonlogical_type_expr(t_expr) or is_has_type_expr(t_expr):
            continue
        hyps.append(t_expr)
    return hyps


def extract_target(goal_data: Any, normalizer: VariableNormalizer) -> Any:
    g = to_plain(goal_data)
    if not isinstance(g, dict):
        return normalize_logic(expr_from_field(g, normalizer))
    for key in ("target", "goal", "target_ast", "type"):
        if key in g and g[key] is not None:
            return normalize_logic(expr_from_field(g[key], normalizer))
    if "sexp" in g:
        return normalize_logic(expr_from_field({"sexp": g["sexp"]}, normalizer))
    if "pp" in g:
        return normalize_logic(parse_simple_pp(str(g["pp"]), normalizer))
    return "UnknownGoal"


def recurry(hyps: Iterable[Any], target: Any) -> Any:
    out = target
    for h in reversed(list(hyps)):
        out = ["Implication", h, out]
    return out


def essentialize_subgoal(
    goal_data: Any,
    *,
    kb: str = "kb",
    mode: str = "recurry",
    depth: int = 10,
    normalize_variables: bool = False,
) -> tuple[str, str, list[dict[str, Any]], str]:
    """
    Return:
      script: MeTTa commands
      cleanup: cleanup comments
      hyp_records: structured local hypotheses
      query_formula: rendered target formula
    """
    normalizer = VariableNormalizer(normalize=normalize_variables)
    hyps = extract_hypotheses(goal_data, normalizer)
    target = extract_target(goal_data, normalizer)
    query_formula = render_formula(target)

    if mode == "recurry":
        formula = recurry(hyps, target)
        return build_query(formula, kb=kb, depth=depth), "", [], render_formula(formula)

    if mode == "dynamic":
        branch = uuid.uuid4().hex[:8]
        output_lines: list[str] = []
        cleanup_lines: list[str] = []
        hyp_records: list[dict[str, Any]] = []

        for i, hyp in enumerate(hyps):
            validate_formula_shape(hyp)
            formula = render_formula(hyp)
            label = f"local-hyp-{branch}-{i}"
            atom = f"(: {label} {formula} (STV 1.0 1.0))"
            output_lines.append(f"!(compileadd {kb} {atom})")
            cleanup_lines.append(f";; cleanup needed for {label}")
            hyp_records.append({
                "index": i,
                "label": label,
                "formula": formula,
                "atom": atom,
            })

        query_cmd = build_query(target, kb=kb, depth=depth)
        output_lines.append(query_cmd)

        return "\n".join(output_lines), "\n".join(cleanup_lines), hyp_records, query_formula

    raise ValueError("mode must be 'recurry' or 'dynamic'")


def proof_state_after_tactics(server, goal_type: str, tactics: list[str]):
    state = server.goal_start(goal_type)

    for tac in tactics:
        state = server.goal_tactic(state, tactic=tac)

    return state
