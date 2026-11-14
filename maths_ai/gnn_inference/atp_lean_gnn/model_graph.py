"""Convert ``lean-model-sexp-v2`` payloads into the policy DAG format.

The conversion reuses the canonical Pantograph S-expression builder
(``graph.sexp_to_dag`` / ``graph._sexp_walk``) so live inference builds
exactly the graphs the policy was trained on:

- ``(:c X)`` becomes a single constant-typed node labeled ``X``;
- ``(:fv FVi)`` stays one free-variable node (the pointer target);
- structured locals become 4-child ``Hyp(FV{i}, name, HypRole:role, type)``
  nodes, matching the training-time hypothesis schema ``fv-name-role-type-v1``.
"""

from __future__ import annotations

from maths_ai.data_models.proof_components import Goal
from maths_ai.pln_inference.metta.translator.translator_modules.parser import (
    parse_sexp_string,
)

from .graph import DAGBuilder, _sexp_walk, sexp_to_dag


class ModelSExprGraphError(ValueError):
    """Raised when a model S-expression cannot be converted safely."""


def _parse(text: str):
    try:
        return parse_sexp_string(text)
    except ValueError as exc:
        raise ModelSExprGraphError(str(exc)) from exc


def _walk(sexp, dag: DAGBuilder) -> int:
    try:
        return _sexp_walk(sexp, [], dag)
    except ValueError as exc:
        raise ModelSExprGraphError(str(exc)) from exc


def _build_hypothesis(dag: DAGBuilder, local: Goal) -> int:
    """Build one ``Hyp`` node in the trained fv-name-role-type-v1 schema.

    Local types share the goal's hash-consed DAG, so a type repeated across
    hypotheses is one node. Lets keep their ``let`` role label; the value
    S-expression is not part of the hypothesis schema the policy trained on.
    """
    name_node = dag.get_or_create(local.user_name or "_", (), node_type="sconst")
    type_node = _walk(_parse(local.type_model_sexp), dag)
    context_node = dag.get_or_create(f"FV{local.context_index}", (), node_type="var")
    role = (
        "instance"
        if local.is_instance
        else "let"
        if local.is_let
        else local.binder_role.removeprefix(":")
    )
    role_node = dag.get_or_create(f"HypRole:{role}", (), node_type="sconst")
    return dag.get_or_create("Hyp", (context_node, name_node, role_node, type_node))


def goal_to_model_dag(goal: Goal) -> DAGBuilder:
    """Build a graph from structured goal state, retaining local hypotheses."""
    goal.require_model_state()
    assert goal.goal_model_sexp is not None
    assert goal.locals is not None
    try:
        dag = sexp_to_dag(goal.goal_model_sexp)
    except ValueError as exc:
        raise ModelSExprGraphError(str(exc)) from exc
    if dag.expression_root_id is None:
        raise ModelSExprGraphError("Goal S-expression did not produce an expression root.")

    root_ids: list[int] = []
    for local in goal.locals:
        root_ids.append(_build_hypothesis(dag, local))
    root_ids.append(dag.get_or_create("Goal", (dag.expression_root_id,)))
    dag.get_or_create("State", tuple(root_ids))
    return dag
