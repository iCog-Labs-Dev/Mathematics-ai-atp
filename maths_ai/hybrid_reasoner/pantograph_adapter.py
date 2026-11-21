"""Adapter from the standalone Pantograph client to this repository's models.

The client returns immutable execution types and owns every wire version.
Canonical ``Goal``/``GoalLocal`` values, translator projections, and the graph
representation check live here, so the dependency only points app -> client.
"""

from __future__ import annotations

from typing import Any

from pantograph_async import (
    MODEL_SEXP_VERSION,
    ExecutionGoal,
    ExecutionLocal,
    PantographConfigurationError,
    TacticState,
    TacticSuccess,
    WireExpression,
)

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import (
    MODEL_SEXPR_GRAPH_SPEC,
    GraphRepresentationSpec,
)
from maths_ai.lean_replay import StructuredLocal


def require_client_compatibility(
    spec: GraphRepresentationSpec = MODEL_SEXPR_GRAPH_SPEC,
) -> None:
    """The graph representation must match the client's model-expression wire."""
    if spec.model_sexp_version != MODEL_SEXP_VERSION:
        raise PantographConfigurationError(
            "Graph representation expects modelSexpVersion "
            f"{spec.model_sexp_version}, but the installed client decodes "
            f"{MODEL_SEXP_VERSION}."
        )


def execution_local_to_goal_local(local: ExecutionLocal) -> GoalLocal:
    return GoalLocal(
        user_name=local.user_name or "_",
        internal_name=local.internal_name,
        context_index=local.context_index,
        binder_role=local.binder_role,
        is_instance=local.is_instance,
        is_let=local.is_let,
        type_pp=local.type.pp,
        type_model_sexp=local.type.model_sexp,
        value_pp=local.value.pp if local.value else None,
        value_model_sexp=local.value.model_sexp if local.value else None,
        model_sexp_version=local.type.model_sexp_version,
        type_sexp=local.type.sexp,
        value_sexp=local.value.sexp if local.value else None,
    )


def execution_goal_to_goal(goal: ExecutionGoal) -> Goal:
    """Return a fresh canonical copy; mutations cannot alter the execution snapshot."""
    return Goal(
        expression=goal.target.pp,
        goal_model_sexp=goal.target.model_sexp,
        goal_sexp=goal.target.sexp,
        locals=[execution_local_to_goal_local(local) for local in goal.locals],
        case_tag=goal.case_tag,
        model_sexp_version=goal.target.model_sexp_version,
    )


def execution_state_to_goals(state: TacticState | TacticSuccess) -> list[Goal]:
    return [execution_goal_to_goal(goal) for goal in state.goals]


def execution_goal_to_structured_locals(goal: ExecutionGoal) -> list[StructuredLocal]:
    """The replayed context in the form ``validate_replay`` compares."""
    return [
        StructuredLocal(
            internal_id=local.internal_name,
            name=local.user_name,
            context_index=local.context_index,
            binder_role=local.binder_role,
            is_instance=local.is_instance,
            is_let=local.is_let,
            type_sexp=local.type.sexp,
            value_sexp=local.value.sexp if local.value is not None else None,
        )
        for local in goal.locals
    ]


def execution_goal_to_translator(goal: ExecutionGoal) -> dict[str, Any]:
    """Named projection consumed by the MeTTa translator."""

    def expression(value: WireExpression) -> dict[str, Any]:
        return {
            "pp": value.pp,
            "sexp": value.sexp,
            "model_sexp": value.model_sexp,
            "model_sexp_version": value.model_sexp_version,
        }

    return {
        "goal_name": goal.goal_name,
        "case_tag": goal.case_tag,
        "is_conversion": goal.is_conversion,
        "fragment": goal.fragment,
        "target": expression(goal.target),
        "locals": [
            {
                "internal_name": local.internal_name,
                "user_name": local.user_name,
                "context_index": local.context_index,
                "binder_role": local.binder_role,
                "is_instance": local.is_instance,
                "is_let": local.is_let,
                "is_inaccessible": local.is_inaccessible,
                "type": expression(local.type),
                "value": expression(local.value) if local.value is not None else None,
            }
            for local in goal.locals
        ],
    }
