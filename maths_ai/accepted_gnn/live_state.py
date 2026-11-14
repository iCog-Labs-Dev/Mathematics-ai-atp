"""Conversion of strict Pantograph model-S-expression payloads into Goals."""

from __future__ import annotations

from typing import Any

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC


class ModelSExprProtocolError(ValueError):
    """Raised when a live state does not satisfy the v2 input contract."""


def _expression(payload: Any, owner: str) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise ModelSExprProtocolError(f"{owner} expression payload is missing.")
    pp = payload.get("pp")
    model_sexp = payload.get("modelSexp")
    version = payload.get("modelSexpVersion")
    if not isinstance(pp, str) or not isinstance(model_sexp, str):
        raise ModelSExprProtocolError(f"{owner} requires pp and modelSexp strings.")
    if version != MODEL_SEXPR_GRAPH_SPEC.model_sexp_version:
        raise ModelSExprProtocolError(
            f"{owner} modelSexpVersion is {version!r}; expected "
            f"{MODEL_SEXPR_GRAPH_SPEC.model_sexp_version}."
        )
    return pp, model_sexp


def goal_payload_to_goal(payload: dict[str, Any]) -> Goal:
    """Convert one strict v2 goal payload while preserving local identities."""
    target_pp, target_sexp = _expression(payload.get("target"), "goal target")
    variables = payload.get("variables")
    if not isinstance(variables, list):
        raise ModelSExprProtocolError("Goal payload requires a variables list.")

    locals_: list[GoalLocal] = []
    for variable in variables:
        if not isinstance(variable, dict):
            raise ModelSExprProtocolError("Every goal local must be an object.")
        type_pp, type_sexp = _expression(variable.get("type"), "local type")
        context_index = variable.get("contextIndex")
        binder_role = variable.get("binderRole")
        if not isinstance(context_index, int) or not isinstance(binder_role, str):
            raise ModelSExprProtocolError(
                "Every local requires integer contextIndex and string binderRole."
            )
        value_payload = variable.get("value")
        value_pp = value_sexp = None
        if value_payload is not None:
            value_pp, value_sexp = _expression(value_payload, "local value")
        is_let = bool(variable.get("isLet", False))
        if is_let != (value_payload is not None):
            raise ModelSExprProtocolError(
                "Local let declarations must agree with the presence of value data."
            )
        locals_.append(
            GoalLocal(
                user_name=str(variable.get("userName") or "_"),
                internal_name=str(variable.get("name") or ""),
                context_index=context_index,
                binder_role=binder_role,
                is_instance=bool(variable.get("isInstance", False)),
                is_let=is_let,
                type_pp=type_pp,
                type_model_sexp=type_sexp,
                value_pp=value_pp,
                value_model_sexp=value_sexp,
            )
        )

    return Goal(
        expression=target_pp,
        goal_model_sexp=target_sexp,
        locals=locals_,
        model_sexp_version=MODEL_SEXPR_GRAPH_SPEC.model_sexp_version,
        case_tag=str(payload.get("name") or "") or None,
    )
