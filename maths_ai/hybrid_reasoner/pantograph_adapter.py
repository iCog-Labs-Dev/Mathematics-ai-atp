"""Pantograph protocol helpers for the live ``lean-model-sexp-v2`` state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from maths_ai.data_models.proof_components import Goal
from maths_ai.hybrid_reasoner.live_state import goal_payload_to_goal

MODEL_STATE_OPTIONS: dict[str, bool] = {
    "printExprAST": True,
    "printExprModelAST": True,
}


def model_state_options(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return server options required to preserve v2 model-S-expression data."""
    merged = dict(options or {})
    merged.update(MODEL_STATE_OPTIONS)
    return merged


def _read(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _read_first(value: Any, *names: str) -> Any:
    for name in names:
        result = _read(value, name)
        if result is not None:
            return result
    return None


def _expression_payload(value: Any, owner: str) -> dict[str, Any]:
    payload = {
        "pp": _read(value, "pp"),
        "modelSexp": _read(value, "modelSexp"),
        "modelSexpVersion": _read(value, "modelSexpVersion"),
    }
    if not all(isinstance(payload[key], str) for key in ("pp", "modelSexp")):
        raise ValueError(f"{owner} is missing Pantograph pp/modelSexp fields.")
    return payload


def _goal_payload(value: Any) -> dict[str, Any]:
    target = _expression_payload(_read(value, "target"), "goal target")
    variables = _read_first(value, "variables", "vars")
    if not isinstance(variables, list):
        raise ValueError("Pantograph goal is missing its variables list.")

    normalized_variables: list[dict[str, Any]] = []
    for variable in variables:
        type_payload = _expression_payload(_read(variable, "type"), "local type")
        raw_value = _read(variable, "value")
        normalized_variables.append(
            {
                "userName": _read_first(variable, "userName", "name"),
                "name": _read(variable, "name"),
                "contextIndex": _read(variable, "contextIndex"),
                "binderRole": _read(variable, "binderRole"),
                "isInstance": _read(variable, "isInstance"),
                "isLet": _read(variable, "isLet"),
                "type": type_payload,
                "value": None if raw_value is None else _expression_payload(raw_value, "local value"),
            }
        )
    return {
        "name": _read(value, "name"),
        "target": target,
        "variables": normalized_variables,
    }


def pantograph_goal_to_goal(value: Any) -> Goal:
    """Convert one Pantograph goal object into the strict internal contract."""
    return goal_payload_to_goal(_goal_payload(value))


def has_model_state_payload(state: Any) -> bool:
    """Return whether a state appears to carry v2 metadata."""
    goals = _read(state, "goals")
    if not isinstance(goals, list) or not goals:
        return False
    target = _read(goals[0], "target")
    return _read(target, "modelSexp") is not None


def pantograph_state_to_goals(state: Any) -> list[Goal]:
    """Convert every goal in a Pantograph state, failing closed on omissions."""
    goals = _read(state, "goals")
    if not isinstance(goals, list):
        raise ValueError("Pantograph state is missing its goals list.")
    return [pantograph_goal_to_goal(goal) for goal in goals]


async def probe_model_state(server: Any, expression: str = "∀ (p : Prop), p → p") -> None:
    """Verify that the connected server returns the fields required by v2."""
    state = await server.goal_start_async(expression)
    pantograph_state_to_goals(state)
