"""Pantograph protocol helpers for the live ``lean-model-sexp-v2`` state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CapabilityProbe:
    """One live state exercising a v2 payload capability."""

    name: str
    expression: str
    expectation: str


DEFAULT_CAPABILITY_PROBES: tuple[CapabilityProbe, ...] = (
    CapabilityProbe("binders", "∀ (a : Nat), a = a", "binders"),
    CapabilityProbe("shadowing", "∀ (a : Nat), ∀ (a : Nat), a = a", "shadowing"),
    CapabilityProbe("local_lets", "let x := 1; x = x", "local_lets"),
    # Instance binders cannot be constructed through goal_start alone, so the
    # probe checks that the payload *reports* the isInstance field. Full
    # instance semantics still require a compatible extractor deployment.
    CapabilityProbe("instances", "∀ (a : Nat), a = a", "instances"),
)


@dataclass(frozen=True)
class CapabilityProbeResult:
    """Outcome of one live capability probe: fail-closed by construction."""

    name: str
    ok: bool
    detail: str = ""


def _variables_of(goal: Any) -> list[Any] | None:
    variables = _read_first(goal, "variables", "vars")
    return variables if isinstance(variables, list) else None


def capability_findings(goal: Any) -> dict[str, bool]:
    """Check one raw goal payload against each probed payload capability.

    The checkers look at the *raw client payload*, because a live release may
    carry the fields without yet satisfying strict conversion; capability is
    reported per field so failure detail stays actionable.
    """
    variables = _variables_of(goal)
    findings: dict[str, bool] = {
        "binders": False,
        "shadowing": False,
        "local_lets": False,
        "instances": False,
    }
    if not variables:
        return findings
    binder_ok = True
    instance_field_present = True
    names: list[str] = []
    internal_names: list[str] = []
    let_with_value = False
    for variable in variables:
        context_index = _read(variable, "contextIndex")
        binder_role = _read(variable, "binderRole")
        user_name = _read_first(variable, "userName", "name")
        internal_name = _read(variable, "name")
        if not isinstance(context_index, int) or not isinstance(binder_role, str) or not binder_role:
            binder_ok = False
        if not isinstance(user_name, str) or not user_name:
            binder_ok = False
        is_instance = _read(variable, "isInstance")
        if not isinstance(is_instance, bool):
            instance_field_present = False
        is_let = _read(variable, "isLet")
        if is_let and _read(variable, "value") is not None:
            let_with_value = True
        if isinstance(user_name, str) and user_name:
            names.append(user_name)
        if isinstance(internal_name, str) and internal_name:
            internal_names.append(internal_name)
    findings["binders"] = binder_ok
    findings["shadowing"] = len(names) != len(set(names)) and len(internal_names) == len(set(internal_names))
    findings["local_lets"] = let_with_value
    findings["instances"] = instance_field_present
    return findings


async def probe_live_capabilities(
    server: Any,
    *,
    probes: tuple[CapabilityProbe, ...] = DEFAULT_CAPABILITY_PROBES,
) -> dict[str, CapabilityProbeResult]:
    """Fail-closed capability report over representative live states."""
    results: dict[str, CapabilityProbeResult] = {}
    for probe in probes:
        try:
            state = await server.goal_start_async(probe.expression)
            goals = _read(state, "goals")
            if not isinstance(goals, list) or not goals:
                raise ValueError("state carries no goals.")
            findings = capability_findings(goals[0])
            # Strict conversion must also succeed: fail closed on omissions.
            pantograph_state_to_goals(state)
            ok = bool(findings.get(probe.expectation))
            detail = findings if not ok else ""
            results[probe.name] = CapabilityProbeResult(probe.name, ok, str(detail))
        except Exception as exc:  # noqa: BLE001 - every probe failure is data, not a crash.
            results[probe.name] = CapabilityProbeResult(probe.name, False, f"{type(exc).__name__}: {exc}")
    return results


def all_capabilities_verified(results: Mapping[str, CapabilityProbeResult]) -> bool:
    """True only when every probed capability passed."""
    return bool(results) and all(result.ok for result in results.values())
