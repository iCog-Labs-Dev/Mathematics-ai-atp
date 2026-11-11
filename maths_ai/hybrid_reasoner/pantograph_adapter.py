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


_PINNED_EXPRESSION_KEYS = ("pp", "sexp")


def _expression_style(expression: Any) -> str:
    if _read(expression, "modelSexp") is not None:
        return "model-sexp-v2"
    if _read(expression, "sexp") is not None:
        # The pinned PyPantograph rev emits plain ``sexp`` under
        # ``printExprAST`` even when ``printExprModelAST`` is requested; the
        # normalized-extraction fork is required for the v2 fields.
        return "plain-sexp (pinned rev)"
    return "print-only"


def protocol_gap_report(state: Any) -> dict[str, Any]:
    """Inventory the gaps between a live payload and the v2 input contract.

    Returns the strict-conversion error plus a field-level gap inventory so
    deployments get actionable diagnostics instead of a bare_fail message.
    """
    goals = _read(state, "goals")
    goal = goals[0] if isinstance(goals, list) and goals else None
    target = _read(goal, "target") if goal is not None else None
    report: dict[str, Any] = {
        "conversion": "ok",
        "expression_style": _expression_style(target) if target is not None else "no-target",
        "missing_expression_fields": [],
        "missing_variable_fields": [],
    }
    try:
        pantograph_state_to_goals(state)
    except Exception as exc:  # noqa: BLE001 - the error text is the report payload.
        report["conversion"] = f"{type(exc).__name__}: {exc}"
    missing_expression: set[str] = set()
    expressions: list[Any] = []
    if target is not None:
        expressions.append(target)
    variables = _read_first(goal, "variables", "vars") if goal is not None else None
    if isinstance(variables, list):
        for variable in variables:
            for owner_key in ("type", "value"):
                payload = _read(variable, owner_key)
                if payload is not None:
                    expressions.append(payload)
    for expression in expressions:
        missing_expression.update(
            field
            for field in ("modelSexp", "modelSexpVersion")
            if _read(expression, field) is None
        )
    report["missing_expression_fields"] = sorted(missing_expression)
    if not isinstance(variables, list) or not variables:
        report["missing_variable_fields"] = [
            "contextIndex",
            "binderRole",
            "isInstance",
            "isLet",
            "userName",
        ]
        return report
    missing_variables: set[str] = set()
    for variable in variables:
        type_payload = _read(variable, "type")
        checks = {
            "userName": _read_first(variable, "userName", "name") is not None,
            "name": _read(variable, "name") is not None,
            "contextIndex": isinstance(_read(variable, "contextIndex"), int),
            "binderRole": isinstance(_read(variable, "binderRole"), str),
            "isInstance": isinstance(_read(variable, "isInstance"), bool),
            "isLet": isinstance(_read(variable, "isLet"), bool),
            "type.modelSexp": type_payload is not None and _read(type_payload, "modelSexp") is not None,
            "type.modelSexpVersion": type_payload is not None
            and _read(type_payload, "modelSexpVersion") is not None,
        }
        missing_variables.update(field for field, present in checks.items() if not present)
    report["missing_variable_fields"] = sorted(missing_variables)
    return report


async def _raw_goal_payload(server: Any, expression: str, replay: tuple[str, ...]) -> dict[str, Any]:
    """Drive the server and return its raw ``goal.tactic`` wire payload.

    The pinned PyPantograph client strips the v2 model-S-expression fields
    when it parses responses into ``GoalState`` objects, so capability checks
    and strict conversion must operate on the raw payload obtained through
    ``run_async``. Each replay tactic advances the state first.
    """
    run_async = getattr(server, "run_async", None)
    if not callable(run_async):
        raise ValueError(
            "server does not expose raw payload access (run_async); "
            "the pinned client parses and strips v2 fields from GoalState."
        )
    state_id = (await run_async("goal.start", {"expr": expression})).get("stateId")
    if not isinstance(state_id, int):
        raise ValueError("goal.start did not return a numeric stateId.")
    raw: dict[str, Any] = {}
    for tactic in replay:
        raw = dict(await run_async("goal.tactic", {"stateId": state_id, "tactic": tactic}))
        if raw.get("error") or raw.get("tacticErrors"):
            raise ValueError(f"replay tactic {tactic!r} failed: {raw}")
        next_state_id = raw.get("nextStateId")
        # Keep advancing from freshly minted states; replay tactic order matters.
        state_id = next_state_id if isinstance(next_state_id, int) else state_id
    if not raw:
        raise ValueError("replay produced no goal payload; extend the replay tactics.")
    return raw


def _hypothesis_binder_name(hypothesis: str) -> str:
    return hypothesis.split(":", 1)[0].strip()


async def structured_goal_from_text(
    server: Any,
    expression: str,
    hypotheses: list[str] | None = None,
) -> Goal:
    """Materialize a structured ``Goal`` for arbitrary goal text via raw Pantograph.

    Both standalone prediction and proof search need the structured
    ``lean-model-sexp-v2`` state, and neither can get it from the parsed
    ``GoalState`` (the pinned client strips those fields) nor from
    ``goal.start`` (which returns no goals payload at all). This drives the
    raw wire protocol instead: the goal is started with its hypotheses
    universally quantified, one ``intro`` replay binds them back — reproducing
    exactly the live state the caller describes — and hypothesis-free goals
    materialize their payload through a no-op ``skip`` replay.

    Servers without ``run_async`` fail closed with an actionable message.
    """
    run_async = getattr(server, "run_async", None)
    if not callable(run_async):
        raise RuntimeError(
            "server does not expose raw payload access (run_async); "
            "the pinned client parses and strips v2 fields from GoalState."
        )
    start_expression = expression
    replay: tuple[str, ...] = ()
    if hypotheses:
        for hypothesis in reversed(hypotheses):
            start_expression = f"∀ ({hypothesis}), {start_expression}"
        names = [_hypothesis_binder_name(hypothesis) for hypothesis in hypotheses]
        replay = (f"intro {' '.join(names)}",)
    else:
        # goal.start alone returns no goals payload; a no-op replay
        # materializes the goal without changing the state.
        replay = ("skip",)
    raw = await _raw_goal_payload(server, start_expression, replay)
    goals = pantograph_state_to_goals(raw)
    if not goals:
        raise ValueError("Live state carries no goals; cannot build a structured Goal.")
    return goals[0]


async def probe_model_state(
    server: Any,
    expression: str = "∀ (p : Prop), p → p",
    *,
    replay: tuple[str, ...] = ("intro p",),
) -> None:
    """Verify that the connected server returns the fields required by v2."""
    raw = await _raw_goal_payload(server, expression, replay)
    pantograph_state_to_goals(raw)


@dataclass(frozen=True)
class CapabilityProbe:
    """One live state exercising a v2 payload capability.

    ``expression`` starts the state; the ``replay`` tactics drive it through
    raw ``goal.tactic`` calls so the probed payload contains the fields under
    test (``goal.start`` alone returns no goals and the pinned client strips
    v2 fields when it parses responses).
    """

    name: str
    expression: str
    expectation: str
    replay: tuple[str, ...] = ()


# Replays are live-verified against the pinned Pantograph fork
# (gnn-sexpr-v410 branch on jajos12/Pantograph) with Lean v4.10.0-rc1.
DEFAULT_CAPABILITY_PROBES: tuple[CapabilityProbe, ...] = (
    CapabilityProbe("binders", "∀ (a : Nat), a = a", "binders", ("intro a",)),
    CapabilityProbe("shadowing", "∀ (a : Nat), ∀ (a : Nat), a = a", "shadowing", ("intro a", "intro a")),
    CapabilityProbe("local_lets", "∀ (a : Nat), a = a", "local_lets", ("intro a", "let x := 2")),
    # Instance binders cannot be constructed through goal_start alone, so the
    # probe checks that the payload *reports* the isInstance field (and the
    # implicit binder role). Full instance semantics still require a compatible
    # extractor deployment.
    CapabilityProbe("instances", "∀ {a : Nat}, a = a", "instances", ("intro a",)),
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
    shadowed = False
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
    all_names = [
        _read_first(v, "userName", "name")
        for v in variables
        if isinstance(_read_first(v, "userName", "name"), str)
    ]
    for variable in variables:
        if not (
            _read(variable, "isInaccessible") and isinstance(_read(variable, "userName"), str)
        ):
            continue
        # Pantograph renames a shadowed binder to 'a✝' (inaccessible); the
        # underlying user name re-appears on another binder in the context.
        base = _read(variable, "userName").rstrip("✝")
        other_names = [n for other in variables if other is not variable
                       for n in [_read_first(other, "userName", "name")] if n]
        if base and base in other_names:
            shadowed = True
            break
    findings["binders"] = binder_ok
    findings["shadowing"] = shadowed or (
        len(names) != len(set(names)) and len(internal_names) == len(set(internal_names))
    )
    findings["local_lets"] = let_with_value
    findings["instances"] = instance_field_present
    return findings


async def probe_live_capabilities(
    server: Any,
    *,
    probes: tuple[CapabilityProbe, ...] = DEFAULT_CAPABILITY_PROBES,
) -> dict[str, CapabilityProbeResult]:
    """Fail-closed capability report over representative live states.

    Probes run against the raw wire payload (the pinned client strips v2
    fields when it parses responses into GoalState objects), so servers must
    expose raw payload access; the report fails closed with an actionable
    message otherwise.
    """
    results: dict[str, CapabilityProbeResult] = {}
    for probe in probes:
        raw: dict[str, Any] | None = None
        try:
            raw = await _raw_goal_payload(server, probe.expression, probe.replay)
            goals = _read(raw, "goals")
            if not isinstance(goals, list) or not goals:
                raise ValueError("state carries no goals.")
            findings = capability_findings(goals[0])
            # Strict conversion must also succeed: fail closed on omissions.
            pantograph_state_to_goals(raw)
            ok = bool(findings.get(probe.expectation))
            detail = findings if not ok else ""
            results[probe.name] = CapabilityProbeResult(probe.name, ok, str(detail))
        except Exception as exc:  # noqa: BLE001 - every probe failure is data, not a crash.
            detail = f"{type(exc).__name__}: {exc}"
            if raw is not None:
                # Attach the field-level gap inventory when the server did
                # produce a payload; this distinguishes a pinned-rev client
                # (plain sexp, missing variable fields) from a one-field gap.
                try:
                    detail += f" | gap_report={protocol_gap_report(raw)}"
                except Exception:  # noqa: BLE001 - reporting must never crash the probe.
                    pass
            results[probe.name] = CapabilityProbeResult(probe.name, False, detail)
    return results


def all_capabilities_verified(results: Mapping[str, CapabilityProbeResult]) -> bool:
    """True only when every probed capability passed."""
    return bool(results) and all(result.ok for result in results.values())
