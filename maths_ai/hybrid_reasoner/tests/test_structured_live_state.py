"""Structured live-state wiring: raw Pantograph extraction and transitions."""

import asyncio
from types import SimpleNamespace

import pytest

from maths_ai.data_models.proof_components import (
    Goal,
    GoalLocal,
    TacticCandidate,
)
from maths_ai.hybrid_reasoner.joint_inference import (
    PantographExecutor,
    _legacy_goals_from_raw_payload,
    _sanitize_inaccessible_names,
)
from maths_ai.hybrid_reasoner.pantograph_adapter import structured_goal_from_text


def _goal_payload(sexp: str = "(:app (:c Eq) (:fv FV0) (:fv FV0))") -> dict:
    return {
        "target": {"pp": "p = p", "modelSexp": sexp, "modelSexpVersion": 1},
        "variables": [
            {
                "userName": "p",
                "name": "p",
                "contextIndex": 0,
                "binderRole": ":explicit",
                "isInstance": False,
                "isLet": False,
                "type": {"pp": "Nat", "modelSexp": "(:c Nat)", "modelSexpVersion": 1},
            }
        ],
    }


class _FakeServer:
    """Raw wire-protocol fake: records goal.start/goal.tactic commands."""

    def __init__(self, goals_payload: dict | None = None, fail_without_raw: bool = False):
        self.commands: list[tuple[str, dict]] = []
        self._goals_payload = goals_payload
        self._fail_without_raw = fail_without_raw

    async def run_async(self, cmd: str, payload: dict) -> dict:
        self.commands.append((cmd, payload))
        if cmd == "goal.start":
            return {"stateId": 1}
        if cmd == "goal.tactic":
            if self._fail_without_raw:
                return {"nextStateId": 2, "messages": []}
            return {"nextStateId": 2, **(self._goals_payload or {"goals": []})}
        raise AssertionError(f"unexpected command {cmd}")


def test_structured_goal_from_text_fails_closed_without_raw_access() -> None:
    server = SimpleNamespace()  # no run_async attribute
    with pytest.raises(RuntimeError, match="raw payload access"):
        asyncio.run(structured_goal_from_text(server, "p = p", None))


def test_structured_goal_from_text_quantifies_and_intros_hypotheses() -> None:
    server = _FakeServer({"goals": [_goal_payload()]})
    goal = asyncio.run(structured_goal_from_text(server, "p = p", ["p : Nat"]))

    assert server.commands[0] == ("goal.start", {"expr": "∀ (p : Nat), p = p"})
    assert server.commands[1][1]["tactic"] == "intro p"
    assert goal.expression == "p = p"
    assert goal.has_model_state
    assert goal.goal_model_sexp == "(:app (:c Eq) (:fv FV0) (:fv FV0))"
    assert goal.locals[0].context_index == 0
    assert goal.locals[0].user_name == "p"


def test_structured_goal_from_text_skips_hypothesis_free_goals() -> None:
    payload = {
        "goals": [
            {
                "target": {"pp": "1 = 1", "modelSexp": "(:lit 1)", "modelSexpVersion": 1},
                "variables": [],
            }
        ]
    }
    server = _FakeServer(payload)
    goal = asyncio.run(structured_goal_from_text(server, "1 = 1", None))

    # goal.start alone returns no goals payload; the no-op replay materializes
    # it without changing the state.
    assert server.commands[1][1]["tactic"] == "skip"
    assert goal.expression == "1 = 1"
    assert goal.locals == []


def test_structured_goal_from_text_fails_closed_on_stripped_fields() -> None:
    server = _FakeServer({"goals": [{"target": {"pp": "p = p"}, "variables": []}]})
    with pytest.raises(ValueError, match="modelSexp"):
        asyncio.run(structured_goal_from_text(server, "p = p", None))


def test_executor_raw_payload_preserves_structured_metadata() -> None:
    server = _FakeServer({"goals": [_goal_payload()]})
    executor = PantographExecutor(server=server)
    state = SimpleNamespace(state_id=7)
    tactic = TacticCandidate(tactic_name="exact", arguments=["h"], probability=0.9)

    outcome = asyncio.run(executor.apply(server, state, tactic))

    assert outcome.success is True
    assert server.commands[0] == ("goal.tactic", {"stateId": 7, "tactic": "exact h"})
    assert outcome.subgoals[0].goal_model_sexp == "(:app (:c Eq) (:fv FV0) (:fv FV0))"
    assert outcome.subgoals[0].locals[0].user_name == "p"


def test_executor_raw_payload_reports_tactic_failures() -> None:
    server = _FakeServer()
    executor = PantographExecutor(server=server)
    state = SimpleNamespace(state_id=7)
    tactic = TacticCandidate(tactic_name="exact", arguments=["nope"], probability=0.9)

    async def run_async(cmd: str, payload: dict) -> dict:
        return {"parseError": "unknown identifier 'nope'"}

    server.run_async = run_async
    outcome = asyncio.run(executor.apply(server, state, tactic))
    assert outcome.success is False
    assert "nope" in (outcome.error or "")


@pytest.mark.parametrize("flag", ["hasSorry", "hasUnsafe"])
def test_executor_rejects_flagged_proof_even_when_no_goals_remain(flag: str) -> None:
    server = _FakeServer()

    async def run_async(cmd: str, payload: dict) -> dict:
        return {"nextStateId": 2, "goals": [], flag: True}

    server.run_async = run_async
    outcome = asyncio.run(PantographExecutor(server).apply(
        server, SimpleNamespace(state_id=1),
        TacticCandidate(tactic_name="exact", arguments=["h"], probability=1.0),
    ))

    assert not outcome.success
    assert outcome.subgoals == []
    assert "admitted or unsafe" in outcome.error


def test_executor_accepts_unflagged_closed_proof() -> None:
    server = _FakeServer()

    async def run_async(cmd: str, payload: dict) -> dict:
        return {"nextStateId": 2, "goals": [], "hasSorry": False, "hasUnsafe": False}

    server.run_async = run_async
    outcome = asyncio.run(PantographExecutor(server).apply(
        server, SimpleNamespace(state_id=1),
        TacticCandidate(tactic_name="rfl", arguments=[], probability=1.0),
    ))

    assert outcome.success
    assert outcome.subgoals == []


def test_legacy_goals_from_raw_payload_keeps_search_alive() -> None:
    raw = {
        "goals": [
            {
                "target": {"pp": "q ∨ p"},
                "variables": [
                    {"userName": "h", "type": {"pp": "p ∨ q"}},
                ],
            }
        ]
    }
    goals = _legacy_goals_from_raw_payload(raw)
    assert goals[0].expression == "q ∨ p"
    assert goals[0].hypotheses == ["h : p ∨ q"]
    assert not goals[0].has_model_state


def test_sanitize_inaccessible_names_preserves_structured_state() -> None:
    goal = Goal(
        expression="p✝ = p✝",
        goal_model_sexp="(:app (:c Eq) (:fv FV0) (:fv FV0))",
        locals=[
            GoalLocal(
                user_name="p✝",
                context_index=0,
                binder_role="explicit",
                type_pp="Nat",
                type_model_sexp="(:c Nat)",
            )
        ],
        model_sexp_version=1,
    )

    sanitized = _sanitize_inaccessible_names(goal)

    # The structured lean-model-sexp-v2 state survives sanitization, with the
    # inaccessible display name renamed consistently in expression,
    # hypotheses, and locals.
    assert sanitized.has_model_state
    assert sanitized.goal_model_sexp == goal.goal_model_sexp
    assert "✝" not in sanitized.expression
    assert sanitized.locals[0].user_name == sanitized.expression.split(" ")[0]
