"""Check that only invalid client goals become input errors."""

import asyncio
from types import SimpleNamespace

import pytest
from pantograph.server import ServerError

from maths_ai.data_models.proof_components import Goal
from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner


def _expand_with_error(payload: dict, depth: int, goal: Goal = None) -> None:
    reasoner = HybridReasoner.__new__(HybridReasoner)
    reasoner.max_depth = 10
    reasoner.predict_next_tactic = lambda goal: [object()]

    async def fail_start(g):
        raise ServerError(payload)

    reasoner._start_state = fail_start
    node = SimpleNamespace(depth=depth, goal=goal or Goal(expression="test goal"))
    asyncio.run(reasoner._expand(None, node))


@pytest.mark.parametrize("payload", [
    {"error": "main", "desc": "expected '_' or identifier"},
    {"error": "main", "desc": "unexpected end of input"},
    {"error": "main", "desc": "Application type mismatch"},
    {"error": "parse", "desc": "syntax error: unexpected token"},
    {"error": "elab", "desc": "unknown identifier 'x'"},
])
def test_invalid_initial_goal_becomes_input_error(payload):
    with pytest.raises(ValueError, match="Lean parsing error"):
        _expand_with_error(payload, depth=0)


@pytest.mark.parametrize("payload, depth", [
    ({"error": "index", "desc": "Invalid goal index 0"}, 0),
    ({"error": "state", "desc": "Invalid state ID 42"}, 0),
    ({"error": "server failed while starting a goal"}, 0),
    ({"error": "main", "desc": "expected '_' or identifier"}, 1),
    ({"error": "parse", "desc": "syntax error: unexpected token"}, 1),
])
def test_server_and_generated_subgoal_errors_remain_internal(payload, depth):
    with pytest.raises(ServerError):
        _expand_with_error(payload, depth)


def test_initial_hypothesis_restoration_index_error_remains_internal():
    # Simulates goal_tactic_async failing with Invalid goal index during hypothesis intro
    payload = {"error": "index", "desc": "Invalid goal index 0"}
    goal_with_hyps = Goal(expression="P", hypotheses=["h : P"])
    with pytest.raises(ServerError):
        _expand_with_error(payload, depth=0, goal=goal_with_hyps)
