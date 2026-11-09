"""Check that only invalid client goals become input errors."""

import asyncio
from types import SimpleNamespace

import pytest
from pantograph.server import ServerError

from maths_ai.data_models.proof_components import Goal
from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner


def _expand_with_error(message: str, depth: int) -> None:
    reasoner = HybridReasoner.__new__(HybridReasoner)
    reasoner.max_depth = 10
    reasoner.predict_next_tactic = lambda goal: [object()]

    async def fail_start(goal):
        raise ServerError({"error": message})

    reasoner._start_state = fail_start
    node = SimpleNamespace(depth=depth, goal=Goal(expression="test goal"))
    asyncio.run(reasoner._expand(None, node))


def test_invalid_initial_goal_becomes_input_error():
    with pytest.raises(ValueError, match="unexpected token"):
        _expand_with_error("parse error: unexpected token", depth=0)


@pytest.mark.parametrize("message, depth", [
    ("server failed while starting a goal", 0),
    ("parse error: unexpected token", 1),
])
def test_server_and_generated_subgoal_errors_remain_internal(message, depth):
    with pytest.raises(ServerError):
        _expand_with_error(message, depth)
