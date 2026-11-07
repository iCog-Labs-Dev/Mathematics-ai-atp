from __future__ import annotations

import asyncio
from types import SimpleNamespace

from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor


def test_pantograph_executor_renders_only_resolved_arguments() -> None:
    class FakeServer:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def goal_tactic_async(self, state, goal_index: int, command: str):
            self.commands.append(command)
            return SimpleNamespace(goals=[])

    server = FakeServer()
    executor = PantographExecutor(server)
    state = SimpleNamespace(goals=[SimpleNamespace()])

    outcome = asyncio.run(
        executor.apply(
            server,
            state,
            TacticCandidate(tactic_name="rfl", arguments=[], probability=1.0),
        )
    )

    assert outcome.success is True
    assert server.commands == ["rfl"]
