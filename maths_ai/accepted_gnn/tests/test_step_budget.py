"""The accepted search stops on its tactic-step allowance, swallowed or not."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from maths_ai.accepted_gnn.joint_inference import HybridReasoner
from maths_ai.accepted_gnn.step_budget import CountingServer
from maths_ai.hybrid_reasoner.tactic_budget import TacticAllowance, TacticBudgetExhausted


class _Server:
    def __init__(self):
        self.tactics = 0

    async def goal_tactic_async(self, *_args):
        self.tactics += 1

    async def run_async(self, cmd, payload):
        if cmd == "goal.tactic":
            self.tactics += 1
        return {}


class _Stub(HybridReasoner):
    """Expands forever: each expansion sends one intro and two executor tactics."""

    def __init__(self, server, *, swallow: bool):
        self.max_depth, self.max_nodes = 10, 500
        self.server = server
        self.executor = SimpleNamespace(server=server)
        self.swallow = swallow

    async def _expand(self, graph, node):
        await self.server.goal_tactic_async(None, 0, "intro h")
        for _ in range(2):
            try:
                await self.executor.server.run_async("goal.tactic", {})
            except Exception:
                if not self.swallow:
                    raise


@pytest.mark.parametrize("swallow", [True, False])
def test_the_search_stops_when_the_allowance_is_used_up(swallow):
    server = _Server()
    reasoner = _Stub(server, swallow=swallow)
    allowance = TacticAllowance(7)
    graph = asyncio.run(reasoner.prove("p", allowance=allowance))
    assert server.tactics == 7 and allowance.issued == 7 and allowance.refused
    assert graph.termination_reason == "step_budget"
    assert graph.is_exhausted() and not graph.is_solved()
    assert graph.tactic_steps_issued == 7
    # The proxies are removed after the search.
    assert reasoner.server is server and reasoner.executor.server is server


def test_the_proxy_counts_only_tactic_commands():
    server = _Server()
    counted = CountingServer(server, TacticAllowance(1))
    asyncio.run(counted.run_async("goal.start", {}))
    asyncio.run(counted.run_async("goal.tactic", {}))
    with pytest.raises(TacticBudgetExhausted):
        asyncio.run(counted.goal_tactic_async(None, 0, "rfl"))
    assert server.tactics == 1
