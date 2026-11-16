"""Count the accepted search's Lean tactic commands against one allowance.

The accepted search sends tactics in two places: the executor (through the
raw ``goal.tactic`` wire call) and ``_start_state``, which re-introduces a
node's hypotheses with ``intro``. Both reach the server, so the allowance is
enforced there: a proxy reserves one step before every ``goal.tactic``. The
executor turns errors into failed tactics, so the search loop also checks
``allowance.refused`` and stops instead of reading a refusal as evidence.
"""

from __future__ import annotations

from typing import Any

from maths_ai.hybrid_reasoner.tactic_budget import TacticAllowance


class CountingServer:
    """A Pantograph server whose tactic commands draw on ``allowance``."""

    def __init__(self, server: Any, allowance: TacticAllowance) -> None:
        self._server = server
        self.allowance = allowance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._server, name)

    async def goal_tactic_async(self, *args, **kwargs):
        self.allowance.reserve()
        return await self._server.goal_tactic_async(*args, **kwargs)

    async def run_async(self, cmd: str, payload: dict, *args, **kwargs):
        if cmd == "goal.tactic":
            self.allowance.reserve()
        return await self._server.run_async(cmd, payload, *args, **kwargs)
