"""Replay bookkeeping against scripted snapshots: leases, budgets and conflicts."""

from __future__ import annotations

import asyncio
import unittest

from pantograph_async import CertificationResult

from maths_ai.hybrid_reasoner.proof_replay import (
    ReplayCoordinator,
    ReplayLimits,
    ReplayStatus,
)
from maths_ai.hybrid_reasoner.execution_registry import SnapshotRegistry
from maths_ai.gnn_inference.tests.snapshot_explorer import Explorer
from maths_ai.gnn_inference.tests.snapshot_fake import FakeSnapshotScope, Spec, Step, table
from maths_ai.gnn_inference.tests.test_execution_registry import ROOT, admitted

RULES = table(
    constructor=Step(children=(Spec("p"), Spec("q"))),
    trivial=Step(),
    exact_hp=Step(),
)


def run(body):
    return asyncio.run(body())


async def explorer_for(rules, **scope_kwargs):
    scope = FakeSnapshotScope(rules, roots={"root": ROOT}, **scope_kwargs)
    registry = SnapshotRegistry(scope)
    return scope, Explorer(registry, await admitted(scope, registry))


class ReplayBookkeepingTests(unittest.TestCase):
    def test_a_certified_replay_releases_every_replay_state(self):
        async def body():
            scope, explorer = await explorer_for(RULES)
            left, right = await explorer.expand(0, "constructor")
            await explorer.expand(right, "trivial")
            await explorer.expand(left, "trivial")
            held = explorer.registry.live_snapshots
            result = await ReplayCoordinator(explorer.registry).combine(explorer, explorer.root, 0)
            self.assertEqual(result.status, ReplayStatus.CERTIFIED)
            self.assertEqual(result.steps, 3)
            self.assertEqual(explorer.registry.live_snapshots, held)
            self.assertEqual(scope.count("goal.start"), 1)
            await explorer.registry.close()
            self.assertEqual(scope.live, 0)

        run(body)

    def test_the_step_budget_stops_with_unknown_evidence(self):
        async def body():
            _scope, explorer = await explorer_for(RULES)
            left, right = await explorer.expand(0, "constructor")
            await explorer.expand(left, "trivial")
            await explorer.expand(right, "trivial")
            held = explorer.registry.live_snapshots
            coordinator = ReplayCoordinator(explorer.registry, limits=ReplayLimits(max_tactic_steps=2))
            result = await coordinator.combine(explorer, explorer.root, 0)
            self.assertEqual(result.status, ReplayStatus.BUDGET)
            self.assertIsNone(result.certificate)
            self.assertEqual(result.continuations, [])
            self.assertEqual(explorer.registry.live_snapshots, held)

        run(body)

    def test_a_shape_change_is_a_conflict_and_failed_prefixes_are_not_retried(self):
        calls = {"constructor": 0}

        def rules(_spec, command):
            if command == "constructor":
                calls["constructor"] += 1
                # Explored with two children; any replay produces three.
                count = 2 if calls["constructor"] == 1 else 3
                return Step(children=tuple(Spec(f"g{i}") for i in range(count)))
            return Step() if command == "trivial" else None

        async def body():
            scope, explorer = await explorer_for(rules)
            children = await explorer.expand(0, "constructor")
            for child in children:
                await explorer.expand(child, "trivial")
            coordinator = ReplayCoordinator(explorer.registry)
            first = await coordinator.combine(explorer, explorer.root, 0)
            self.assertEqual(first.status, ReplayStatus.CONFLICT)
            self.assertEqual([c.kind for c in first.conflicts], ["shape"])
            tactics = scope.count("goal.tactic")
            again = await coordinator.combine(explorer, explorer.root, 0)
            self.assertEqual(again.status, ReplayStatus.CONFLICT)
            self.assertEqual(scope.count("goal.tactic"), tactics)

        run(body)

    def test_an_argument_without_a_declaration_is_a_binding_conflict(self):
        async def body():
            _scope, explorer = await explorer_for(RULES)
            await explorer.expand(0, "exact", "hp")
            step = explorer.nodes[0].edges[0].step
            # Forget the explored context's key for hp: the declaration is unknown.
            hp = step.arguments[0].internal_name
            explorer.nodes[0].occurrence.declarations.pop(hp)
            result = await ReplayCoordinator(explorer.registry).combine(explorer, explorer.root, 0)
            self.assertEqual(result.status, ReplayStatus.CONFLICT)
            self.assertEqual([c.kind for c in result.conflicts], ["binding"])
            self.assertEqual(result.steps, 0)

        run(body)

    def test_an_inconclusive_kernel_check_is_not_a_conflict(self):
        async def body():
            scope, explorer = await explorer_for(RULES)
            await explorer.expand(0, "trivial")

            async def inconclusive(state, *, max_heartbeats=None):
                return CertificationResult("inconclusive", "Kernel heartbeat budget exhausted", None)

            scope.certify = inconclusive
            result = await ReplayCoordinator(explorer.registry).combine(explorer, explorer.root, 0)
            self.assertEqual(result.status, ReplayStatus.INCONCLUSIVE)
            self.assertEqual(result.conflicts, [])

        run(body)

    def test_an_incomplete_assignment_is_never_ready(self):
        async def body():
            _scope, explorer = await explorer_for(table(close_partly=Step(leave_holes=True)))
            await explorer.expand(0, "close partly")
            self.assertEqual(explorer.alternatives(0), [])

        run(body)

    def test_limits_are_validated(self):
        with self.assertRaises(ValueError):
            ReplayLimits(max_tactic_steps=0)
        with self.assertRaises(ValueError):
            ReplayLimits(max_combinations=True)


if __name__ == "__main__":
    unittest.main()


class SharedBudgetTests(unittest.TestCase):
    def test_the_step_budget_covers_every_replay_of_a_search(self):
        async def body():
            _scope, explorer = await explorer_for(RULES)
            await explorer.expand(0, "trivial")
            coordinator = ReplayCoordinator(explorer.registry, limits=ReplayLimits(max_tactic_steps=1))
            first = await coordinator.combine(explorer, explorer.root, 0)
            self.assertEqual(first.status, ReplayStatus.CERTIFIED)
            again = await coordinator.combine(explorer, explorer.root, 0)
            self.assertEqual(again.status, ReplayStatus.BUDGET)
            self.assertEqual(coordinator.total_steps, 1)

        run(body)
