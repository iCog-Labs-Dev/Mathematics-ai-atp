"""Regressions for conditional candidates and resource checks during replay."""

import asyncio
from unittest.mock import patch

import pytest

from maths_ai.gnn_inference.tests.snapshot_fake import Spec, Step
from maths_ai.gnn_inference.tests.test_retained_search import (
    SEED, _Client, _Scripted, _admit, _commands,
)
from maths_ai.hybrid_reasoner import joint_inference
from maths_ai.hybrid_reasoner.execution_limits import RetentionLimits
from maths_ai.hybrid_reasoner.hypergraph import SearchEndReason
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor
from maths_ai.hybrid_reasoner.retained_executor import RetainedOutcome


def conjunction(*, limit=4, memory=None):
    rules = _commands({
        ("P ∧ Q", "constructor"): Step(children=(Spec("P"), Spec("Q"))),
        ("P", "trivial"): Step(),
        ("Q", "trivial"): Step(),
    })
    client = _Client(rules, Spec("P ∧ Q"))
    reasoner = _Scripted(
        {"P ∧ Q": [("constructor", .9)], "P": [("trivial", .8)], "Q": [("trivial", .7)]},
        executor=PantographExecutor(client),
        retention=RetentionLimits(max_retained_snapshots=limit),
        collection_memory=memory,
    )
    with patch.object(joint_inference, "materialize_seed", _admit):
        graph = asyncio.run(reasoner.prove(SEED))
    return graph, client.scopes[0]


def test_replay_never_exceeds_the_retained_allocation_limit():
    graph, scope = conjunction(limit=3)
    assert graph.execution["peak_snapshots"] <= 3
    assert graph.is_solved()
    assert scope.live == 0


def test_replay_certifies_with_sufficient_small_capacity():
    graph, scope = conjunction(limit=4)
    assert graph.is_solved()
    assert graph.execution["peak_snapshots"] <= 4
    assert scope.live == 0


def test_memory_is_checked_during_replay():
    class Monitor:
        peak_growth = 0

        def exceeded(self):
            # Three exploration requests precede the first replay request.
            return self.calls > 3

        calls = 0

    monitor = Monitor()
    original = PantographExecutor.apply

    async def counted(self, *args, **kwargs):
        monitor.calls += 1
        return await original(self, *args, **kwargs)

    with patch.object(PantographExecutor, "apply", counted):
        graph, scope = conjunction(memory=monitor)
    assert graph.search_end_reason == SearchEndReason.RESOURCE_LIMIT
    assert not graph.is_solved()
    assert scope.live == 0


def test_unknown_holes_do_not_make_a_candidate_or_a_proof():
    client = _Client(_commands({("P", "partial"): Step(leave_holes=True)}), Spec("P"))
    reasoner = _Scripted({"P": [("partial", 1.0)]}, executor=PantographExecutor(client))
    with patch.object(joint_inference, "materialize_seed", _admit):
        graph = asyncio.run(reasoner.prove(SEED))
    assert not graph.has_candidate()
    assert graph.execution["replay_attempts"] == 0
    assert not graph.is_solved()


@pytest.mark.parametrize("persistent", [True, False])
def test_memory_pressure_is_rechecked_after_pruning(persistent):
    client = _Client(_commands({
        ("P", "trivial"): Step(),
        ("P", "branch"): Step(children=(Spec("Unexplored"),)),
    }), Spec("P"))

    class Monitor:
        peak_growth = 0
        checks = []

        def exceeded(self):
            high = (reasoner._counters.get("replay_attempts", 0) > 0
                    and (persistent or not reasoner._counters.get("pruned_nodes", 0)))
            self.checks.append(high)
            return high

    monitor = Monitor()
    reasoner = _Scripted(
        {"P": [("trivial", .9), ("branch", .1)]},
        executor=PantographExecutor(client), collection_memory=monitor,
    )
    with patch.object(joint_inference, "materialize_seed", _admit):
        graph = asyncio.run(reasoner.prove(SEED))
    assert graph.execution["pruned_nodes"] == 1
    if persistent:
        assert monitor.checks.count(True) >= 2
        assert graph.search_end_reason == SearchEndReason.RESOURCE_LIMIT
        assert not graph.is_solved()
    else:
        assert monitor.checks.count(True) == 1
        assert monitor.checks[-1] is False
        assert graph.is_solved()
    assert client.scopes[0].live == 0


@pytest.mark.parametrize("continuation", [False, True])
def test_replay_defers_siblings_assigned_through_a_later_obligation(continuation):
    rules = _commands({
        ("root", "split"): Step(children=(Spec("h"), Spec("v"), Spec("w"))),
        ("h", "assign"): Step(
            closes=(0,), depends=("w",), sibling_depends=((0, ("w",)),),
            changes=((1, "Changed"),) if continuation else (),
        ),
        ("v", "finish"): Step(),
        ("w", "finish"): Step(),
        ("w", "adapt"): Step(),
    })

    class Executor(PantographExecutor):
        async def apply(self, registry, occurrence, command, **kwargs):
            if occurrence.goal.expression == "Changed" and command == "finish":
                return RetainedOutcome(False, command, error="changed observation")
            return await super().apply(registry, occurrence, command, **kwargs)

    client = _Client(rules, Spec("root"))
    reasoner = _Scripted(
        {"root": [("split", 1.)], "h": [("assign", 1.)],
         "v": [("finish", 1.)], "w": [("finish", 1.)],
         "Changed": [("adapt", 1.)]},
        executor=Executor(client),
    )
    with patch.object(joint_inference, "materialize_seed", _admit):
        graph = asyncio.run(reasoner.prove(SEED))
    assert graph.is_solved()
    # Replay skips the assigned sibling's independent recipe and finishes w.
    assert graph.execution["replay_tactic_calls"] == (4 if continuation else 3)
    scope = client.scopes[0]
    assert any(operation == "goal.inspect" and len(names) == 2
               for operation, names in scope.calls)
    assert scope.live == 0
