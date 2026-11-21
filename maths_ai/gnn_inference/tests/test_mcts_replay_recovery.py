"""Recovery after incompatible candidates and cleanup of closed sources."""

import asyncio
from unittest.mock import patch

import pytest
from pantograph_async import CertificationResult

from maths_ai.gnn_inference.tests.snapshot_fake import FakeSnapshotScope, Spec, Step
from maths_ai.gnn_inference.tests.test_retained_search import (
    SEED, _Client, _Scripted, _Widening, _admit, _commands,
)
from maths_ai.hybrid_reasoner import joint_inference
from maths_ai.hybrid_reasoner.execution_limits import RetentionLimits
from maths_ai.hybrid_reasoner.hypergraph import (
    NodeClosureReason, SearchEndReason, UnsupportedObservation,
)
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor


@pytest.mark.parametrize("internal", [False, True])
def test_an_uncertified_candidate_does_not_close_widening(internal):
    client = _Client(_commands({
        ("root", "enter"): Step(children=(Spec("P"),)),
        ("P", "bad"): Step(),
        ("P", "good"): Step(),
    }), Spec("root" if internal else "P"))
    reasoner = _Widening(
        {"root": ((("enter", .6), ("missing", .4)),),
         "P": ((("bad", 1.),), (("good", 1.),))},
        executor=PantographExecutor(client), selection_policy="puct",
        num_simulations=40, sim_batch_size=1, top_k_tactics=2,
        widening_initial_actions=2, widening_max_actions=2,
    )
    original = FakeSnapshotScope.certify

    async def certify(scope, state, **kwargs):
        command = next(value for method, value in reversed(scope.calls)
                       if method == "goal.tactic")
        if command == "bad":
            return CertificationResult("rejected", "dependency policy", None)
        return await original(scope, state, **kwargs)

    with patch.object(joint_inference, "materialize_seed", _admit), \
         patch.object(FakeSnapshotScope, "certify", certify):
        graph = asyncio.run(reasoner.prove(SEED))
    assert graph.is_solved()
    assert graph.execution["replay_attempts"] >= 2
    assert graph.execution["replay_conflicts"] >= 1
    source = next(n for n in graph.nodes.values() if n.goal.expression == "P")
    assert source.proposal_rounds == 2
    assert graph.completed_simulations < reasoner.num_simulations
    assert client.scopes[0].live == 0


def test_unsupported_batched_leaf_releases_capacity_before_supported_work():
    client = _Client(_commands({
        ("root", "left"): Step(children=(Spec("P"),)),
        ("root", "right"): Step(children=(Spec("Q"),)),
        ("Q", "trivial"): Step(),
    }), Spec("root"))

    class Reasoner(_Scripted):
        def predict_next_tactics_for_nodes(self, nodes, draws=None):
            if any(node.goal.expression == "P" for node in nodes):
                raise UnsupportedObservation("unsupported P")
            return super().predict_next_tactics_for_nodes(nodes, draws)

    reasoner = Reasoner(
        {"root": [("left", .6), ("right", .4)], "Q": [("trivial", 1.)]},
        executor=PantographExecutor(client), selection_policy="puct",
        num_simulations=20, sim_batch_size=2,
        retention=RetentionLimits(max_retained_snapshots=3),
    )
    with patch.object(joint_inference, "materialize_seed", _admit):
        graph = asyncio.run(reasoner.prove(SEED))
    assert graph.is_solved()
    assert graph.end_reason is SearchEndReason.ROOT_CERTIFIED
    assert graph.execution["peak_snapshots"] <= 3
    unsupported = next(n for n in graph.nodes.values() if n.goal.expression == "P")
    assert unsupported.closure_reason is NodeClosureReason.UNSUPPORTED_OBSERVATION
    assert client.scopes[0].live == 0


def test_root_and_continuation_both_progress_with_single_simulation_batches():
    from maths_ai.hybrid_reasoner.retained_executor import RetainedOutcome

    client = _Client(_commands({
        ("root", "split"): Step(children=(Spec("S"), Spec("T"))),
        ("S", "assign"): Step(changes=((0, "Changed"),)),
        ("T", "finish"): Step(),
        ("T", "adapt"): Step(),
    }), Spec("root"))

    class Executor(PantographExecutor):
        async def apply(self, registry, occurrence, command, **kwargs):
            if occurrence.goal.expression == "Changed" and command == "finish":
                return RetainedOutcome(False, command, error="changed goal")
            return await super().apply(registry, occurrence, command, **kwargs)

    class Reasoner(_Widening):
        starts = []
        observed_anchor = False

        def _select_partial_hypertree(self, graph, start=None):
            simulation = super()._select_partial_hypertree(graph, start)
            if self._continuations and simulation is not None:
                self.starts.append(start)
            return simulation

        async def _try_certify(self, graph):
            for node_id, continuation in self._continuations.items():
                node = graph.nodes[node_id]
                if node.exhausted and node.status == joint_inference.NodeStatus.READY:
                    assert node.occurrence.lease.released
                    assert not continuation.occurrence.lease.released
                    assert node.occurrence.lease is not continuation.occurrence.lease
                    self.observed_anchor = True
            await super()._try_certify(graph)

    reasoner = Reasoner(
        {"root": ((("split", 1.),),), "S": ((("assign", 1.),),),
         "T": ((("finish", 1.),),),
         "Changed": ((("missing", 1.),), (("adapt", 1.),))},
        executor=Executor(client), selection_policy="puct", num_simulations=80,
        sim_batch_size=1, top_k_tactics=1, widening_initial_actions=1,
        widening_max_actions=2,
    )
    with patch.object(joint_inference, "materialize_seed", _admit):
        graph = asyncio.run(reasoner.prove(SEED))
    assert graph.is_solved()
    assert 0 in reasoner.starts
    assert any(start != 0 for start in reasoner.starts)
    assert reasoner.observed_anchor
    continuation = next(n for n in graph.nodes.values() if n.continuation_of is not None)
    assert continuation.proposal_rounds == 2
    assert all(edge.visit_stats.virtual_loss == 0 for edge in graph.edges.values())
    assert client.scopes[0].live == 0


def test_closing_one_unsupported_sibling_keeps_the_shared_allocation_usable():
    client = _Client(_commands({
        ("root", "split"): Step(children=(Spec("P"), Spec("Q"))),
        ("Q", "trivial"): Step(),
    }), Spec("root"))

    class Reasoner(_Scripted):
        def predict_next_tactics_for_nodes(self, nodes, draws=None):
            if any(node.goal.expression == "P" for node in nodes):
                raise UnsupportedObservation("unsupported P")
            return super().predict_next_tactics_for_nodes(nodes, draws)

    reasoner = Reasoner(
        {"root": [("split", 1.)], "Q": [("trivial", 1.)]},
        executor=PantographExecutor(client), selection_policy="puct",
        num_simulations=12, sim_batch_size=2,
        retention=RetentionLimits(max_retained_snapshots=3),
    )
    with patch.object(joint_inference, "materialize_seed", _admit):
        graph = asyncio.run(reasoner.prove(SEED))
    assert any(edge.tactic.tactic_name == "trivial" for edge in graph.edges.values())
    assert graph.execution["peak_snapshots"] <= 3
    assert graph.end_reason is not SearchEndReason.RESOURCE_LIMIT
    assert client.scopes[0].live == 0


@pytest.mark.parametrize("ready", [False, True])
def test_closed_continuation_releases_graph_lease_and_retires_unused_anchor(ready):
    from dataclasses import replace
    from types import SimpleNamespace
    from maths_ai.gnn_inference.tests.test_execution_registry import ROOT, RULES, admitted
    from maths_ai.hybrid_reasoner.execution_registry import LeaseKind, SnapshotRegistry
    from maths_ai.hybrid_reasoner.hypergraph import NodeStatus, ProofHypergraph

    async def run():
        scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
        registry = SnapshotRegistry(scope, retire_session=scope.aclose)
        root = await admitted(scope, registry)
        anchor = replace(root, lease=registry.share(root.lease, LeaseKind.REPLAY, "resume"))
        graph_occurrence = replace(root, lease=registry.share(root.lease, LeaseKind.GRAPH, "propose"))
        graph = ProofHypergraph(root.goal)
        node = graph.add_continuation(graph_occurrence, 0)
        graph.mark_node_exhausted(node.id, reason=NodeClosureReason.UNSUPPORTED_OBSERVATION)
        if ready:
            node.status = NodeStatus.READY
        reasoner = _Scripted({}, executor=PantographExecutor(object()))
        reasoner._registry = registry
        reasoner._continuations = {node.id: SimpleNamespace(occurrence=anchor)}
        await reasoner._release_closed_sources(graph)
        assert graph_occurrence.lease.released
        assert anchor.lease.released is not ready
        assert root.is_retained
        if ready:
            await reasoner._release_closed_sources(graph, attempted_continuation=node.id)
            assert anchor.lease.released
        assert not reasoner._continuations
        await registry.close()
        assert scope.live == 0

    asyncio.run(run())


@pytest.mark.pantograph_live
def test_live_conflicting_candidate_allows_a_later_root_action(live_env):
    from maths_ai.hybrid_reasoner.mcts_diagnostics import scripted_action
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
    from maths_ai.hybrid_reasoner.proof_replay import reverify
    from maths_ai.lean_replay import seed_from_text

    class Reasoner(_Scripted):
        after_conflict = 0

        def _proposal_source_is_finite(self):
            return False

        def predict_next_tactics_for_nodes(self, nodes, draws=None):
            proposals = []
            for node in nodes:
                text = node.goal.expression
                if node.id == 0:
                    conflicted = self._graph.execution.get("last_replay") == "conflict"
                    self.after_conflict += int(conflicted)
                    command = "right" if conflicted else "left"
                elif "∃" in text:
                    command = "refine ⟨?_, ?_, ?_⟩"
                elif text == "ℕ":
                    command = "exact 0"
                elif text.endswith("= 0"):
                    command = "exact (rfl : 0 = 0)"
                elif text.endswith("= 1"):
                    command = "exact (rfl : 1 = 1)"
                else:
                    command = "trivial"
                proposals.append([scripted_action(command, 1.)])
            return proposals

    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            reasoner = Reasoner(
                {}, executor=PantographExecutor(client), selection_policy="puct",
                num_simulations=50, sim_batch_size=1, top_k_tactics=1, max_nodes=100,
            )
            graph = await reasoner.prove(
                seed_from_text(goal="(∃ n : Nat, n = 0 ∧ n = 1) ∨ True")
            )
            assert graph.is_solved(), graph.execution
            assert reasoner.after_conflict > 0
            assert graph.proof_trace()["command"] == "right"
            assert graph.execution["replay_conflicts"] >= 1
            assert graph.execution["intermediate_reconstructions"] == 0
            assert (await client.stat()).n_goals == 0
        async with await create_model_sexpr_client(live_env) as fresh:
            async with fresh.state_scope() as scope:
                assert (await reverify(scope, graph.artifact, retire_session=fresh.aclose)).is_certified
            assert (await fresh.stat()).n_goals == 0

    asyncio.run(run())


@pytest.mark.pantograph_live
def test_live_unsupported_branch_does_not_consume_the_other_branch_capacity(live_env):
    from maths_ai.hybrid_reasoner.mcts_diagnostics import scripted_action
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
    from maths_ai.lean_replay import seed_from_text

    class Reasoner(_Scripted):
        def predict_next_tactics_for_nodes(self, nodes, draws=None):
            if any(node.id == 1 for node in nodes):
                raise UnsupportedObservation("controlled unsupported leaf")
            return [
                [scripted_action("left", .6), scripted_action("right", .4)]
                if node.id == 0 else [scripted_action("trivial", 1.)]
                for node in nodes
            ]

    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            reasoner = Reasoner(
                {}, executor=PantographExecutor(client), selection_policy="puct",
                num_simulations=20, sim_batch_size=2,
                retention=RetentionLimits(max_retained_snapshots=3),
            )
            graph = await reasoner.prove(seed_from_text(goal="True ∨ True"))
            assert graph.is_solved(), graph.execution
            assert graph.execution["peak_snapshots"] <= 3
            assert graph.execution["unsupported_observations"] == 1
            assert (await client.stat()).n_goals == 0

    asyncio.run(run())


@pytest.mark.parametrize("search", [{}, {"selection_policy": "puct", "num_simulations": 8}])
def test_a_round_retains_its_input_for_every_candidate(search):
    client = _Client(_commands({
        ("root", "enter"): Step(children=(Spec("P"),)),
        **{("P", name): Step() for name in ("first", "second", "third")},
    }), Spec("root"))
    reasoner = _Scripted(
        {"root": [("enter", 1.)],
         "P": [("first", .4), ("second", .35), ("third", .25)]},
        executor=PantographExecutor(client), top_k_tactics=3, **search,
    )
    with patch.object(joint_inference, "materialize_seed", _admit):
        graph = asyncio.run(reasoner.prove(SEED))
    assert graph.is_solved()
    node = next(node for node in graph.nodes.values() if node.goal.expression == "P")
    assert len(node.outgoing_edge_ids) == 3
    assert graph.execution["original_tactic_calls"] == 4
    assert client.scopes[0].live == 0
