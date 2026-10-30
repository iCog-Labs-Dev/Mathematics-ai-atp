"""MCTS and best-first search over the installed client's scoped execution API.

A recording scoped client stands in for the REPL: it rejects operations on
released states the way the real scope does, and records every allocation and
release, so these tests check ownership as well as search results. The
``pantograph_live`` cases run the same search against the actual executable.
"""

from __future__ import annotations

import asyncio
import itertools
import unittest
from dataclasses import replace
from types import MethodType

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import LocalCandidateProvider
from maths_ai.lean_replay import seed_from_text
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import extract_critic_samples
from maths_ai.gnn_inference.tests.model_helpers import (
    actor_critic,
    pantograph_goal,
    structured_goal,
)
from maths_ai.hybrid_reasoner.hypergraph import (
    BackupSource,
    NodeClosureReason,
    NodeStatus,
    SearchEndReason,
    UnsupportedObservation,
)
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor
from maths_ai.hybrid_reasoner.mcts_diagnostics import (
    verify_learning_round,
    verify_mcts_search,
    verify_shared_device_workers,
    verify_widening,
)
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from pantograph_async import (
    CertificateReceipt,
    CertificationResult,
    ObligationInspection,
    TransitionProvenance,
    PantographGoalRejected,
    PantographProtocolError,
    PantographStateError,
    PantographTacticRejected,
    PantographTransportError,
)
from maths_ai.hybrid_reasoner.proposal import ProposalSpec

SAMPLE = ProposalSpec(mode="sample", seed=0)
GREEDY = ProposalSpec(mode="greedy", seed=0)


def accepted(result):
    """Each accepted edge's first raw action."""
    actions = {}
    for observation in result.observations:
        if observation.edge_id is not None:
            actions.setdefault(observation.edge_id, observation.action)
    return actions


def failed(result):
    return [o for o in result.observations if o.edge_id is None]


TACTIC_VOCAB = {"trivial": 0, "intro": 1, "exact": 2}
ROOT = "p → p"
HYPS = ["p : Prop"]
SEED = seed_from_text(goal=ROOT, hypotheses=HYPS)


class _State:
    _ids = itertools.count()

    def __init__(self, goals, *, expression=None, transition=None, carried=()):
        # A stable identity: id() values are reused once states are freed.
        self.sid = next(self._ids)
        # Each obligation is its own metavariable, so names are unique.
        self.goals = [
            replace(pantograph_goal(goal), goal_name=f"_g{self.sid}.{index}")
            for index, goal in enumerate(goals)
        ] + list(carried)
        self.expression = expression
        self.transition = transition


class _RecordingScope:
    def __init__(self, client):
        self.client = client
        self.live: dict[int, _State] = {}
        self.allocated: dict[int, _State] = {}
        self.is_usable = True

    async def __aenter__(self):
        self.client.scopes.append(self)
        return self

    async def __aexit__(self, *_exc):
        self.client.left_at_exit.append(len(self.live))
        self.live.clear()
        self.is_usable = False

    def _allocate(self, state):
        self.allocated[state.sid] = state
        self.live[state.sid] = state
        self.client.events.append(("allocate", state.sid))
        return state

    async def start_goal(self, expression, *, levels=()):
        self.client.started_levels = tuple(levels)
        if not self.is_usable:
            raise PantographStateError("State operations require an active scope.")
        if expression in self.client.unelaborated:
            raise PantographGoalRejected(f"cannot elaborate {expression}")
        if expression in self.client.protocol_failures:
            raise PantographProtocolError("malformed goal.start response")
        return self._allocate(_State([], expression=expression))

    async def apply_tactic(self, state, tactic, *, goal_index=0):
        if self.live.get(state.sid) is not state:
            raise PantographStateError("Cannot apply a tactic to a released state.")
        if state.expression is not None:
            # Admission of the seed: serialize its canonical goal.
            goal = self.client.goals[state.expression]
            admitted = _State([goal])
            admitted.transition = TransitionProvenance(
                "", (admitted.goals[0].goal_name,), ((),), (), (), (), (), ()
            )
            return self._allocate(admitted)
        selected = state.goals[goal_index]
        target = selected.target.pp
        node_id = self.client.current_node
        if node_id is None:
            # Replay re-runs an explored attempt; Lean answers it the same way.
            self.client.events.append(("replay", state.sid, tactic))
            outcome = self.client.played.get((target, tactic), "reject")
        else:
            self.client.events.append(("apply", node_id, state.sid, tactic))
            await self.client.before_apply(node_id)
            outcome = self.client.outcome_for(node_id, target)
        if isinstance(outcome, BaseException):
            raise outcome
        if node_id is not None:
            self.client.played[(target, tactic)] = outcome
        if outcome == "reject":
            raise PantographTacticRejected(f"{tactic} failed")
        subgoals = [structured_goal(expression, HYPS) for expression in outcome]
        siblings = [goal for goal in state.goals if goal is not selected]
        result = _State(subgoals, carried=siblings)
        children = tuple(goal.goal_name for goal in result.goals[: len(subgoals)])
        result.transition = TransitionProvenance(
            selected.goal_name,
            children,
            tuple(() for _ in children),
            tuple(goal.goal_name for goal in siblings),
            (), (), (), (),
        )
        return self._allocate(result)

    async def inspect(self, state, goals):
        listed = {goal.goal_name: goal for goal in state.goals}
        return tuple(
            ObligationInspection(name, "active", listed[name], (), False)
            if name in listed
            else ObligationInspection(name, "complete", None, (), False)
            for name in goals
        )

    async def certify(self, state, *, max_heartbeats=None):
        if state.goals:
            return CertificationResult("incomplete", f"{len(state.goals)} goals remain", None)
        receipt = CertificateReceipt("root", (), "0" * 16, "1" * 16, "2" * 16, "fake", (), ())
        return CertificationResult("certified", None, receipt)

    async def release(self, state):
        if self.live.pop(state.sid, None) is not None:
            self.client.events.append(("release", state.sid))


class _RecordingClient:
    is_usable = True

    def __init__(self, outcomes=None):
        root = structured_goal(ROOT, HYPS)
        # The seed's replay, and the canonical root's own replay for a later
        # widening round at the root.
        self.goals = {"∀ (p : Prop), p → p": root, root.replay_spec().expression: root}
        self.outcomes = outcomes or {}
        # (goal text, command) -> what exploration observed, for replay.
        self.played: dict[tuple[str, str], object] = {}
        self.unelaborated: set[str] = set()
        self.protocol_failures: set[str] = set()
        self.current_node: int | None = None
        self.events: list[tuple] = []
        self.scopes: list[_RecordingScope] = []
        self.left_at_exit: list[int] = []
        self.gate: asyncio.Event | None = None
        self.entered_apply: asyncio.Event | None = None

    def state_scope(self):
        return _RecordingScope(self)

    def outcome_for(self, node_id, expression):
        """Script by node id first, then by goal text; unscripted goals close."""
        if node_id in self.outcomes:
            return self.outcomes[node_id]
        return self.outcomes.get(expression, [])

    async def before_apply(self, node_id):
        if self.gate is not None:
            self.entered_apply.set()
            await self.gate.wait()

    async def aclose(self):
        self.is_usable = False


class _NodeTrackingReasoner(RLHybridReasoner):
    """Tell the fake which graph node an application belongs to; replay
    applications belong to no node. Goals in ``unobservable`` cannot be read
    by the model."""

    unobservable: frozenset = frozenset()

    async def _execute_and_link(self, graph, node, candidates, *, draws=None):
        self.server.current_node = node.id
        self.executed.append((node.id, len({candidate.command for candidate in candidates})))
        try:
            return await super()._execute_and_link(graph, node, candidates, draws=draws)
        finally:
            self.server.current_node = None

    def predict_next_tactics_for_nodes(self, nodes, draws=None):
        self.proposal_batches.append([node.id for node in nodes])
        if any(node.goal.expression in self.unobservable for node in nodes):
            raise UnsupportedObservation("the vocabulary lacks a structural label")
        return super().predict_next_tactics_for_nodes(nodes, draws)


def _reasoner(client, *, top_k=2, seed=0, **search):
    torch.manual_seed(seed)
    node_vocab = build_vocab([model_goal_to_dag(structured_goal(ROOT, HYPS))])
    model = actor_critic(len(node_vocab), len(TACTIC_VOCAB))
    reasoner = _NodeTrackingReasoner(
        model,
        node_vocab,
        TACTIC_VOCAB,
        executor=PantographExecutor(client),
        candidate_provider=LocalCandidateProvider(),
        top_k_tactics=top_k,
        max_depth=3,
        max_nodes=20,
        use_pln=False,
        **_one_round_capacity(search, top_k),
    )
    reasoner.executed = []
    reasoner.proposal_batches = []
    return reasoner


def _one_round_capacity(search, top_k):
    """Under PUCT, let a node's first round use its whole widening capacity."""
    if search.get("selection_policy") == "puct":
        return {"widening_initial_actions": top_k, **search}
    return search


def _puct(**overrides):
    """PUCT whose widening stays at the first round's capacity for these short
    searches, so they test state ownership rather than widening."""
    settings = dict(
        selection_policy="puct",
        num_simulations=6,
        sim_batch_size=2,
        widening_coefficient=0.5,
    )
    settings.update(overrides)
    return settings


def _assert_scoped_cleanly(test, client):
    """Every allocation was released by the search or its scope."""
    test.assertEqual(len(client.scopes), 1)
    test.assertFalse(client.scopes[0].live)
    test.assertFalse(client.scopes[0].is_usable)


def _assert_alternatives_released_promptly(test, client):
    """A result with no open obligations is released before the next
    application; results with obligations are retained for their nodes."""
    states = {state.sid: state for scope in client.scopes for state in _allocated(scope)}
    live_results: set[int] = set()
    applied = False
    for event in client.events:
        if event[0] == "apply":
            test.assertFalse(live_results, "a closed alternative's result outlived its use")
            applied = True
        elif event[0] == "allocate" and applied:
            if not states[event[1]].goals:
                live_results.add(event[1])
            applied = False
        elif event[0] == "release":
            live_results.discard(event[1])


def _allocated(scope):
    return scope.allocated.values()


class SearchModeOwnershipTests(unittest.TestCase):
    def _assert_owned_execution(self, client, reasoner, result):
        _assert_scoped_cleanly(self, client)
        _assert_alternatives_released_promptly(self, client)
        # The seed is admitted once; no node is rebuilt from its printed goal.
        self.assertEqual(sum(1 for event in client.events if event[0] == "allocate")
                         - sum(1 for event in client.events if event[0] in ("apply", "replay")), 2)
        self.assertEqual(result.graph.execution["intermediate_reconstructions"], 0)
        self.assertFalse(reasoner._proving)
        self.assertIsNone(reasoner._scope)
        self.assertIsNone(reasoner._registry)
        self.assertEqual(reasoner._pending, {})
        # Graph nodes keep canonical goals only.
        for node in result.graph.nodes.values():
            node.goal.require_model_state()

    def test_best_first_uses_scoped_execution(self):
        client = _RecordingClient({0: ["A", "B"]})
        reasoner = _reasoner(client)
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        self.assertTrue(result.graph.is_solved())
        self._assert_owned_execution(client, reasoner, result)
        for edge in result.graph.edges.values():
            self.assertEqual(edge.visit_stats.N, 0)

    def test_puct_uses_scoped_execution(self):
        client = _RecordingClient({0: ["A", "B"]})
        reasoner = _reasoner(client, **_puct())
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        self.assertTrue(result.graph.is_solved())
        self._assert_owned_execution(client, reasoner, result)
        self.assertTrue(any(e.visit_stats.N for e in result.graph.edges.values()))
        for edge in result.graph.edges.values():
            self.assertEqual(edge.visit_stats.virtual_loss, 0)

    def test_multi_leaf_batch_keeps_each_leaf_state_until_all_alternatives_finish(self):
        client = _RecordingClient({0: ["A", "B"], "A": "reject"})
        reasoner = _reasoner(client, top_k=3, **_puct())
        asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))

        # One policy forward proposed several leaves together.
        self.assertTrue(any(len(batch) > 1 for batch in reasoner.proposal_batches))
        applies: dict[int, list[int]] = {}
        for event in client.events:
            if event[0] == "apply":
                applies.setdefault(event[1], []).append(event[2])
        leaves = [node_id for node_id in applies if node_id != 0]
        self.assertTrue(leaves)
        self.assertTrue(any(len(applies[node_id]) > 1 for node_id in leaves))
        for node_id in leaves:
            # Every alternative of a leaf ran against its one retained
            # snapshot, which was not released before the last of them.
            self.assertEqual(len(applies[node_id]), dict(reasoner.executed)[node_id])
            self.assertEqual(len(set(applies[node_id])), 1)
            last_apply = max(
                index
                for index, event in enumerate(client.events)
                if event[0] == "apply" and event[1] == node_id
            )
            release = ("release", applies[node_id][0])
            released = (
                client.events.index(release) if release in client.events else len(client.events)
            )
            self.assertGreater(released, last_apply)
        _assert_scoped_cleanly(self, client)
        _assert_alternatives_released_promptly(self, client)

    def test_identical_goal_text_in_different_nodes_keeps_node_action_alignment(self):
        # Every root alternative yields two copies of the same goal. The first
        # copy executed can be closed; every other copy rejects all tactics.
        client = _RecordingClient({0: ["A", "A"]})
        first_copy: list[int] = []

        def outcome_for(node_id, expression):
            if node_id == 0:
                return ["A", "A"]
            if not first_copy:
                first_copy.append(node_id)
            return [] if node_id == first_copy[0] else "reject"

        client.outcome_for = outcome_for
        reasoner = _reasoner(client, top_k=3, **_puct(num_simulations=4))
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        graph = result.graph

        executed = dict(reasoner.executed)
        copies = [node_id for node_id in executed if node_id != 0]
        self.assertGreater(len(copies), 1)
        fingerprints = {graph.nodes[node_id].goal.state_fingerprint() for node_id in copies}
        self.assertEqual(len(fingerprints), 1)
        solved, rejected = first_copy[0], [n for n in copies if n != first_copy[0]]
        self.assertEqual(graph.nodes[solved].status, NodeStatus.READY)
        for node_id in rejected:
            # Every known alternative was rejected, but a sampled source can
            # still widen, so the copy is not a complete failure.
            self.assertEqual(graph.nodes[node_id].status, NodeStatus.OPEN)
            self.assertFalse(graph.nodes[node_id].exhausted)

        # Accepted actions belong to the solved copy's edges (or the root's),
        # and sampled-action multiplicity is kept per node.
        solved_edges = set(graph.nodes[solved].outgoing_edge_ids)
        root_edges = set(graph.nodes[0].outgoing_edge_ids)
        self.assertLessEqual(result.accepted_edge_ids(), solved_edges | root_edges)
        self.assertEqual(
            sum(
                o.action.multiplicity for o in result.observations if o.edge_id in solved_edges
            ),
            reasoner.top_k_tactics,
        )
        self.assertEqual(
            sum(record.action.multiplicity for record in failed(result)),
            reasoner.top_k_tactics * len(rejected),
        )
        fingerprint = str(reasoner.dag_featurize(graph.nodes[solved].goal)[1].graph_fingerprint)
        for record in failed(result):
            self.assertEqual(record.action.graph_fingerprint, fingerprint)
        _assert_scoped_cleanly(self, client)


class _SkippingReasoner(_NodeTrackingReasoner):
    """Propose nothing for goal ``B``; sample normally for every other node."""

    def predict_next_tactics_for_nodes(self, nodes, draws=None):
        self.leaf_batches.append([node.id for node in nodes])
        if any(node.goal.expression in self.unobservable for node in nodes):
            raise UnsupportedObservation("the vocabulary lacks a structural label")
        kept = [index for index, node in enumerate(nodes) if node.goal.expression != "B"]
        sampled = iter(
            super().predict_next_tactics_for_nodes(
                [nodes[index] for index in kept],
                None if draws is None else [draws[index] for index in kept],
            )
            if kept else ()
        )
        return [[] if node.goal.expression == "B" else next(sampled) for node in nodes]


class BatchCompositionTests(unittest.TestCase):
    def test_mixed_batch_settles_each_node(self):
        # The model cannot read A, B receives no candidates, C closes.
        client = _RecordingClient({0: ["A", "B", "C"]})
        torch.manual_seed(0)
        node_vocab = build_vocab([model_goal_to_dag(structured_goal(ROOT, HYPS))])
        reasoner = _SkippingReasoner(
            actor_critic(len(node_vocab), len(TACTIC_VOCAB)),
            node_vocab,
            TACTIC_VOCAB,
            executor=PantographExecutor(client),
            candidate_provider=LocalCandidateProvider(),
            top_k_tactics=1,
            max_depth=3,
            max_nodes=20,
            use_pln=False,
            **_puct(num_simulations=4, widening_initial_actions=1),
        )
        reasoner.executed, reasoner.proposal_batches, reasoner.leaf_batches = [], [], []
        reasoner.unobservable = frozenset({"A"})
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        graph = result.graph

        a, b, c = (graph.nodes[child] for child in graph.edges[0].child_ids)
        # A's goal failed the batched forward, so B and C were proposed alone.
        self.assertIn([b.id], reasoner.leaf_batches)
        self.assertIn([c.id], reasoner.leaf_batches)
        self.assertEqual(a.closure_reason, NodeClosureReason.UNSUPPORTED_OBSERVATION)
        self.assertNotIn(a.id, dict(reasoner.executed))
        # A sampled round that proposed nothing does not prove the source
        # empty: B stays open for a later round, with no failure recorded.
        self.assertEqual(b.closure_reason, NodeClosureReason.NONE)
        self.assertEqual(b.status, NodeStatus.OPEN)
        self.assertGreaterEqual(b.proposal_rounds, 1)
        self.assertEqual(c.status, NodeStatus.READY)
        # Only executed alternatives settle as evidence: C's accepted edge.
        self.assertEqual(failed(result), [])
        self.assertTrue(result.accepted_edge_ids() <= {0, *c.outgoing_edge_ids})
        _assert_scoped_cleanly(self, client)

    def test_depth_limited_leaves_are_closed_without_execution(self):
        client = _RecordingClient({0: ["A", "B"]})
        # Enough simulations to reach every leaf of every root edge.
        reasoner = _reasoner(client, **_puct(num_simulations=8))
        reasoner.max_depth = 1
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        leaves = [
            node for node in result.graph.nodes.values() if node.id != result.graph.root_id
        ]
        self.assertTrue(leaves)
        for node in leaves:
            self.assertEqual(node.closure_reason, NodeClosureReason.DEPTH_LIMIT)
        # Only the seed was admitted: one start plus its serialization.
        self.assertEqual(
            sum(1 for event in client.events if event[0] == "allocate") - sum(
                1 for event in client.events if event[0] == "apply"
            ),
            2,
        )
        self.assertEqual([node_id for node_id, _ in reasoner.executed], [0])
        self.assertTrue(all(record.goal.expression == ROOT for record in failed(result)))
        _assert_scoped_cleanly(self, client)


class OverlappingSearchTests(unittest.TestCase):
    def test_overlapping_search_is_rejected_before_touching_the_running_search(self):
        client = _RecordingClient({0: []})
        reasoner = _reasoner(client, **_puct())

        async def run():
            client.gate = asyncio.Event()
            client.entered_apply = asyncio.Event()
            first = asyncio.create_task(
                reasoner.prove(SEED, proposal=ProposalSpec(mode="sample", seed=1))
            )
            await client.entered_apply.wait()
            pending, result, proposal = reasoner._pending, reasoner._result, reasoner._proposal
            pending_snapshot = {key: dict(value.draws) for key, value in pending.items()}
            with self.assertRaises(PantographStateError):
                await reasoner.prove(SEED, proposal=SAMPLE)
            self.assertIs(reasoner._pending, pending)
            self.assertEqual(
                {key: dict(value.draws) for key, value in pending.items()},
                pending_snapshot,
            )
            self.assertIs(reasoner._result, result)
            self.assertIs(reasoner._proposal, proposal)
            self.assertEqual(len(client.scopes), 1)
            client.gate.set()
            return await first

        result = asyncio.run(run())
        self.assertTrue(result.graph.is_solved())
        _assert_scoped_cleanly(self, client)


class FailureInjectionTests(unittest.TestCase):
    def _capture_graph(self, reasoner):
        captured = {}
        original = reasoner._prove_mcts

        async def capture(self, graph, deadline):
            captured["graph"] = graph
            return await original(graph, deadline)

        reasoner._prove_mcts = MethodType(capture, reasoner)
        return captured

    def _assert_released(self, reasoner, client, graph):
        for edge in graph.edges.values():
            self.assertEqual(edge.visit_stats.virtual_loss, 0)
        self.assertEqual(reasoner._pending, {})
        self.assertIsNone(reasoner._result)
        self.assertFalse(reasoner._proving)
        self.assertIsNone(reasoner._proposal)
        _assert_scoped_cleanly(self, client)

    def test_cancellation_during_leaf_execution_releases_everything(self):
        # One draw per node gives the root one edge, so both simulations of a
        # batch traverse it and it holds two virtual losses at once.
        client = _RecordingClient({0: ["A", "B"]})
        reasoner = _reasoner(client, top_k=1, **_puct())
        captured = self._capture_graph(reasoner)

        async def run():
            client.gate = None
            first_round = asyncio.Event()
            original = client.before_apply

            async def block_on_leaf(node_id):
                if node_id != 0:
                    first_round.set()
                    await asyncio.Event().wait()
                await original(node_id)

            client.before_apply = block_on_leaf
            task = asyncio.create_task(reasoner.prove(SEED, proposal=SAMPLE))
            await first_round.wait()
            graph = captured["graph"]
            self.assertEqual(
                max(edge.visit_stats.virtual_loss for edge in graph.edges.values()), 2
            )
            self.assertTrue(reasoner._pending)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        graph = captured["graph"]
        self.assertEqual(graph.end_reason, SearchEndReason.EXTERNAL_ABORT)
        self._assert_released(reasoner, client, graph)

    def test_transport_failure_aborts_without_fabricated_failure_records(self):
        client = _RecordingClient(
            {0: ["A", "B"], 1: PantographTransportError("process exited")}
        )
        reasoner = _reasoner(client, **_puct())
        captured = self._capture_graph(reasoner)
        with self.assertRaises(PantographTransportError):
            asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        graph = captured["graph"]
        self.assertEqual(graph.end_reason, SearchEndReason.EXTERNAL_ABORT)
        self.assertEqual(graph.nodes[1].closure_reason, NodeClosureReason.EXTERNAL_ABORT)
        self.assertEqual(
            [s for s in extract_critic_samples(graph) if s.node_id == 1], []
        )
        self._assert_released(reasoner, client, graph)

    def test_protocol_failure_during_a_leaf_tactic_propagates(self):
        client = _RecordingClient(
            {0: ["A", "B"], 1: PantographProtocolError("malformed goal.tactic response")}
        )
        reasoner = _reasoner(client, **_puct())
        captured = self._capture_graph(reasoner)
        with self.assertRaises(PantographProtocolError):
            asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        graph = captured["graph"]
        self.assertEqual(graph.nodes[1].closure_reason, NodeClosureReason.EXTERNAL_ABORT)
        self._assert_released(reasoner, client, graph)

    def test_unobservable_goal_rejection_and_unknown_backup_stay_distinct(self):
        # The model cannot read A; every tactic on B is rejected by Lean.
        # One draw per node is its whole widening budget, so B's rejected round
        # closes its source and its failure is genuine for this search.
        client = _RecordingClient({0: ["A", "B"], "B": "reject"})
        reasoner = _reasoner(client, top_k=1, **_puct(num_simulations=4, widening_max_draws=1))
        reasoner.unobservable = frozenset({"A"})
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        graph = result.graph

        self.assertEqual(len(graph.nodes[0].outgoing_edge_ids), 1)
        root_edge = graph.edges[graph.nodes[0].outgoing_edge_ids[0]]
        elaboration, rejected = (graph.nodes[child] for child in root_edge.child_ids)
        self.assertEqual(elaboration.closure_reason, NodeClosureReason.UNSUPPORTED_OBSERVATION)
        self.assertEqual(rejected.closure_reason, NodeClosureReason.WIDENING_BOUND)
        # Ordinary rejection produces actor failure records for B only; the
        # unobservable node never sampled or executed anything.
        self.assertTrue(failed(result))
        for record in failed(result):
            self.assertEqual(record.goal.state_fingerprint(), rejected.goal.state_fingerprint())
        self.assertNotIn(elaboration.id, dict(reasoner.executed))
        # The AND-backup over (unknown, valid 0.0) is a valid 0.0: completed
        # visits count, and only valid numeric evidence enters W.
        self.assertGreaterEqual(root_edge.visit_stats.N, 1)
        self.assertEqual(root_edge.visit_stats.N, root_edge.visit_stats.N_v)
        self.assertEqual(root_edge.visit_stats.W, 0.0)
        # Learning: B is a valid search failure, A has no critic label.
        samples = {sample.node_id: sample for sample in extract_critic_samples(graph)}
        self.assertNotIn(elaboration.id, samples)
        self.assertEqual(samples[rejected.id].target, 0.0)
        self.assertEqual(samples[rejected.id].source, BackupSource.SEARCH_FAILURE)
        _assert_scoped_cleanly(self, client)

    def test_unknown_backup_counts_the_visit_without_numeric_evidence(self):
        client = _RecordingClient({0: ["A"]})
        reasoner = _reasoner(client, **_puct(num_simulations=2, sim_batch_size=1))
        reasoner.unobservable = frozenset({"A"})
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        root_edge = result.graph.edges[result.graph.nodes[0].outgoing_edge_ids[0]]
        self.assertEqual(root_edge.visit_stats.N, 1)
        self.assertEqual(root_edge.visit_stats.N_v, 0)
        self.assertEqual(root_edge.visit_stats.W, 0.0)
        self.assertEqual(failed(result), [])
        _assert_scoped_cleanly(self, client)


class CooperativeDeadlineTests(unittest.TestCase):
    def test_deadline_after_first_batch_returns_completed_experience(self):
        client = _RecordingClient({0: ["A", "B"]})
        reasoner = _reasoner(client, **_puct())
        checks = 0

        def within_deadline(_deadline):
            nonlocal checks
            checks += 1
            return checks == 1

        reasoner._within_deadline = within_deadline
        result = asyncio.run(reasoner.prove(SEED, deadline=0.0, proposal=SAMPLE))
        graph = result.graph
        self.assertEqual(graph.end_reason, SearchEndReason.DEADLINE)
        self.assertFalse(graph.is_solved())
        # The completed root expansion is kept as experience.
        self.assertTrue(graph.nodes[0].outgoing_edge_ids)
        self.assertTrue(result.observations)
        for edge in graph.edges.values():
            self.assertEqual(edge.visit_stats.virtual_loss, 0)
        _assert_scoped_cleanly(self, client)


@pytest.mark.pantograph_live
def test_actual_puct_and_best_first_search_over_the_installed_client(live_env):
    async def run():
        client = await create_model_sexpr_client(live_env)
        async with client:
            baseline = (await client.stat()).n_goals
            checks = await verify_mcts_search(client, searches=20)
            assert (await client.stat()).n_goals == baseline
        assert client.phase.value == "CLOSED"
        return checks

    checks = asyncio.run(run())
    assert {
        "puct-multi-leaf-batch",
        "checked-rejection-of-alternatives",
        "puct-graph-equivalence",
        "best-first-graph-equivalence",
        "cooperative-deadline-scope-closure",
        "repeated-searches-20",
    } <= set(checks)


@pytest.mark.pantograph_live
def test_actual_widening_finds_a_later_action_after_a_checked_rejection(live_env):
    async def run():
        client = await create_model_sexpr_client(live_env)
        async with client:
            return await verify_widening(client)

    assert asyncio.run(run()) == [
        "widening-later-action-after-checked-rejection",
        "widening-rejection-served-from-cache",
        "widening-and-obligations-kept",
        "widening-priors-normalized",
    ]


@pytest.mark.pantograph_live
def test_actual_pool_workers_share_one_model_device(live_env):
    assert asyncio.run(verify_shared_device_workers(live_env)) == [
        "two-workers-one-model-device"
    ]


@pytest.mark.pantograph_live
def test_actual_bounded_updates_with_htps_disabled_and_enabled(live_env):
    async def run():
        client = await create_model_sexpr_client(live_env)
        async with client:
            baseline = (await client.stat()).n_goals
            learning = await verify_learning_round(client)
            assert (await client.stat()).n_goals == baseline
        assert client.phase.value == "CLOSED"
        return learning

    learning = asyncio.run(run())
    disabled, enabled = learning["htps_disabled"], learning["htps_enabled"]
    assert disabled["onpolicy_optimizer_steps"] == 1
    assert disabled["htps_optimizer_steps"] == 0
    assert disabled["model_generation"] == 1
    assert enabled["onpolicy_optimizer_steps"] == 1
    assert enabled["htps_optimizer_steps"] == 1
    assert enabled["model_generation"] == 2
    assert enabled["imitation_samples"] or enabled["critic_samples"]


if __name__ == "__main__":
    unittest.main()
