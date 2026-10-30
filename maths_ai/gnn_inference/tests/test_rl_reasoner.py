from __future__ import annotations

import asyncio
import time
import unittest
from dataclasses import replace
from types import MethodType, SimpleNamespace

import torch
from torch.optim import AdamW
from torch_geometric.data import Batch

from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import LocalCandidateProvider
from maths_ai.data_models.proof_components import STV, TacticCandidate
from maths_ai.lean_replay import seed_from_text
from maths_ai.hybrid_reasoner.execution_types import ExecutionOccurrence
from maths_ai.hybrid_reasoner.execution_registry import LeaseKind, SnapshotRegistry
from maths_ai.hybrid_reasoner.hypergraph import (
    NodeClosureReason,
    ProofHypergraph,
    SearchEndReason,
    UnsupportedObservation,
)
from maths_ai.pln_inference.model import PLNResult

from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import (
    ActionSample,
    ActorCriticWithArgsClassifier,
)
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import (
    ArgumentReference,
    CandidatePoolSnapshot,
)
from maths_ai.gnn_inference.tests.model_helpers import (
    FakeResultState,
    FakeStateScope,
    ScopedFakeClient,
    ScriptedExecutor,
    actor_critic,
    pantograph_goal,
    scored,
    structured_goal,
)
from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    EdgeAction,
    ObservationOutcome,
    PolicyObservation,
    compute_onpolicy_loss,
    make_dag_featurizer,
    train_step_onpolicy,
)
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import extract_transitions
from maths_ai.hybrid_reasoner.proposal import ProposalSpec

SAMPLE = ProposalSpec(mode="sample", seed=0)


def accepted(result):
    """Each accepted edge's first raw action, as the old join table had it."""
    actions = {}
    for observation in result.observations:
        if observation.edge_id is not None:
            actions.setdefault(observation.edge_id, observation.action)
    return actions


def failed(result):
    return [o for o in result.observations if o.outcome != ObservationOutcome.ACCEPTED]


# ---------------------------------------------------------------------------
# Fakes: no Lean/Pantograph backend, no petta subprocess
# ---------------------------------------------------------------------------


class _FakeGoalState:
    def __init__(self, goals=None, *, expression=None):
        self.expression = expression
        self.goals = [pantograph_goal(goal) for goal in (goals or [])]


class _FakeServer(ScopedFakeClient):
    def __init__(self):
        self.goals_by_expression = {}
        self._last_expression = None

    async def _start_goal(self, expression):
        self._last_expression = expression
        goal = self.goals_by_expression.get(expression)
        if goal is None:
            goal = structured_goal(expression, HYPS if expression == GOAL_EXPR else [])
        return _FakeGoalState([goal] if goal is not None else [], expression=expression)

    async def _apply_tactic(self, state, tactic):
        if tactic.startswith("intro "):
            if state.expression == "∀ (p : Prop), p → p":
                return _FakeGoalState(
                    [structured_goal(GOAL_EXPR, HYPS)], expression=GOAL_EXPR
                )
            goal = self.goals_by_expression.get(state.expression)
            if goal is not None:
                return _FakeGoalState([goal], expression=state.expression)
        return state


def _QEDExecutor():
    """Every tactic application succeeds with no child obligations."""
    return ScriptedExecutor(lambda _occurrence, _command: [], server=_FakeServer())


def _RejectExecutor():
    """Every tactic application fails (Lean rejects the tactic)."""
    return ScriptedExecutor(lambda _occurrence, _command: None, server=_FakeServer())


class _SubgoalExecutor(ScriptedExecutor):
    """One application on a goal in ``depth_map`` yields its subgoal
    expressions; an application on any other goal closes it.

    Each application runs on an occurrence, so the routing key is the
    occurrence's own goal expression.
    """

    def __init__(self, depth_map: dict[str, list[str]]):
        super().__init__(self.produce, server=_FakeServer())
        self.depth_map = depth_map
        self.applications = 0

    def produce(self, occurrence, command):
        self.applications += 1
        return [structured_goal(e, HYPS) for e in self.depth_map.get(occurrence.goal.expression, [])]


class _StubPLN:
    """PLN stand-in: a real (non-fallback) low score, no subprocess."""

    async def evaluate_async(self, expression, hypotheses=None, **kwargs):
        return PLNResult(stv=STV(strength=0.1, confidence=1.0), status="ok", is_fallback=False)


TACTIC_VOCAB = {"trivial": 0, "intro": 1, "exact": 2}
GOAL_EXPR = "p → p"
HYPS = ["p : Prop"]
SEED = seed_from_text(goal=GOAL_EXPR, hypotheses=HYPS)


def _make_reasoner(executor, *, top_k=3, seed=0, reasoner_cls=RLHybridReasoner, **search_kwargs):
    torch.manual_seed(seed)
    goal = structured_goal(GOAL_EXPR, HYPS)
    # Vocab from the test goal's own DAG so featurization is non-degenerate.
    from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
    from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab

    node_vocab = build_vocab([model_goal_to_dag(goal)])
    model = actor_critic(len(node_vocab), len(TACTIC_VOCAB))
    reasoner = reasoner_cls(
        model,
        node_vocab,
        TACTIC_VOCAB,
        executor=executor,
        candidate_provider=LocalCandidateProvider(),
        top_k_tactics=top_k,
        max_depth=3,
        max_nodes=20,
        **search_kwargs,
    )
    reasoner.petta_chainer = _StubPLN()  # keep unit tests free of the petta subprocess
    return reasoner, model, node_vocab


def _local_sample(indices, *, tactic_id=2, arg_logp=-0.3):
    """One draw per batch row of ``tactic_id`` with one local argument
    ``indices[row]``; the argument's whole log-probability is ``arg_logp``."""
    rows = len(indices)
    return ActionSample(
        tactic_action=torch.tensor([tactic_id] * rows),
        tactic_logp=torch.tensor([-0.5] * rows),
        tactic_entropy=torch.tensor([1.0] * rows),
        arguments=tuple((ArgumentReference("local", index),) for index in indices),
        count_logp=torch.tensor([arg_logp] * rows),
        source_logp=torch.tensor([0.0] * rows),
        identity_logp=torch.tensor([0.0] * rows),
        value=torch.tensor([0.0] * rows),
        tactic_logits=torch.zeros(rows, 3),
    )


def _pool_of(dag) -> CandidatePoolSnapshot:
    """A pool admitting every node of ``dag``; decoding does not narrow it."""
    return CandidatePoolSnapshot(tuple(range(len(dag.nodes))))


def _goal_pool(reasoner, goal) -> CandidatePoolSnapshot:
    """The pool live collection retrieves for ``goal``."""
    _dag, data = reasoner.dag_featurize(goal)
    local_ids = data.premise_mask.nonzero().flatten().tolist()
    return asyncio.run(
        LocalCandidateProvider().retrieve(goal, local_ids, str(data.graph_fingerprint))
    )


class DecodeTests(unittest.TestCase):
    def test_decode_tactic_and_argument(self):
        reasoner, _model, node_vocab = _make_reasoner(_QEDExecutor())
        goal = structured_goal(GOAL_EXPR, HYPS)
        dag, data = reasoner.dag_featurize(goal)

        # The hypothesis node's first child is its name node ("p").
        hyp_idx = next(i for i, n in enumerate(dag.nodes) if n.label == "Hyp")

        sample = _local_sample([hyp_idx])  # "exact"
        from maths_ai.gnn_inference.atp_lean_gnn.inference import local_declarations
        pool = _pool_of(dag)
        candidate, action = reasoner._decode(
            sample, dag, local_declarations(goal), str(data.graph_fingerprint), pool
        )
        self.assertEqual(candidate.candidate.tactic_name, "exact")
        self.assertEqual(candidate.candidate.arguments, ["p"])
        # The prior is the complete action: the tactic and its argument.
        self.assertAlmostEqual(candidate.log_prior, -0.8, places=6)
        self.assertEqual(candidate.raw_action_id, (2, (("local", hyp_idx),)))
        self.assertEqual(action.tactic_id, 2)
        self.assertEqual(action.arguments, (ArgumentReference("local", hyp_idx),))
        self.assertIs(action.candidate_pool, pool)

    def test_decode_out_of_range_argument_dropped_from_command(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        goal = structured_goal(GOAL_EXPR, HYPS)
        dag, data = reasoner.dag_featurize(goal)

        oob = len(dag.nodes) + 5  # outside the graph
        sample = _local_sample([oob], tactic_id=1, arg_logp=0.0)  # "intro"
        from maths_ai.gnn_inference.atp_lean_gnn.inference import local_declarations
        candidate, action = reasoner._decode(
            sample, dag, local_declarations(goal), str(data.graph_fingerprint), _pool_of(dag)
        )
        self.assertIsNone(candidate)
        self.assertEqual(action.arguments, (ArgumentReference("local", oob),))
        self.assertEqual(action.graph_fingerprint, str(data.graph_fingerprint))


class DeclarationResolutionTests(unittest.TestCase):
    """A sampled local resolves to the live declaration it denotes, not by name."""

    @staticmethod
    def _sample(indices):
        """One ``exact`` draw per batch row, each selecting ``indices[row]``."""
        return _local_sample(indices)

    @staticmethod
    def _positions(dag, label):
        """The FV node with ``label`` and the structured ``Hyp`` that declares it."""
        fv = next(i for i, n in enumerate(dag.nodes) if n.label == label)
        hyp = next(
            i
            for i, n in enumerate(dag.nodes)
            if n.label == "Hyp" and n.children and n.children[0] == fv
        )
        return hyp, fv

    def test_review_case_through_rl_decoding_for_hyp_and_fv_candidates(self):
        from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
        from maths_ai.gnn_inference.atp_lean_gnn.inference import local_declarations
        from maths_ai.gnn_inference.tests.test_model_identity import (
            after_intro_x,
            outer_aliased,
        )

        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        child = after_intro_x(outer_aliased())
        dag = model_goal_to_dag(child)
        declarations = local_declarations(child)

        for label, live in (("FV5", "x_1"), ("FV6", "x")):
            for index in self._positions(dag, label):
                candidate, _ = reasoner._decode(
                    self._sample([index]), dag, declarations, "fingerprint", _pool_of(dag)
                )
                self.assertEqual(candidate.candidate.arguments, [live])

    def test_batched_leaves_resolve_through_their_own_declarations(self):
        """Leaves with the same model names but different live names, and a leaf
        with fewer locals, in one policy batch: each row resolves only through its
        own goal's locals, before and after re-materialization."""
        from maths_ai.data_models.model_identity import with_model_identity
        from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
        from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
        from maths_ai.gnn_inference.tests.test_model_identity import (
            after_intro_x,
            goal,
            local,
            outer_aliased,
        )

        def replayed(names, internals, identities):
            """A goal as Lean returns it, given the model identity it replays."""
            live = goal(
                [local(name, internal, index) for index, (name, internal) in enumerate(zip(names, internals))],
                target="(:app (:c Eq) (:fv FV0) (:fv FV0))",
            )
            return with_model_identity(live, identities)

        first = after_intro_x(outer_aliased())
        identities = [(l.user_name, l.context_index) for l in first.locals]
        second = replayed(["y_1", "z"], ["_live.3", "_live.4"], identities)
        third = replayed(["w"], ["_live.5"], identities[:1])
        self.assertEqual(
            [l.user_name for l in first.locals], [l.user_name for l in second.locals]
        )
        # Re-materialization: Lean issues new internal ids; the model identity
        # comes back from each goal's own replay seed.
        again = [
            replayed(
                [l.name for l in leaf.replay_seed().locals],
                [f"_new.{row}.{i}" for i in range(len(leaf.locals))],
                [l.model_identity for l in leaf.replay_seed().locals],
            )
            for row, leaf in enumerate((first, second, third))
        ]
        leaves_by_round = ((first, second, third), tuple(again))
        node_vocab = build_vocab(
            [model_goal_to_dag(leaf) for leaves in leaves_by_round for leaf in leaves]
        )
        reasoner = RLHybridReasoner(
            actor_critic(len(node_vocab), len(TACTIC_VOCAB)),
            node_vocab,
            TACTIC_VOCAB,
            executor=_QEDExecutor(),
            candidate_provider=LocalCandidateProvider(),
            top_k_tactics=1,
        )
        reasoner.petta_chainer = _StubPLN()
        # Proposals outside prove() need the search's proposal and generator.
        reasoner._proposal, reasoner._generator = SAMPLE, torch.Generator()

        # FV5 is in every leaf; FV6 only in the two-local leaves.
        cases = (("FV5", ["x_1", "y_1", "w"], 3), ("FV6", ["x", "z"], 2))
        for leaves in leaves_by_round:
            for label, expected, rows in cases:
                for kind in (0, 1):  # the Hyp candidate, then the free variable
                    batch = list(leaves[:rows])
                    indices = [
                        self._positions(model_goal_to_dag(leaf), label)[kind] for leaf in batch
                    ]
                    reasoner._pending = {}
                    reasoner._pools = {
                        node_id: _pool_of(model_goal_to_dag(leaf))
                        for node_id, leaf in zip(range(1, rows + 1), batch)
                    }
                    reasoner.model.propose_actions = lambda *_args, **_kwargs: (
                        self._sample(indices), torch.arange(len(indices))
                    )
                    proposals = reasoner._predict_next_tactics_batch(
                        batch, node_ids=list(range(1, rows + 1))
                    )
                    self.assertEqual(
                        [row[0].candidate.arguments[0] for row in proposals], expected, (label, kind)
                    )
                    # Each row's draw is owned by its own node.
                    self.assertEqual(sorted(reasoner._pending), list(range(1, rows + 1)))


class ProposalSeedTests(unittest.TestCase):
    """A search's draws come from its own seed, never from the global RNG."""

    @staticmethod
    def _draws(result):
        return sorted(
            (action.raw_action_id, action.multiplicity)
            for action in (o.action for o in result.observations)
        )

    def test_the_same_seed_repeats_a_search_and_leaves_the_global_rng_alone(self):
        reasoner, _model, _vocab = _make_reasoner(_RejectExecutor(), top_k=4)
        proposal = ProposalSpec(mode="sample", seed=11)
        before = torch.get_rng_state()
        first = asyncio.run(reasoner.prove(SEED, proposal=proposal))
        second = asyncio.run(reasoner.prove(SEED, proposal=proposal))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertEqual(self._draws(first), self._draws(second))
        other = [
            self._draws(asyncio.run(reasoner.prove(SEED, proposal=ProposalSpec(mode="sample", seed=s))))
            for s in range(1, 6)
        ]
        self.assertTrue(any(draws != self._draws(first) for draws in other))

    def test_greedy_draws_one_action_k_times(self):
        reasoner, _model, _vocab = _make_reasoner(_RejectExecutor(), top_k=4)
        result = asyncio.run(reasoner.prove(SEED, proposal=ProposalSpec(mode="greedy", seed=0)))
        draws = self._draws(result)
        self.assertEqual(len(draws), 1)
        self.assertEqual(draws[0][1], 4)

    def test_prove_requires_a_proposal_spec(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        with self.assertRaises(TypeError):
            asyncio.run(reasoner.prove(SEED, proposal="sample"))


class RolloutTests(unittest.TestCase):
    def _prove(self, reasoner):
        return asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))

    def test_qed_rollout_stash_migrates_to_edge_ids(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        result = self._prove(reasoner)

        self.assertTrue(result.graph.is_solved())
        self.assertGreater(len(accepted(result)), 0)
        for edge_id, action in accepted(result).items():
            edge = result.graph.edges[edge_id]
            self.assertEqual(edge.tactic.tactic_name, reasoner.id_to_tactic[action.tactic_id])

    def test_multiplicity_accounts_for_every_draw(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        result = self._prove(reasoner)
        total = sum(o.action.multiplicity for o in result.observations)
        # One node expanded (root, immediately solved): every i.i.d. draw is accounted for.
        self.assertEqual(total, reasoner.top_k_tactics)

    def test_rejected_samples_become_rejected_observations(self):
        reasoner, _model, _vocab = _make_reasoner(_RejectExecutor())
        result = self._prove(reasoner)

        self.assertFalse(result.graph.is_solved())
        self.assertEqual(len(accepted(result)), 0)
        self.assertGreater(len(failed(result)), 0)
        total = sum(f.action.multiplicity for f in failed(result))
        self.assertEqual(total, reasoner.top_k_tactics)

    def test_onpolicy_edge_filter(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        result = self._prove(reasoner)
        transitions = extract_transitions(
            result.graph, RewardConfig(step_penalty=0.0),
            edge_ids=list(accepted(result).keys()),
        )
        # Only the certified witness is positive evidence; the other closing
        # attempts are candidates that no replay certified, so they stay unknown.
        witnessed = result.graph.witnessed_edges()
        self.assertEqual(len(witnessed), 1)
        self.assertEqual({t.edge_id for t in transitions}, witnessed)
        for t in transitions:
            self.assertIn(t.edge_id, accepted(result))


class OnPolicyLossTests(unittest.TestCase):
    def test_evaluate_actions_gradient_reaches_pointer(self):
        reasoner, model, _vocab = _make_reasoner(_QEDExecutor())
        goal = structured_goal(GOAL_EXPR, HYPS)
        dag, data = reasoner.dag_featurize(goal)
        batch = Batch.from_data_list([data])
        hyp_idx = next(i for i, n in enumerate(dag.nodes) if n.label == "Hyp")

        model.train()
        evaluated = model.evaluate_actions(
            batch,
            torch.tensor([2]),
            [(ArgumentReference("local", hyp_idx),)],
            candidate_pools=(CandidatePoolSnapshot((hyp_idx,)),),
        )
        loss = -evaluated.logp.sum()
        loss.backward()
        grad = model.decoder.argument_selector.init_proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_train_step_onpolicy_updates_params(self):
        reasoner, model, _vocab = _make_reasoner(_QEDExecutor())
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))

        optimizer = AdamW(model.parameters(), lr=0.01)
        before = [p.detach().clone() for p in model.parameters()]
        metrics = train_step_onpolicy(
            model, optimizer, [result], reasoner.dag_featurize_data,
            reward_cfg=RewardConfig(step_penalty=0.0), bc_weight=0.1,
        )
        self.assertGreater(metrics["num_transitions"], 0.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        after = list(model.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        self.assertTrue(changed, "on-policy training step did not update any parameters")

    def test_failure_only_batch_is_actor_only(self):
        reasoner, model, _vocab = _make_reasoner(_RejectExecutor())
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        self.assertGreater(len(failed(result)), 0)

        loss_result = compute_onpolicy_loss(
            model, [], [], failed(result), reasoner.dag_featurize_data,
        )
        self.assertIsNotNone(loss_result)
        loss, metrics = loss_result
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["critic_loss"], 0.0)  # a failed ACTION carries no state value
        self.assertEqual(metrics["num_transitions"], 0.0)
        self.assertGreater(metrics["num_failures"], 0.0)

    def test_multiplicity_weights_actor_term(self):
        reasoner, model, _vocab = _make_reasoner(_QEDExecutor())
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))
        transitions = extract_transitions(
            result.graph, RewardConfig(step_penalty=0.0),
            edge_ids=list(accepted(result).keys()),
        )
        goal = structured_goal(GOAL_EXPR, HYPS)

        # Success rows (return ≈ 1) against a failure row (return ≈ 0) give a nonzero
        # advantage contrast, so re-weighting the failure by m must change the loss.
        def loss_with(m: int) -> float:
            model.eval()  # deterministic forward so only multiplicity differs
            failures = [PolicyObservation(
                node_id=0,
                goal=goal,
                action=EdgeAction(
                    tactic_id=1,
                    candidate_pool=_goal_pool(reasoner, goal),
                    multiplicity=m,
                    graph_fingerprint=str(reasoner.dag_featurize(goal)[1].graph_fingerprint),
                ),
                outcome=ObservationOutcome.REJECTED,
                first_round=0,
                log_prior=-1.0,
                provider="test",
            )]
            out = compute_onpolicy_loss(
                model, transitions, [], [*result.observations, *failures],
                reasoner.dag_featurize_data, reward_cfg=RewardConfig(step_penalty=0.0),
            )
            self.assertIsNotNone(out)
            return float(out[0].item())

        self.assertNotAlmostEqual(loss_with(1), loss_with(3))


class MCTSSearchTests(unittest.TestCase):
    """Multi-simulation search (selection_policy="puct") on the RL reasoner,
    and the legacy/puct mode gate."""

    ROOT = GOAL_EXPR  # "p → p"

    def _prove(self, reasoner):
        return asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))

    def test_default_legacy_loop_never_touches_visit_stats(self):
        # selection_policy="legacy" (default): the best-first loop runs
        # verbatim and no edge accumulates visit statistics — the regression
        # guarantee.
        executor = _SubgoalExecutor({self.ROOT: ["A", "B"]})
        reasoner, _model, _vocab = _make_reasoner(
            executor
        )
        result = self._prove(reasoner)
        self.assertTrue(result.graph.is_solved())
        for edge in result.graph.edges.values():
            self.assertEqual(edge.visit_stats.N, 0)
            self.assertEqual(edge.visit_stats.W, 0.0)
            self.assertEqual(edge.visit_stats.virtual_loss, 0)

    def test_default_equals_explicit_legacy_structurally(self):
        # A default-constructed reasoner and one with explicit
        # selection_policy="legacy" run the same code path: same seed and
        # executor script must produce structurally identical graphs. This
        # pins the default to the legacy loop.
        def run(**kwargs):
            executor = _SubgoalExecutor({self.ROOT: ["A", "B"]})
            reasoner, _model, _vocab = _make_reasoner(
                executor, **kwargs
            )
            return self._prove(reasoner).graph

        default_graph = run()
        legacy_graph = run(selection_policy="legacy")

        self.assertEqual(
            [(n.id, n.goal.expression, n.status) for n in default_graph.nodes.values()],
            [(n.id, n.goal.expression, n.status) for n in legacy_graph.nodes.values()],
        )
        self.assertEqual(
            [
                (e.id, e.tactic.tactic_name, tuple(e.tactic.arguments), tuple(e.child_ids), e.status)
                for e in default_graph.edges.values()
            ],
            [
                (e.id, e.tactic.tactic_name, tuple(e.tactic.arguments), tuple(e.child_ids), e.status)
                for e in legacy_graph.edges.values()
            ],
        )

    def test_legacy_with_explicit_budget_raises(self):
        # The gate is the policy: an explicit simulation budget under
        # "legacy" would be silently ignored, so construction rejects it.
        executor = _SubgoalExecutor({self.ROOT: ["A", "B"]})
        with self.assertRaises(ValueError):
            _make_reasoner(
                executor,
                selection_policy="legacy", num_simulations=6,
            )

    def test_puct_single_simulation_runs_simulation_loop(self):
        # selection_policy="puct" with num_simulations=1 runs the simulation
        # loop, not the legacy loop — proving the gate is the policy, not the
        # simulation count. One simulation selects the unexpanded root as its
        # only leaf and stops after expanding it (its backup traverses no
        # edges, since the descent chose none), so the graph holds exactly the
        # root plus its subgoals, unsolved — where the legacy loop on the same
        # script (see test_default_legacy_loop_never_touches_visit_stats)
        # continues to root resolution.
        executor = _SubgoalExecutor({self.ROOT: ["A", "B"]})
        reasoner, _model, _vocab = _make_reasoner(
            executor, top_k=1,
            selection_policy="puct", num_simulations=1,
        )
        result = self._prove(reasoner)
        self.assertFalse(result.graph.is_solved())
        expanded = [
            n for n in result.graph.nodes.values() if n.outgoing_edge_ids or n.exhausted
        ]
        self.assertEqual([n.id for n in expanded], [result.graph.root_id])

    def test_multi_sim_accumulates_stats_and_releases_virtual_loss(self):
        # Every root command splits into A and B, so the root keeps widening
        # for its first simulations before they descend to its children.
        executor = _SubgoalExecutor({self.ROOT: ["A", "B"]})
        reasoner, _model, _vocab = _make_reasoner(
            executor, top_k=1,
            selection_policy="puct", num_simulations=12, sim_batch_size=2,
        )
        result = self._prove(reasoner)
        self.assertTrue(result.graph.is_solved())
        # Simulations that descended through the root's edge backed values into
        # its statistics (W ≤ N: every backup value is a product of [0,1] terms).
        visited = [e for e in result.graph.edges.values() if e.visit_stats.N > 0]
        self.assertGreater(len(visited), 0)
        for edge in visited:
            self.assertLessEqual(edge.visit_stats.W, edge.visit_stats.N)
        # Every increment during selection was released during backup.
        for edge in result.graph.edges.values():
            self.assertEqual(edge.visit_stats.virtual_loss, 0)

    def test_expired_deadline_returns_partial_graph(self):
        executor = _SubgoalExecutor({self.ROOT: ["A", "B"]})
        reasoner, _model, _vocab = _make_reasoner(
            executor,
            selection_policy="puct", num_simulations=6, sim_batch_size=2,
        )
        result = asyncio.run(
            reasoner.prove(SEED, deadline=time.monotonic() - 1.0, proposal=SAMPLE)
        )
        # The loop stopped before the first simulation batch: only the root
        # exists, unexpanded, and the partial graph came back instead of an error.
        self.assertEqual(len(result.graph.nodes), 1)
        self.assertFalse(result.graph.is_solved())
        self.assertEqual(executor.applications, 0)
        self.assertEqual(result.graph.end_reason, SearchEndReason.DEADLINE)

    def test_cancellation_releases_in_flight_virtual_losses(self):
        reasoner, _model, _vocab = _make_reasoner(
            _QEDExecutor(),
            selection_policy="puct",
            num_simulations=6,
            sim_batch_size=2,
        )
        graph, _leaf_ids = self._two_leaf_graph()
        expansion_started = asyncio.Event()

        async def block_expansion(self, graph, leaf_ids):
            del self, graph, leaf_ids
            expansion_started.set()
            await asyncio.Event().wait()

        reasoner._expand_leaves = MethodType(block_expansion, reasoner)

        async def run_and_cancel():
            task = asyncio.create_task(reasoner._prove_mcts(graph, None))
            await expansion_started.wait()
            self.assertGreater(
                sum(edge.visit_stats.virtual_loss for edge in graph.edges.values()),
                0,
            )
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run_and_cancel())
        for edge in graph.edges.values():
            self.assertEqual(edge.visit_stats.virtual_loss, 0)

    def test_batched_expansion_attributes_failures_to_the_right_node(self):
        # Regression for the per-node stash (Decision 1.1): one simulation's
        # leaves A and B are proposed in one batched call; A's applications
        # succeed (QED) while B's are rejected — every failure record must
        # carry B's goal, never A's or the root's.
        class _SelectiveExecutor(_SubgoalExecutor):
            def produce(self, occurrence, command):
                if occurrence.goal.expression == "B":
                    self.applications += 1
                    return None
                return super().produce(occurrence, command)

        executor = _SelectiveExecutor({self.ROOT: ["A", "B"]})
        reasoner, _model, _vocab = _make_reasoner(
            executor, top_k=2,
            selection_policy="puct", num_simulations=12, sim_batch_size=2,
        )
        result = self._prove(reasoner)
        self.assertGreater(len(failed(result)), 0)
        for record in failed(result):
            self.assertEqual(record.goal.expression, "B")
        # A's QED edge is in the on-policy join table with a matching tactic.
        solved_edges = [
            eid for eid, e in result.graph.edges.items()
            if result.graph.nodes[e.source_id].goal.expression == "A"
        ]
        self.assertTrue(any(eid in accepted(result) for eid in solved_edges))

    def _two_leaf_graph(self):
        root = structured_goal(self.ROOT, HYPS)
        graph = ProofHypergraph(root)
        edge = graph.add_edge(
            graph.root_id,
            scored(TacticCandidate(tactic_name="intro", arguments=[], probability=1.0)),
            [(structured_goal("A", HYPS), None), (structured_goal("B", HYPS), None)],
        )
        # The root's proposals are spent, so simulations descend to A and B.
        graph.mark_node_exhausted(graph.root_id, reason=NodeClosureReason.CANDIDATES_EXHAUSTED)
        return graph, edge.child_ids

    @staticmethod
    def _retain(reasoner, graph, node_ids, *, skip=()):
        """Give each leaf a registry-owned occurrence of its own."""
        registry = SnapshotRegistry(FakeStateScope(ScopedFakeClient()))
        reasoner._registry = registry
        leases = {}
        for node_id in node_ids:
            if node_id in skip:
                continue
            node = graph.nodes[node_id]
            state = FakeResultState([
                replace(pantograph_goal(node.goal), goal_name=f"g{node_id}")
            ])
            leases[node_id] = registry.reserve().adopt(
                state, LeaseKind.GRAPH, f"leaf-{node_id}"
            )
            node.occurrence = ExecutionOccurrence(
                node_id, node.key, f"g{node_id}", node.goal, {}, leases[node_id]
            )
        return leases

    def test_leaf_batch_is_proposed_once_and_executed_on_its_own_snapshot(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        graph, leaf_ids = self._two_leaf_graph()
        leases = self._retain(reasoner, graph, leaf_ids)
        policy_rows = []
        executions = []

        def propose(self, nodes, draws=None):
            policy_rows.append([node.id for node in nodes])
            return [
                [scored(TacticCandidate(tactic_name="skip", arguments=[], probability=1.0))]
                for _ in nodes
            ]

        async def execute(self, graph, node, candidates, *, draws=None):
            executions.append((node.id, node.occurrence.lease))

        reasoner.predict_next_tactics_for_nodes = MethodType(propose, reasoner)
        reasoner._execute_and_link = MethodType(execute, reasoner)
        asyncio.run(reasoner._expand_leaves(graph, leaf_ids))
        self.assertEqual(policy_rows, [leaf_ids])
        self.assertEqual(executions, [(node_id, leases[node_id]) for node_id in leaf_ids])

    def test_an_unretained_leaf_does_not_shift_policy_rows(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        graph, leaf_ids = self._two_leaf_graph()
        self._retain(reasoner, graph, leaf_ids, skip={leaf_ids[0]})
        proposed = []

        def propose(self, nodes, draws=None):
            proposed.extend(node.id for node in nodes)
            return [[] for _ in nodes]

        reasoner.predict_next_tactics_for_nodes = MethodType(propose, reasoner)
        asyncio.run(reasoner._expand_leaves(graph, leaf_ids))
        self.assertEqual(proposed, [leaf_ids[1]])
        self.assertEqual(
            graph.nodes[leaf_ids[0]].closure_reason, NodeClosureReason.RESOURCE_PRUNED
        )

    def test_an_unobservable_leaf_is_closed_without_failing_the_batch(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        graph, leaf_ids = self._two_leaf_graph()
        self._retain(reasoner, graph, leaf_ids)
        reasoner._counters = {}
        proposed = []

        def propose(self, nodes, draws=None):
            if any(node.id == leaf_ids[0] for node in nodes):
                raise UnsupportedObservation("missing structural label")
            proposed.extend(node.id for node in nodes)
            return [[] for _ in nodes]

        reasoner.predict_next_tactics_for_nodes = MethodType(propose, reasoner)
        asyncio.run(reasoner._expand_leaves(graph, leaf_ids))
        self.assertEqual(proposed, [leaf_ids[1]])
        self.assertEqual(
            graph.nodes[leaf_ids[0]].closure_reason,
            NodeClosureReason.UNSUPPORTED_OBSERVATION,
        )
        self.assertEqual(reasoner._counters["unsupported_observations"], 1)


class PLNDisabledTests(unittest.TestCase):
    """use_pln=False: no PLN objects exist, no petta subprocess, terminal-only reward."""

    def _prove(self, reasoner):
        return asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))

    def test_no_pln_objects_constructed(self):
        # use_pln=False: petta_chainer and dts_sampler must both be None —
        # the test must NOT reassign _StubPLN here, because the point is that
        # nothing ever dereferences the chainer.
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor(), use_pln=False)
        # _make_reasoner assigns _StubPLN after construction; undo that
        # assignment to confirm the constructor truly skipped PLNInference.
        # The easiest way: build without the helper's post-assignment.
        from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
        from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
        goal = structured_goal(GOAL_EXPR, HYPS)
        node_vocab = build_vocab([model_goal_to_dag(goal)])
        model = actor_critic(len(node_vocab), len(TACTIC_VOCAB))
        raw = RLHybridReasoner(
            model, node_vocab, TACTIC_VOCAB,
            executor=_QEDExecutor(), top_k_tactics=3, max_depth=3, max_nodes=20,
            candidate_provider=LocalCandidateProvider(),
            use_pln=False,
        )
        self.assertIsNone(raw.petta_chainer)
        self.assertIsNone(raw.dts_sampler)

    def test_subgoal_nodes_have_stv_none_and_executor_order(self):
        # With PLN off, subgoals keep Lean's order; stv=None on each node.
        # Every Lean-returned subgoal must remain in the AND-edge.
        root_expr = GOAL_EXPR
        executor = _SubgoalExecutor({root_expr: ["A", "B", "C"]})
        reasoner, _model, _vocab = _make_reasoner(
            executor,
            top_k=1,
            use_pln=False,
        )
        result = self._prove(reasoner)
        child_nodes = [
            n for n in result.graph.nodes.values() if n.id != result.graph.root_id
        ]
        child_exprs = [n.goal.expression for n in child_nodes]
        self.assertEqual(child_exprs, ["A", "B", "C"])
        for n in child_nodes:
            self.assertIsNone(n.stv)

    def test_no_pln_fallback_edge_on_total_rejection(self):
        # With PLN off, a node whose every tactic is rejected goes straight to
        # exhausted — no PLN_fallback edge is ever added.
        reasoner, _model, _vocab = _make_reasoner(_RejectExecutor(), use_pln=False)
        result = self._prove(reasoner)
        self.assertFalse(result.graph.is_solved())
        pln_edges = [
            e for e in result.graph.edges.values()
            if e.tactic.tactic_name == "PLN_fallback"
        ]
        self.assertEqual(len(pln_edges), 0)
        root = result.graph.nodes[result.graph.root_id]
        self.assertTrue(root.exhausted)
        self.assertEqual(root.note, "executor rejected every renderable candidate tactic")

    def test_reward_is_terminal_only(self):
        # With stv=None on every subgoal node, potential() returns 0.0 for each
        # node, so edge_shaped_reward equals edge_terminal_reward everywhere.
        from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import (
            RewardConfig,
            edge_shaped_reward,
            edge_terminal_reward,
        )
        root_expr = GOAL_EXPR
        executor = _SubgoalExecutor({root_expr: ["A", "B"]})
        reasoner, _model, _vocab = _make_reasoner(
            executor, top_k=1, use_pln=False
        )
        result = self._prove(reasoner)
        reward_cfg = RewardConfig(
            gamma=0.99, step_penalty=0.01,
            terminal_success=1.0, terminal_failure=0.0,
        )
        for edge in result.graph.edges.values():
            shaped = edge_shaped_reward(edge, result.graph, reward_cfg)
            terminal = edge_terminal_reward(edge, result.graph, reward_cfg)
            self.assertAlmostEqual(shaped, terminal, places=9)

    def test_rank_subgoals_raises_when_pln_disabled(self):
        # Calling rank_subgoals on a use_pln=False reasoner raises RuntimeError
        # with a named message — not an AttributeError on None.
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor(), use_pln=False)
        goal = structured_goal(GOAL_EXPR, HYPS)
        from maths_ai.data_models.proof_components import TacticCandidate as TC
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(
                reasoner.rank_subgoals(
                    GOAL_EXPR, [goal], TC(tactic_name="intro", arguments=[], probability=1.0)
                )
            )
        self.assertIn("use_pln=False", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class LibraryCandidateTests(unittest.TestCase):
    """A node whose pool holds a library lemma, through search and observation."""

    def test_a_library_argument_renders_by_name_and_keeps_its_pool(self):
        import numpy as np

        from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import fingerprint
        from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import (
            LibraryMemory,
            goal_query,
        )

        library = LibraryMemory([7], ["Nat.le_refl"], np.zeros((1, 4), dtype=np.float32), "lib")

        class Provider:
            async def retrieve(self, goal, local_ids, graph_fingerprint, *, excluded=()):
                return CandidatePoolSnapshot(
                    tuple(local_ids), (7,), (0.5,),
                    graph_fingerprint=graph_fingerprint,
                    query_fingerprint=fingerprint(goal_query(goal)),
                    artifact_fingerprint="lib",
                    library=library,
                )

        executor = ScriptedExecutor(lambda _occurrence, _command: [], server=_FakeServer())
        reasoner, model, _vocab = _make_reasoner(executor, top_k=1)
        reasoner.candidate_provider = Provider()
        sample = ActionSample(
            tactic_action=torch.tensor([2]),  # "exact"
            tactic_logp=torch.tensor([-0.5]),
            tactic_entropy=torch.tensor([1.0]),
            arguments=((ArgumentReference("library", 7),),),
            count_logp=torch.tensor([-0.2]),
            source_logp=torch.tensor([-0.1]),
            identity_logp=torch.tensor([0.0]),
            value=torch.tensor([0.0]),
            tactic_logits=torch.zeros(1, 3),
        )
        model.propose_actions = lambda *_args, **_kwargs: (sample, torch.tensor([0]))
        result = asyncio.run(reasoner.prove(SEED, proposal=SAMPLE))

        self.assertEqual(executor.calls, ["exact Nat.le_refl"])
        (edge,) = result.graph.edges.values()
        (ref,) = edge.tactic.argument_refs
        self.assertEqual((ref.kind, ref.text), ("global", "Nat.le_refl"))
        self.assertEqual(set(edge.raw_log_priors), {(2, (("library", 7),))})
        self.assertAlmostEqual(edge.raw_log_priors[(2, (("library", 7),))], -0.8, places=6)
        (observation,) = result.observations
        self.assertEqual(observation.action.candidate_pool.lemma_ids, (7,))
        self.assertIs(observation.action.candidate_pool.library, library)
