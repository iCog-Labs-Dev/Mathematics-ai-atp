from __future__ import annotations

import asyncio
import time
import unittest
from dataclasses import replace
from types import MethodType

import torch
from torch.optim import AdamW
from torch_geometric.data import Batch

from maths_ai.data_models.proof_components import STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph, TacticOutcome
from maths_ai.hybrid_reasoner.joint_inference import MaterializedGoal
from maths_ai.pln_inference.model import PLNResult

from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import (
    ActionSample,
    ActorCriticWithArgsClassifier,
)
from maths_ai.gnn_inference.tests.model_helpers import (
    actor_critic,
    pantograph_goal,
    structured_goal,
)
from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    EdgeAction,
    FailureRecord,
    compute_onpolicy_loss,
    make_dag_featurizer,
    train_step_onpolicy,
)
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import extract_transitions


# ---------------------------------------------------------------------------
# Fakes: no Lean/Pantograph backend, no petta subprocess
# ---------------------------------------------------------------------------


class _FakeGoalState:
    def __init__(self, goals=None, *, expression=None):
        self.expression = expression
        self.goals = [pantograph_goal(goal) for goal in (goals or [])]


class _FakeServer:
    def __init__(self):
        self.proc = object()
        self.goals_by_expression = {}
        self._last_expression = None

    async def goal_start_async(self, expression):
        self._last_expression = expression
        goal = self.goals_by_expression.get(expression)
        if goal is None:
            goal = structured_goal(expression, HYPS if expression == GOAL_EXPR else [])
        return _FakeGoalState([goal] if goal is not None else [], expression=expression)

    async def goal_tactic_async(self, state, tactic):
        if tactic.startswith("intro "):
            if state.expression == "∀ (p : Prop), p → p":
                return _FakeGoalState(
                    [structured_goal(GOAL_EXPR, HYPS)], expression=GOAL_EXPR
                )
            goal = self.goals_by_expression.get(state.expression)
            if goal is not None:
                return _FakeGoalState([goal], expression=state.expression)
        return state


class _QEDExecutor:
    """Every tactic application succeeds with no subgoals (immediate QED)."""

    def __init__(self):
        self.server = _FakeServer()

    async def apply(self, server, state, tactic):
        return TacticOutcome(success=True, subgoals=[])


class _RejectExecutor:
    """Every tactic application fails (Lean rejects the tactic)."""

    def __init__(self):
        self.server = _FakeServer()

    async def apply(self, server, state, tactic):
        return TacticOutcome(success=False, subgoals=[], error="rejected")


class _SubgoalExecutor:
    """Yield configurable subgoals for the first ``depth`` applications per
    branch, then QED.

    ``depth_map`` maps a goal expression to the subgoal expressions one tactic
    application on it produces; an expression absent from the map QEDs. The
    goal state is not consulted (the fakes carry no goals), so the routing key
    is the tactic's *parent* expression — passed in at ``apply`` time via
    ``self.current_goal``, set by the reasoner hook below.
    """

    def __init__(self, depth_map: dict[str, list[str]]):
        self.server = _FakeServer()
        self.depth_map = depth_map
        self.current_goal: str | None = None
        self.applications = 0

    async def apply(self, server, state, tactic):
        self.applications += 1
        subgoal_exprs = self.depth_map.get(self.current_goal, [])
        subgoals = [structured_goal(e, HYPS) for e in subgoal_exprs]
        for subgoal in subgoals:
            self.server.goals_by_expression[subgoal.replay_spec().expression] = subgoal
        return TacticOutcome(success=True, subgoals=subgoals)


class _GoalTrackingReasoner(RLHybridReasoner):
    """Route the executor by the expanding node's goal expression.

    ``_SubgoalExecutor`` cannot see which node an application belongs to (the
    fake goal states carry no goals), so this override records it before the
    per-node execution stage runs.
    """

    async def _execute_and_link(self, graph, node, candidates, *, materialized):
        self.executor.current_goal = node.goal.expression
        return await super()._execute_and_link(
            graph, node, candidates, materialized=materialized
        )


class _StubPLN:
    """PLN stand-in: a real (non-fallback) low score, no subprocess."""

    async def evaluate_async(self, expression, hypotheses=None, **kwargs):
        return PLNResult(stv=STV(strength=0.1, confidence=1.0), status="ok", is_fallback=False)


TACTIC_VOCAB = {"trivial": 0, "intro": 1, "exact": 2}
GOAL_EXPR = "p → p"
HYPS = ["p : Prop"]


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
        top_k_tactics=top_k,
        max_depth=3,
        max_nodes=20,
        **search_kwargs,
    )
    reasoner.petta_chainer = _StubPLN()  # keep unit tests free of the petta subprocess
    return reasoner, model, node_vocab


class DecodeTests(unittest.TestCase):
    def test_decode_tactic_and_argument(self):
        reasoner, _model, node_vocab = _make_reasoner(_QEDExecutor())
        goal = structured_goal(GOAL_EXPR, HYPS)
        dag, data = reasoner.dag_featurize(goal)

        # The hypothesis node's first child is its name node ("p").
        hyp_idx = next(i for i, n in enumerate(dag.nodes) if n.label == "Hyp")

        sample = ActionSample(
            tactic_action=torch.tensor([2]),  # "exact"
            tactic_logp=torch.tensor([-0.5]),
            tactic_entropy=torch.tensor([1.0]),
            arg_actions=[torch.tensor([hyp_idx])],
            arg_logp=torch.tensor([-0.3]),
            value=torch.tensor([0.0]),
            tactic_logits=torch.zeros(1, 3),
        )
        from maths_ai.gnn_inference.atp_lean_gnn.inference import _local_names_by_fv_label
        candidate, action = reasoner._decode(
            sample, dag, _local_names_by_fv_label(dag), str(data.graph_fingerprint)
        )
        self.assertEqual(candidate.tactic_name, "exact")
        self.assertEqual(candidate.arguments, ["p"])
        self.assertEqual(action.tactic_id, 2)
        self.assertEqual(action.arg_indices, (hyp_idx,))

    def test_decode_out_of_range_argument_dropped_from_command(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        goal = structured_goal(GOAL_EXPR, HYPS)
        dag, data = reasoner.dag_featurize(goal)

        oob = len(dag.nodes) + 5  # padding position
        sample = ActionSample(
            tactic_action=torch.tensor([1]),  # "intro"
            tactic_logp=torch.tensor([-0.5]),
            tactic_entropy=torch.tensor([1.0]),
            arg_actions=[torch.tensor([oob])],
            arg_logp=torch.tensor([0.0]),
            value=torch.tensor([0.0]),
            tactic_logits=torch.zeros(1, 3),
        )
        from maths_ai.gnn_inference.atp_lean_gnn.inference import _local_names_by_fv_label
        candidate, action = reasoner._decode(
            sample, dag, _local_names_by_fv_label(dag), str(data.graph_fingerprint)
        )
        self.assertIsNone(candidate)
        self.assertEqual(action.arg_indices, (oob,))
        self.assertEqual(action.graph_fingerprint, str(data.graph_fingerprint))


class RolloutTests(unittest.TestCase):
    def _prove(self, reasoner):
        return asyncio.run(reasoner.prove(GOAL_EXPR, hypotheses=HYPS))

    def test_qed_rollout_stash_migrates_to_edge_ids(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        result = self._prove(reasoner)

        self.assertTrue(result.graph.is_solved())
        self.assertGreater(len(result.edge_actions), 0)
        for edge_id, action in result.edge_actions.items():
            edge = result.graph.edges[edge_id]
            self.assertEqual(edge.tactic.tactic_name, reasoner.id_to_tactic[action.tactic_id])

    def test_multiplicity_accounts_for_every_draw(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        result = self._prove(reasoner)
        total = sum(a.multiplicity for a in result.edge_actions.values())
        total += sum(f.action.multiplicity for f in result.failure_actions)
        # One node expanded (root, immediately solved): every i.i.d. draw is accounted for.
        self.assertEqual(total, reasoner.top_k_tactics)

    def test_rejected_samples_flush_to_failure_actions(self):
        reasoner, _model, _vocab = _make_reasoner(_RejectExecutor())
        result = self._prove(reasoner)

        self.assertFalse(result.graph.is_solved())
        self.assertEqual(len(result.edge_actions), 0)
        self.assertGreater(len(result.failure_actions), 0)
        total = sum(f.action.multiplicity for f in result.failure_actions)
        self.assertEqual(total, reasoner.top_k_tactics)

    def test_onpolicy_edge_filter(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        result = self._prove(reasoner)
        transitions = extract_transitions(
            result.graph, RewardConfig(step_penalty=0.0),
            edge_ids=list(result.edge_actions.keys()),
        )
        self.assertEqual(len(transitions), len(result.edge_actions))
        for t in transitions:
            self.assertIn(t.edge_id, result.edge_actions)


class OnPolicyLossTests(unittest.TestCase):
    def test_evaluate_actions_gradient_reaches_pointer(self):
        reasoner, model, _vocab = _make_reasoner(_QEDExecutor())
        goal = structured_goal(GOAL_EXPR, HYPS)
        dag, data = reasoner.dag_featurize(goal)
        batch = Batch.from_data_list([data])
        hyp_idx = next(i for i, n in enumerate(dag.nodes) if n.label == "Hyp")

        model.train()
        tactic_logp, arg_logp, _entropy, _values, _logits = model.evaluate_actions(
            batch, torch.tensor([2]), torch.tensor([[hyp_idx]])
        )
        loss = -(tactic_logp + arg_logp).sum()
        loss.backward()
        grad = model.argument_selector.query_proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_train_step_onpolicy_updates_params(self):
        reasoner, model, _vocab = _make_reasoner(_QEDExecutor())
        result = asyncio.run(reasoner.prove(GOAL_EXPR, hypotheses=HYPS))

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
        result = asyncio.run(reasoner.prove(GOAL_EXPR, hypotheses=HYPS))
        self.assertGreater(len(result.failure_actions), 0)

        loss_result = compute_onpolicy_loss(
            model, [], [], {}, result.failure_actions, reasoner.dag_featurize_data,
        )
        self.assertIsNotNone(loss_result)
        loss, metrics = loss_result
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["critic_loss"], 0.0)  # a failed ACTION carries no state value
        self.assertEqual(metrics["num_transitions"], 0.0)
        self.assertGreater(metrics["num_failures"], 0.0)

    def test_multiplicity_weights_actor_term(self):
        reasoner, model, _vocab = _make_reasoner(_QEDExecutor())
        result = asyncio.run(reasoner.prove(GOAL_EXPR, hypotheses=HYPS))
        transitions = extract_transitions(
            result.graph, RewardConfig(step_penalty=0.0),
            edge_ids=list(result.edge_actions.keys()),
        )
        goal = structured_goal(GOAL_EXPR, HYPS)

        # Success rows (return ≈ 1) against a failure row (return ≈ 0) give a nonzero
        # advantage contrast, so re-weighting the failure by m must change the loss.
        def loss_with(m: int) -> float:
            model.eval()  # deterministic forward so only multiplicity differs
            failures = [FailureRecord(
                goal=goal,
                action=EdgeAction(
                    tactic_id=1,
                    multiplicity=m,
                    graph_fingerprint=str(reasoner.dag_featurize(goal)[1].graph_fingerprint),
                ),
            )]
            out = compute_onpolicy_loss(
                model, transitions, [], result.edge_actions, failures,
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
        return asyncio.run(reasoner.prove(GOAL_EXPR, hypotheses=HYPS))

    def test_default_legacy_loop_never_touches_visit_stats(self):
        # selection_policy="legacy" (default): the best-first loop runs
        # verbatim and no edge accumulates visit statistics — the regression
        # guarantee.
        executor = _SubgoalExecutor({self.ROOT: ["A", "B"]})
        reasoner, _model, _vocab = _make_reasoner(
            executor, reasoner_cls=_GoalTrackingReasoner
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
                executor, reasoner_cls=_GoalTrackingReasoner, **kwargs
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
                executor, reasoner_cls=_GoalTrackingReasoner,
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
            executor, top_k=1, reasoner_cls=_GoalTrackingReasoner,
            selection_policy="puct", num_simulations=1,
        )
        result = self._prove(reasoner)
        self.assertFalse(result.graph.is_solved())
        expanded = [
            n for n in result.graph.nodes.values() if n.outgoing_edge_ids or n.exhausted
        ]
        self.assertEqual([n.id for n in expanded], [result.graph.root_id])

    def test_multi_sim_accumulates_stats_and_releases_virtual_loss(self):
        executor = _SubgoalExecutor({self.ROOT: ["A", "B"]})
        reasoner, _model, _vocab = _make_reasoner(
            executor, top_k=1, reasoner_cls=_GoalTrackingReasoner,
            selection_policy="puct", num_simulations=6, sim_batch_size=2,
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
            executor, reasoner_cls=_GoalTrackingReasoner,
            selection_policy="puct", num_simulations=6, sim_batch_size=2,
        )
        result = asyncio.run(
            reasoner.prove(GOAL_EXPR, hypotheses=HYPS, deadline=time.monotonic() - 1.0)
        )
        # The loop stopped before the first simulation batch: only the root
        # exists, unexpanded, and the partial graph came back instead of an error.
        self.assertEqual(len(result.graph.nodes), 1)
        self.assertFalse(result.graph.is_solved())
        self.assertEqual(executor.applications, 0)

    def test_batched_expansion_attributes_failures_to_the_right_node(self):
        # Regression for the per-node stash (Decision 1.1): one simulation's
        # leaves A and B are proposed in one batched call; A's applications
        # succeed (QED) while B's are rejected — every failure record must
        # carry B's goal, never A's or the root's.
        class _SelectiveExecutor(_SubgoalExecutor):
            async def apply(self, server, state, tactic):
                self.applications += 1
                if self.current_goal == "B":
                    return TacticOutcome(success=False, subgoals=[], error="rejected")
                subgoal_exprs = self.depth_map.get(self.current_goal, [])
                subgoals = [structured_goal(e, HYPS) for e in subgoal_exprs]
                for subgoal in subgoals:
                    self.server.goals_by_expression[subgoal.replay_spec().expression] = subgoal
                return TacticOutcome(success=True, subgoals=subgoals)

        executor = _SelectiveExecutor({self.ROOT: ["A", "B"]})
        reasoner, _model, _vocab = _make_reasoner(
            executor, top_k=2, reasoner_cls=_GoalTrackingReasoner,
            selection_policy="puct", num_simulations=6, sim_batch_size=2,
        )
        result = self._prove(reasoner)
        self.assertGreater(len(result.failure_actions), 0)
        for record in result.failure_actions:
            self.assertEqual(record.goal.expression, "B")
        # A's QED edge is in the on-policy join table with a matching tactic.
        solved_edges = [
            eid for eid, e in result.graph.edges.items()
            if result.graph.nodes[e.source_id].goal.expression == "A"
        ]
        self.assertTrue(any(eid in result.edge_actions for eid in solved_edges))

    def _two_leaf_graph(self):
        root = structured_goal(self.ROOT, HYPS)
        graph = ProofHypergraph(root)
        edge = graph.add_edge(
            graph.root_id,
            TacticCandidate(tactic_name="intro", arguments=[], probability=1.0),
            [(structured_goal("A", HYPS), None), (structured_goal("B", HYPS), None)],
        )
        return graph, edge.child_ids

    def test_puct_materializes_leaves_before_policy_forward(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        graph, leaf_ids = self._two_leaf_graph()
        events = []

        async def materialize(self, graph, node):
            events.append(("materialize", node.id))
            return MaterializedGoal(
                goal=node.goal,
                state=object(),
                server_epoch=self._server_epoch,
                fingerprint=node.goal.state_fingerprint(),
            )

        def propose(self, nodes):
            events.append(("policy", [node.id for node in nodes]))
            return [[] for _ in nodes]

        reasoner._materialize_for_expansion = MethodType(materialize, reasoner)
        reasoner.predict_next_tactics_for_nodes = MethodType(propose, reasoner)
        asyncio.run(reasoner._expand_leaves(graph, leaf_ids))
        self.assertEqual(events[:2], [("materialize", leaf_ids[0]), ("materialize", leaf_ids[1])])
        self.assertEqual(events[2], ("policy", leaf_ids))

    def test_failed_materialization_does_not_shift_policy_rows(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        graph, leaf_ids = self._two_leaf_graph()
        proposed = []

        async def materialize(self, graph, node):
            if node.id == leaf_ids[0]:
                return None
            return MaterializedGoal(
                goal=node.goal,
                state=object(),
                server_epoch=self._server_epoch,
                fingerprint=node.goal.state_fingerprint(),
            )

        def propose(self, nodes):
            proposed.extend(node.id for node in nodes)
            return [[] for _ in nodes]

        reasoner._materialize_for_expansion = MethodType(materialize, reasoner)
        reasoner.predict_next_tactics_for_nodes = MethodType(propose, reasoner)
        asyncio.run(reasoner._expand_leaves(graph, leaf_ids))
        self.assertEqual(proposed, [leaf_ids[1]])

    def test_restart_resamples_stale_leaf_before_execution(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        graph, leaf_ids = self._two_leaf_graph()
        policy_rows = []
        executions = []
        calls = 0

        async def materialize(self, graph, node):
            return MaterializedGoal(
                goal=node.goal,
                state=f"state-{self._server_epoch}",
                server_epoch=self._server_epoch,
                fingerprint=node.goal.state_fingerprint(),
            )

        def propose(self, nodes):
            nonlocal calls
            calls += 1
            policy_rows.append((calls, [node.id for node in nodes]))
            if calls == 1:
                self._server_epoch += 1
            return [[TacticCandidate(tactic_name="skip", arguments=[], probability=1.0)] for _ in nodes]

        async def execute(self, graph, node, candidates, *, materialized):
            executions.append((node.id, materialized.server_epoch, materialized.state))

        reasoner._materialize_for_expansion = MethodType(materialize, reasoner)
        reasoner.predict_next_tactics_for_nodes = MethodType(propose, reasoner)
        reasoner._execute_and_link = MethodType(execute, reasoner)
        asyncio.run(reasoner._expand_leaves(graph, [leaf_ids[0]]))
        self.assertEqual(policy_rows, [(1, [leaf_ids[0]]), (2, [leaf_ids[0]])])
        self.assertEqual(executions, [(leaf_ids[0], 1, "state-1")])


class PLNDisabledTests(unittest.TestCase):
    """use_pln=False: no PLN objects exist, no petta subprocess, terminal-only reward."""

    def _prove(self, reasoner):
        return asyncio.run(reasoner.prove(GOAL_EXPR, hypotheses=HYPS))

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
            reasoner_cls=_GoalTrackingReasoner,
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
        self.assertEqual(root.note, "executor rejected every candidate tactic")

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
            executor, reasoner_cls=_GoalTrackingReasoner, top_k=1, use_pln=False
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
