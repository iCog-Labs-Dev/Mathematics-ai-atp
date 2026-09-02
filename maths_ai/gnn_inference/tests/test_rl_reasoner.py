from __future__ import annotations

import asyncio
import unittest

import torch
from pantograph.server import ServerError
from torch.optim import AdamW
from torch_geometric.data import Batch

from maths_ai.data_models.proof_components import Goal, STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import (
    NodeClosureReason,
    SearchEndReason,
    TacticOutcome,
)
from maths_ai.pln_inference.model import PLNResult

from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import (
    ActionSample,
    ActorCriticWithArgsClassifier,
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
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    extract_actor_transitions,
    extract_critic_samples,
)
from maths_ai.gnn_inference.tests.model_helpers import actor_critic


# ---------------------------------------------------------------------------
# Fakes: no Lean/Pantograph backend, no petta subprocess
# ---------------------------------------------------------------------------


class _FakeGoalState:
    goals: list = []


class _FakeServer:
    def __init__(self):
        self.proc = object()

    async def goal_start_async(self, expression):
        return _FakeGoalState()

    async def goal_tactic_async(self, state, tactic):
        return _FakeGoalState()


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


class _ElaborationServer(_FakeServer):
    async def goal_start_async(self, expression):
        raise ServerError("cannot elaborate reconstructed goal")


class _ElaborationExecutor:
    def __init__(self):
        self.server = _ElaborationServer()

    async def apply(self, server, state, tactic):
        raise AssertionError("tactics must not run when goal elaboration fails")


class _InfrastructureFailureExecutor:
    def __init__(self):
        self.server = _FakeServer()

    async def apply(self, server, state, tactic):
        raise BrokenPipeError("Pantograph pipe closed")


class _StubPLN:
    """PLN stand-in: a real (non-fallback) low score, no subprocess."""

    async def evaluate_async(self, expression, hypotheses=None, **kwargs):
        return PLNResult(stv=STV(strength=0.1, confidence=1.0), status="ok", is_fallback=False)


TACTIC_VOCAB = {"trivial": 0, "intro": 1, "exact": 2}
GOAL_EXPR = "p → p"
HYPS = ["p : Prop"]


def _make_reasoner(executor, *, top_k=3, seed=0):
    torch.manual_seed(seed)
    goal = Goal(expression=GOAL_EXPR, hypotheses=HYPS)
    # Vocab from the test goal's own DAG so featurization is non-degenerate.
    from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
    from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
    from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import goal_to_state

    node_vocab = build_vocab([proof_state_to_dag(goal_to_state(goal))])
    model = actor_critic(len(node_vocab), len(TACTIC_VOCAB))
    reasoner = RLHybridReasoner(
        model,
        node_vocab,
        TACTIC_VOCAB,
        executor=executor,
        top_k_tactics=top_k,
        max_depth=3,
        max_nodes=20,
    )
    reasoner.petta_chainer = _StubPLN()  # keep unit tests free of the petta subprocess
    return reasoner, model, node_vocab


class DecodeTests(unittest.TestCase):
    def test_decode_tactic_and_argument(self):
        reasoner, _model, node_vocab = _make_reasoner(_QEDExecutor())
        goal = Goal(expression=GOAL_EXPR, hypotheses=HYPS)
        dag, _data = reasoner.dag_featurize(goal)

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
        candidate, action = reasoner._decode(sample, dag)
        self.assertEqual(candidate.tactic_name, "exact")
        self.assertEqual(candidate.arguments, ["p"])
        self.assertEqual(action.tactic_id, 2)
        self.assertEqual(action.arg_indices, (hyp_idx,))

    def test_decode_out_of_range_argument_dropped_from_command(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        goal = Goal(expression=GOAL_EXPR, hypotheses=HYPS)
        dag, _data = reasoner.dag_featurize(goal)

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
        candidate, action = reasoner._decode(sample, dag)
        self.assertEqual(candidate.arguments, [])           # not sent to Lean
        self.assertEqual(action.arg_indices, (oob,))        # but kept for the recompute


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
        transitions = extract_actor_transitions(
            result.graph, RewardConfig(step_penalty=0.0),
            edge_ids=list(result.edge_actions.keys()),
        )
        self.assertEqual(len(transitions), len(result.edge_actions))
        for t in transitions:
            self.assertIn(t.edge_id, result.edge_actions)

    def test_elaboration_failure_discards_unexecuted_actions(self):
        reasoner, _model, _vocab = _make_reasoner(_ElaborationExecutor())
        result = self._prove(reasoner)

        self.assertEqual(result.graph.root.closure_reason, NodeClosureReason.ELABORATION_ERROR)
        self.assertEqual(result.edge_actions, {})
        self.assertEqual(result.failure_actions, [])
        self.assertEqual(extract_actor_transitions(result.graph), [])
        self.assertEqual(extract_critic_samples(result.graph), [])

    def test_infrastructure_failure_discards_unobserved_actions(self):
        reasoner, _model, _vocab = _make_reasoner(_InfrastructureFailureExecutor())
        result = self._prove(reasoner)

        self.assertEqual(result.graph.search_end_reason, SearchEndReason.EXTERNAL_ABORT)
        self.assertEqual(result.graph.root.closure_reason, NodeClosureReason.INFRASTRUCTURE_FAILURE)
        self.assertEqual(result.edge_actions, {})
        self.assertEqual(result.failure_actions, [])
        self.assertEqual(extract_actor_transitions(result.graph), [])
        self.assertEqual(extract_critic_samples(result.graph), [])

    def test_cancellation_clears_pending_actions(self):
        class _BlockingExecutor:
            def __init__(self):
                self.server = _FakeServer()
                self.started = asyncio.Event()

            async def apply(self, server, state, tactic):
                self.started.set()
                await asyncio.Event().wait()

        executor = _BlockingExecutor()
        reasoner, _model, _vocab = _make_reasoner(executor)

        async def cancel_search():
            task = asyncio.create_task(reasoner.prove(GOAL_EXPR, hypotheses=HYPS))
            await executor.started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_search())
        self.assertEqual(reasoner._pending, {})
        self.assertIsNone(reasoner._pending_goal)


class OnPolicyLossTests(unittest.TestCase):
    def test_evaluate_actions_gradient_reaches_pointer(self):
        reasoner, model, _vocab = _make_reasoner(_QEDExecutor())
        goal = Goal(expression=GOAL_EXPR, hypotheses=HYPS)
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
        transitions = extract_actor_transitions(
            result.graph, RewardConfig(step_penalty=0.0),
            edge_ids=list(result.edge_actions.keys()),
        )
        critic_samples = extract_critic_samples(result.graph)
        goal = Goal(expression=GOAL_EXPR, hypotheses=HYPS)

        # Success rows (return ≈ 1) against a failure row (return ≈ 0) give a nonzero
        # advantage contrast, so re-weighting the failure by m must change the loss.
        def loss_with(m: int) -> float:
            model.eval()  # deterministic forward so only multiplicity differs
            failures = [FailureRecord(goal=goal, action=EdgeAction(tactic_id=1, multiplicity=m))]
            out = compute_onpolicy_loss(
                model, transitions, critic_samples, result.edge_actions, failures,
                reasoner.dag_featurize_data, reward_cfg=RewardConfig(step_penalty=0.0),
            )
            self.assertIsNotNone(out)
            return float(out[0].item())

        self.assertNotAlmostEqual(loss_with(1), loss_with(3))


class PLNDisabledTests(unittest.TestCase):
    """Tests for the use_pln=False kill switch in RLHybridReasoner."""

    def _make_no_pln(self, executor, *, top_k=3, seed=0):
        """Build an RLHybridReasoner with use_pln=False.

        _make_reasoner installs a _StubPLN to keep the petta subprocess out of
        unit tests; here we intentionally do NOT do that — the point is that
        petta_chainer is None and nothing dereferences it.
        """
        torch.manual_seed(seed)
        goal = Goal(expression=GOAL_EXPR, hypotheses=HYPS)
        from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
        from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
        from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import goal_to_state

        node_vocab = build_vocab([proof_state_to_dag(goal_to_state(goal))])
        model = actor_critic(len(node_vocab), len(TACTIC_VOCAB))
        reasoner = RLHybridReasoner(
            model,
            node_vocab,
            TACTIC_VOCAB,
            executor=executor,
            top_k_tactics=top_k,
            max_depth=3,
            max_nodes=20,
            use_pln=False,
        )
        return reasoner, model, node_vocab

    def _prove(self, reasoner):
        return asyncio.run(reasoner.prove(GOAL_EXPR, hypotheses=HYPS))

    def test_pln_objects_are_none(self):
        """use_pln=False means neither petta_chainer nor dts_sampler is constructed."""
        reasoner, _, _ = self._make_no_pln(_QEDExecutor())
        self.assertIsNone(reasoner.petta_chainer)
        self.assertIsNone(reasoner.dts_sampler)

    def test_subgoal_nodes_have_none_stv_in_executor_order(self):
        """With PLN off, every Lean subgoal is retained in executor order."""
        sg1 = Goal(expression="p", hypotheses=HYPS)
        sg2 = Goal(expression="True", hypotheses=[])
        sg3 = Goal(expression="False", hypotheses=[])

        class _TwoSubgoalThenQEDExecutor:
            """First apply: returns two subgoals; subsequent applies: QED."""
            def __init__(self):
                self.server = _FakeServer()
                self._calls = 0

            async def apply(self, server, state, tactic):
                self._calls += 1
                if self._calls == 1:
                    return TacticOutcome(success=True, subgoals=[sg1, sg2, sg3])
                return TacticOutcome(success=True, subgoals=[])

        executor = _TwoSubgoalThenQEDExecutor()
        reasoner, _, _ = self._make_no_pln(executor, top_k=1)
        result = self._prove(reasoner)

        # Verify subgoal children carry stv=None (PLN never scored them).
        child_stvs = [
            result.graph.nodes[nid].stv
            for nid in result.graph.nodes
            if result.graph.nodes[nid].incoming_edge_id is not None
        ]
        self.assertTrue(all(stv is None for stv in child_stvs),
                        f"expected all stv=None, got {child_stvs}")
        root_edge = next(edge for edge in result.graph.edges.values() if edge.source_id == 0)
        child_goals = [result.graph.nodes[node_id].goal for node_id in root_edge.child_ids]
        self.assertEqual(child_goals, [sg1, sg2, sg3])

    def test_no_pln_fallback_qed_on_total_rejection(self):
        """With PLN off, a fully-rejected node is exhausted, not fake-closed."""
        reasoner, _, _ = self._make_no_pln(_RejectExecutor())
        result = self._prove(reasoner)

        self.assertFalse(result.graph.is_solved())
        pln_fb_edges = [
            e for e in result.graph.edges.values()
            if e.tactic.tactic_name == "PLN_fallback"
        ]
        self.assertEqual(pln_fb_edges, [], "PLN_fallback edge must not exist when use_pln=False")
        root = result.graph.root
        self.assertTrue(root.exhausted)
        self.assertIn("executor rejected", root.note or "")

    def test_reward_is_terminal_only_when_pln_off(self):
        """With PLN off every node has stv=None, so edge_shaping==0 for all edges."""
        from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import (
            edge_shaped_reward,
            edge_terminal_reward,
            RewardConfig,
        )
        reasoner, _, _ = self._make_no_pln(_QEDExecutor())
        result = self._prove(reasoner)
        cfg = RewardConfig(step_penalty=0.0)
        for edge in result.graph.edges.values():
            shaped = edge_shaped_reward(edge, result.graph, cfg)
            terminal = edge_terminal_reward(edge, result.graph, cfg)
            self.assertAlmostEqual(shaped, terminal, places=6,
                                   msg=f"edge {edge.id}: shaping should be 0 but got {shaped - terminal}")

    def test_rank_subgoals_guard_raises(self):
        """Calling rank_subgoals on a use_pln=False reasoner raises RuntimeError."""
        reasoner, _, _ = self._make_no_pln(_QEDExecutor())
        goal = Goal(expression=GOAL_EXPR, hypotheses=HYPS)
        from maths_ai.data_models.proof_components import TacticCandidate
        tactic = TacticCandidate(tactic_name="intro", arguments=[], probability=1.0)
        with self.assertRaises(RuntimeError):
            asyncio.run(reasoner.rank_subgoals(goal.expression, [goal], tactic))


if __name__ == "__main__":
    unittest.main()
