from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace

import torch
from pantograph_async import (
    PantographGoalRejected,
    PantographTransportError,
)
from maths_ai.gnn_inference.tests.model_helpers import ScopedFakeClient
from torch.optim import AdamW
from torch_geometric.data import Batch

from maths_ai.data_models.proof_components import STV, TacticCandidate
from maths_ai.lean_replay import ReplayFailure, ReplayInputError, seed_from_text
from maths_ai.hybrid_reasoner.hypergraph import (
    NodeClosureReason,
    SearchEndReason,
)
from maths_ai.hybrid_reasoner.retained_executor import RetainedTacticExecutor
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
from maths_ai.gnn_inference.tests.model_helpers import (
    ScriptedExecutor,
    actor_critic,
    pantograph_goal,
    structured_goal,
)


# ---------------------------------------------------------------------------
# Fakes: no Lean/Pantograph backend, no petta subprocess
# ---------------------------------------------------------------------------


class _FakeGoalState:
    def __init__(self, goal=None):
        goal = goal or structured_goal(GOAL_EXPR, HYPS)
        self.goals = [pantograph_goal(goal)]


class _FakeServer(ScopedFakeClient):
    def __init__(self):
        self.goals_by_expression = {}
        self._last_expression = None

    async def _start_goal(self, expression):
        self._last_expression = expression
        return _FakeGoalState()

    async def _apply_tactic(self, state, tactic):
        return _FakeGoalState(self.goals_by_expression.get(self._last_expression))


def _QEDExecutor():
    """Every tactic application succeeds with no child obligations."""
    return ScriptedExecutor(lambda _occurrence, _command: [], server=_FakeServer())


def _RejectExecutor():
    """Every tactic application fails (Lean rejects the tactic)."""
    return ScriptedExecutor(lambda _occurrence, _command: None, server=_FakeServer())


class _ElaborationServer(_FakeServer):
    async def _start_goal(self, expression):
        raise PantographGoalRejected("cannot elaborate reconstructed goal")


class _ElaborationExecutor(RetainedTacticExecutor):
    def __init__(self):
        self.server = _ElaborationServer()

    async def apply(self, registry, occurrence, command, *, holder="operation"):
        raise AssertionError("tactics must not run when goal elaboration fails")


class _InfrastructureFailureExecutor(RetainedTacticExecutor):
    def __init__(self):
        self.server = _FakeServer()

    async def apply(self, registry, occurrence, command, *, holder="operation"):
        raise PantographTransportError("Pantograph pipe closed")


class _StubPLN:
    """PLN stand-in: a real (non-fallback) low score, no subprocess."""

    async def evaluate_async(self, expression, hypotheses=None, **kwargs):
        return PLNResult(
            stv=STV(strength=0.1, confidence=1.0), status="ok", is_fallback=False
        )


TACTIC_VOCAB = {"trivial": 0, "intro": 1, "exact": 2}
GOAL_EXPR = "p → p"
HYPS = ["p : Prop"]


def _make_reasoner(executor, *, top_k=3, seed=0):
    torch.manual_seed(seed)
    goal = structured_goal(GOAL_EXPR, HYPS)
    # Vocab from the test goal's own DAG so featurization is non-degenerate.
    from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
    from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab

    node_vocab = build_vocab([model_goal_to_dag(goal)])
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
        from maths_ai.gnn_inference.atp_lean_gnn.inference import local_declarations

        candidate, action = reasoner._decode(
            sample, dag, local_declarations(goal), data.graph_fingerprint
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
        from maths_ai.gnn_inference.atp_lean_gnn.inference import local_declarations

        candidate, action = reasoner._decode(
            sample, dag, local_declarations(goal), data.graph_fingerprint
        )
        self.assertIsNone(candidate)
        self.assertEqual(action.arg_indices, (oob,))

    def test_anonymous_local_is_unplayable_but_sample_is_retained(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        goal = structured_goal(GOAL_EXPR, ["_ : Prop"])
        dag, data = reasoner.dag_featurize(goal)
        hyp_idx = next(i for i, node in enumerate(dag.nodes) if node.label == "Hyp")
        sample = ActionSample(
            tactic_action=torch.tensor([2]),
            tactic_logp=torch.tensor([-0.5]),
            tactic_entropy=torch.tensor([1.0]),
            arg_actions=[torch.tensor([hyp_idx])],
            arg_logp=torch.tensor([-0.3]),
            value=torch.tensor([0.0]),
            tactic_logits=torch.zeros(1, 3),
        )
        from maths_ai.gnn_inference.atp_lean_gnn.inference import local_declarations

        candidate, action = reasoner._decode(
            sample, dag, local_declarations(goal), data.graph_fingerprint
        )
        self.assertIsNone(candidate)
        self.assertEqual(action.arg_indices, (hyp_idx,))
        self.assertEqual(action.graph_fingerprint, data.graph_fingerprint)


class DeclarationResolutionTests(unittest.TestCase):
    """A sampled local resolves to the live declaration it denotes, not by name."""

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

        def decode(index):
            sample = ActionSample(
                tactic_action=torch.tensor([2]),  # "exact"
                tactic_logp=torch.tensor([-0.5]),
                tactic_entropy=torch.tensor([1.0]),
                arg_actions=[torch.tensor([index])],
                arg_logp=torch.tensor([-0.3]),
                value=torch.tensor([0.0]),
                tactic_logits=torch.zeros(1, 3),
            )
            candidate, _ = reasoner._decode(sample, dag, declarations, "fingerprint")
            return candidate.arguments

        for label, live in (("FV5", "x_1"), ("FV6", "x")):
            fv = next(i for i, n in enumerate(dag.nodes) if n.label == label)
            hyp = next(
                i
                for i, n in enumerate(dag.nodes)
                if n.label == "Hyp" and n.children and n.children[0] == fv
            )
            self.assertEqual(decode(hyp), [live])
            self.assertEqual(decode(fv), [live])

    def test_decoded_arguments_keep_the_chosen_declaration_id(self):
        from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
        from maths_ai.gnn_inference.atp_lean_gnn.inference import (
            local_declaration_ids,
            local_declarations,
        )
        from maths_ai.gnn_inference.tests.test_model_identity import (
            after_intro_x,
            outer_aliased,
        )

        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        child = after_intro_x(outer_aliased())
        dag = model_goal_to_dag(child)
        ids = local_declaration_ids(child)
        for label in ("FV5", "FV6"):
            fv = next(i for i, n in enumerate(dag.nodes) if n.label == label)
            sample = ActionSample(
                tactic_action=torch.tensor([2]),
                tactic_logp=torch.tensor([-0.5]),
                tactic_entropy=torch.tensor([1.0]),
                arg_actions=[torch.tensor([fv])],
                arg_logp=torch.tensor([-0.3]),
                value=torch.tensor([0.0]),
                tactic_logits=torch.zeros(1, 3),
            )
            candidate, _ = reasoner._decode(
                sample, dag, local_declarations(child), "fp", declaration_ids=ids
            )
            (ref,) = candidate.argument_refs
            self.assertEqual((ref.kind, ref.internal_name), ("local", ids[label]))
            self.assertEqual(ref.text, candidate.arguments[0])


class RolloutTests(unittest.TestCase):
    def _prove(self, reasoner):
        return asyncio.run(reasoner.prove(seed_from_text(goal=GOAL_EXPR, hypotheses=HYPS)))

    def test_qed_rollout_stash_migrates_to_edge_ids(self):
        reasoner, _model, _vocab = _make_reasoner(_QEDExecutor())
        result = self._prove(reasoner)

        self.assertTrue(result.graph.is_solved())
        self.assertGreater(len(result.edge_actions), 0)
        for edge_id, action in result.edge_actions.items():
            edge = result.graph.edges[edge_id]
            self.assertEqual(
                edge.tactic.tactic_name, reasoner.id_to_tactic[action.tactic_id]
            )

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
            result.graph,
            RewardConfig(step_penalty=0.0),
            edge_ids=list(result.edge_actions.keys()),
        )
        # Only the certified witness is positive evidence; the other closing
        # attempts are candidates that no replay certified, so they stay unknown.
        witnessed = result.graph.witnessed_edges()
        self.assertEqual(len(witnessed), 1)
        self.assertEqual({t.edge_id for t in transitions}, witnessed)
        for t in transitions:
            self.assertIn(t.edge_id, result.edge_actions)

    def test_elaboration_failure_discards_unexecuted_actions(self):
        reasoner, _model, _vocab = _make_reasoner(_ElaborationExecutor())
        with self.assertRaises(ReplayInputError) as error:
            self._prove(reasoner)
        self.assertIs(error.exception.reason, ReplayFailure.LEAN_REJECTED)
        self.assertIsInstance(error.exception.__cause__, PantographGoalRejected)
        self.assertEqual(reasoner._pending, {})
        self.assertIsNone(reasoner._result)

    def test_infrastructure_failure_discards_unobserved_actions(self):
        reasoner, _model, _vocab = _make_reasoner(_InfrastructureFailureExecutor())
        with self.assertRaises(PantographTransportError) as error:
            self._prove(reasoner)
        graph = error.exception.graph
        self.assertEqual(graph.search_end_reason, SearchEndReason.EXTERNAL_ABORT)
        self.assertEqual(
            graph.root.closure_reason, NodeClosureReason.INFRASTRUCTURE_FAILURE
        )
        self.assertEqual(reasoner._pending, {})
        self.assertIsNone(reasoner._result)
        self.assertEqual(extract_actor_transitions(graph), [])
        self.assertEqual(extract_critic_samples(graph), [])

    def test_cancellation_clears_pending_actions(self):
        class _BlockingExecutor(RetainedTacticExecutor):
            def __init__(self):
                self.server = _FakeServer()
                self.started = asyncio.Event()

            async def apply(self, registry, occurrence, command, *, holder="operation"):
                self.started.set()
                await asyncio.Event().wait()

        executor = _BlockingExecutor()
        reasoner, _model, _vocab = _make_reasoner(executor)

        async def cancel_search():
            task = asyncio.create_task(reasoner.prove(seed_from_text(goal=GOAL_EXPR, hypotheses=HYPS)))
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
        result = asyncio.run(reasoner.prove(seed_from_text(goal=GOAL_EXPR, hypotheses=HYPS)))

        optimizer = AdamW(model.parameters(), lr=0.01)
        before = [p.detach().clone() for p in model.parameters()]
        metrics = train_step_onpolicy(
            model,
            optimizer,
            [result],
            reasoner.dag_featurize_data,
            reward_cfg=RewardConfig(step_penalty=0.0),
            bc_weight=0.1,
        )
        self.assertGreater(metrics["num_transitions"], 0.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        after = list(model.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        self.assertTrue(
            changed, "on-policy training step did not update any parameters"
        )

    def test_failure_only_batch_is_actor_only(self):
        reasoner, model, _vocab = _make_reasoner(_RejectExecutor())
        result = asyncio.run(reasoner.prove(seed_from_text(goal=GOAL_EXPR, hypotheses=HYPS)))
        self.assertGreater(len(result.failure_actions), 0)

        loss_result = compute_onpolicy_loss(
            model,
            [],
            [],
            {},
            result.failure_actions,
            reasoner.dag_featurize_data,
        )
        self.assertIsNotNone(loss_result)
        loss, metrics = loss_result
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            metrics["critic_loss"], 0.0
        )  # a failed ACTION carries no state value
        self.assertEqual(metrics["num_transitions"], 0.0)
        self.assertGreater(metrics["num_failures"], 0.0)

    def test_graph_fingerprint_mismatch_fails_before_policy_recompute(self):
        reasoner, model, _vocab = _make_reasoner(_QEDExecutor())
        result = asyncio.run(reasoner.prove(seed_from_text(goal=GOAL_EXPR, hypotheses=HYPS)))
        transitions = extract_actor_transitions(
            result.graph,
            RewardConfig(step_penalty=0.0),
            edge_ids=list(result.edge_actions),
        )
        edge_id = next(iter(result.edge_actions))
        actions = dict(result.edge_actions)
        actions[edge_id] = replace(actions[edge_id], graph_fingerprint="different")
        with self.assertRaisesRegex(ValueError, "graph changed"):
            compute_onpolicy_loss(
                model,
                transitions,
                extract_critic_samples(result.graph),
                actions,
                [],
                reasoner.dag_featurize_data,
            )

    def test_multiplicity_weights_actor_term(self):
        reasoner, model, _vocab = _make_reasoner(_QEDExecutor())
        result = asyncio.run(reasoner.prove(seed_from_text(goal=GOAL_EXPR, hypotheses=HYPS)))
        transitions = extract_actor_transitions(
            result.graph,
            RewardConfig(step_penalty=0.0),
            edge_ids=list(result.edge_actions.keys()),
        )
        critic_samples = extract_critic_samples(result.graph)
        goal = structured_goal(GOAL_EXPR, HYPS)

        # Success rows (return ≈ 1) against a failure row (return ≈ 0) give a nonzero
        # advantage contrast, so re-weighting the failure by m must change the loss.
        def loss_with(m: int) -> float:
            model.eval()  # deterministic forward so only multiplicity differs
            failures = [
                FailureRecord(goal=goal, action=EdgeAction(tactic_id=1, multiplicity=m))
            ]
            out = compute_onpolicy_loss(
                model,
                transitions,
                critic_samples,
                result.edge_actions,
                failures,
                reasoner.dag_featurize_data,
                reward_cfg=RewardConfig(step_penalty=0.0),
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
        goal = structured_goal(GOAL_EXPR, HYPS)
        from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
        from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab

        node_vocab = build_vocab([model_goal_to_dag(goal)])
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
        return asyncio.run(reasoner.prove(seed_from_text(goal=GOAL_EXPR, hypotheses=HYPS)))

    def test_pln_objects_are_none(self):
        """use_pln=False means neither petta_chainer nor dts_sampler is constructed."""
        reasoner, _, _ = self._make_no_pln(_QEDExecutor())
        self.assertIsNone(reasoner.petta_chainer)
        self.assertIsNone(reasoner.dts_sampler)

    def test_subgoal_nodes_have_none_stv_in_executor_order(self):
        """With PLN off, every Lean subgoal is retained in executor order."""
        sg1 = structured_goal("p", HYPS)
        sg2 = structured_goal("True")
        sg3 = structured_goal("False")

        calls = []

        def script(_occurrence, _command):
            """First apply: three child obligations; later applies: closed."""
            calls.append(_command)
            return [sg1, sg2, sg3] if len(calls) == 1 else []

        executor = ScriptedExecutor(script, server=_FakeServer())
        reasoner, _, _ = self._make_no_pln(executor, top_k=1)
        reasoner.max_nodes = 4
        result = self._prove(reasoner)

        # Verify subgoal children carry stv=None (PLN never scored them).
        child_stvs = [
            result.graph.nodes[nid].stv
            for nid in result.graph.nodes
            if result.graph.nodes[nid].incoming_edge_id is not None
        ]
        self.assertTrue(
            all(stv is None for stv in child_stvs),
            f"expected all stv=None, got {child_stvs}",
        )
        root_edge = next(
            edge for edge in result.graph.edges.values() if edge.source_id == 0
        )
        child_goals = [
            result.graph.nodes[node_id].goal for node_id in root_edge.child_ids
        ]
        self.assertEqual(child_goals, [sg1, sg2, sg3])

    def test_no_pln_fallback_qed_on_total_rejection(self):
        """With PLN off, a fully-rejected node is exhausted, not fake-closed."""
        reasoner, _, _ = self._make_no_pln(_RejectExecutor())
        result = self._prove(reasoner)

        self.assertFalse(result.graph.is_solved())
        pln_fb_edges = [
            e
            for e in result.graph.edges.values()
            if e.tactic.tactic_name == "PLN_fallback"
        ]
        self.assertEqual(
            pln_fb_edges, [], "PLN_fallback edge must not exist when use_pln=False"
        )
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
            self.assertAlmostEqual(
                shaped,
                terminal,
                places=6,
                msg=f"edge {edge.id}: shaping should be 0 but got {shaped - terminal}",
            )

    def test_rank_subgoals_guard_raises(self):
        """Calling rank_subgoals on a use_pln=False reasoner raises RuntimeError."""
        reasoner, _, _ = self._make_no_pln(_QEDExecutor())
        goal = structured_goal(GOAL_EXPR, HYPS)
        from maths_ai.data_models.proof_components import TacticCandidate

        tactic = TacticCandidate(tactic_name="intro", arguments=[], probability=1.0)
        with self.assertRaises(RuntimeError):
            asyncio.run(reasoner.rank_subgoals(goal.expression, [goal], tactic))


if __name__ == "__main__":
    unittest.main()
