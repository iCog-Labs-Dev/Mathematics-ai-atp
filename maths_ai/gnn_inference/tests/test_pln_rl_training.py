from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch
from torch.optim import AdamW

from maths_ai.data_models.proof_components import STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import BackupSource, ProofHypergraph, NodeStatus

from maths_ai.gnn_inference.atp_lean_gnn.graph import dag_fingerprint, model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab, build_vocab_from_labels
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import ActorCriticWithArgsClassifier
from maths_ai.gnn_inference.tests.model_helpers import (
    actor_critic,
    certify_witness,
    scored,
    structured_goal,
)
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    _is_structural_label,
    make_dag_featurizer,
    make_featurizer,
    supported_structural_labels,
    train_step,
    train_step_htps_style,
    compute_transition_loss,
)
from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    CriticSample,
    TacticImitationSample,
    extract_critic_samples,
    extract_transitions,
)


def _tac(name: str = "apply", p: float = 1.0) -> TacticCandidate:
    return TacticCandidate(tactic_name=name, arguments=[], probability=p)


def _goal(expression: str):
    return structured_goal(expression, ["n : Nat"])


def _fingerprint(goal) -> str:
    return str(dag_fingerprint(model_goal_to_dag(goal)))


class PLNRLTrainingTests(unittest.TestCase):
    # Parseable proof-state strings (same format the GNN featurizer expects).
    ROOT = "n : Nat\n⊢ Even n ∧ Odd n"
    SUB_A = "n : Nat\n⊢ Even n"
    SUB_B = "n : Nat\n⊢ Odd n"

    def _solved_and_graph(self):
        root = _goal(self.ROOT)
        sub_a = _goal(self.SUB_A)
        sub_b = _goal(self.SUB_B)
        g = ProofHypergraph(root)
        edge = g.add_edge(
            g.root_id,
            scored(_tac()),
            ranked_subgoals=[
                (sub_a, STV(strength=0.6, confidence=1.0)),
                (sub_b, STV(strength=0.4, confidence=1.0)),
            ],
        )
        a_id, b_id = edge.child_ids
        closing_a = g.add_edge(a_id, scored(_tac()), ranked_subgoals=[])  # QED
        closing_b = g.add_edge(b_id, scored(_tac()), ranked_subgoals=[])  # QED
        # The replay of all three attempts was kernel-certified.
        certify_witness(g, edge.id, closing_a.id, closing_b.id)
        return g

    def _setup(self):
        g = self._solved_and_graph()
        dags = [model_goal_to_dag(_goal(s)) for s in (self.ROOT, self.SUB_A, self.SUB_B)]
        vocab = build_vocab(dags)
        featurize = make_featurizer(vocab)
        model = actor_critic(len(vocab), 3)
        tactic_to_id = {"apply": 0}
        return g, featurize, model, tactic_to_id

    def test_harvest_then_loss_is_finite(self):
        g, featurize, model, tactic_to_id = self._setup()
        transitions = extract_transitions(g, RewardConfig(step_penalty=0.0))
        critic_samples = extract_critic_samples(g)
        self.assertEqual(len(transitions), 3)
        result = compute_transition_loss(
            model, transitions, critic_samples, featurize, tactic_to_id
        )
        self.assertIsNotNone(result)
        loss, metrics = result
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["num_transitions"], 3.0)
        # Every attempt is on the certified witness ⇒ value target 1.0.
        self.assertGreater(metrics["mean_value_target"], 0.0)

    def test_train_step_updates_params(self):
        g, featurize, model, tactic_to_id = self._setup()
        optimizer = AdamW(model.parameters(), lr=0.01)
        before = [p.detach().clone() for p in model.parameters()]
        forward_modes = []
        handle = model.encoder.register_forward_pre_hook(
            lambda module, args: forward_modes.append(
                (module.training, torch.is_grad_enabled())
            )
        )
        try:
            with torch.no_grad():
                metrics = train_step(
                    model, optimizer, [g], featurize, tactic_to_id,
                    reward_cfg=RewardConfig(step_penalty=0.0), bc_weight=0.1,
                )
                self.assertFalse(torch.is_grad_enabled())
        finally:
            handle.remove()
        self.assertTrue(forward_modes)
        self.assertTrue(all(mode == (False, True) for mode in forward_modes))
        self.assertTrue(all(not module.training for module in model.modules()))
        self.assertEqual(metrics["num_transitions"], 3.0)
        after = list(model.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        self.assertTrue(changed, "training step did not update any parameters")

    def test_unknown_tactic_dropped(self):
        # A transition whose tactic isn't in the vocab is dropped, not crashed.
        g, featurize, model, tactic_to_id = self._setup()
        transitions = extract_transitions(g, RewardConfig(step_penalty=0.0))
        # Empty vocab ⇒ every tactic unknown ⇒ None result.
        result = compute_transition_loss(model, transitions, [], featurize, {})
        self.assertIsNone(result)


class HTPSStyleStepTests(unittest.TestCase):
    """Decoupled imitation + soft-critic step (Phase 2+3)."""

    GOAL_A = "n : Nat\n⊢ Even n"
    GOAL_B = "n : Nat\n⊢ Odd n"

    def _setup(self, seed: int = 0):
        torch.manual_seed(seed)
        dags = [model_goal_to_dag(_goal(s)) for s in (self.GOAL_A, self.GOAL_B)]
        vocab = build_vocab(dags)
        featurize = make_featurizer(vocab)
        model = actor_critic(len(vocab), 3, dropout=0.0)
        tactic_batch = [
            TacticImitationSample(goal=_goal(self.GOAL_A), graph_fingerprint=_fingerprint(_goal(self.GOAL_A)), tactic_id=1, arg_indices=(0,)),
            TacticImitationSample(goal=_goal(self.GOAL_B), graph_fingerprint=_fingerprint(_goal(self.GOAL_B)), tactic_id=2),
        ]
        critic_batch = [
            CriticSample(
                node_id=0, goal=_goal(self.GOAL_A), graph_fingerprint=_fingerprint(_goal(self.GOAL_A)), target=1.0,
                source=BackupSource.LEAN_STATUS,
            ),
            CriticSample(
                node_id=1, goal=_goal(self.GOAL_B), graph_fingerprint=_fingerprint(_goal(self.GOAL_B)), target=0.25,
                source=BackupSource.VISIT_MEAN,
            ),
        ]
        return model, featurize, tactic_batch, critic_batch

    def test_loss_decreases_on_fixed_batch(self):
        model, featurize, tactic_batch, critic_batch = self._setup()
        opt = AdamW(model.parameters(), lr=0.01)
        first = train_step_htps_style(model, opt, tactic_batch, critic_batch, featurize)
        for _ in range(20):
            last = train_step_htps_style(model, opt, tactic_batch, critic_batch, featurize)
        self.assertLess(last["htps_total_loss"], first["htps_total_loss"])
        self.assertEqual(first["num_imitation_rows"], 2.0)
        self.assertEqual(first["num_critic_rows"], 2.0)
        self.assertEqual(first["htps_optimizer_step"], 1.0)

    def test_htps_restores_training_mode_after_onpolicy_update(self):
        model, featurize, tactic_batch, critic_batch = self._setup()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.2
        onpolicy_opt = AdamW(model.parameters(), lr=0.001)
        htps_opt = AdamW(model.parameters(), lr=0.001)
        goal = tactic_batch[0].goal
        # A rejected observation supplies an actor row without requiring Lean.
        from types import SimpleNamespace
        from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
            EdgeAction, ObservationOutcome, PolicyObservation, train_step_onpolicy,
        )
        observation = PolicyObservation(
            node_id=0, goal=goal,
            action=EdgeAction(
                tactic_id=1, graph_fingerprint=_fingerprint(goal),
            ),
            outcome=ObservationOutcome.REJECTED, first_round=0,
            log_prior=-1.0, provider="test",
        )
        graph = ProofHypergraph(goal)
        result = SimpleNamespace(
            graph=graph, observations=[observation], accepted_edge_ids=lambda: set(),
        )
        metrics = train_step_onpolicy(
            model, onpolicy_opt, [result], featurize,
        )
        self.assertEqual(metrics["onpolicy_optimizer_step"], 1)
        self.assertTrue(all(not module.training for module in model.modules()))
        modes = []
        handle = model.encoder.register_forward_pre_hook(
            lambda module, args: modes.append((module.training, torch.is_grad_enabled()))
        )
        try:
            metrics = train_step_htps_style(
                model, htps_opt, tactic_batch, critic_batch, featurize,
            )
        finally:
            handle.remove()
        self.assertEqual(metrics["htps_optimizer_step"], 1)
        self.assertTrue(modes)
        self.assertTrue(all(mode == (True, True) for mode in modes))
        self.assertTrue(all(module.training for module in model.modules()))

    def test_does_not_touch_the_onpolicy_optimizer(self):
        model, featurize, tactic_batch, critic_batch = self._setup()
        onpolicy_opt = AdamW(model.parameters(), lr=0.01)
        htps_opt = AdamW(model.parameters(), lr=0.01)
        train_step_htps_style(model, htps_opt, tactic_batch, critic_batch, featurize)
        # The decoupled step must not create Adam moments in the on-policy
        # optimizer: shared moments would corrupt the on-policy update scale.
        self.assertEqual(len(onpolicy_opt.state), 0)
        self.assertGreater(len(htps_opt.state), 0)

    def test_empty_batches_return_zero_metrics_without_stepping(self):
        model, featurize, _tactic, _critic = self._setup()
        opt = AdamW(model.parameters(), lr=0.01)
        before = [p.detach().clone() for p in model.parameters()]
        metrics = train_step_htps_style(model, opt, [], [], featurize)
        self.assertEqual(metrics["htps_total_loss"], 0.0)
        self.assertEqual(metrics["num_imitation_rows"], 0.0)
        self.assertEqual(metrics["htps_optimizer_step"], 0.0)
        after = list(model.parameters())
        self.assertTrue(all(torch.equal(b, a) for b, a in zip(before, after)))

    def test_critic_only_batch_regresses_the_value_head(self):
        model, featurize, _tactic, critic_batch = self._setup()
        opt = AdamW(model.parameters(), lr=0.01)
        metrics = train_step_htps_style(model, opt, [], critic_batch, featurize)
        self.assertEqual(metrics["num_imitation_rows"], 0.0)
        self.assertGreater(metrics["critic_soft_loss"], 0.0)
        # No imitation rows ⇒ the CE over labels==-1 must be exactly 0.
        self.assertEqual(metrics["tactic_imitation_loss"], 0.0)

    def test_out_of_range_arg_indices_are_masked(self):
        model, featurize, _tactic, _critic = self._setup()
        opt = AdamW(model.parameters(), lr=0.01)
        # 10_000 exceeds every graph's node count; forced_step must treat it as
        # invalid (log-prob 0), leaving the loss finite.
        bad_goal = _goal(self.GOAL_A)
        bad = [TacticImitationSample(
            goal=bad_goal,
            graph_fingerprint=_fingerprint(bad_goal),
            tactic_id=1,
            arg_indices=(10_000, -1),
        )]
        metrics = train_step_htps_style(model, opt, bad, [], featurize)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["htps_total_loss"])))
        self.assertEqual(metrics["num_imitation_rows"], 1.0)


if __name__ == "__main__":
    unittest.main()


FIXTURES = Path(__file__).parent / "fixtures/pantograph"


class StructuralVocabularyTests(unittest.TestCase):
    def _successor(self):
        """``intro p h`` on ``∀ (p : Prop), p → p``: locals FV0 and FV1."""
        from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_goal
        from pantograph_async import decode_execution_goal

        contract = json.loads((FIXTURES / "future_contract.json").read_text())
        wire = contract["successes"]["identity_intro"]["goals"][0]
        return execution_goal_to_goal(decode_execution_goal(wire))

    def _semantic_labels(self, goal):
        return {
            node.label
            for node in model_goal_to_dag(goal).nodes
            if not _is_structural_label(node.label)
        }

    def test_seed_only_vocabulary_rejects_an_introduced_local(self):
        goal = self._successor()
        seed_labels = {node.label for node in model_goal_to_dag(goal).nodes} - {"FV1"}
        featurize = make_dag_featurizer(build_vocab_from_labels(seed_labels))
        with self.assertRaisesRegex(ValueError, "structural labels: FV1"):
            featurize(goal)

    def test_supported_labels_cover_successors_within_their_bounds(self):
        goal = self._successor()
        vocab = build_vocab_from_labels(
            [
                *self._semantic_labels(goal),
                *supported_structural_labels(max_locals=2, max_arity=0),
            ]
        )
        _dag, data = make_dag_featurizer(vocab)(goal)
        self.assertEqual(int(data.structural_unknown_label_count), 0)
        narrow = build_vocab_from_labels(
            [
                *self._semantic_labels(goal),
                *supported_structural_labels(max_locals=1, max_arity=0),
            ]
        )
        with self.assertRaisesRegex(ValueError, "structural labels: FV1"):
            make_dag_featurizer(narrow)(goal)

    def test_supported_labels_follow_the_contract_and_serializer(self):
        labels = supported_structural_labels(max_locals=3, max_arity=2)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(all(_is_structural_label(label) for label in labels))
        self.assertIn("FV2", labels)
        self.assertNotIn("FV3", labels)
        self.assertIn("ArgPosition:1", labels)
        self.assertNotIn("ArgPosition:2", labels)
        goldens = json.loads((FIXTURES / "canonical_graph_expectations.json").read_text())
        observed = {
            node["label"]
            for goals in goldens["goals"].values()
            for goal in goals
            for node in goal["nodes"]
            if _is_structural_label(node["label"])
        }
        self.assertTrue(observed)
        self.assertLessEqual(
            observed, set(supported_structural_labels(max_locals=8, max_arity=8))
        )
        with self.assertRaises(ValueError):
            supported_structural_labels(max_locals=-1, max_arity=0)

