from __future__ import annotations

import unittest

import torch
from torch.optim import AdamW

from maths_ai.data_models.proof_components import Goal, STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import BackupSource, ProofHypergraph, NodeStatus

from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import ActorCriticWithArgsClassifier
from maths_ai.gnn_inference.tests.model_helpers import actor_critic
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    make_featurizer,
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


class PLNRLTrainingTests(unittest.TestCase):
    # Parseable proof-state strings (same format the GNN featurizer expects).
    ROOT = "n : Nat\n⊢ Even n ∧ Odd n"
    SUB_A = "n : Nat\n⊢ Even n"
    SUB_B = "n : Nat\n⊢ Odd n"

    def _solved_and_graph(self):
        g = ProofHypergraph(Goal(expression=self.ROOT, hypotheses=[]))
        edge = g.add_edge(
            g.root_id,
            _tac(),
            ranked_subgoals=[
                (Goal(expression=self.SUB_A, hypotheses=[]), STV(strength=0.6, confidence=1.0)),
                (Goal(expression=self.SUB_B, hypotheses=[]), STV(strength=0.4, confidence=1.0)),
            ],
        )
        a_id, b_id = edge.child_ids
        g.add_edge(a_id, _tac(), ranked_subgoals=[])  # QED
        g.add_edge(b_id, _tac(), ranked_subgoals=[])  # QED
        return g

    def _setup(self):
        g = self._solved_and_graph()
        dags = [proof_state_to_dag(s) for s in (self.ROOT, self.SUB_A, self.SUB_B)]
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
        # Root's subgoals both solved ⇒ backup value target 1.0 on that transition.
        self.assertGreater(metrics["mean_value_target"], 0.0)

    def test_train_step_updates_params(self):
        g, featurize, model, tactic_to_id = self._setup()
        optimizer = AdamW(model.parameters(), lr=0.01)
        before = [p.detach().clone() for p in model.parameters()]
        metrics = train_step(
            model, optimizer, [g], featurize, tactic_to_id,
            reward_cfg=RewardConfig(step_penalty=0.0), bc_weight=0.1,
        )
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
        dags = [proof_state_to_dag(s) for s in (self.GOAL_A, self.GOAL_B)]
        vocab = build_vocab(dags)
        featurize = make_featurizer(vocab)
        model = actor_critic(len(vocab), 3, dropout=0.0)
        tactic_batch = [
            TacticImitationSample(goal=self.GOAL_A, hypotheses=(), tactic_id=1, arg_indices=(0,)),
            TacticImitationSample(goal=self.GOAL_B, hypotheses=(), tactic_id=2),
        ]
        critic_batch = [
            CriticSample(
                node_id=0, goal=self.GOAL_A, hypotheses=(), target=1.0,
                source=BackupSource.LEAN_STATUS,
            ),
            CriticSample(
                node_id=1, goal=self.GOAL_B, hypotheses=(), target=0.25,
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
        bad = [TacticImitationSample(goal=self.GOAL_A, hypotheses=(), tactic_id=1,
                                     arg_indices=(10_000, -1))]
        metrics = train_step_htps_style(model, opt, bad, [], featurize)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["htps_total_loss"])))
        self.assertEqual(metrics["num_imitation_rows"], 1.0)


if __name__ == "__main__":
    unittest.main()
