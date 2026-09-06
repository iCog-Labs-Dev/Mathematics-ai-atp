from __future__ import annotations

import unittest

import torch
from torch.optim import AdamW

from maths_ai.data_models.proof_components import Goal, STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph, NodeStatus

from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import ActorCriticWithArgsClassifier
from maths_ai.gnn_inference.tests.model_helpers import actor_critic
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    make_featurizer,
    train_step,
    compute_transition_loss,
)
from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    extract_actor_transitions,
    extract_critic_samples,
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
        transitions = extract_actor_transitions(g, RewardConfig(step_penalty=0.0))
        critic_samples = extract_critic_samples(g)
        self.assertEqual(len(transitions), 3)
        result = compute_transition_loss(
            model, transitions, critic_samples, featurize, tactic_to_id
        )
        self.assertIsNotNone(result)
        loss, metrics = result
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["num_transitions"], 3.0)
        self.assertEqual(metrics["num_critic_samples"], 3.0)
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

    def test_unknown_tactic_drops_actor_row_but_keeps_critic_rows(self):
        g, featurize, model, tactic_to_id = self._setup()
        transitions = extract_actor_transitions(g, RewardConfig(step_penalty=0.0))
        critic_samples = extract_critic_samples(g)
        result = compute_transition_loss(model, transitions, critic_samples, featurize, {})
        self.assertIsNotNone(result)
        loss, metrics = result
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["num_transitions"], 0.0)
        self.assertEqual(metrics["num_critic_samples"], 3.0)
        self.assertEqual(metrics["actor_loss"], 0.0)

    def test_actor_only_batch_has_zero_critic_loss(self):
        g, featurize, model, tactic_to_id = self._setup()
        transitions = extract_actor_transitions(g, RewardConfig(step_penalty=0.0))
        result = compute_transition_loss(model, transitions, [], featurize, tactic_to_id)
        self.assertIsNotNone(result)
        loss, metrics = result
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["num_transitions"], 3.0)
        self.assertEqual(metrics["num_critic_samples"], 0.0)
        self.assertEqual(metrics["critic_loss"], 0.0)

    def test_critic_only_batch_has_zero_actor_loss(self):
        g, featurize, model, tactic_to_id = self._setup()
        critic_samples = extract_critic_samples(g)
        result = compute_transition_loss(model, [], critic_samples, featurize, tactic_to_id)
        self.assertIsNotNone(result)
        loss, metrics = result
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["num_transitions"], 0.0)
        self.assertEqual(metrics["num_critic_samples"], 3.0)
        self.assertEqual(metrics["actor_loss"], 0.0)

    def test_update_budget_rejects_the_complete_round(self):
        g, featurize, model, tactic_to_id = self._setup()
        transitions = extract_actor_transitions(g, RewardConfig(step_penalty=0.0))
        critic_samples = extract_critic_samples(g)
        with self.assertRaisesRegex(
            ValueError,
            "Collected RL update exceeds.*nodes=.*max_nodes=1",
        ):
            compute_transition_loss(
                model,
                transitions,
                critic_samples,
                featurize,
                tactic_to_id,
                max_update_nodes=1,
            )


if __name__ == "__main__":
    unittest.main()
