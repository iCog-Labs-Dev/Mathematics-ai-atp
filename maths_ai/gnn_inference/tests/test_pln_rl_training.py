from __future__ import annotations
from types import SimpleNamespace
from torch_geometric.data import Batch
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import local_pools
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import EdgeAction

import json
from pathlib import Path
import unittest

import torch
from torch.optim import AdamW

from maths_ai.data_models.proof_components import STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph, NodeStatus

from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab, build_vocab_from_labels
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import (
    ActorCriticWithArgsClassifier,
)
from maths_ai.gnn_inference.tests.model_helpers import actor_critic, structured_goal
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    _is_structural_label,
    make_dag_featurizer,
    make_featurizer,
    supported_structural_labels,
    train_step,
    compute_transition_loss,
)
from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    extract_actor_transitions,
    extract_critic_samples,
)
from maths_ai.gnn_inference.tests.test_pln_reward import _certify


def _tac(name: str = "apply", p: float = 1.0) -> TacticCandidate:
    return TacticCandidate(tactic_name=name, arguments=[], probability=p)


def recorded_actions(transitions, featurize, tactic_to_id):
    from dataclasses import replace
    from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import fingerprint
    from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import goal_query

    actions = {}
    for t in transitions:
        if t.tactic.tactic_name not in tactic_to_id:
            continue
        data = featurize(t.goal)
        pool = replace(
            local_pools(Batch.from_data_list([data]))[0],
            graph_fingerprint=data.graph_fingerprint,
            query_fingerprint=fingerprint(goal_query(t.goal)),
        )
        actions[t.edge_id] = EdgeAction(
            tactic_id=tactic_to_id[t.tactic.tactic_name],
            candidate_pool=pool,
            graph_fingerprint=data.graph_fingerprint,
        )
    return actions


class PLNRLTrainingTests(unittest.TestCase):
    # Parseable proof-state strings (same format the GNN featurizer expects).
    ROOT = "n : Nat\n⊢ Even n ∧ Odd n"
    SUB_A = "n : Nat\n⊢ Even n"
    SUB_B = "n : Nat\n⊢ Odd n"

    def _solved_and_graph(self):
        root = structured_goal(self.ROOT)
        sub_a = structured_goal(self.SUB_A)
        sub_b = structured_goal(self.SUB_B)
        g = ProofHypergraph(root)
        edge = g.add_edge(
            g.root_id,
            _tac(),
            ranked_subgoals=[
                (sub_a, STV(strength=0.6, confidence=1.0)),
                (sub_b, STV(strength=0.4, confidence=1.0)),
            ],
        )
        a_id, b_id = edge.child_ids
        a = g.add_edge(a_id, _tac(), ranked_subgoals=[])  # closes A
        b = g.add_edge(b_id, _tac(), ranked_subgoals=[])  # closes B
        # The certified replay ran all three attempts.
        _certify(g, edge.id, a.id, b.id)
        return g

    def _setup(self):
        g = self._solved_and_graph()
        dags = [
            model_goal_to_dag(structured_goal(s))
            for s in (self.ROOT, self.SUB_A, self.SUB_B)
        ]
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
            model,
            transitions,
            critic_samples,
            featurize,
            edge_actions=recorded_actions(transitions, featurize, tactic_to_id),
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
            model,
            optimizer,
            [
                SimpleNamespace(
                    graph=g,
                    edge_actions=recorded_actions(
                        extract_actor_transitions(g), featurize, tactic_to_id
                    ),
                    failure_actions=[],
                )
            ],
            featurize,
            reward_cfg=RewardConfig(step_penalty=0.0),
            bc_weight=0.1,
        )
        self.assertEqual(metrics["num_transitions"], 3.0)
        after = list(model.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        self.assertTrue(changed, "training step did not update any parameters")

    def test_missing_collection_action_cannot_silently_drop_actor_rows(self):
        g, featurize, model, tactic_to_id = self._setup()
        transitions = extract_actor_transitions(g, RewardConfig(step_penalty=0.0))
        critic_samples = extract_critic_samples(g)
        with self.assertRaisesRegex(
            ValueError, "missing its complete collection action"
        ):
            compute_transition_loss(
                model, transitions, critic_samples, featurize, edge_actions={}
            )

    def test_actor_only_batch_has_zero_critic_loss(self):
        g, featurize, model, tactic_to_id = self._setup()
        transitions = extract_actor_transitions(g, RewardConfig(step_penalty=0.0))
        result = compute_transition_loss(
            model,
            transitions,
            [],
            featurize,
            edge_actions=recorded_actions(transitions, featurize, tactic_to_id),
        )
        self.assertIsNotNone(result)
        loss, metrics = result
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["num_transitions"], 3.0)
        self.assertEqual(metrics["num_critic_samples"], 0.0)
        self.assertEqual(metrics["critic_loss"], 0.0)

    def test_critic_only_batch_has_zero_actor_loss(self):
        g, featurize, model, tactic_to_id = self._setup()
        critic_samples = extract_critic_samples(g)
        result = compute_transition_loss(
            model, [], critic_samples, featurize, edge_actions={}
        )
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
                edge_actions=recorded_actions(transitions, featurize, tactic_to_id),
                max_update_nodes=1,
            )


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
        goldens = json.loads(
            (FIXTURES / "canonical_graph_expectations.json").read_text()
        )
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
