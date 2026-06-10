"""Unit tests for the premise scoring head (Phase 5)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import CandidatePool
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import (
    PremiseScorer,
    PremiseScorerConfig,
    compute_premise_ranking_loss,
    _find_target_index_in_pool,
)


def _make_pool(
    *,
    num_local: int = 2,
    num_lemma: int = 3,
    hidden_dim: int = 8,
) -> CandidatePool:
    """Create a synthetic CandidatePool for testing."""
    local_vecs = torch.randn(num_local, hidden_dim)
    lemma_vecs = torch.randn(num_lemma, hidden_dim)
    candidate_vectors = torch.cat([local_vecs, lemma_vecs], dim=0)
    candidate_sources = ["local"] * num_local + ["lemma"] * num_lemma
    local_ids = list(range(num_local))
    lemma_ids = [100 + i for i in range(num_lemma)]
    candidate_ids = local_ids + lemma_ids
    return CandidatePool(
        candidate_vectors=candidate_vectors,
        candidate_sources=candidate_sources,
        candidate_ids=candidate_ids,
        local_node_ids=local_ids,
        lemma_ids=lemma_ids,
    )


class TestPremiseScorerConfig(unittest.TestCase):
    def test_default_config(self) -> None:
        config = PremiseScorerConfig()
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.scoring_mode, "dot")
        self.assertEqual(config.tactic_conditioning, "soft")
        self.assertAlmostEqual(config.premise_loss_weight, 0.3)

    def test_to_dict(self) -> None:
        config = PremiseScorerConfig(hidden_dim=64, scoring_mode="mlp")
        d = config.to_dict()
        self.assertEqual(d["hidden_dim"], 64)
        self.assertEqual(d["scoring_mode"], "mlp")


class TestPremiseScorerDot(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_dim = 8
        self.scorer = PremiseScorer(self.hidden_dim, mode="dot")

    def test_score_shape(self) -> None:
        goal = torch.randn(self.hidden_dim)
        tactic = torch.randn(self.hidden_dim)
        candidates = torch.randn(5, self.hidden_dim)
        scores = self.scorer.score(goal, tactic, candidates)
        self.assertEqual(scores.shape, (5,))

    def test_score_batched_shape(self) -> None:
        goal = torch.randn(1, self.hidden_dim)
        tactic = torch.randn(1, self.hidden_dim)
        candidates = torch.randn(10, self.hidden_dim)
        scores = self.scorer.score(goal, tactic, candidates)
        self.assertEqual(scores.shape, (10,))

    def test_forward_with_pools(self) -> None:
        batch_size = 3
        pools = [_make_pool(hidden_dim=self.hidden_dim) for _ in range(batch_size)]
        goal_vecs = torch.randn(batch_size, self.hidden_dim)
        tactic_embs = torch.randn(batch_size, self.hidden_dim)

        scores = self.scorer(goal_vecs, tactic_embs, pools)
        self.assertEqual(len(scores), batch_size)
        for i, s in enumerate(scores):
            expected_len = len(pools[i].candidate_ids)
            self.assertEqual(s.shape, (expected_len,))

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            PremiseScorer(8, mode="invalid")

    def test_batch_size_mismatch_raises(self) -> None:
        pools = [_make_pool(hidden_dim=self.hidden_dim)]
        goal_vecs = torch.randn(2, self.hidden_dim)  # mismatch
        tactic_embs = torch.randn(2, self.hidden_dim)
        with self.assertRaises(ValueError):
            self.scorer(goal_vecs, tactic_embs, pools)


class TestPremiseScorerMLP(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_dim = 8
        self.scorer = PremiseScorer(self.hidden_dim, mode="mlp")

    def test_score_shape(self) -> None:
        goal = torch.randn(self.hidden_dim)
        tactic = torch.randn(self.hidden_dim)
        candidates = torch.randn(7, self.hidden_dim)
        scores = self.scorer.score(goal, tactic, candidates)
        self.assertEqual(scores.shape, (7,))

    def test_mlp_differs_from_dot(self) -> None:
        """MLP and dot should generally produce different scores."""
        dot_scorer = PremiseScorer(self.hidden_dim, mode="dot")
        goal = torch.randn(self.hidden_dim)
        tactic = torch.randn(self.hidden_dim)
        candidates = torch.randn(5, self.hidden_dim)

        dot_scores = dot_scorer.score(goal, tactic, candidates)
        mlp_scores = self.scorer.score(goal, tactic, candidates)

        # They should not be identical (different architectures)
        self.assertFalse(torch.allclose(dot_scores, mlp_scores))


class TestFindTargetIndex(unittest.TestCase):
    def test_local_match(self) -> None:
        pool = _make_pool(num_local=3, num_lemma=2, hidden_dim=4)
        # local IDs are [0, 1, 2], want node 1
        idx = _find_target_index_in_pool(
            pool, arg_node_indices=[1], arg_lemma_ids=[-1]
        )
        self.assertEqual(idx, 1)

    def test_lemma_match(self) -> None:
        pool = _make_pool(num_local=2, num_lemma=3, hidden_dim=4)
        # lemma IDs are [100, 101, 102], want lemma 101
        idx = _find_target_index_in_pool(
            pool, arg_node_indices=[-1], arg_lemma_ids=[101]
        )
        # Pool order: [local0, local1, lemma100, lemma101, lemma102]
        self.assertEqual(idx, 3)

    def test_no_match(self) -> None:
        pool = _make_pool(num_local=2, num_lemma=2, hidden_dim=4)
        idx = _find_target_index_in_pool(
            pool, arg_node_indices=[-1], arg_lemma_ids=[-1]
        )
        self.assertEqual(idx, -1)

    def test_local_preferred_over_lemma(self) -> None:
        pool = _make_pool(num_local=2, num_lemma=2, hidden_dim=4)
        # Both local and lemma match — local should win
        idx = _find_target_index_in_pool(
            pool, arg_node_indices=[0], arg_lemma_ids=[100]
        )
        self.assertEqual(idx, 0)  # local match at position 0


class TestPremiseRankingLoss(unittest.TestCase):
    def test_basic_loss_and_metrics(self) -> None:
        hidden_dim = 8
        pool = _make_pool(num_local=2, num_lemma=3, hidden_dim=hidden_dim)
        # Simulate scores: highest score is at index 0
        scores = torch.tensor([5.0, 1.0, 2.0, -1.0, 0.0], requires_grad=True)

        # Target: local node 0 (pool position 0) -> Rank 1
        arg_node_indices = torch.tensor([[0, -1]])
        arg_lemma_ids = torch.tensor([[-1, -1]])

        loss, metrics = compute_premise_ranking_loss(
            [scores], [pool], arg_node_indices, arg_lemma_ids
        )

        self.assertTrue(loss.requires_grad)
        self.assertEqual(metrics["valid_samples"], 1)
        self.assertEqual(metrics["total_samples"], 1)
        self.assertEqual(metrics["target_present_count"], 1)
        self.assertEqual(metrics["top1_correct"], 1)
        self.assertEqual(metrics["top5_correct"], 1)
        self.assertAlmostEqual(metrics["mrr_sum"], 1.0)
        self.assertGreater(loss.item(), 0.0)  # CE is always positive

    def test_lower_rank_metrics(self) -> None:
        hidden_dim = 8
        pool = _make_pool(num_local=2, num_lemma=3, hidden_dim=hidden_dim)
        # Simulate scores: target at index 1 gets the 3rd highest score
        # Sorted scores will be: index 0 (5.0), index 2 (4.0), index 1 (3.0)...
        scores = torch.tensor([5.0, 3.0, 4.0, -1.0, 0.0], requires_grad=True)

        # Target: local node 1 (pool position 1) -> Rank 3
        arg_node_indices = torch.tensor([[1, -1]])
        arg_lemma_ids = torch.tensor([[-1, -1]])

        loss, metrics = compute_premise_ranking_loss(
            [scores], [pool], arg_node_indices, arg_lemma_ids
        )

        self.assertEqual(metrics["top1_correct"], 0)
        self.assertEqual(metrics["top5_correct"], 1)
        self.assertAlmostEqual(metrics["mrr_sum"], 1.0 / 3.0)

    def test_no_target_present(self) -> None:
        hidden_dim = 8
        pool = _make_pool(num_local=2, num_lemma=3, hidden_dim=hidden_dim)
        scores = torch.randn(5, requires_grad=True)

        # No valid targets at all
        arg_node_indices = torch.tensor([[-1, -1]])
        arg_lemma_ids = torch.tensor([[-1, -1]])

        loss, metrics = compute_premise_ranking_loss(
            [scores], [pool], arg_node_indices, arg_lemma_ids
        )

        self.assertEqual(metrics["target_present_count"], 0)
        self.assertEqual(metrics["valid_samples"], 0)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_target_present_but_not_retrieved(self) -> None:
        hidden_dim = 8
        pool = _make_pool(num_local=2, num_lemma=3, hidden_dim=hidden_dim)
        scores = torch.randn(5, requires_grad=True)

        # Target is a lemma that isn't in the pool
        arg_node_indices = torch.tensor([[-1, -1]])
        arg_lemma_ids = torch.tensor([[999, -1]])

        loss, metrics = compute_premise_ranking_loss(
            [scores], [pool], arg_node_indices, arg_lemma_ids
        )

        self.assertEqual(metrics["target_present_count"], 1)
        self.assertEqual(metrics["valid_samples"], 0)
        self.assertEqual(metrics["mrr_sum"], 0.0)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_batch_loss(self) -> None:
        hidden_dim = 8
        pools = [_make_pool(num_local=2, num_lemma=3, hidden_dim=hidden_dim) for _ in range(3)]
        scores = [torch.randn(5) for _ in range(3)]

        # Sample 0: valid target, retrieved.
        # Sample 1: no target.
        # Sample 2: valid target, retrieved.
        arg_node_indices = torch.tensor([[0, -1], [-1, -1], [1, -1]])
        arg_lemma_ids = torch.tensor([[-1, -1], [-1, -1], [-1, -1]])

        loss, metrics = compute_premise_ranking_loss(
            scores, pools, arg_node_indices, arg_lemma_ids
        )

        self.assertEqual(metrics["target_present_count"], 2)
        self.assertEqual(metrics["valid_samples"], 2)
        self.assertEqual(metrics["total_samples"], 3)

    def test_tactic_conditioning_changes_output(self) -> None:
        """Changing the tactic embedding should change the scores."""
        hidden_dim = 8
        scorer = PremiseScorer(hidden_dim, mode="dot")
        goal = torch.randn(1, hidden_dim)
        pool = _make_pool(num_local=2, num_lemma=3, hidden_dim=hidden_dim)

        tactic1 = torch.randn(1, hidden_dim)
        tactic2 = torch.randn(1, hidden_dim)

        scores1 = scorer(goal, tactic1, [pool])
        scores2 = scorer(goal, tactic2, [pool])

        # Different tactic embeddings should produce different scores
        self.assertFalse(torch.allclose(scores1[0], scores2[0]))


if __name__ == "__main__":
    unittest.main()
