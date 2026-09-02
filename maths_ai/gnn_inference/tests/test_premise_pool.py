from __future__ import annotations

import unittest

import numpy as np
import torch

from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import build_unified_pools


class _FakeLemmaIndex:
    def search(self, goal_vecs, *, k):
        batch_size = int(goal_vecs.size(0))
        dim = int(goal_vecs.size(1))
        lemma_ids = []
        lemma_vecs = []
        for b in range(batch_size):
            lemma_ids.append([100 + 2 * b, 100 + 2 * b + 1])
            lemma_vecs.append(
                np.full((2, dim), fill_value=float(b + 1), dtype=np.float32)
            )
        lemma_vecs = np.stack(lemma_vecs, axis=0)
        scores = np.zeros((batch_size, 2), dtype=np.float32)
        return lemma_ids, lemma_vecs, scores


class PremisePoolTests(unittest.TestCase):
    def test_builds_unified_pool(self) -> None:
        goal_vecs = torch.randn(2, 4)
        node_embeddings = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float,
        )
        premise_mask = torch.tensor([True, False, True, False, True, True])
        batch_index = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        pools = build_unified_pools(
            goal_vecs,
            node_embeddings,
            premise_mask,
            batch_index,
            lemma_index=_FakeLemmaIndex(),
            k=2,
        )

        self.assertEqual(len(pools), 2)

        pool0 = pools[0]
        self.assertEqual(pool0.local_node_ids, [0, 2])
        self.assertEqual(pool0.lemma_ids, [100, 101])
        self.assertEqual(pool0.candidate_sources.count("local"), 2)
        self.assertEqual(pool0.candidate_sources.count("lemma"), 2)
        self.assertEqual(pool0.candidate_vectors.shape[0], 4)

        pool1 = pools[1]
        self.assertEqual(pool1.local_node_ids, [4, 5])
        self.assertEqual(pool1.lemma_ids, [102, 103])
        self.assertEqual(pool1.candidate_sources.count("local"), 2)
        self.assertEqual(pool1.candidate_sources.count("lemma"), 2)
        self.assertEqual(pool1.candidate_vectors.shape[0], 4)
