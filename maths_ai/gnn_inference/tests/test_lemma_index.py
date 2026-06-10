from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import faiss

from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex


class TestLemmaIndex(unittest.TestCase):
    def test_load_accepts_index_file_path(self) -> None:
        dim = 2
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            index_path = index_dir / "faiss.index"
            vectors_path = index_dir / "lemma_vectors.npy"
            ids_path = index_dir / "lemma_ids.json"

            index = faiss.IndexFlatL2(dim)
            index.add(vectors)
            faiss.write_index(index, str(index_path))
            np.save(vectors_path, vectors)
            ids_path.write_text(json.dumps([7, 11]), encoding="utf-8")

            loaded = LemmaIndex.load(index_path)

            self.assertEqual(loaded.lemma_ids, [7, 11])
            self.assertTrue(np.array_equal(loaded.lemma_vectors, vectors))
            self.assertEqual(loaded.index.ntotal, 2)

    def test_search_empty_index(self) -> None:
        dim = 8
        index = faiss.IndexFlatL2(dim)
        lemma_ids = []
        lemma_vectors = np.empty((0, dim), dtype=np.float32)

        lemma_index = LemmaIndex(index, lemma_ids, lemma_vectors)
        
        query = np.random.randn(2, dim).astype(np.float32)
        # Search for 5 neighbors, which is more than the index size (0)
        retrieved_ids, retrieved_vecs, scores = lemma_index.search(query, k=5)
        
        self.assertEqual(len(retrieved_ids), 2)
        self.assertEqual(len(retrieved_ids[0]), 5)
        self.assertEqual(retrieved_ids, [[-1] * 5, [-1] * 5])
        self.assertEqual(retrieved_vecs.shape, (2, 5, dim))
        self.assertTrue(np.all(retrieved_vecs == 0.0))

    def test_search_partial_index(self) -> None:
        dim = 8
        index = faiss.IndexFlatL2(dim)
        # Add 2 items
        vectors = np.random.randn(2, dim).astype(np.float32)
        index.add(vectors)
        lemma_ids = [100, 101]
        
        lemma_index = LemmaIndex(index, lemma_ids, vectors)
        
        query = np.random.randn(1, dim).astype(np.float32)
        # Search for 5 neighbors, which is more than the index size (2)
        retrieved_ids, retrieved_vecs, scores = lemma_index.search(query, k=5)
        
        self.assertEqual(len(retrieved_ids), 1)
        self.assertEqual(len(retrieved_ids[0]), 5)
        
        # The first 2 elements should be valid IDs (100 or 101)
        # The remaining 3 elements should be -1
        valid_ids = retrieved_ids[0][:2]
        self.assertTrue(all(x in [100, 101] for x in valid_ids))
        self.assertEqual(retrieved_ids[0][2:], [-1, -1, -1])
        
        self.assertEqual(retrieved_vecs.shape, (1, 5, dim))
        # Check that the invalid indices' vectors are zeroed
        self.assertTrue(np.all(retrieved_vecs[0, 2:] == 0.0))


if __name__ == "__main__":
    unittest.main()
