from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LemmaIndexConfig:
    index_dir: Path
    k: int = 500
    normalize_queries: bool = False


class LemmaIndex:
    """Load and query a FAISS index of precomputed lemma embeddings."""

    def __init__(
        self,
        index: Any,
        lemma_ids: list[int],
        lemma_vectors: np.ndarray,
        *,
        normalize_queries: bool = False,
    ) -> None:
        self.index = index
        self.lemma_ids = lemma_ids
        self.lemma_vectors = lemma_vectors
        self.normalize_queries = normalize_queries

        if self.lemma_vectors.ndim != 2:
            raise ValueError("lemma_vectors must be 2D (num_lemmas, dim).")
        if len(self.lemma_ids) != self.lemma_vectors.shape[0]:
            raise ValueError("lemma_ids length must match lemma_vectors rows.")
        if hasattr(self.index, "d") and int(self.index.d) != int(self.lemma_vectors.shape[1]):
            raise ValueError("FAISS index dimension does not match lemma_vectors.")

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        *,
        normalize_queries: bool = False,
    ) -> "LemmaIndex":
        import faiss

        index_path = Path(index_dir) / "faiss.index"
        vectors_path = Path(index_dir) / "lemma_vectors.npy"
        ids_path = Path(index_dir) / "lemma_ids.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at '{index_path}'.")
        if not vectors_path.exists():
            raise FileNotFoundError(f"Lemma vectors not found at '{vectors_path}'.")
        if not ids_path.exists():
            raise FileNotFoundError(f"Lemma id map not found at '{ids_path}'.")

        lemma_vectors = np.load(vectors_path)
        lemma_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(lemma_ids, list):
            raise ValueError("lemma_ids.json must contain a JSON list of ids.")

        index = faiss.read_index(str(index_path))
        return cls(
            index,
            [int(x) for x in lemma_ids],
            lemma_vectors,
            normalize_queries=normalize_queries,
        )

    def search(
        self,
        state_vecs: np.ndarray | "torch.Tensor",
        *,
        k: int = 500,
    ) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
        """Return (lemma_ids, lemma_vecs, scores) for each query."""
        query = self._to_numpy(state_vecs)
        if self.normalize_queries:
            query = _normalize_rows(query)

        scores, indices = self.index.search(query, k)
        lemma_ids = [[self.lemma_ids[int(idx)] for idx in row] for row in indices]
        lemma_vecs = self.lemma_vectors[indices]
        return lemma_ids, lemma_vecs, scores

    @staticmethod
    def _to_numpy(state_vecs: np.ndarray | "torch.Tensor") -> np.ndarray:
        if isinstance(state_vecs, np.ndarray):
            array = state_vecs
        else:
            import torch

            if not torch.is_tensor(state_vecs):
                raise TypeError("state_vecs must be a numpy array or torch Tensor.")
            array = state_vecs.detach().cpu().numpy()
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        return array


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return array / norms
