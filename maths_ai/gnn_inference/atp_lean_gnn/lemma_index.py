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
        lemma_names: list[str] | None = None,
        encoder_fingerprint: str | None = None,
        normalize_queries: bool = False,
    ) -> None:
        self.index = index
        self.lemma_ids = lemma_ids
        self.lemma_vectors = lemma_vectors
        self.normalize_queries = normalize_queries
        self.lemma_names = lemma_names or []
        self.name_to_id = {
            name: lemma_id for name, lemma_id in zip(self.lemma_names, self.lemma_ids)
        }
        self.encoder_fingerprint = encoder_fingerprint

        if self.lemma_vectors.ndim != 2:
            raise ValueError("lemma_vectors must be 2D (num_lemmas, dim).")
        if len(self.lemma_ids) != self.lemma_vectors.shape[0]:
            raise ValueError("lemma_ids length must match lemma_vectors rows.")
        if self.lemma_names and len(self.lemma_names) != len(self.lemma_ids):
            raise ValueError("lemma_names length must match lemma_ids length.")
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

        input_path = Path(index_dir)

        if input_path.is_file():
            index_path = input_path
            vectors_path = input_path.with_name("lemma_vectors.npy")
            ids_path = input_path.with_name("lemma_ids.json")
            names_path = input_path.with_name("lemma_names.json")
            manifest_path = input_path.with_name("manifest.json")
        else:
            index_path = input_path / "faiss.index"
            vectors_path = input_path / "lemma_vectors.npy"
            ids_path = input_path / "lemma_ids.json"
            names_path = input_path / "lemma_names.json"
            manifest_path = input_path / "manifest.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at '{index_path}'.")
        if not vectors_path.exists():
            raise FileNotFoundError(f"Lemma vectors not found at '{vectors_path}'.")
        if not ids_path.exists():
            raise FileNotFoundError(f"Lemma id map not found at '{ids_path}'.")
        if not names_path.exists():
            raise FileNotFoundError(f"Lemma name map not found at '{names_path}'.")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Lemma index manifest not found at '{manifest_path}'.")

        lemma_vectors = np.load(vectors_path)
        lemma_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(lemma_ids, list):
            raise ValueError("lemma_ids.json must contain a JSON list of ids.")
        lemma_names = json.loads(names_path.read_text(encoding="utf-8"))
        if not isinstance(lemma_names, list):
            raise ValueError("lemma_names.json must contain a JSON list of names.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        encoder_fingerprint = manifest.get("encoder_fingerprint")
        if not isinstance(encoder_fingerprint, str) or not encoder_fingerprint:
            raise ValueError("Lemma index manifest is missing 'encoder_fingerprint'.")

        index = faiss.read_index(str(index_path))
        return cls(
            index,
            [int(x) for x in lemma_ids],
            lemma_vectors,
            lemma_names=[str(x) for x in lemma_names],
            encoder_fingerprint=encoder_fingerprint,
            normalize_queries=normalize_queries,
        )

    def validate_encoder_fingerprint(self, expected: str) -> None:
        if self.encoder_fingerprint != expected:
            raise ValueError(
                "Lemma index encoder fingerprint does not match the active tactic model."
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
        
        num_lemmas = len(self.lemma_ids)
        lemma_ids = [
            [self.lemma_ids[int(idx)] if 0 <= int(idx) < num_lemmas else -1 for idx in row]
            for row in indices
        ]
        
        valid_mask = (indices >= 0) & (indices < num_lemmas)
        safe_indices = np.where(valid_mask, indices, 0)
        
        if num_lemmas > 0:
            lemma_vecs = self.lemma_vectors[safe_indices]
            lemma_vecs[~valid_mask] = 0.0
        else:
            lemma_vecs = np.zeros(
                (indices.shape[0], indices.shape[1], self.lemma_vectors.shape[1]),
                dtype=self.lemma_vectors.dtype
            )
            
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
