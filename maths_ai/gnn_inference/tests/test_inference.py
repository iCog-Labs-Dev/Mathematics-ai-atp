from __future__ import annotations

import torch

from maths_ai.gnn_inference.atp_lean_gnn.inference import _top_tactic_candidates


def test_top_tactic_candidates_returns_sorted_top_k() -> None:
    tactic_probs = torch.tensor([0.1, 0.5, 0.4], dtype=torch.float32)
    id_to_tactic = {0: "simp", 1: "rw", 2: "exact"}

    candidates = _top_tactic_candidates(tactic_probs, id_to_tactic, top_k=2)

    assert [(item["tactic_id"], item["tactic_name"], item["probability"]) for item in candidates] == [
        (1, "rw", 0.5),
        (2, "exact", 0.4),
    ]


def test_top_tactic_candidates_caps_at_vocab_size() -> None:
    tactic_probs = torch.tensor([0.2, 0.3], dtype=torch.float32)
    id_to_tactic = {0: "simp", 1: "rw"}

    candidates = _top_tactic_candidates(tactic_probs, id_to_tactic, top_k=10)

    assert len(candidates) == 2
