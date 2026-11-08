from __future__ import annotations

import faiss
import numpy as np
import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn import inference as inference_module
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.inference import (
    InferencePipeline,
    _fresh_arguments_for_tactic,
    _top_tactic_candidates,
)
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab_from_labels
from maths_ai.gnn_inference.atp_lean_gnn.state import parse_state


def test_fresh_arguments_allows_bare_intro() -> None:
    assert _fresh_arguments_for_tactic("intro", [], 1) == []


@pytest.mark.parametrize("include_local_candidates", [True, False])
def test_pipeline_preserves_bare_intro_for_implication(
    monkeypatch: pytest.MonkeyPatch, include_local_candidates: bool
) -> None:
    state = "p q : Prop\n⊢ p → q → p"
    dag = proof_state_to_dag(parse_state(state))
    node_vocab = build_vocab_from_labels([node.label for node in dag.nodes])
    model = TacticWithArgsClassifier(
        num_node_labels=len(node_vocab), num_tactics=1, hidden_dim=8, num_layers=1
    )
    index = LemmaIndex(faiss.IndexFlatL2(8), [], np.empty((0, 8), dtype=np.float32))
    if not include_local_candidates:
        monkeypatch.setattr(
            inference_module, "build_premise_mask", lambda graph: [False] * len(graph.nodes)
        )
    pipeline = InferencePipeline(
        model=model,
        scorer=PremiseScorer(hidden_dim=8),
        lemma_index=index,
        node_vocab=node_vocab,
        tactic_vocab={"intro": 0},
        device=torch.device("cpu"),
    )

    result = pipeline.predict_tactic_result(state)

    assert result.predicted_tactic == "intro"
    assert len(result.top_tactic_predictions) == 1
    assert result.top_tactic_predictions[0]["selected_arguments"] == []
    assert pipeline.filtered_arity_count == 0


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
