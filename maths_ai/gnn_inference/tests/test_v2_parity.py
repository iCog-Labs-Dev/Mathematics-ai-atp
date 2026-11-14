"""Parity tests for the published Mathlib-Sexpr-GNN v2 forward computations.

These lock the repo's inference modules to the reference implementation the
published bundle was trained with (Mathematics-ai-atp @ d9e09cc):
reference attention readout equations, binder-depth bucketing, the trained
lemma_adapter Tanh, dual-encoder tower splitting, and the published
``policy/inference.json`` retrieval width.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import JointSourcePolicy
from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import (
    JointBundle,
    load_joint_source_bundle,
)
from maths_ai.gnn_inference.atp_lean_gnn.joint_loader import (
    JointNeuralBundle,
    split_dual_encoder_weights,
)
from maths_ai.gnn_inference.atp_lean_gnn.model import PublishedAttentionReadout, PublishedGATBackbone


# ---------------------------------------------------------------------------
# Reference attention readout equations (state_mean_attention)
# ---------------------------------------------------------------------------


def test_published_attention_readout_matches_reference_equations() -> None:
    torch.manual_seed(0)
    hidden = 8
    readout = PublishedAttentionReadout(hidden)
    node_embeddings = torch.randn(5, hidden)
    state_embeddings = torch.randn(2, hidden)
    batch_index = torch.tensor([0, 0, 0, 1, 1])

    fused, weights = readout(node_embeddings, state_embeddings, batch_index)

    # Reference StateMeanAttentionReadout: per-node state context, tanh
    # scores, graph softmax, add+mean pooling, LayerNorm+GELU fusion.
    state_per_node = state_embeddings.index_select(0, batch_index)
    scores = readout.attention_score(
        torch.tanh(
            readout.node_projection(node_embeddings)
            + readout.state_projection(state_per_node)
        )
    ).squeeze(-1)
    reference_weights = torch.cat(
        [torch.softmax(scores[batch_index == graph], dim=0) for graph in range(2)]
    )
    reference_weighted = torch.stack(
        [
            (reference_weights.unsqueeze(-1) * node_embeddings)[batch_index == graph].sum(0)
            for graph in range(2)
        ]
    )
    reference_mean = torch.stack(
        [node_embeddings[batch_index == graph].mean(0) for graph in range(2)]
    )
    reference_combined = torch.cat(
        [state_embeddings, reference_mean, reference_weighted], dim=-1
    )
    reference_fused = F.gelu(readout.normalization(readout.fusion(reference_combined)))

    assert torch.allclose(weights, reference_weights, atol=1e-6)
    assert torch.allclose(fused, reference_fused, atol=1e-6)
    assert fused.shape == (2, hidden)


def test_published_attention_readout_fuses_three_summaries() -> None:
    # The published checkpoint's fusion.weight is [hidden, 3 * hidden]:
    # state + mean + attention-weighted add pool.
    readout = PublishedAttentionReadout(16)
    assert readout.fusion.weight.shape == (16, 48)
    assert readout.attention_score.weight.shape == (1, 16)


def test_published_policy_keeps_trained_lemma_adapter_tanh() -> None:
    torch.manual_seed(0)
    policy = JointSourcePolicy(
        num_node_labels=10,
        num_tactics=6,
        num_node_types=4,
        hidden_dim=8,
        num_layers=2,
        heads=2,
        dropout=0.0,
        use_node_type=True,
        max_args=3,
    )
    # Reference layout: Linear → LayerNorm → Tanh (Tanh is parameter-less, so
    # the published checkpoint's .0/.1 keys strict-load over it).
    assert len(policy.lemma_adapter) == 3
    assert isinstance(policy.lemma_adapter[2], torch.nn.Tanh)
    adapted = policy.lemma_adapter(torch.randn(2, 384))
    assert adapted.abs().max() <= 1.0 + 1e-6


def test_published_backbone_buckets_out_of_range_binder_depth() -> None:
    backbone = PublishedGATBackbone(
        num_node_labels=5,
        num_tactics=3,
        num_node_types=11,
        hidden_dim=8,
        num_layers=1,
        heads=2,
        dropout=0.0,
        use_node_type=True,
    )
    data = Data(
        x=torch.tensor([0, 1]),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        node_type=torch.zeros(2, dtype=torch.long),
        is_bound=torch.tensor([1, 0]),
        # Depth 15 exceeds the 10-row embedding trained on; the reference
        # clamps instead of crashing the lookup.
        binder_depth=torch.tensor([15, 2]),
        binder_kind=torch.zeros(2, dtype=torch.long),
        state_node_index=torch.tensor([0]),
    )
    embeddings = backbone.encode_nodes(data)
    assert embeddings.shape == (2, 8)


def test_published_backbone_readout_without_batch_vector() -> None:
    backbone = PublishedGATBackbone(
        num_node_labels=5,
        num_tactics=3,
        num_node_types=11,
        hidden_dim=8,
        num_layers=1,
        heads=2,
        dropout=0.0,
        use_node_type=True,
    )
    data = Data(
        x=torch.tensor([0, 1]),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        node_type=torch.zeros(2, dtype=torch.long),
        is_bound=torch.zeros(2, dtype=torch.long),
        binder_depth=torch.zeros(2, dtype=torch.long),
        binder_kind=torch.zeros(2, dtype=torch.long),
        state_node_index=torch.tensor([0]),
    )
    node_embeddings = backbone.encode_nodes(data)
    state = backbone.readout(node_embeddings, data)
    assert state.shape == (1, 8)


# ---------------------------------------------------------------------------
# Dual-encoder tower splitting
# ---------------------------------------------------------------------------


def test_split_dual_encoder_weights_separates_towers() -> None:
    flat = {
        "state_encoder.embeddings.word_embeddings.weight": torch.randn(4, 4),
        "state_encoder.pooler.dense.weight": torch.randn(4, 4),
        "lemma_encoder.embeddings.word_embeddings.weight": torch.randn(4, 4),
        "lemma_encoder.pooler.dense.weight": torch.randn(4, 4),
    }
    state_tower, lemma_tower = split_dual_encoder_weights(flat)
    assert set(state_tower) == {"embeddings.word_embeddings.weight", "pooler.dense.weight"}
    assert set(lemma_tower) == {"embeddings.word_embeddings.weight", "pooler.dense.weight"}
    assert (
        state_tower["pooler.dense.weight"].data_ptr()
        != lemma_tower["pooler.dense.weight"].data_ptr()
    )


def test_split_dual_encoder_weights_fail_closed_on_foreign_prefix() -> None:
    with pytest.raises(ValueError, match="outside the dual-encoder towers"):
        split_dual_encoder_weights({"encoder.weight": torch.randn(4, 4)})
    with pytest.raises(ValueError, match="missing the state_encoder tower"):
        split_dual_encoder_weights({"lemma_encoder.weight": torch.randn(4, 4)})


# ---------------------------------------------------------------------------
# Published policy/inference.json retrieval width
# ---------------------------------------------------------------------------


def _stub_bundle(retrieval_k: int | None) -> JointNeuralBundle:
    inference_config = {} if retrieval_k is None else {"retrieval_k": retrieval_k}
    artifact = JointBundle(
        root=Path("."),
        manifest={},
        policy_config={},
        retriever_config={},
        index_manifest={},
        inference_config=inference_config,
    )
    return JointNeuralBundle(
        artifact=artifact,
        policy=None,
        tokenizer=None,
        text_encoder=None,
        lemma_encoder=None,
        lemma_index=SimpleNamespace(lemma_ids=list(range(1000))),
        lemma_corpus={},
        node_vocab={},
        tactic_vocab={},
        device=torch.device("cpu"),
    )


def test_neural_bundle_honors_published_retrieval_k() -> None:
    # policy/inference.json declares retrieval_k: 200 — the manifest has no
    # "k" field, so the old index_manifest.get("k", 500) default was wrong.
    assert _stub_bundle(200).pool_size() == 200
    assert _stub_bundle(None).pool_size() == 500  # legacy fallback
    assert _stub_bundle(200).pool_size(k=5) == 5


def test_load_joint_source_bundle_reads_inference_json(tmp_path: Path) -> None:
    from maths_ai.gnn_inference.tests.test_joint_bundle import _write_bundle

    _write_bundle(tmp_path)
    (tmp_path / "policy" / "inference.json").write_text(
        '{"retrieval_k": 200, "max_args": 3}', encoding="utf-8"
    )
    bundle = load_joint_source_bundle(tmp_path, expected_revision="revision-v2")
    assert bundle.inference_config["retrieval_k"] == 200
    assert bundle.default_retrieval_k == 200


def test_load_joint_source_bundle_rejects_invalid_retrieval_k(tmp_path: Path) -> None:
    from maths_ai.gnn_inference.tests.test_joint_bundle import _write_bundle

    _write_bundle(tmp_path)
    (tmp_path / "policy" / "inference.json").write_text(
        '{"retrieval_k": -5}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="retrieval_k"):
        load_joint_source_bundle(tmp_path, expected_revision="revision-v2")
