import json
from types import SimpleNamespace
import torch
import pytest
from maths_ai.gnn_inference.tests.test_source_aware_integration import fixture
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import file_hash
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import (
    build_model_from_checkpoint,
)
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC
from maths_ai.gnn_inference.scripts.convert_source_aware_checkpoint import convert


def test_source_layout_maps_all_trained_heads_and_binds_vocabulary(tmp_path):
    model, batch, pools = fixture()
    source = {}
    for key, value in model.state_dict().items():
        if key.startswith("encoder.node_features."):
            target = "pointer.backbone." + key.removeprefix("encoder.node_features.")
        elif key.startswith("encoder."):
            target = "pointer.backbone." + key.removeprefix("encoder.")
        elif key.startswith("actor.base."):
            target = "pointer.backbone.classifier." + key.removeprefix("actor.base.")
        elif key.startswith(
            ("decoder.tactic_embedding.", "decoder.argument_selector.")
        ):
            target = "pointer." + key.removeprefix("decoder.")
        elif key.startswith("decoder."):
            target = key.removeprefix("decoder.")
        else:
            continue
        source[target] = value
    source["pointer.stop_head.weight"] = torch.zeros(1, 8)
    source["pointer.stop_head.bias"] = torch.zeros(1)
    fields = {
        "node_vocab_sha256": "vocab/node_vocab.json",
        "tactic_vocab_sha256": "vocab/tactic_vocab.json",
        "argument_label_binding_sha256": "argument_labels.json",
        "prepared_packed_manifest_sha256": "packed/bidirectional/manifest.json",
    }
    for path in fields.values():
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("{}")
    manifest = {key: file_hash(tmp_path / path) for key, path in fields.items()}
    candidate = tmp_path / "candidates.json"
    candidate.write_text(json.dumps(manifest))
    metadata = SimpleNamespace(
        root=tmp_path,
        node_vocab={str(i): i for i in range(5)},
        tactic_vocab={str(i): i for i in range(3)},
        graph_representation=MODEL_SEXPR_GRAPH_SPEC,
    )
    checkpoint = dict(
        model_type="joint_source_flat_decoder_v2",
        config=dict(
            gnn_type="gat",
            model=dict(
                hidden_dim=8,
                dropout=0.5,
                num_layers=1,
                heads=2,
                readout="state_mean_attention",
            ),
            max_args=3,
            use_node_type=True,
        ),
        binding=dict(
            text_index_manifest={},
            text_checkpoint_sha256="text-fixture",
            candidate_manifest_sha256=file_hash(candidate),
        ),
        model_state_dict=source,
    )
    converted = convert(
        checkpoint,
        metadata=metadata,
        index_manifest={
            "source_manifest": {},
            "text_checkpoint_sha256": "text-fixture",
        },
        candidate_manifest_path=candidate,
    )
    loaded, _, _ = build_model_from_checkpoint(
        converted,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        graph_representation=MODEL_SEXPR_GRAPH_SPEC,
    )
    for key, value in model.decoder.state_dict().items():
        torch.testing.assert_close(value, loaded.decoder.state_dict()[key])
    first = model.act(batch, candidate_pools=pools, greedy=True)
    from dataclasses import replace

    bound_pools = tuple(
        replace(p, artifact_fingerprint=loaded.retrieval_binding) for p in pools
    )
    second = loaded.act(batch, candidate_pools=bound_pools, greedy=True)
    torch.testing.assert_close(first.logp, second.logp)
    assert converted["conversion"]["new_components"] == ["critic"]
    (tmp_path / "vocab/node_vocab.json").write_text('{"changed":1}')
    with pytest.raises(ValueError, match="prepared artifact"):
        convert(
            checkpoint,
            metadata=metadata,
            index_manifest={
                "source_manifest": {},
                "text_checkpoint_sha256": "text-fixture",
            },
            candidate_manifest_path=candidate,
        )
