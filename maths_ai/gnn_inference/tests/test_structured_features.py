"""Normalized graph encoding and exact frozen-cache identity alignment."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import _build_policy
from maths_ai.gnn_inference.atp_lean_gnn import structured_features
from maths_ai.gnn_inference.atp_lean_gnn.structured_features import FeatureStore, MatchedRetrievalCache, build_feature_cache
from maths_ai.gnn_inference.scripts.compile_structured_supervision import compile_candidates
from maths_ai.gnn_inference.scripts.prepare_structured_supervision import prepare_candidates
from maths_ai.gnn_inference.tests.test_structured_supervision import frozen_candidates


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def feature_inputs(tmp_path, frozen_candidates):
    recovery, _, bundle = frozen_candidates
    vectors = np.arange(8, dtype=np.float32).reshape(1, 8) / 10
    vector_path = bundle / "retriever/index/lemma_vectors.npy"
    np.save(vector_path, vectors)
    policy_dir = bundle / "policy"
    policy_dir.mkdir()
    (policy_dir / "config.json").write_text(json.dumps({"input_contract": {"edge_mode": "bidirectional"}}))
    cache = tmp_path / "text_candidates"
    cache.mkdir()
    index_manifest = {"model_name": "local_encoder", "lemma_encoder_sha256": "fixture_lemma_encoder"}
    metadata = {"schema": "text_library_candidates_v1", "dataset_name": "fixture", "k": 1,
        "query_policy": "goal_first_v1", "selection_policy": "exact_cosine_no_gold_injection",
        "text_index_manifest": {**index_manifest, "model_name": "/original/training/model"}, "splits": {}}
    for split in ("train", "val"):
        arrays = {"rows": np.arange(7, dtype=np.int64), "ids": np.full((7, 1), 71, dtype=np.int32),
                  "scores": np.full((7, 1), 0.5, dtype=np.float32)}
        for name, array in arrays.items():
            np.save(cache / f"{split}_{name}.npy", array)
        metadata["splits"][split] = {"count": 7, **{f"{name}_sha256": sha(cache / f"{split}_{name}.npy") for name in arrays}}
    (cache / "manifest.json").write_text(json.dumps(metadata))
    provenance = bundle / "provenance"
    provenance.mkdir()
    (provenance / "training.json").write_text(json.dumps({"source_candidate_manifest_sha256": sha(cache / "manifest.json")}))
    manifest = json.loads((bundle / "bundle.json").read_text())
    manifest["files"].update({"retriever/index/lemma_vectors.npy": sha(vector_path),
                              "policy/config.json": sha(policy_dir / "config.json"),
                              "provenance/training.json": sha(provenance / "training.json")})
    manifest.update(policy_state_sha256="fixture_policy", retriever_state_sha256="fixture_retriever")
    (bundle / "bundle.json").write_text(json.dumps(manifest))
    candidate = tmp_path / "candidates"
    source = prepare_candidates(recovery, bundle, candidate, train_limit=0, val_limit=0)
    labels = [json.loads(line) for line in (candidate / "val.jsonl").read_text().splitlines()]
    required = ("input_model_match", "original_executed", "original_transition_match", "original_text_executed",
                "original_text_transition_match", "rendered_executed", "rendered_transition_match")
    native = {"complete": True, "manifest": {"schema": "native_gold_step_replay_v2", "renderer": "template-v9",
        "split": "val", "original_replay_mode": "captured_syntax", "cache_root": source["manifest"]["cache_root"],
        "row_indices_sha256": sha(candidate / "native_val_pilot_rows.json"),
        "rows": [{"row_index": row["row_index"], "cache_hashes": row["cache_sha256"]} for row in labels]},
        "files": {"Fixture.lean": {"rows": [{"row_index": row["row_index"], "rendered_tactic": row["rendered_tactic"],
            "status": "rendered_transition_match", "original_control": "captured_syntax", "alignment_kind": "exact",
            "dataset_target_matches_source": True, **{key: True for key in required}} for row in labels]}}}
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    (native_dir / "report.json").write_text(json.dumps(native))
    compiled = tmp_path / "compiled"
    compile_candidates(candidate, native_dir / "report.json", compiled)
    node_vocab = {name: index for index, name in enumerate(["<UNK>", "FV0", "State", "True", "Hyp", "HypRole:explicit", "h0", "Goal"])}
    tactics = {"<UNK_TACTIC>": 0, "exact": 1, "rfl": 2}
    policy = _build_policy({"pointer_config": {"prepared_root": ".", "run_root": ".", "gnn_type": "sage", "max_args": 3,
        "model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0, "max_args": 3}}}, node_vocab, tactics, 8).eval().requires_grad_(False)
    system = SimpleNamespace(root=bundle, policy=policy, retriever=nn.Linear(1, 1),
        lemma_index=SimpleNamespace(manifest=index_manifest, lemma_vectors=vectors, id_to_position={71: 0}),
        node_vocab=node_vocab, tactic_vocab=tactics, manifest=manifest, inference_config={"retrieval_k": 1})
    return compiled, cache, system


def test_features_use_real_gnn_and_preserve_frozen_components(tmp_path, feature_inputs, monkeypatch):
    compiled, cache, system = feature_inputs
    before = {name: value.clone() for name, value in system.policy.state_dict().items()}
    monkeypatch.setattr(system.retriever, "forward", lambda *args: pytest.fail("frozen retrieval pool was re-queried"))
    store = build_feature_cache(compiled, cache, system, tmp_path / "features", shard_size=2, graph_batch_size=2)
    assert all(torch.equal(value, system.policy.state_dict()[name]) for name, value in before.items())
    assert all(not parameter.requires_grad for parameter in system.policy.parameters())
    assert all(not parameter.requires_grad for parameter in system.retriever.parameters())
    rows = [row for batch in store.batches("train", batch_size=2, max_tokens=4096, seed=42, epoch=0) for row in batch]
    assert len(rows) == 3
    assert all(row["state"].shape == (8,) and row["local_vectors"].shape == (1, 8) for row in rows)
    assert all(row["local_ids"] == (0,) and row["library_ids"] == (71,) for row in rows)
    assert all(not row["state"].requires_grad for row in rows)
    assert store.manifest["binding"]["gold_injected_into_retrieval_pool"] is False
    assert store.manifest["ready_for_training"] and store.manifest["retrieval_reachability_measured"]
    assert store.manifest["splits"]["train"]["retrieval_counts"]["nonempty_library_rows"] == 1
    monkeypatch.setattr(system.policy.pointer.backbone, "encode_nodes", lambda *args: pytest.fail("completed cache re-encoded"))
    resumed = build_feature_cache(compiled, cache, system, tmp_path / "features", shard_size=2, graph_batch_size=2, resume=True)
    assert resumed.manifest == store.manifest


def test_retrieval_arrays_and_feature_shards_are_hash_bound(tmp_path, feature_inputs):
    compiled, cache, system = feature_inputs
    store = build_feature_cache(compiled, cache, system, tmp_path / "features")
    shard = store.root / store.manifest["splits"]["train"]["shards"][0]["file"]
    shard.write_bytes(shard.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest"):
        FeatureStore(store.root)
    np.save(cache / "train_scores.npy", np.zeros((7, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="array changed"):
        MatchedRetrievalCache(cache, system)


def test_interrupted_feature_write_adopts_only_bound_orphan(tmp_path, feature_inputs, monkeypatch):
    compiled, cache, system = feature_inputs
    writer = structured_features.atomic_json

    def interrupted(path, payload):
        if payload.get("splits", {}).get("train", {}).get("next_row", 0):
            raise RuntimeError("simulated interruption after shard write")
        writer(path, payload)

    monkeypatch.setattr(structured_features, "atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_feature_cache(compiled, cache, system, tmp_path / "features")
    monkeypatch.setattr(structured_features, "atomic_json", writer)
    store = build_feature_cache(compiled, cache, system, tmp_path / "features", resume=True)
    assert store.manifest["complete"]
    assert store.manifest["splits"]["train"]["feature_rows"] == store.manifest["splits"]["val"]["feature_rows"] == 3


def test_training_cli_performs_real_update_and_writes_resumable_checkpoint(tmp_path, feature_inputs, monkeypatch):
    from maths_ai.gnn_inference.scripts import train_structured_decoder
    compiled, cache, system = feature_inputs
    monkeypatch.setattr(train_structured_decoder, "load_joint_source_bundle", lambda *args, **kwargs: system)
    run = tmp_path / "training_run"
    args = ["--compiled-dir", str(compiled), "--bundle-dir", str(system.root),
        "--candidate-cache", str(cache), "--feature-cache", str(tmp_path / "features"),
        "--run-dir", str(run), "--device", "cpu", "--batch-size", "2",
        "--epochs", "2", "--max-updates", "1", "--checkpoint-every", "1"]
    assert train_structured_decoder.main(args) == 0
    report = json.loads((run / "report.json").read_text())
    assert report["status"] == "update_budget_reached" and report["global_updates"] == 1
    assert report["val_metrics"]["candidate_rows"] == 3
    checkpoint = torch.load(run / "last.pt", weights_only=False)
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["model_type"] == "structured_source_decoder_v1"
    assert train_structured_decoder.main([*args, "--resume"]) == 0
    assert json.loads((run / "report.json").read_text())["global_updates"] == 1
