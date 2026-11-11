"""Actual decoder updates, frozen initialization, scoped heads and resume."""

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import _build_policy
from maths_ai.gnn_inference.atp_lean_gnn.math_templates import MATH_SCHEMA, MathTerm
from maths_ai.gnn_inference.atp_lean_gnn.scoped_templates import BinderGroup, ScopeBinder
from maths_ai.gnn_inference.atp_lean_gnn.structured_codec import StructuredActionCodec
from maths_ai.gnn_inference.atp_lean_gnn.structured_decoder import StructuredDecoder
from maths_ai.gnn_inference.atp_lean_gnn.structured_features import FORMAT, FeatureStore, atomic_json, atomic_torch
from maths_ai.gnn_inference.atp_lean_gnn.structured_training import StructuredTrainingConfig, evaluate, train_decoder
from maths_ai.gnn_inference.atp_lean_gnn.structured_vocabulary import StructuredVocabulary
from maths_ai.gnn_inference.atp_lean_gnn.wrapper_templates import WrappedTactic


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def learning_data(tmp_path):
    torch.manual_seed(11)
    config = {"pointer_config": {"prepared_root": ".", "run_root": ".", "gnn_type": "sage", "max_args": 3,
        "model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0, "max_args": 3}}}
    policy = _build_policy(config, {"<UNK>": 0, "State": 1, "FV0": 2}, {"<UNK_TACTIC>": 0, "exact": 1}, 8)
    policy.eval().requires_grad_(False)
    context = [{"context_index": 0, "user_name": "h0"}, {"context_index": 1, "user_name": "h1"}]
    codec = StructuredActionCodec({"A.first": 71, "A.second": 72, "A.missing": 73})
    local = {"schema": "flat_reference_v2", "family": "exact", "arguments": [{"source": "local", "identity": 0}]}
    library = {"schema": "flat_reference_v2", "family": "exact", "arguments": [{"source": "library", "identity": "A.second"}]}
    missing = {**library, "arguments": [{"source": "library", "identity": "A.missing"}]}
    expression = MathTerm("lambda", (MathTerm("lambda", (MathTerm("bound_reference", value="0"),),
        groups=(BinderGroup((ScopeBinder(1, "y"),)),)),), groups=(BinderGroup((ScopeBinder(0, "x"),)),))
    bound = WrappedTactic("exact", terms=(expression,)).to_target()
    bound["schema"] = MATH_SCHEMA
    targets = [local, library, bound, missing]
    operations = [codec.encode(target, context) for target in targets]
    vocabulary = StructuredVocabulary.fit(operations[:3])
    vectors = np.random.default_rng(11).normal(size=(3, 8)).astype(np.float32)
    records = []
    for index, ops in enumerate(operations):
        teacher_ids = tuple(dict.fromkeys(op["value"] for op in ops if op["token"] == "LIBRARY"))
        records.append({"row_index": index, "family": "exact", "tactic_id": 1, "baseline_tactic_id": 1,
            "state": torch.randn(8), "local_ids": (0, 1), "local_vectors": torch.randn(2, 8),
            "library_ids": (71, 72), "library_positions": (0, 1), "library_scores": torch.tensor([0.2, 0.8]),
            "teacher_library_ids": teacher_ids, "teacher_library_positions": tuple(identity - 71 for identity in teacher_ids),
            "library_positions_total": len(teacher_ids), "library_positions_missing": int(73 in teacher_ids),
            **vocabulary.pack(ops)})
    bundle = tmp_path / "bundle"
    vector_path = bundle / "retriever/index/lemma_vectors.npy"
    vector_path.parent.mkdir(parents=True)
    np.save(vector_path, vectors)
    atomic_json(bundle / "bundle.json", {"files": {"retriever/index/lemma_vectors.npy": sha(vector_path)}})
    root = tmp_path / "features"
    root.mkdir()
    binding = {"format": FORMAT, "hidden_dim": 8, "text_dim": 8, "bundle_dir": str(bundle), "bundle_sha256": sha(bundle / "bundle.json")}
    binding_sha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    splits = {}
    for split, rows, denominator in (("train", records[:3], 3), ("val", [records[0], records[3]], 3)):
        path = root / f"{split}.pt"
        atomic_torch(path, {"binding_sha256": binding_sha, "split": split, "rows": [{**row, "split": split} for row in rows]})
        splits[split] = {"candidate_rows": denominator, "compiled_rows": len(rows), "feature_rows": len(rows),
            "shards": [{"file": path.name, "count": len(rows), "sha256": sha(path)}]}
    atomic_json(root / "manifest.json", {"binding": binding, "binding_sha256": binding_sha, "complete": True, "splits": splits})
    return policy, vocabulary, FeatureStore(root)


def test_actual_updates_reduce_loss_without_changing_frozen_policy(learning_data):
    policy, vocabulary, features = learning_data
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    torch.manual_seed(19)
    model = StructuredDecoder(policy, vocabulary)
    rows = next(features.batches("train", batch_size=3, max_tokens=4096, seed=42, epoch=0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    first, _ = model(rows)
    initial_loss = float(first.detach())
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(rows)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    final, stats = model(rows)
    assert float(final.detach()) < initial_loss * 0.8
    assert stats["bound_positions"] and model.lexical_key[0].weight.grad is not None
    assert all(not parameter.requires_grad for parameter in policy.parameters())
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())
    assert model.gru.weight_hh.data_ptr() != policy.pointer.argument_selector.gru.weight_hh.data_ptr()


def test_missing_library_is_never_injected_or_counted_as_exact(learning_data):
    policy, vocabulary, features = learning_data
    rows = next(features.batches("val", batch_size=2, max_tokens=4096, seed=0, epoch=0))
    row = next(row for row in rows if row["library_positions_missing"])
    model = StructuredDecoder(policy, vocabulary)

    class PerfectTokens(nn.Module):
        def __init__(self):
            super().__init__()
            self.position = 1

        def forward(self, hidden):
            logits = hidden.new_full((1, len(vocabulary.tokens)), -100)
            logits[0, row["token_ids"][self.position]] = 100
            self.position += 1
            return logits

    model.token_head = PerfectTokens()
    _, stats = model([row])
    assert stats["token_hits"] == stats["tokens"]
    assert stats["library_missing_positions"] == 1 and stats["teacher_forced_exact_hits"] == 0
    assert row["library_ids"] == (71, 72) and row["teacher_library_ids"] == (73,)


def test_validation_denominator_keeps_uncompiled_and_unretrieved_rows(learning_data):
    policy, vocabulary, features = learning_data
    result = evaluate(StructuredDecoder(policy, vocabulary), features, StructuredTrainingConfig(batch_size=2))
    assert result["rows"] == 2 and result["candidate_rows"] == 3
    assert result["uncompiled_misses"] == 1 and result["library_missing_positions"] == 1
    assert result["teacher_forced_exact_rate"] <= 1 / 3
    assert not result["proof_success_measured"] and result["previous_arguments"] == "gold_teacher_forcing"


def test_resume_matches_uninterrupted_updates_and_rejects_binding_changes(tmp_path, learning_data):
    policy, vocabulary, features = learning_data
    torch.manual_seed(31)
    initial = StructuredDecoder(policy, vocabulary)
    initial_state = copy.deepcopy(initial.state_dict())
    config = StructuredTrainingConfig(epochs=2, batch_size=2, max_batch_tokens=4096, learning_rate=0.01,
        checkpoint_every=1, max_updates=1)
    paused = train_decoder(initial, features, tmp_path / "resumed", config)
    assert paused["status"] == "update_budget_reached" and paused["global_updates"] == 1
    continuation = StructuredDecoder(policy, vocabulary)
    resumed = train_decoder(continuation, features, tmp_path / "resumed", replace(config, max_updates=0), resume=True)
    straight = StructuredDecoder(policy, vocabulary)
    straight.load_state_dict(initial_state)
    complete = train_decoder(straight, features, tmp_path / "straight", replace(config, max_updates=0))
    assert resumed["global_updates"] == complete["global_updates"] == 4
    for name, value in straight.state_dict().items():
        torch.testing.assert_close(value, continuation.state_dict()[name], atol=1e-7, rtol=0)
    repeated = train_decoder(continuation, features, tmp_path / "resumed", replace(config, max_updates=0), resume=True)
    assert repeated["status"] == "complete" and repeated["global_updates"] == 4
    with pytest.raises(ValueError, match="bindings"):
        train_decoder(continuation, features, tmp_path / "resumed", replace(config, learning_rate=0.02), resume=True)


def test_illegal_local_scope_is_not_silently_skipped(learning_data):
    policy, vocabulary, features = learning_data
    row = next(batch[0] for batch in features.batches("train", batch_size=1, max_tokens=4096, seed=42, epoch=0)
               if any(vocabulary.tokens[token] == "LOCAL" for token in batch[0]["token_ids"]))
    positions = [index for index, token in enumerate(row["token_ids"]) if vocabulary.tokens[token] == "LOCAL"]
    assert positions
    row["input_local_scopes"][positions[0]] = [1]
    with pytest.raises(ValueError, match="legal input scope"):
        StructuredDecoder(policy, vocabulary)([row])
