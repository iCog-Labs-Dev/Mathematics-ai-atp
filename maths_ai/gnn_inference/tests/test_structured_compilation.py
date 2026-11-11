"""Candidate/native bindings, exact codec rendering, and train-only vocabulary."""

import hashlib
import json

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.structured_codec import BYTE_TOKENS, StructuredActionCodec
from maths_ai.gnn_inference.atp_lean_gnn.structured_vocabulary import StructuredVocabulary, VocabularyError
from maths_ai.gnn_inference.scripts.compile_structured_supervision import compile_candidates
from maths_ai.gnn_inference.scripts.prepare_structured_supervision import prepare_candidates
from maths_ai.gnn_inference.tests.test_structured_supervision import frozen_candidates
from maths_ai.gnn_inference.tests.test_template_coverage import _snapshot


def test_vocabulary_fits_only_train_and_unseen_names_use_fixed_bytes():
    codec = StructuredActionCodec({})
    train = {"schema": "intro_binders_v1", "family": "intro", "binders": [
        {"slot": 0, "kind": "named", "gold_name": "hp"}], "explicit_binder_count": 1, "mode": "explicit", "reference_count": 0}
    validation = {**train, "binders": [{"slot": 0, "kind": "named", "gold_name": "新β"}]}
    vocabulary = StructuredVocabulary.fit([codec.encode(train, [])])
    before = vocabulary.to_dict()
    packed = vocabulary.pack(codec.encode(validation, []))
    assert vocabulary.to_dict() == before
    assert all(token in vocabulary.tokens for token in BYTE_TOKENS)
    assert (len(packed["token_ids"]) == len(packed["pointer_targets"]) == len(packed["declaration_slots"])
            == len(packed["lexical_scopes"]) == len(packed["input_local_scopes"]))
    assert vocabulary.tokens[packed["token_ids"][0]] == "<BOS>"
    assert all(value == -1 for value in packed["pointer_targets"])
    assert StructuredVocabulary.from_dict(vocabulary.to_dict()) == vocabulary
    with pytest.raises(VocabularyError, match="only fit train"):
        StructuredVocabulary.fit([codec.encode(validation, [])], split="val")


def test_validation_only_family_is_an_explicit_oov():
    codec = StructuredActionCodec({})
    train = {"schema": "flat_reference_v2", "family": "rfl", "arguments": []}
    validation = {**train, "family": "assumption"}
    vocabulary = StructuredVocabulary.fit([codec.encode(train, [])])
    with pytest.raises(VocabularyError, match="FAMILY:assumption"):
        vocabulary.pack(codec.encode(validation, []))


@pytest.fixture
def candidate_package(tmp_path, frozen_candidates):
    recovery, _, bundle = frozen_candidates
    candidate = tmp_path / "candidates"
    source = prepare_candidates(recovery, bundle, candidate, train_limit=0, val_limit=0)
    labels = [json.loads(line) for line in (candidate / "val.jsonl").read_text().splitlines()]
    selected_path = candidate / "native_val_pilot_rows.json"
    required = ("input_model_match", "original_executed", "original_transition_match", "original_text_executed",
                "original_text_transition_match", "rendered_executed", "rendered_transition_match")
    native = {"complete": True, "manifest": {"schema": "native_gold_step_replay_v2", "renderer": "template-v9",
        "split": "val", "original_replay_mode": "captured_syntax", "cache_root": source["manifest"]["cache_root"],
        "row_indices_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "rows": [{"row_index": label["row_index"], "cache_hashes": label["cache_sha256"]} for label in labels]},
        "files": {"Fixture.lean": {"rows": [{"row_index": label["row_index"], "rendered_tactic": label["rendered_tactic"],
            "status": "rendered_transition_match", "original_control": "captured_syntax", "alignment_kind": "exact",
            "dataset_target_matches_source": True, **{key: True for key in required}} for label in labels]}}}
    directory = tmp_path / "native"
    directory.mkdir()
    path = directory / "report.json"
    path.write_text(json.dumps(native))
    return candidate, path


def test_compilation_preserves_sources_and_marks_only_measured_native_rows(tmp_path, candidate_package):
    candidate, native = candidate_package
    before = _snapshot(tmp_path)
    output = tmp_path / "compiled"
    report = compile_candidates(candidate, native, output)
    assert report["complete"] and report["native_codec_gate_passed"]
    assert report["native_codec_matched_rows"] == report["native_pilot_expected_rows"] == 3
    assert report["ready_for_training"] is False and not report["manifest"]["retrieval_reachability_measured"]
    assert report["manifest"]["vocabulary_fit_split"] == "train"
    for split in ("train", "val"):
        assert report["splits"][split]["counts"]["compiled_rows"] == 3
        labels = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        assert all(label["neural_target_compiled"] and not label["candidate_pool_positions_aligned"] for label in labels)
        assert {label["native_validation"] for label in labels} == {"matched" if split == "val" else "not_measured"}
        assert all(len(label["token_ids"]) == len(label["operations"]) + 1 for label in labels)
    after = _snapshot(tmp_path)
    assert all(after[name] == digest for name, digest in before.items())
    with pytest.raises(FileExistsError):
        compile_candidates(candidate, native, output)


@pytest.mark.parametrize("damage", ["original_failure", "cache_binding", "candidate_digest"])
def test_compilation_rejects_untrusted_pilot_or_changed_candidates(tmp_path, candidate_package, damage):
    candidate, native = candidate_package
    report = json.loads(native.read_text())
    if damage == "original_failure":
        report["files"]["Fixture.lean"]["rows"][0]["original_transition_match"] = False
        native.write_text(json.dumps(report))
    elif damage == "cache_binding":
        report["manifest"]["rows"][0]["cache_hashes"]["sexpr"] = "stale"
        native.write_text(json.dumps(report))
    else:
        path = candidate / "val.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        compile_candidates(candidate, native, tmp_path / "invalid")
