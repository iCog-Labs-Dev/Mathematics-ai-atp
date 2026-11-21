"""File-source decoding preserves executable and model context together."""

from dataclasses import replace
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.goal_file_records import decode_goal_record, STRUCTURED_GOAL_SCHEMA
from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import AdmissionReport, _file_row_events
from maths_ai.lean_replay import ReplayInputError, render_replay


def simple_record():
    # Raw constant syntax is verified by the shared raw decoder.
    return {"schema": STRUCTURED_GOAL_SCHEMA, "target_sexp": "(:c True)",
            "model_target": "(:c True)", "locals": [], "source": {"split": "val"}}


def decode(row):
    return decode_goal_record(row, path=Path("goals.jsonl"), line=7)


def test_text_provenance_cannot_override_file_identity():
    seed = decode({"goal": "True", "source": {"file": "forged", "line": 99}})
    assert seed.source_dict["file"] == "goals.jsonl"
    assert seed.source_dict["line"] == 7
    assert seed.source_dict["record_source"]["file"] == "forged"
    assert render_replay(seed).expression == "True"


@pytest.mark.parametrize("row", [[], {}, {"goal": 3}, {"goal": "True", "hypotheses": [1]},
    {"goal": "True", "source": "bad"}, {"goal": "True", "locals": []},
    {"schema": "unknown", "goal": "True"},
    {"schema": STRUCTURED_GOAL_SCHEMA, "target_sexp": "x", "locals": []}])
def test_invalid_records_fail_without_text_recovery(row):
    with pytest.raises(ReplayInputError):
        decode(row)


def test_structured_model_metadata_is_part_of_validation_identity():
    seed = decode(simple_record())
    assert seed.origin == "structured"
    assert seed.source_dict["record_source"]["split"] == "val"
    assert render_replay(seed).fingerprint != render_replay(replace(seed, model_target="changed")).fingerprint


def test_mixed_structured_record_rejected():
    row = simple_record()
    row["goal"] = "False"
    with pytest.raises(ReplayInputError, match="mixed"):
        decode(row)


def test_malformed_file_rows_are_accounted_and_next_row_survives(tmp_path):
    path = tmp_path / "goals.jsonl"
    path.write_text('invalid\n[]\n\n{"goal":"True"}\n')
    report = AdmissionReport()
    events = list(_file_row_events(path, max_scan_rows=4, max_state_chars=None, report=report))
    assert report.scanned == 4
    assert len(events) == 4
    assert events[-1].item.seed.target == "True"
    assert sum(report.quarantined.values()) == 2


def test_local_types_and_identity_are_checked_before_lean():
    row = simple_record()
    row["locals"] = [{"internal_id":"x", "name":"x", "context_index":0,
        "binder_role":":implicit", "is_instance":False, "is_let":False,
        "type_sexp":"(:const Nat)", "model_name":"other", "model_index":0,
        "model_type_sexp":"(:const Nat)"}]
    with pytest.raises(ReplayInputError, match="model identity"):
        decode(row)
    row["locals"][0]["model_name"] = "x"
    row["locals"][0]["is_instance"] = "false"
    with pytest.raises(ReplayInputError, match="boolean"):
        decode(row)


def test_text_record_size_is_its_printed_state():
    from maths_ai.gnn_inference.atp_lean_gnn.goal_file_records import record_size
    labelled = {"goal": "p → p", "hypotheses": ["p : Prop"], "tactic": "intro"}
    assert record_size(labelled) == record_size({"goal": "p → p", "hypotheses": ["p : Prop"]}) == 13
    structured = {"schema": "lean-structured-goal-v1", "target_sexp": "x", "model_target": "x", "locals": []}
    assert record_size(structured) > len("x")
