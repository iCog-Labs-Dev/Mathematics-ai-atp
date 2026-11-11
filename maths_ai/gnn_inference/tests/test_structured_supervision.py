"""Candidate preparation keeps exact alignment, bindings, and deferred tiers."""

import copy
import hashlib
import json

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.structured_supervision import SupervisionExclusion, make_structured_candidate
from maths_ai.gnn_inference.scripts import (
    audit_scoped_recovery, audit_selector_recovery, audit_template_coverage,
    audit_template_recovery, audit_wrapper_recovery, prepare_structured_supervision,
)
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref, trace
from maths_ai.gnn_inference.tests.test_template_coverage import _cache_row, _prepared, _snapshot


def candidate_input(root, tactic):
    data = trace(root)
    raw = {"split": "train", "row_index": 0, "tactic": tactic,
           "alignment_kind": "exact", "target_state_matches_invocation": True}
    model = {"goal_sexp": "(:c True)", "hyp_sexps": [
        {"context_index": index, "name": f"h{index}", "sexp": "(:c True)"} for index in range(5)]}
    return raw, data, model


def test_typed_local_and_library_references_keep_separate_id_spaces():
    raw, data, model = candidate_input(node("Lean.Parser.Tactic.exact", atom("exact"), ref(2)), "exact h2")
    local = make_structured_candidate(raw, data, model, {})
    assert local["required_local_indices"] == [2] and not local["required_library"]
    assert local["native_validation"] == local["retrieval_reachability"] == "not_measured"
    assert local["neural_target_compiled"] is False
    raw, data, model = candidate_input(node("Lean.Parser.Tactic.exact", atom("exact"), ref(name="True.intro")), "exact True.intro")
    library = make_structured_candidate(raw, data, model, {"True.intro": 71})
    assert library["required_library"] == [{"name": "True.intro", "lemma_id": 71}]
    assert not library["required_local_indices"]
    with pytest.raises(SupervisionExclusion, match="library_outside_matched_corpus"):
        make_structured_candidate(raw, data, model, {})


def test_binder_declarations_are_not_input_local_pointers():
    declaration = {"tag": "identifier", "source": "new_h", "semanticRole": "scoped_local"}
    raw, data, model = candidate_input(node("Lean.Parser.Tactic.intro", atom("intro"), node("null", declaration)), "intro new_h")
    result = make_structured_candidate(raw, data, model, {})
    assert result["target"]["binders"][0]["slot"] == 0
    assert result["gold_declaration_name_fields"] == 1 and not result["required_local_indices"]


def test_source_only_and_mismatched_input_context_are_excluded():
    raw, data, model = candidate_input(node("Lean.Parser.Tactic.exact", atom("exact"), ref(0)), "exact h0")
    nonexact = copy.deepcopy(raw)
    nonexact["target_state_matches_invocation"] = False
    with pytest.raises(SupervisionExclusion, match="nonexact_source_alignment"):
        make_structured_candidate(nonexact, data, model, {})
    model["hyp_sexps"].pop()
    with pytest.raises(SupervisionExclusion, match="model_context_mismatch"):
        make_structured_candidate(raw, data, model, {})


@pytest.fixture
def frozen_candidates(tmp_path):
    prepared, cache = tmp_path / "prepared", tmp_path / "cache"
    unknown = node("Lean.Parser.Tactic.custom", atom("custom"), {"tag": "identifier", "source": "freeName"})
    unknown["source"] = "custom freeName"
    scoped = {"tag": "identifier", "source": "h", "semanticRole": "scoped_local"}
    suffices = node("Lean.Parser.Tactic.tacticSuffices_", atom("suffices"), node("Lean.Parser.Term.sufficesDecl",
        node("group", scoped, atom(":")), ref(name="True"), node("Lean.Parser.Term.fromTerm", atom("from"), scoped)))
    suffices["source"] = "suffices h : True from h"
    graphs = {}
    for split in ("train", "val"):
        graphs[split] = [
            _cache_row(cache, split, 0, node("Lean.Parser.Tactic.exact", atom("exact"), ref(0)), "exact h0"),
            _cache_row(cache, split, 1, node("Lean.Parser.Tactic.rfl", atom("rfl")), "rfl"),
            _cache_row(cache, split, 2, node("Lean.Parser.Tactic.exact", atom("exact"), ref(name="True.intro")), "exact True.intro"),
            _cache_row(cache, split, 3, unknown, "custom freeName"),
            _cache_row(cache, split, 4, node("Lean.Parser.Tactic.exact", atom("exact"), ref(99)), "exact h99"),
            _cache_row(cache, split, 5, node("Lean.Parser.Tactic.exact", atom("exact"), ref(0)), "exact h0", alignment="branch_opener", target=False),
            _cache_row(cache, split, 6, suffices, "suffices h : True from h"),
        ]
        for index in range(7):
            path = cache / split / "model_sexpr_v2" / f"{index:09d}.json"
            model = json.loads(path.read_text())
            model["hyp_sexps"] = [{"context_index": 0, "name": "h0", "sexp": "(:c True)"}]
            path.write_text(json.dumps(model))
    _prepared(prepared, graphs)
    baseline = tmp_path / "baseline"
    assert audit_template_coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(baseline)]) == 0
    previous = tmp_path / "v4"
    assert audit_template_recovery.main(["--baseline-coverage-dir", str(baseline), "--renderer", "template-v4", "--output-dir", str(previous)]) == 0
    for name, runner in (("v5", audit_wrapper_recovery), ("v6", audit_scoped_recovery), ("v7", audit_selector_recovery)):
        output = tmp_path / name
        assert runner.main(["--previous-recovery-dir", str(previous), "--output-dir", str(output)]) == 0
        previous = output
    recovery = tmp_path / "v9"
    assert audit_selector_recovery.main(["--previous-recovery-dir", str(previous), "--renderer", "template-v9", "--output-dir", str(recovery)]) == 0
    bundle = tmp_path / "bundle"
    files = {"retriever/corpus/lemmas.jsonl": json.dumps({"name": "True.intro", "lemma_id": 71}) + "\n",
             "retriever/index/lemma_ids.json": "[71]", "retriever/index/lemma_names.json": '["True.intro"]'}
    for name, content in files.items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (bundle / "bundle.json").write_text(json.dumps({"format": "joint_source_system_v1",
        "files": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}}))
    return recovery, cache, bundle


def test_preparation_preserves_sources_and_keeps_exclusions_visible(tmp_path, frozen_candidates):
    recovery, _, bundle = frozen_candidates
    before = _snapshot(tmp_path)
    output = tmp_path / "supervision"
    report = prepare_structured_supervision.prepare_candidates(recovery, bundle, output, train_limit=0, val_limit=0)
    assert report["complete"] and report["manifest"]["splits"] == ["train", "val"]
    for split in ("train", "val"):
        stats = report["splits"][split]["counts"]
        assert stats["population"] == 7 and stats["exported"] == stats["typed_candidate_pool"] == 3
        assert stats["excluded:quarantined"] == stats["excluded:nonexact_source_alignment"] == 1
        assert stats["excluded:deferred_source_program"] == stats["excluded:source_scope_program"] == 1
        labels = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        assert {label["row_index"] for label in labels} == {0, 1, 2}
        assert all(not label["neural_target_compiled"] for label in labels)
    after = _snapshot(tmp_path)
    assert all(after[name] == digest for name, digest in before.items())
    with pytest.raises(FileExistsError):
        prepare_structured_supervision.prepare_candidates(recovery, bundle, output)


def test_preparation_rejects_changed_snapshot_bytes(tmp_path, frozen_candidates):
    recovery, cache, bundle = frozen_candidates
    path = cache / "train/model_sexpr_v2/000000000.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="cache changed"):
        prepare_structured_supervision.prepare_candidates(recovery, bundle, tmp_path / "changed", train_limit=0)


def test_pilot_limits_are_deterministic_and_keep_splits_separate(tmp_path, frozen_candidates):
    recovery, _, bundle = frozen_candidates
    for name in ("one", "two"):
        report = prepare_structured_supervision.prepare_candidates(recovery, bundle, tmp_path / name,
            train_limit=2, val_limit=1, seed=73)
        assert report["splits"]["train"]["counts"]["exported"] == 2
        assert report["splits"]["val"]["counts"]["exported"] == 1
    for split in ("train", "val"):
        assert (tmp_path / "one" / f"{split}.jsonl").read_bytes() == (tmp_path / "two" / f"{split}.jsonl").read_bytes()
    assert json.loads((tmp_path / "one/native_val_pilot_rows.json").read_text()) == json.loads(
        (tmp_path / "one/val_row_indices.json").read_text())


def test_preparation_rejects_changed_library_identity_table(tmp_path, frozen_candidates):
    recovery, _, bundle = frozen_candidates
    (bundle / "retriever/index/lemma_ids.json").write_text("[99]")
    with pytest.raises(ValueError, match="bundle member binding mismatch"):
        prepare_structured_supervision.prepare_candidates(recovery, bundle, tmp_path / "wrong_index")
