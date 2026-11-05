"""Reject stale/ambiguous semantic recovery and preserve expression receivers."""

import json

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.preparation import ActionTraceCache
from maths_ai.gnn_inference.atp_lean_gnn.scoped_templates import ScopedTermParser
from maths_ai.gnn_inference.atp_lean_gnn.semantic_templates import decode_semantic_term, load_semantic_trace
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import PANTOGRAPH_COMMIT
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import TemplateError, prepare_gold_rendering
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import canonical_form
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, trace
from maths_ai.gnn_inference.tests.test_template_coverage import _cache_row
from maths_ai.gnn_inference.tests.test_template_v6 import identifier, lambda_term


def method_trace():
    expression = {"tag": "identifier", "source": "h0.symm", "semanticRole": "global", "name": "Eq.symm",
                  "sourceStart": 10, "sourceEnd": 17}
    data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), expression))
    term = {"source_start": 10, "source_end": 17, "source": "h0.symm", "syntax_kind": "ident",
            "action_sexp": "(:app (:global Eq.symm) (:local FV0))"}
    return data, term


def test_semantic_recovery_requires_bound_exact_owned_range_and_retains_receiver():
    data, term = method_trace()
    assert prepare_gold_rendering("exact h0.symm", data, renderer="template-v6", semantic_trace={"terms": [term]})["rendered_tactic"] is None
    result = prepare_gold_rendering("exact h0.symm", data, renderer="template-v6", semantic_trace={"status": "bound", "terms": [term]})
    assert result["rendered_tactic"] == "exact (_root_.«Eq».«symm» «h0»)"
    assert result["semantic_reconstructions"] == 1
    for terms in ([{**term, "source_end": 18}], [term, {**term, "action_sexp": "(:global Eq.refl)"}]):
        assert prepare_gold_rendering("exact h0.symm", data, renderer="template-v6",
                                      semantic_trace={"status": "bound", "terms": terms})["rendered_tactic"] is None
    # V2 cannot bind scoped locals by name; never reinterpret input FV0 as a lambda binder.
    parser = ScopedTermParser(data["local_context"], [term])
    with pytest.raises(TemplateError):
        parser.parse(lambda_term("h0", data["source_syntax"]["children"][1]))


@pytest.mark.parametrize("source", ["(:local FV99)", "(:local FV?)", "(:bound 0)", "(:metavar)",
    "(:lambda x (:global Nat) (:bound 0))", "(:sort Type)", "(:lit \"123\")", "(:global x) (:global y)",
    "(:app (:global f) " + " ".join("(:local FV0)" for _ in range(17)) + ")"])
def test_insufficient_or_excessive_semantic_payload_is_rejected(source):
    with pytest.raises(TemplateError):
        decode_semantic_term(source, ScopedTermParser([{"context_index": 0, "user_name": "h0"}]), 0)


def test_semantic_sidecar_is_bound_to_raw_source_context_pins_and_ranges(tmp_path):
    data, term = method_trace()
    _cache_row(tmp_path, "val", 0, data["source_syntax"], "exact h0.symm")
    raw = json.loads((tmp_path / "val/sexpr/000000000.json").read_text())
    source_path = tmp_path / "val/action_trace_v3/000000000.json"
    source = json.loads(source_path.read_text())
    source.update(pantograph_commit=PANTOGRAPH_COMMIT, state_sha256="state", tactic_sha256="tactic", target_state_sha256="after",
                  term_ranges=[{key: value for key, value in term.items() if key != "action_sexp"}])
    assert load_semantic_trace(tmp_path, "val", 0, raw, source)["status"] == "missing"
    sidecar = {**source, "schema_version": ActionTraceCache.SCHEMA_VERSION, "extractor_version": ActionTraceCache.EXTRACTOR_VERSION, "terms": [term]}
    path = tmp_path / "val/action_trace_v2/000000000.json"
    path.parent.mkdir()
    path.write_text(json.dumps(sidecar))
    assert load_semantic_trace(tmp_path, "val", 0, raw, source)["status"] == "bound"
    for key, value in (("raw_record_sha256", "stale"), ("pantograph_commit", "stale"), ("state_sha256", "stale"),
                       ("local_context", []), ("terms", [{**term, "source_start": 11}])):
        path.write_text(json.dumps({**sidecar, key: value}))
        result = load_semantic_trace(tmp_path, "val", 0, raw, source)
        assert result["status"] == "invalid" and not result["terms"] and result["sha256"]


def test_canonical_scope_forms_preserve_shadow_and_repeated_name_structure():
    def shape(outer, inner):
        data = trace(node("Lean.Parser.Tactic.exact", atom("exact"), lambda_term(outer, lambda_term(inner, identifier(inner)))))
        result = prepare_gold_rendering("exact fun x => fun y => y", data, renderer="template-v6")
        return canonical_form(result["template_target"], result["rendered_tactic"], data["local_context"])

    assert shape("x", "y") == shape("a", "b")
    assert shape("x", "x") != shape("x", "y")
    assert shape("h0", "x") != shape("fresh", "x")


def test_native_runner_binds_optional_semantic_hashes_into_resume(tmp_path, monkeypatch):
    from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
    from maths_ai.gnn_inference.atp_lean_gnn.preparation import ModelSExprCache, SExprCache, SourceSyntaxTraceCache
    from maths_ai.gnn_inference.scripts import replay_gold_actions as runner
    from maths_ai.gnn_inference.tests.test_gold_replay import _goal, _model_record

    source_root, cache = tmp_path / "source", tmp_path / "cache"
    source_root.mkdir()
    (source_root / "Example.lean").write_text("-- mocked frontend; native receiver replay is checked separately\n")
    row = DatasetRow("h : P\n⊢ P", "T", "exact h0.symm", "val", 0,
                     repo_commit=runner.DATASET_MATHLIB_COMMIT, file_path="Example.lean")
    raw = {"schema_version": SExprCache.SCHEMA_VERSION, "extractor_version": SExprCache.EXTRACTOR_VERSION,
        "dataset": row.dataset_name, "split": row.split, "row_index": 0, "theorem": row.theorem, "tactic": row.tactic,
        "repo_commit": row.repo_commit, "file_path": row.file_path, "unit_index": 0, "invocation_index": 0,
        "alignment_kind": "exact", "target_state_matches_invocation": True,
        "state_sha256": SExprCache.row_state_sha256(row), "tactic_sha256": SExprCache.row_tactic_sha256(row),
        "target_state_sha256": SExprCache.row_target_state_sha256(row)}
    data, term = method_trace()
    data["source_syntax"].update(sourceStart=0, sourceEnd=17)
    data = {**raw, **data, "schema_version": SourceSyntaxTraceCache.SCHEMA_VERSION,
            "extractor_version": SourceSyntaxTraceCache.EXTRACTOR_VERSION,
            "raw_record_sha256": SourceSyntaxTraceCache.raw_record_sha256(raw), "pantograph_commit": PANTOGRAPH_COMMIT,
            "term_ranges": [{key: value for key, value in term.items() if key != "action_sexp"}]}
    SExprCache(cache, project_path=str(source_root)).save("val", 0, raw)
    SourceSyntaxTraceCache(cache).save("val", 0, data)
    ModelSExprCache(cache).save("val", 0, {**_model_record(), "schema_version": ModelSExprCache.SCHEMA_VERSION,
        "normalization": ModelSExprCache.NORMALIZATION, "raw_record_sha256": ModelSExprCache.raw_record_sha256(raw)})
    semantic = {**data, "schema_version": ActionTraceCache.SCHEMA_VERSION,
                "extractor_version": ActionTraceCache.EXTRACTOR_VERSION, "terms": [term]}
    path = cache / "val/action_trace_v2/000000000.json"
    path.parent.mkdir()
    path.write_text(json.dumps(semantic))
    monkeypatch.setattr(runner, "_verify_environment", lambda *_: None)
    monkeypatch.setattr(runner, "iter_dataset_rows", lambda **_: iter([row]))
    monkeypatch.setattr(runner, "run_native_file", lambda *_, **__: {"frontend_errors": [], "rows": [{
        "row_index": 0, "original_control": "captured_syntax", "source_before": [_goal()], "source_after": [],
        "original": {"status": "executed", "goals": []}, "rendered": {"status": "executed", "goals": []}}]})
    output = tmp_path / "replay"
    argv = ["--source-root", str(source_root), "--pantograph-root", str(source_root), "--cache-root", str(cache),
            "--output-dir", str(output), "--renderer", "template-v6"]
    assert runner.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["manifest"]["rows"][0]["cache_hashes"]["action_trace_v2"]
    assert report["summary"]["semantic_trace_status_counts"] == {"bound": 1}
    assert report["summary"]["semantic_reconstructed_steps"] == 1
    assert runner.main([*argv, "--resume"]) == 0
    path.write_bytes(path.read_bytes() + b" ")  # Same parsed content still has a different immutable binding.
    with pytest.raises(ValueError, match="resume bindings changed"):
        runner.main([*argv, "--resume"])
