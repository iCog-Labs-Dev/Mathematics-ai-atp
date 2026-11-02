"""Coverage accounting must preserve denominators, bindings and alignment."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn.preparation import ModelSExprCache, SExprCache, SourceSyntaxTraceCache
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import DATASET_MATHLIB_COMMIT
from maths_ai.gnn_inference.scripts import audit_template_coverage as audit


def _node(kind, *children):
    return {"tag": "node", "kind": kind, "children": list(children)}


def _atom(source):
    return {"tag": "atom", "source": source}


def _local(index=0):
    return {"tag": "identifier", "source": f"h{index}", "semanticRole": "local", "contextIndex": index}


def _rw():
    return _node("Lean.Parser.Tactic.rwSeq", _atom("rw"), _node("null"),
                 _node("Lean.Parser.Tactic.rwRuleSeq", _atom("["),
                       _node("null", _node("Lean.Parser.Tactic.rwRule", _node("null", _atom("←")), _local())),
                       _atom("]")), _node("null"))


def _cache_row(cache, split, index, root=None, tactic="rw [← h0]", *, alignment="exact", target=True, legacy=False):
    data = SimpleNamespace(row_index=index, split=split, dataset_name="fixture", theorem=f"T{index}",
                           tactic_raw=tactic, action_target_supported=legacy)
    raw = {"schema_version": SExprCache.SCHEMA_VERSION, "extractor_version": SExprCache.EXTRACTOR_VERSION,
           "dataset": "fixture", "split": split, "row_index": index, "theorem": data.theorem,
           "tactic": tactic, "repo_commit": DATASET_MATHLIB_COMMIT, "alignment_kind": alignment,
           "target_state_matches_invocation": target, "file_path": "Fixture.lean"}
    raw_digest = ModelSExprCache.raw_record_sha256(raw)
    trace = {**raw, "schema_version": SourceSyntaxTraceCache.SCHEMA_VERSION,
             "extractor_version": SourceSyntaxTraceCache.EXTRACTOR_VERSION,
             "raw_record_sha256": raw_digest, "source_syntax": root or _rw(),
             "syntax_args": [], "local_context": [{"context_index": 0, "user_name": "h0"}]}
    model = {"schema_version": ModelSExprCache.SCHEMA_VERSION, "normalization": ModelSExprCache.NORMALIZATION,
             "raw_record_sha256": raw_digest, "goal_sexp": "(:c True)", "hyp_sexps": []}
    SExprCache(cache, project_path="unused").save(split, index, raw)
    ModelSExprCache(cache).save(split, index, model)
    SourceSyntaxTraceCache(cache).save(split, index, trace)
    data.argument_trace_sha256 = SourceSyntaxTraceCache.raw_record_sha256(trace)
    return data


def _prepared(root, graphs):
    packed = root / "packed/bidirectional"
    splits = {}
    for split, items in graphs.items():
        (packed / split).mkdir(parents=True)
        torch.save(items, packed / split / "chunk.pt")
        splits[split] = {"count": len(items), "chunks": ["chunk.pt"]}
    (packed / "manifest.json").write_text(json.dumps({"edge_mode": "bidirectional", "splits": splits}))
    (root / "argument_labels.json").write_text("{}")


def _snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_full_cohort_audit_keeps_invalid_rows_and_source_only_cases(tmp_path):
    prepared, cache, output = (tmp_path / name for name in ("prepared", "cache", "audit"))
    exact = _node("Lean.Parser.Tactic.exact", _atom("exact"), _local())
    private = {"tag": "identifier", "source": "_private.A.0.foo", "semanticRole": "global",
               "name": "_private.A.0.foo"}
    simp = _node("Lean.Parser.Tactic.simp", _atom("simp"), _node("null"), _node("null"),
                 _node("null", _atom("only")), _node("null", _atom("["),
                 _node("null", _node("Lean.Parser.Tactic.simpLemma", _node("null"), _node("null"), private)),
                 _atom("]")), _node("null"))
    derived = _node("Lean.Parser.Tactic.exact", _atom("exact"), {
        "tag": "identifier", "source": "witness.finish", "semanticRole": "global", "name": "Receiver.finish"})
    train = [
        _cache_row(cache, "train", 0),
        _cache_row(cache, "train", 1, exact, "exact h0", legacy=True),
        _cache_row(cache, "train", 2, alignment="branch_opener", target=False),
        _cache_row(cache, "train", 3, target=False),
        _cache_row(cache, "train", 4, simp, "simp only [_private.A.0.foo]"),
        _cache_row(cache, "train", 5, derived, "exact witness.finish", legacy=True),
        _cache_row(cache, "train", 6, _node("Lean.Parser.Tactic.exact", _atom("exact"), _local(99)),
                   "exact h99", legacy=True),
        _cache_row(cache, "train", 7),
        SimpleNamespace(row_index=8, split="train", dataset_name="fixture", theorem="T8",
                        tactic_raw="simp", action_target_supported=False),
    ]
    stale_path = cache / "train/action_trace_v3/000000007.json"
    stale = json.loads(stale_path.read_text())
    stale["raw_record_sha256"] = "stale"
    stale_path.write_text(json.dumps(stale))
    _prepared(prepared, {"train": train, "val": [_cache_row(cache, "val", 0)]})
    before = _snapshot(tmp_path)
    argv = ["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(output)]
    assert audit.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] and all(entry["complete"] for entry in report["splits"].values())
    assert not report["manifest"]["native_execution_measured"]
    summary = report["splits"]["train"]["summary"]
    assert summary["selected_steps"] == 9
    assert summary["status_counts"] == {"renderable": 4, "render_unsupported": 2,
                                        "source_trace_invalid": 1, "cache_invalid": 2}
    assert summary["legacy_label_supported_steps"] == 3
    assert summary["trace_compiled_steps"] == 6
    assert summary["flat_renderable_steps"] == 1
    assert summary["template_v1_parsed_steps"] == 4  # Includes unrenderable private reference.
    assert summary["template_mode_renderable_steps"] == 4
    assert summary["newly_renderable_vs_flat"] == 3
    assert summary["dataset_exact_alignment_steps"] == 6
    assert summary["exact_aligned_template_mode_renderable_steps"] == 2
    assert summary["exact_aligned_template_v1_parsed_steps"] == 2
    assert summary["exact_aligned_newly_renderable_vs_flat"] == 1
    assert summary["template_error_counts"] == {"reference_resolution": 1}
    assert summary["parsed_template_features"]["directions=B"] == 3
    assert report["splits"]["train"]["by_family"]["rw"]["selected_steps"] == 4
    records = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(records) == 10 and all("template_target" not in row for row in records)
    train_records = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in records if row["split"] == "train")
    assert report["splits"]["train"]["row_records_sha256"] == hashlib.sha256(train_records.encode()).hexdigest()
    after = _snapshot(tmp_path)
    assert all(after[path] == digest for path, digest in before.items())
    assert all(path in before or path.startswith("audit/") for path in after)
    with pytest.raises(FileExistsError):
        audit.main(argv)


@pytest.mark.parametrize("change, expected", [
    ("model", "model_binding_mismatch"), ("prepared", "prepared_trace_binding_mismatch"),
])
def test_stale_bindings_are_quarantined_without_rendering(tmp_path, change, expected):
    data = _cache_row(tmp_path, "train", 0)
    if change == "prepared":
        data.argument_trace_sha256 = "stale"
    else:
        path = tmp_path / "train/model_sexpr_v2/000000000.json"
        model = json.loads(path.read_text())
        model["raw_record_sha256"] = "stale"
        path.write_text(json.dumps(model))
    row = audit.classify_row(data, "train", tmp_path)
    assert row["status"] == "cache_invalid" and row["error"] == expected
    assert not row["template_mode_renderable"]


@pytest.mark.parametrize("failure", ["duplicate_row", "count_mismatch"])
def test_incomplete_or_duplicated_prepared_population_cannot_report_success(tmp_path, failure):
    prepared, cache, output = (tmp_path / name for name in ("prepared", "cache", "audit"))
    train = _cache_row(cache, "train", 0)
    _prepared(prepared, {"train": [train, train] if failure == "duplicate_row" else [train],
                         "val": [_cache_row(cache, "val", 0)]})
    if failure == "count_mismatch":
        path = prepared / "packed/bidirectional/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["splits"]["train"]["count"] = 2
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate/invalid|count differs"):
        audit.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(output)])
    assert not json.loads((output / "report.json").read_text())["complete"]
