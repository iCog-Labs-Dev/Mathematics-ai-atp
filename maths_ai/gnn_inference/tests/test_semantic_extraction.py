"""Bounded extraction preserves frozen inputs and binds recaptured expressions."""

import asyncio
import copy
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import goal_signature, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.preparation import ModelSExprCache, SExprCache, SourceSyntaxTraceCache
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import (
    DATASET_MATHLIB_COMMIT, PantographInvocationClient, _make_model_record, _make_record, _make_source_syntax_trace_record,
)
from maths_ai.gnn_inference.scripts import audit_scoped_recovery as scoped
from maths_ai.gnn_inference.scripts import audit_template_coverage as coverage
from maths_ai.gnn_inference.scripts import audit_template_recovery as recursive
from maths_ai.gnn_inference.scripts import audit_wrapper_recovery as wrappers
from maths_ai.gnn_inference.scripts import extract_semantic_recovery as extraction
from maths_ai.gnn_inference.tests.test_scoped_recovery import frozen_v5
from maths_ai.gnn_inference.tests.test_template_coverage import _prepared, _snapshot


def test_selection_uses_only_exact_unsupported_reference_rows_and_frozen_digests(tmp_path, frozen_v5):
    previous, cache = frozen_v5
    v6 = tmp_path / "v6"
    assert scoped.main(["--previous-recovery-dir", str(previous), "--output-dir", str(v6)]) == 0
    before = _snapshot(cache)
    _, _, _, selected, _, dataset = extraction.select_candidates(v6, "val", 1, 1)
    assert dataset == "fixture"
    assert [row["row_index"] for row in selected] == [5]
    assert _snapshot(cache) == before
    path = v6 / "rows.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable V6"):
        extraction.select_candidates(v6, "val", 1, 1)


def test_native_isolated_semantic_extraction_replay_and_resume(tmp_path, monkeypatch):
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for pinned extraction/replay")
    root = Path(configured).resolve()
    prepared, cache, baseline, v4, v5, v6, overlay = (tmp_path / name for name in
        ("prepared", "cache", "baseline", "v4", "v5", "v6", "overlay"))
    with tempfile.TemporaryDirectory(prefix="SemanticOverlay_", dir=root) as directory:
        path = Path(directory) / "Example.lean"
        path.write_text("import Lean\nset_option linter.unusedVariables false\n"
                        "theorem known : True := True.intro\n"
                        "theorem True.finish (receiver h : True) : True := h\n"
                        "theorem swap (h : True) : True := by\n  exact known.finish h\n")

        async def capture():
            client = PantographInvocationClient(source_root=root, pantograph_repl=root / ".lake/build/bin/repl", capture_model_sexprs=True)
            try:
                return await client.process_file(str(path.relative_to(root)))
            finally:
                await client.close()

        units = asyncio.run(capture())
        u, v, invocation = next((u, v, inv) for u, unit in enumerate(units)
            for v, inv in enumerate(unit.get("invocations", [])) if inv["tactic"].strip() == "exact known.finish h")
        candidate = {"invocation": invocation, "unit_index": u, "invocation_index": v, "unit_boundary": units[u]["boundary"], "alignment_kind": "exact"}
        dataset_rows, graphs, artifacts = {}, {}, {}
        for split in ("train", "val"):
            row = DatasetRow(invocation["goalBefore"], "swap", "exact known.finish h", split, 0,
                target_state=invocation["goalAfter"], repo_commit=DATASET_MATHLIB_COMMIT, file_path=str(path.relative_to(root)), dataset_name="fixture")
            raw = _make_record(row, candidate)
            model = _make_model_record(row, candidate, raw)
            trace = _make_source_syntax_trace_record(row, candidate, raw)
            SExprCache(cache, project_path=str(root)).save(split, 0, raw)
            ModelSExprCache(cache).save(split, 0, model)
            SourceSyntaxTraceCache(cache).save(split, 0, trace)
            graphs[split] = [SimpleNamespace(row_index=0, split=split, dataset_name="fixture", theorem="swap",
                tactic_raw=row.tactic, action_target_supported=False,
                argument_trace_sha256=SourceSyntaxTraceCache.raw_record_sha256(trace))]
            dataset_rows[split] = row
            artifacts[split] = raw, trace, model
        _prepared(prepared, graphs)
        assert coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(baseline)]) == 0
        assert recursive.main(["--baseline-coverage-dir", str(baseline), "--renderer", "template-v4", "--output-dir", str(v4)]) == 0
        assert wrappers.main(["--previous-recovery-dir", str(v4), "--output-dir", str(v5)]) == 0
        assert scoped.main(["--previous-recovery-dir", str(v5), "--output-dir", str(v6)]) == 0
        before = _snapshot(tmp_path)
        # Fixture compiles in Pantograph rather than Mathlib; production still verifies both checkout pins.
        monkeypatch.setattr(extraction, "_verify_environment", lambda *_: None)
        monkeypatch.setattr(extraction, "iter_dataset_rows", lambda **kw: iter([dataset_rows[kw["split"]]]))
        argv = ["--recovery-dir", str(v6), "--source-root", str(root), "--pantograph-root", str(root),
                "--output-dir", str(overlay), "--max-actions", "1", "--max-files", "1"]
        assert extraction.main(argv) == 0
        report = json.loads((overlay / "report.json").read_text())
        assert report["complete"] and report["summary"]["bound_sidecars"] == report["summary"]["semantic_rescues"] == 1
        assert json.loads((overlay / "native_pilot_rows.json").read_text()) == [0]
        after = _snapshot(tmp_path)
        assert all(after[name] == digest for name, digest in before.items())
        assert all(name in before or name.startswith("overlay/") for name in after)
        assert extraction.main([*argv, "--resume"]) == 0
        assert _snapshot(cache) == {name.removeprefix("cache/"): digest for name, digest in before.items() if name.startswith("cache/")}

        raw, trace, model = artifacts["val"]
        action = report["files"][str(path.relative_to(root))]["rows"][0]["rendered_tactic"]
        native = run_native_file(path, [{"rowIndex": 0, "unitIndex": u, "invocationIndex": v,
            "sourceStart": trace["source_syntax"]["sourceStart"], "sourceEnd": trace["source_syntax"]["sourceEnd"],
            "renderedTactic": action}], source_root=root, pantograph_root=root, timeout=120)
        assert not native["frontend_errors"]
        result = native["rows"][0]
        assert result["original"]["status"] == result["rendered"]["status"] == "executed"
        assert goal_signature(result["original"]["goals"]) == goal_signature(result["rendered"]["goals"]) == goal_signature(result["source_after"])
        changed_trace = copy.deepcopy(trace)
        changed_trace["source_syntax"]["sourceEnd"] += 1
        with pytest.raises(ValueError, match="annotated syntax differs"):
            extraction.capture_sidecar(dataset_rows["val"], raw, changed_trace, model, units)
        changed_model = {**model, "goal_sexp": "(:c False)"}
        with pytest.raises(ValueError, match="input model differs"):
            extraction.capture_sidecar(dataset_rows["val"], raw, trace, changed_model, units)
        sidecar = overlay / "val/action_trace_v2/000000000.json"
        sidecar.write_bytes(sidecar.read_bytes() + b" ")
        with pytest.raises(ValueError, match="completed overlay sidecar changed"):
            extraction.main([*argv, "--resume"])
        overlaid = tmp_path / "overlaid_audit"
        assert scoped.main(["--previous-recovery-dir", str(v5), "--semantic-cache-root", str(overlay), "--output-dir", str(overlaid)]) == 0
        audited = json.loads((overlaid / "report.json").read_text())
        assert audited["splits"]["val"]["summary"]["semantic_rescues"] == 1
        assert audited["splits"]["train"]["summary"]["semantic_rescues"] == 0
