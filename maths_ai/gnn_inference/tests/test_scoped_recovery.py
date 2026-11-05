"""V6 keeps frozen denominators, quarantines and exact-only native sampling."""

import json

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.preparation import ActionTraceCache, SourceSyntaxTraceCache
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import PANTOGRAPH_COMMIT
from maths_ai.gnn_inference.scripts import audit_scoped_recovery as scoped
from maths_ai.gnn_inference.scripts import audit_template_coverage as coverage
from maths_ai.gnn_inference.scripts import audit_template_recovery as recursive
from maths_ai.gnn_inference.scripts import audit_wrapper_recovery as wrappers
from maths_ai.gnn_inference.tests.test_semantic_templates import method_trace
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref, simp_root
from maths_ai.gnn_inference.tests.test_template_coverage import _cache_row, _prepared, _snapshot
from maths_ai.gnn_inference.tests.test_template_v6 import identifier, lambda_term


@pytest.fixture
def frozen_v5(tmp_path):
    prepared, cache, baseline, v4, v5 = (tmp_path / name for name in ("prepared", "cache", "baseline", "v4", "v5"))
    expression = lambda_term("hp", identifier("hp"))
    pair = node("Lean.Parser.Tactic.rcasesPat.tuple", atom("⟨"), node("null",
                node("Lean.Parser.Tactic.rcasesPat.one", identifier("hp")), atom(","),
                node("Lean.Parser.Tactic.rcasesPat.one", identifier("hq"))), atom("⟩"))
    cases = node("Lean.Parser.Tactic.rcases", atom("rcases"), node("null",
                 node("Lean.Parser.Tactic.casesTarget", node("null"), ref(0))), node("null", atom("with"), pair))
    method, term = method_trace()
    graphs = {}
    for split in ("train", "val"):
        items = [
            _cache_row(cache, split, 0, simp_root(ref(0)), "simp [h0]"),
            _cache_row(cache, split, 1, node("Lean.Parser.Tactic.exact", atom("exact"), expression), "exact fun hp => hp"),
            _cache_row(cache, split, 2, cases, "rcases h0 with ⟨hp, hq⟩"),
            _cache_row(cache, split, 3, method["source_syntax"], "exact h0.symm"),
            _cache_row(cache, split, 4, node("Lean.Parser.Tactic.exact", atom("exact"), ref(99)), "exact h99"),
            _cache_row(cache, split, 5, method["source_syntax"], "exact h0.symm"),
            _cache_row(cache, split, 6, node("Lean.Parser.Tactic.exact", atom("exact"), expression), "exact fun hp => hp",
                       alignment="branch_opener", target=False),
        ]
        for index in (3, 5):
            source_path = cache / split / f"action_trace_v3/{index:09d}.json"
            source = json.loads(source_path.read_text())
            source.update(pantograph_commit=PANTOGRAPH_COMMIT, state_sha256="state", tactic_sha256="tactic", target_state_sha256="after",
                          term_ranges=[{key: value for key, value in term.items() if key != "action_sexp"}])
            source_path.write_text(json.dumps(source))
            items[index].argument_trace_sha256 = SourceSyntaxTraceCache.raw_record_sha256(source)
            if index == 3:
                semantic_source = source
        semantic = {**semantic_source, "schema_version": ActionTraceCache.SCHEMA_VERSION,
                    "extractor_version": ActionTraceCache.EXTRACTOR_VERSION, "terms": [term]}
        path = cache / split / "action_trace_v2/000000003.json"
        path.parent.mkdir()
        path.write_text(json.dumps(semantic))
        graphs[split] = items
    _prepared(prepared, graphs)
    assert coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(baseline)]) == 0
    assert recursive.main(["--baseline-coverage-dir", str(baseline), "--renderer", "template-v4", "--output-dir", str(v4)]) == 0
    assert wrappers.main(["--previous-recovery-dir", str(v4), "--output-dir", str(v5)]) == 0
    return v5, cache


def test_v6_pairing_semantic_availability_and_balanced_exact_pilot(tmp_path, frozen_v5):
    previous, _ = frozen_v5
    output = tmp_path / "v6"
    before = _snapshot(tmp_path)
    assert scoped.main(["--previous-recovery-dir", str(previous), "--output-dir", str(output), "--pilot-actions", "3"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] and report["semantic_inputs_sha256"]
    for split in ("train", "val"):
        summary = report["splits"][split]["summary"]
        assert summary["selected_steps"] == 7 and summary["quarantined_steps"] == 1
        assert summary["baseline_v5_renderable_steps"] == 1
        assert summary["v6_syntax_renderable_steps"] == 4 and summary["template_v6_renderable_steps"] == 5
        assert summary["v6_rescues_vs_v5"] == 4 and summary["v6_losses_vs_v5"] == 0
        assert summary["exact_aligned_v6_steps"] == 4 and summary["exact_aligned_v6_rescues"] == 3
        assert summary["semantic_rescues"] == summary["pattern_rescues"] == 1
        assert summary["scoped_term_rescues"] == 2
        assert report["splits"][split]["semantic_trace_status_counts"] == {"bound": 1, "missing": 5, "quarantined": 1}
    assert set(json.loads((output / "new_val_pilot_rows.json").read_text())) == {1, 2, 3}
    after = _snapshot(tmp_path)
    assert all(after[path] == digest for path, digest in before.items())
    assert all(path in before or path.startswith("v6/") for path in after)


def test_v6_rejects_changed_frozen_cache_and_counts_stale_optional_sidecars(tmp_path, frozen_v5):
    previous, cache = frozen_v5
    path = cache / "train/action_trace_v2/000000003.json"
    value = json.loads(path.read_text())
    path.write_text(json.dumps({**value, "raw_record_sha256": "stale"}))
    output = tmp_path / "stale"
    assert scoped.main(["--previous-recovery-dir", str(previous), "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["splits"]["train"]["semantic_trace_status_counts"]["invalid"] == 1
    assert report["splits"]["train"]["semantic_trace_error_counts"] == {"semantic trace raw binding mismatch": 1}
    assert report["splits"]["train"]["summary"]["semantic_rescues"] == 0
    path = cache / "val/action_trace_v3/000000001.json"
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="cache changed"):
        scoped.main(["--previous-recovery-dir", str(previous), "--output-dir", str(output)])
    assert not json.loads((output / "report.json").read_text())["complete"]


def test_v6_reads_isolated_overlay_without_falling_back_to_original_sidecars(tmp_path, frozen_v5):
    previous, cache = frozen_v5
    overlay = tmp_path / "overlay"
    target = overlay / "val/action_trace_v2/000000003.json"
    target.parent.mkdir(parents=True)
    target.write_bytes((cache / "val/action_trace_v2/000000003.json").read_bytes())
    before = _snapshot(cache)
    output = tmp_path / "overlaid"
    assert scoped.main(["--previous-recovery-dir", str(previous), "--semantic-cache-root", str(overlay),
                        "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["manifest"]["semantic_cache_root"] == str(overlay)
    assert report["splits"]["val"]["summary"]["semantic_rescues"] == 1
    assert report["splits"]["train"]["summary"]["semantic_rescues"] == 0
    assert _snapshot(cache) == before
