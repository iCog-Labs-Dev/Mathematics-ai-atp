"""Combined V9 audit skips intermediate full scans and accounts for deferred names."""

import json

import pytest

from maths_ai.gnn_inference.atp_lean_gnn import semantic_templates
from maths_ai.gnn_inference.scripts import audit_scoped_recovery as scoped
from maths_ai.gnn_inference.scripts import audit_selector_recovery as audit
from maths_ai.gnn_inference.scripts import audit_template_coverage as coverage
from maths_ai.gnn_inference.scripts import audit_template_recovery as recursive
from maths_ai.gnn_inference.scripts import audit_wrapper_recovery as wrappers
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref
from maths_ai.gnn_inference.tests.test_template_coverage import _cache_row, _prepared, _snapshot


@pytest.fixture
def frozen_program_v7(tmp_path):
    prepared, cache, baseline, v4, v5, v6, v7 = (tmp_path / name for name in ("prepared", "cache", "baseline", "v4", "v5", "v6", "v7"))
    scoped_name = {"tag": "identifier", "source": "h", "semanticRole": "scoped_local"}
    suffices = node("Lean.Parser.Tactic.tacticSuffices_", atom("suffices"), node("Lean.Parser.Term.sufficesDecl",
        node("group", scoped_name, atom(":")), {"tag": "identifier", "source": "True", "semanticRole": "global", "name": "True"},
        node("Lean.Parser.Term.fromTerm", atom("from"), scoped_name)))
    suffices["source"] = "suffices h : True from h"
    unknown = node("Lean.Parser.Tactic.custom", atom("custom"), {"tag": "identifier", "source": "unknownName"})
    unknown["source"] = "custom unknownName"
    _prepared(prepared, {split: [
        _cache_row(cache, split, 0, node("Lean.Parser.Tactic.exact", atom("exact"), ref(0)), "exact h0"),
        _cache_row(cache, split, 1, suffices, "suffices h : True from h"),
        _cache_row(cache, split, 2, unknown, "custom unknownName"),
        _cache_row(cache, split, 3, node("Lean.Parser.Tactic.exact", atom("exact"), ref(99)), "exact h99"),
        _cache_row(cache, split, 4, suffices, "suffices h : True from h", alignment="branch_opener", target=False),
    ] for split in ("train", "val")})
    assert coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(baseline)]) == 0
    assert recursive.main(["--baseline-coverage-dir", str(baseline), "--renderer", "template-v4", "--output-dir", str(v4)]) == 0
    assert wrappers.main(["--previous-recovery-dir", str(v4), "--output-dir", str(v5)]) == 0
    assert scoped.main(["--previous-recovery-dir", str(v5), "--output-dir", str(v6)]) == 0
    assert audit.main(["--previous-recovery-dir", str(v6), "--output-dir", str(v7)]) == 0
    return v7, cache


def test_combined_program_audit_directly_compares_v7_without_semantic_calls(tmp_path, frozen_program_v7, monkeypatch):
    previous, _ = frozen_program_v7
    before = _snapshot(tmp_path)
    monkeypatch.setattr(semantic_templates, "load_semantic_trace", lambda *_: pytest.fail("V9 must not read V2 data"))
    output = tmp_path / "v9"
    assert audit.main(["--renderer", "template-v9", "--previous-recovery-dir", str(previous), "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] and report["manifest"]["schema"] == "program_recovery_v1"
    for split in ("train", "val"):
        summary = report["splits"][split]["summary"]
        assert summary["selected_steps"] == 5 and summary["quarantined_steps"] == 1
        assert summary["baseline_v7_renderable_steps"] == 1 and summary["template_v9_renderable_steps"] == 4
        assert summary["v9_rescues_vs_v7"] == summary["source_program_rescues"] == 3
        assert summary["exact_aligned_v9_rescues"] == 2 and summary["v9_losses_vs_v7"] == 0
        assert summary["deferred_identifier_rescues"] == 1 and summary["no_deferred_identifier_program_rescues"] == 2
    assert json.loads((output / "new_val_pilot_rows.json").read_text()) == [1, 2]
    after = _snapshot(tmp_path)
    assert all(after[name] == digest for name, digest in before.items())
    assert all(name in before or name.startswith("v9/") for name in after)


def test_program_audit_rejects_changed_frozen_cache(tmp_path, frozen_program_v7):
    previous, cache = frozen_program_v7
    path = cache / "val/action_trace_v3/000000001.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="cache changed"):
        audit.main(["--renderer", "template-v9", "--previous-recovery-dir", str(previous), "--output-dir", str(tmp_path / "changed")])
