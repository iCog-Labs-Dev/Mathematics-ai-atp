"""V8 math recovery pairs fixed populations without semantic extraction."""

import json

import pytest

from maths_ai.gnn_inference.atp_lean_gnn import semantic_templates
from maths_ai.gnn_inference.scripts import audit_scoped_recovery as scoped
from maths_ai.gnn_inference.scripts import audit_selector_recovery as selectors
from maths_ai.gnn_inference.scripts import audit_template_coverage as coverage
from maths_ai.gnn_inference.scripts import audit_template_recovery as recursive
from maths_ai.gnn_inference.scripts import audit_wrapper_recovery as wrappers
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref
from maths_ai.gnn_inference.tests.test_template_coverage import _cache_row, _prepared, _snapshot


@pytest.fixture
def frozen_v7(tmp_path):
    prepared, cache, baseline, v4, v5, v6, v7 = (tmp_path / name for name in
        ("prepared", "cache", "baseline", "v4", "v5", "v6", "v7"))
    norm = node("Lean.Parser.Tactic.exact", atom("exact"), node("«term‖_‖»", atom("‖"), ref(0), atom("‖")))
    negative = node("«term¬_»", atom("¬"), {"tag": "identifier", "source": "True", "semanticRole": "global", "name": "True"})
    declaration = node("Lean.Parser.Tactic.tacticHave_", atom("have"), node("Lean.Parser.Term.haveDecl",
        node("Lean.Parser.Term.haveIdDecl", node("Lean.Parser.Term.haveId", {
            "tag": "identifier", "source": "hx", "semanticRole": "scoped_local"}), node("null"),
            node("null", node("Lean.Parser.Term.typeSpec", atom(":"), negative)), atom(":="), ref(0))))
    graphs = {split: [
        _cache_row(cache, split, 0, node("Lean.Parser.Tactic.exact", atom("exact"), ref(0)), "exact h0"),
        _cache_row(cache, split, 1, norm, "exact ‖h0‖"),
        _cache_row(cache, split, 2, declaration, "have hx : ¬ True := h0"),
        _cache_row(cache, split, 3, node("Lean.Parser.Tactic.exact", atom("exact"), ref(99)), "exact h99"),
        _cache_row(cache, split, 4, norm, "exact ‖h0‖", alignment="branch_opener", target=False),
        _cache_row(cache, split, 5, node("Lean.Parser.Tactic.exact", atom("exact"), node("Test.unknown", ref(0))), "exact custom h0"),
    ] for split in ("train", "val")}
    _prepared(prepared, graphs)
    assert coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(baseline)]) == 0
    assert recursive.main(["--baseline-coverage-dir", str(baseline), "--renderer", "template-v4", "--output-dir", str(v4)]) == 0
    assert wrappers.main(["--previous-recovery-dir", str(v4), "--output-dir", str(v5)]) == 0
    assert scoped.main(["--previous-recovery-dir", str(v5), "--output-dir", str(v6)]) == 0
    assert selectors.main(["--previous-recovery-dir", str(v6), "--output-dir", str(v7)]) == 0
    return v7, cache


def test_math_audit_tracks_all_rows_gains_and_exact_only_pilot(tmp_path, frozen_v7, monkeypatch):
    previous, _ = frozen_v7
    before = _snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("V8 must not load semantic sidecars")

    monkeypatch.setattr(semantic_templates, "load_semantic_trace", forbidden)
    output = tmp_path / "v8"
    assert selectors.main(["--renderer", "template-v8", "--previous-recovery-dir", str(previous), "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] and report["manifest"]["schema"] == "math_recovery_v1"
    assert report["manifest"]["semantic_cache_used"] is False
    for split in ("train", "val"):
        summary = report["splits"][split]["summary"]
        assert summary["selected_steps"] == 6 and summary["quarantined_steps"] == 1
        assert summary["baseline_v7_renderable_steps"] == 1 and summary["template_v8_renderable_steps"] == 4
        assert summary["v8_rescues_vs_v7"] == summary["math_notation_rescues"] == 3
        assert summary["exact_aligned_v8_rescues"] == 2
        assert summary["v8_losses_vs_v7"] == summary["exact_aligned_v8_losses"] == summary["other_source_repair_rescues"] == 0
        assert summary["math_notation_steps"] == summary["math_notation_nodes"] == 3
    assert json.loads((output / "new_val_pilot_rows.json").read_text()) == [1, 2]
    after = _snapshot(tmp_path)
    assert all(after[name] == digest for name, digest in before.items())
    assert all(name in before or name.startswith("v8/") for name in after)


def test_math_audit_rejects_changed_cache_with_incomplete_report(tmp_path, frozen_v7):
    previous, cache = frozen_v7
    path = cache / "val/action_trace_v3/000000001.json"
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="cache changed"):
        selectors.main(["--renderer", "template-v8", "--previous-recovery-dir", str(previous), "--output-dir", str(output)])
    assert not json.loads((output / "report.json").read_text())["complete"]


def test_math_audit_requires_saved_v7_not_an_older_stage(tmp_path, frozen_v7):
    previous, _ = frozen_v7
    source = json.loads((previous / "report.json").read_text())
    v6 = source["manifest"]["previous_recovery_dir"]
    with pytest.raises(ValueError, match="immutable V7"):
        selectors.main(["--renderer", "template-v8", "--previous-recovery-dir", v6, "--output-dir", str(tmp_path / "invalid")])
