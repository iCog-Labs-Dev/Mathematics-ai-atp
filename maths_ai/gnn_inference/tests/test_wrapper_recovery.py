"""V5 pairs against recorded V4 outcomes without rerunning native/train validation."""

import json

import pytest

from maths_ai.gnn_inference.scripts import audit_template_coverage as coverage
from maths_ai.gnn_inference.scripts import audit_template_recovery as recursive_audit
from maths_ai.gnn_inference.scripts import audit_wrapper_recovery as wrappers
from maths_ai.gnn_inference.tests.test_template_coverage import _cache_row, _prepared, _snapshot
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref, simp_root


@pytest.fixture
def frozen_v4(tmp_path):
    prepared, cache, baseline, v4 = (tmp_path / name for name in ("prepared", "cache", "baseline", "v4"))
    using = node("Lean.Parser.Tactic.simpa", atom("simpa"), node("null"), node("null"),
                 node("Lean.Parser.Tactic.simpaArgsRest", node("null"), node("null"), node("null"), node("null"),
                      node("null", atom("using"), ref(0))))
    declaration = node("Lean.Parser.Tactic.tacticHave_", atom("have"), node("Lean.Parser.Term.haveDecl",
        node("Lean.Parser.Term.haveIdDecl", node("Lean.Parser.Term.haveId", {
            "tag": "identifier", "source": "fresh", "semanticRole": "scoped_local"}), node("null"), node("null"), atom(":="), ref(0))))
    _prepared(prepared, {"train": [
        _cache_row(cache, "train", 0, simp_root(ref(0)), "simp [h0]"),
        _cache_row(cache, "train", 1, using, "simpa using h0"),
        _cache_row(cache, "train", 2, declaration, "have fresh := h0"),
        _cache_row(cache, "train", 3, node("Lean.Parser.Tactic.exact", atom("exact"), ref(99)), "exact h99"),
    ], "val": [
        _cache_row(cache, "val", 0, using, "simpa using h0"),
        _cache_row(cache, "val", 1, declaration, "have fresh := h0"),
        _cache_row(cache, "val", 2, using, "simpa using h0", alignment="unknown", target=False),
        _cache_row(cache, "val", 3, simp_root(ref(0)), "simp [h0]"),
    ]})
    assert coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(baseline)]) == 0
    assert recursive_audit.main(["--baseline-coverage-dir", str(baseline), "--renderer", "template-v4", "--output-dir", str(v4)]) == 0
    return v4, cache


def test_v5_frozen_pairing_and_exact_only_family_balanced_pilot(tmp_path, frozen_v4):
    previous, _ = frozen_v4
    output = tmp_path / "v5"
    before = _snapshot(tmp_path)
    assert wrappers.main(["--previous-recovery-dir", str(previous), "--output-dir", str(output), "--pilot-actions", "2"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["complete"]
    train, val = (report["splits"][split]["summary"] for split in ("train", "val"))
    assert train["selected_steps"] == 4 and train["quarantined_steps"] == 1
    assert train["baseline_v4_renderable_steps"] == 1 and train["template_v5_renderable_steps"] == 3
    assert train["v5_rescues_vs_v4"] == 2 and train["v5_losses_vs_v4"] == 0
    assert train["declaration_rescues"] == train["simplifier_wrapper_rescues"] == 1
    assert val["v5_rescues_vs_v4"] == 3 and val["exact_aligned_v5_rescues"] == 2
    assert json.loads((output / "new_val_pilot_rows.json").read_text()) == [0, 1]
    after = _snapshot(tmp_path)
    assert all(after[path] == digest for path, digest in before.items())
    assert all(path in before or path.startswith("v5/") for path in after)


def test_v5_rejects_changed_inputs_and_does_not_repair_quarantined_contexts(tmp_path, frozen_v4):
    previous, cache = frozen_v4
    path = cache / "val/action_trace_v3/000000001.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="cache changed"):
        wrappers.main(["--previous-recovery-dir", str(previous), "--output-dir", str(tmp_path / "changed")])
    assert not json.loads((tmp_path / "changed/report.json").read_text())["complete"]
