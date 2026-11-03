"""Paired accounting distinguishes lookup recovery, applications and regressions."""

import json

import pytest

from maths_ai.gnn_inference.atp_lean_gnn import tactic_templates
from maths_ai.gnn_inference.scripts import audit_template_coverage as coverage
from maths_ai.gnn_inference.scripts import audit_template_recovery as recovery
from maths_ai.gnn_inference.tests.test_template_coverage import _atom, _cache_row, _local, _node, _prepared, _snapshot


def _rw(term, *, backward=False, count=1):
    rules = []
    for i in range(count):
        if i:
            rules.append(_atom(","))
        rules.append(_node("Lean.Parser.Tactic.rwRule", _node("null", _atom("←")) if backward else _node("null"), term))
    return _node("Lean.Parser.Tactic.rwSeq", _atom("rw"), _node("null"),
                 _node("Lean.Parser.Tactic.rwRuleSeq", _atom("["), _node("null", *rules), _atom("]")), _node("null"))


@pytest.fixture
def frozen_baseline(tmp_path, monkeypatch):
    prepared, cache, output = (tmp_path / name for name in ("prepared", "cache", "baseline"))
    short = _node("Lean.Parser.Tactic.exact", _atom("exact"), {
        "tag": "identifier", "source": "foo", "semanticRole": "global", "name": "A.foo"})
    application = _node("Lean.Parser.Term.app", _local(), _node("null", _local()))
    nested = _node("Lean.Parser.Term.app", _local(), _node("null", application))
    graphs = {"train": [
        _cache_row(cache, "train", 0, _rw(_local(), backward=True), "rw [← h0]"),
        _cache_row(cache, "train", 1, short, "exact <a>A.foo</a>"),
        _cache_row(cache, "train", 2, _rw(application), "rw [h0 h0]"),
        _cache_row(cache, "train", 3, _rw(nested), "rw [h0 (h0 h0)]"),
        _cache_row(cache, "train", 4, _rw(_local(99)), "rw [h99]"),
        _cache_row(cache, "train", 5, _rw(application), "rw [h0 h0]", alignment="branch_opener", target=False),
    ], "val": [
        _cache_row(cache, "val", 0, short, "exact <a>A.foo</a>"),
        _cache_row(cache, "val", 1, _rw(_node("Lean.Parser.Term.paren", _atom("("), _local(), _atom(")"))), "rw [(h0)]"),
    ]}
    _prepared(prepared, graphs)
    original = tactic_templates.prepare_gold_rendering

    def source_only(raw, trace, **kwargs):
        return original(raw, trace, **kwargs, reference_lookup="source")

    with monkeypatch.context() as context:
        context.setattr(coverage.tactic_templates, "prepare_gold_rendering", source_only)
        assert coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(output)]) == 0
    return output, cache


def test_paired_recovery_preserves_frozen_cohort_and_attributes_both_fixes(tmp_path, frozen_baseline):
    baseline, _ = frozen_baseline
    output = tmp_path / "recovery"
    before = _snapshot(tmp_path)
    argv = ["--baseline-coverage-dir", str(baseline), "--output-dir", str(output)]
    assert recovery.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    train, val = (report["splits"][split]["summary"] for split in ("train", "val"))
    assert report["complete"]
    assert train["selected_steps"] == 6 and train["quarantined_steps"] == 1
    assert train["baseline_renderable_steps"] == 1
    assert train["lookup_fixed_v1_renderable_steps"] == 2
    assert train["template_v2_renderable_steps"] == 4
    assert train["lookup_rescues"] == 1 and train["v2_grammar_rescues"] == 2
    assert train["applied_rule_rescues"] == 2 and train["bare_rule_rescues"] == 0
    assert train["combined_rescues"] == 3 and train["combined_losses"] == 0
    assert train["exact_aligned_template_v2_renderable_steps"] == 3
    assert train["exact_aligned_combined_rescues"] == 2
    assert train["template_v2_parsed_steps"] == 3 and train["applied_rule_parsed_steps"] == 2
    assert val["baseline_renderable_steps"] == 0 and val["template_v2_renderable_steps"] == 2
    assert val["lookup_rescues"] == val["v2_grammar_rescues"] == 1
    assert val["applied_rule_rescues"] == 0 and val["bare_rule_rescues"] == 1
    assert val["applied_rule_parsed_steps"] == 0  # Parenthesis-only recovery is not an applied rule.
    assert report["splits"]["train"]["v2_error_counts"] == {"compound_rule_argument": 1}
    assert report["splits"]["train"]["by_previous_error"]["supported"]["selected_steps"] == 1
    records = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(records) == 8
    after = _snapshot(tmp_path)
    assert all(after[path] == digest for path, digest in before.items())
    assert all(path in before or path.startswith("recovery/") for path in after)
    with pytest.raises(FileExistsError):
        recovery.main(argv)


def test_paired_recovery_reports_losses_and_returns_nonzero(tmp_path, frozen_baseline, monkeypatch):
    baseline, _ = frozen_baseline
    original = tactic_templates.prepare_gold_rendering

    def lose_existing(raw, trace, **kwargs):
        result = original(raw, trace, **kwargs)
        if raw == "rw [← h0]" and kwargs["renderer"] == "template-v2":
            result.update(rendered_tactic=None, render_error="forced_rendering_loss")
        return result

    monkeypatch.setattr(tactic_templates, "prepare_gold_rendering", lose_existing)
    output = tmp_path / "loss"
    assert recovery.main(["--baseline-coverage-dir", str(baseline), "--output-dir", str(output)]) == 2
    train = json.loads((output / "report.json").read_text())["splits"]["train"]["summary"]
    assert train["combined_losses"] == train["v2_grammar_losses"] == 1


def test_recovery_rejects_changed_snapshot(tmp_path, frozen_baseline):
    baseline, cache = frozen_baseline
    path = cache / "train/sexpr/000000000.json"
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="cache changed"):
        recovery.main(["--baseline-coverage-dir", str(baseline), "--output-dir", str(output)])
    assert not json.loads((output / "report.json").read_text())["complete"]


def test_v3_recovery_separates_binders_named_roles_and_v2_support(tmp_path):
    prepared, cache, baseline = (tmp_path / name for name in ("prepared", "cache", "baseline"))
    introduction = _node("Lean.Parser.Tactic.intro", _atom("intro"), _node("null", {
        "tag": "identifier", "source": "fresh", "semanticRole": "scoped_local"}))
    parameter = _node("Lean.Parser.Term.namedArgument", _atom("("), {"tag": "identifier", "source": "n"},
                      _atom(":="), _local(), _atom(")"))
    application = _node("Lean.Parser.Term.app", {"tag": "identifier", "source": "Nat.add_zero",
                        "semanticRole": "global", "name": "Nat.add_zero"}, _node("null", parameter))
    train = [
        _cache_row(cache, "train", 0, introduction, "intro fresh"),
        _cache_row(cache, "train", 1, _rw(application), "rw [Nat.add_zero (n := h0)]"),
        _cache_row(cache, "train", 2, _rw(_local()), "rw [h0]"),
        _cache_row(cache, "train", 3, introduction, "intro fresh", alignment="branch_opener", target=False),
        _cache_row(cache, "train", 4, _node("Lean.Parser.Tactic.exact", _atom("exact"),
                   _node("Lean.Parser.Term.paren", _atom("("),
                         _node("Lean.Parser.Term.app", _local(), _node("null", parameter)), _atom(")"))), "exact (h0 (n := h0))"),
        _cache_row(cache, "train", 5, _rw(_local(), count=4), "rw [h0, h0, h0, h0]"),
    ]
    _prepared(prepared, {"train": train, "val": [_cache_row(cache, "val", 0, introduction, "intro fresh")]})
    assert coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(baseline)]) == 0
    output = tmp_path / "v3"
    before = _snapshot(tmp_path)
    assert recovery.main(["--baseline-coverage-dir", str(baseline), "--output-dir", str(output),
                          "--renderer", "template-v3"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["manifest"]["schema"] == "template_recovery_v2"
    train = report["splits"]["train"]["summary"]
    assert train["selected_steps"] == 6 and train["template_v2_renderable_steps"] == 1
    assert train["template_v3_renderable_steps"] == 6 and train["v3_rescues_vs_v2"] == 5
    assert train["intro_rescues"] == 2 and train["named_rule_rescues"] == 1
    assert train["exact_aligned_v3_rescues_vs_v2"] == 4
    assert train["application_term_rescues"] == 1
    assert train["rule_list_rescues"] == train["long_rule_list_parsed_steps"] == 1
    records = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    term = next(row for row in records if row["split"] == "train" and row["row_index"] == 4)
    assert term["named_argument_count"] == 1
    assert train["v3_losses_vs_v2"] == train["overall_v3_losses_vs_original"] == 0
    after = _snapshot(tmp_path)
    assert all(after[path] == digest for path, digest in before.items())
    assert all(path in before or path.startswith("v3/") for path in after)


def test_v4_recovery_measures_recursive_gains_and_selects_only_new_exact_val_pilot(tmp_path):
    prepared, cache, baseline = (tmp_path / name for name in ("prepared", "cache", "baseline"))
    inner = _node("Lean.Parser.Term.app", _local(), _node("null", _local()))
    nested = _node("Lean.Parser.Term.app", _local(), _node("null", inner))
    constructor = _node("Lean.Parser.Term.anonymousCtor", _atom("⟨"), _node("null", _local()), _atom("⟩"))
    train = [_cache_row(cache, "train", 0, _rw(nested), "rw [h0 (h0 h0)]"),
             _cache_row(cache, "train", 1, _rw(_local()), "rw [h0]")]
    val = [_cache_row(cache, "val", 0, _rw(nested), "rw [h0 (h0 h0)]"),
           _cache_row(cache, "val", 1, _rw(nested), "rw [h0 (h0 h0)]", alignment="unknown", target=False),
           _cache_row(cache, "val", 2, _node("Lean.Parser.Tactic.exact", _atom("exact"), constructor), "exact ⟨h0⟩"),
           _cache_row(cache, "val", 3, _rw(_local()), "rw [h0]")]
    _prepared(prepared, {"train": train, "val": val})
    assert coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(baseline)]) == 0
    before = _snapshot(tmp_path)
    output = tmp_path / "v4"
    assert recovery.main(["--baseline-coverage-dir", str(baseline), "--output-dir", str(output),
                          "--renderer", "template-v4", "--pilot-actions", "2"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["manifest"]["schema"] == "template_recovery_v3"
    train, val = (report["splits"][split]["summary"] for split in ("train", "val"))
    assert train["template_v3_renderable_steps"] == 1 and train["template_v4_renderable_steps"] == 2
    assert train["v4_rescues_vs_v3"] == 1 and train["v4_losses_vs_v3"] == 0
    assert val["v4_rescues_vs_v3"] == 3 and val["exact_aligned_v4_rescues_vs_v3"] == 2
    assert val["recursive_rule_rescues"] == 2 and val["recursive_application_rescues"] == 1
    assert val["v4_constructor_rescues"] == 1
    assert json.loads((output / "new_val_pilot_rows.json").read_text()) == [0, 2]
    after = _snapshot(tmp_path)
    assert all(after[path] == digest for path, digest in before.items())
    assert all(path in before or path.startswith("v4/") for path in after)
