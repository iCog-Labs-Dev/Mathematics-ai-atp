"""Inventories retain grammar/equality structure and freeze complete validation selections."""

import copy
import json
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn import template_catalog as catalog
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import prepare_gold_rendering
from maths_ai.gnn_inference.scripts import audit_template_coverage as coverage
from maths_ai.gnn_inference.scripts import audit_template_recovery as recovery
from maths_ai.gnn_inference.scripts import inventory_tactic_templates as inventory
from maths_ai.gnn_inference.tests.test_template_coverage import _cache_row, _prepared, _snapshot
from maths_ai.gnn_inference.tests.test_tactic_templates import atom, node, ref, rw_root, trace
from maths_ai.gnn_inference.tests.test_template_v2 import app
from maths_ai.gnn_inference.tests.test_template_v3 import binder, intro_root, named


def test_form_abstraction_retains_reference_reuse_direction_and_named_parameters():
    data = trace(rw_root(ref(0), ref(0)))
    first = prepare_gold_rendering("rw [h0, h0]", data, renderer="template-v3")
    changed = copy.deepcopy(first["template_target"])
    for rule in changed["rules"]:
        rule["reference"]["identity"] = 19
    shape = catalog.canonical_form(first["template_target"], first["rendered_tactic"])
    assert catalog.form_id(shape) == catalog.form_id(catalog.canonical_form(changed, "unused"))
    changed["rules"][1]["reference"]["identity"] = 20
    assert catalog.form_id(shape) != catalog.form_id(catalog.canonical_form(changed, "unused"))
    named_data = trace(rw_root(app(ref(name="Nat.add_zero"), named("n", ref(0)))))
    target = prepare_gold_rendering("rw [Nat.add_zero (n := h0)]", named_data, renderer="template-v3")["template_target"]
    other = copy.deepcopy(target)
    other["rules"][0]["arguments"][0]["parameter"] = "m"
    assert catalog.canonical_form(target, "unused") != catalog.canonical_form(other, "unused")


def test_flat_forms_parse_emitted_escaped_names_and_binders_keep_shadowing():
    flat = catalog.canonical_form(None, "dsimp [_root_.«A».«space, name», «h», «h»]")
    assert len(flat["references"]) == 3
    assert flat["references"][1] == flat["references"][2]
    target = prepare_gold_rendering("intro h0 h0", trace(intro_root(binder("h0"), binder("h0"))),
                                    renderer="template-v3")["template_target"]
    shadowed = catalog.canonical_form(target, "unused", [{"user_name": "h0"}])
    fresh = catalog.canonical_form(target, "unused", [])
    assert shadowed != fresh
    assert shadowed["binders"][0]["name_slot"] == shadowed["binders"][1]["name_slot"]
    assert shadowed["binders"][0]["shadows_input"] is True


@pytest.fixture
def inventory_artifacts(tmp_path):
    prepared, cache, baseline, recovered, output = (tmp_path / name for name in (
        "prepared", "cache", "baseline", "recovery", "inventory"))
    introduction = intro_root(binder("fresh"))
    automatic = intro_root(family="intros")
    parameter_rule = rw_root(app(ref(name="Nat.add_zero"), named("n", ref(0))))
    _prepared(prepared, {"train": [
        _cache_row(cache, "train", 0), _cache_row(cache, "train", 1),
        _cache_row(cache, "train", 2, introduction, "intro fresh"),
        _cache_row(cache, "train", 3, automatic, "intros", alignment="branch_opener", target=False),
        _cache_row(cache, "train", 4, rw_root(ref(99)), "rw [h99]"),
        _cache_row(cache, "train", 5, node("null", node("Lean.Parser.Tactic.exact", atom("exact"), ref(0)),
                   node("Lean.Parser.Tactic.exact", atom("exact"), ref(0))), "exact h0", alignment="unknown", target=False),
    ], "val": [
        _cache_row(cache, "val", 0),
        _cache_row(cache, "val", 1, parameter_rule, "rw [Nat.add_zero (n := h0)]"),
        _cache_row(cache, "val", 2, automatic, "intros", alignment="branch_opener", target=False),
        _cache_row(cache, "val", 3, rw_root(ref(name="_private.A.0.foo")), "rw [_private.A.0.foo]"),
    ]})
    assert coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache), "--output-dir", str(baseline)]) == 0
    assert recovery.main(["--baseline-coverage-dir", str(baseline), "--renderer", "template-v3", "--output-dir", str(recovered)]) == 2
    return recovered, cache, output


def test_inventory_covers_all_rendered_forms_and_freezes_all_exact_val_rows(tmp_path, inventory_artifacts):
    recovered, _, output = inventory_artifacts
    before = _snapshot(tmp_path)
    assert inventory.main(["--recovery-dir", str(recovered), "--output-dir", str(output)]) == 0
    report = json.loads((output / "catalog.json").read_text())
    assert report["complete"] and report["observed_forms"] == 5
    assert report["forms_without_exact_candidates"] == 2
    assert report["forms_without_renderable_candidates"] == 1
    assert len(report["losses_vs_v2"]) == 1
    assert report["loss_counts_vs_v2"] == {"train": 1, "val": 0}
    assert report["losses_vs_v2"][0]["error_code"] == "multi_tactic_wrapper"
    assert report["splits"]["train"]["renderable_steps"] == 4
    assert report["splits"]["val"]["exact_aligned_renderable_steps"] == 2
    assert any(entry["unrenderable_counts"]["val"] for entry in report["forms"].values())
    val = catalog.load_validation_plan(output / "plan_val_all_exact.json", split="val", renderer="template-v3")
    train = catalog.load_validation_plan(output / "plan_train_forms.json", split="train", renderer="template-v3")
    assert len(val["rows"]) == 2 and {row["row_index"] for row in val["rows"]} == {0, 1}
    assert len(train["rows"]) == len(train["selected_forms"]) == 2
    after = _snapshot(tmp_path)
    assert all(after[path] == digest for path, digest in before.items())
    assert all(path in before or path.startswith("inventory/") for path in after)
    incomplete = copy.deepcopy(val)
    incomplete["rows"].pop()
    incomplete["selected_forms"] = [incomplete["rows"][0]["form_id"]]
    incomplete["manifest"]["selected_steps"] = 1
    (output / "incomplete_plan.json").write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="all exact-aligned validation rows"):
        catalog.load_validation_plan(output / "incomplete_plan.json", split="val", renderer="template-v3")


def test_inventory_rejects_changed_snapshot_and_leaves_incomplete_inventory(inventory_artifacts):
    recovered, cache, output = inventory_artifacts
    path = cache / "val/action_trace_v3/000000001.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="cache changed"):
        inventory.main(["--recovery-dir", str(recovered), "--output-dir", str(output)])
    assert not json.loads((output / "catalog.json").read_text())["complete"]


def test_native_form_summary_never_confuses_sample_success_with_universal_validity():
    selected = [{"form_id": "A"}, {"form_id": "A"}, {"form_id": "B"}]
    partial = catalog.validation_summary([{"template_form_id": "A", "status": "rendered_transition_match"}], selected)
    assert partial["pending_steps"] == 2 and partial["forms_all_selected_instances_matched"] == 0
    final = catalog.validation_summary([
        {"template_form_id": "A", "status": "rendered_transition_match"},
        {"template_form_id": "A", "status": "rendered_replay_failed"},
        {"template_form_id": "B", "status": "rendered_transition_match"},
    ], selected)
    assert final["forms_all_selected_instances_matched"] == final["forms_with_failures"] == 1
    assert not final["universal_template_validity_proved"]
