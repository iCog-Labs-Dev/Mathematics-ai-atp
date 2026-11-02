"""Inventory all extracted V3 forms and freeze exact-aligned native validation cohorts."""

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn import tactic_templates
from maths_ai.gnn_inference.atp_lean_gnn.intro_templates import MAX_INTRO_BINDERS
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import canonical_form, form_id, renderer_fingerprints
from maths_ai.gnn_inference.scripts.audit_unsupported_actions import _rows, read_snapshot, verify_coverage


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def schema_registry():
    shared = {"families": ["rw", "simp"], "rw_directions": ["forward", "backward"],
              "simp_only": True, "location": ["goal", "one_local"], "reference_sources": ["local", "library"]}
    return {
        "rw_simp_template_v1": {**shared, "max_rules": tactic_templates.MAX_RULES, "arguments_per_rule": 0,
            "active_v3": False},
        "rw_simp_template_v2": {**shared, "max_rules": tactic_templates.MAX_RULES,
            "arguments_per_rule": tactic_templates.MAX_RULE_ARGUMENTS,
            "argument_kind": "positional_reference", "active_v3": False},
        "rw_simp_template_v3": {**shared, "max_rules": tactic_templates.MAX_V3_RULES,
            "arguments_per_rule": tactic_templates.MAX_RULE_ARGUMENTS,
            "argument_kind": "positional_or_named_reference", "active_v3": True},
        "intro_binders_v1": {"families": ["intro", "intros"], "max_explicit_binders": MAX_INTRO_BINDERS,
            "binder_kinds": ["named", "anonymous"], "modes": ["explicit", "one", "all"], "active_v3": True},
        "reference_application_v1": {"families": ["exact", "apply", "refine"],
            "max_arguments": tactic_templates.MAX_RULE_ARGUMENTS,
            "argument_kind": "positional_or_named_reference", "active_v3": True},
        "flat_fallback": {"typed_target": False, "max_references": 3,
            "families": "observed admitted flat-grammar tactic names", "active_v3": True},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-examples-per-form", type=int, default=1)
    args = parser.parse_args(argv)
    if args.train_examples_per_form < 1:
        raise ValueError("train examples per form must be positive")
    source_dir, output = args.recovery_dir.resolve(), args.output_dir.resolve()
    source_path, rows_path = source_dir / "report.json", source_dir / "rows.jsonl"
    recovery = json.loads(source_path.read_text(encoding="utf-8"))
    if (recovery.get("complete") is not True or recovery["manifest"].get("schema") != "template_recovery_v2"
            or recovery["manifest"].get("renderer") != "template-v3"):
        raise ValueError("complete V3 recovery report required")
    source_hashes = {"report": file_sha256(source_path), "rows": file_sha256(rows_path)}
    if source_hashes["rows"] != recovery["row_records_sha256"]:
        raise ValueError("recovery row records changed")
    fingerprints = renderer_fingerprints()
    if any(recovery["manifest"]["code_sha256"].get(key) != value for key, value in fingerprints.items()):
        raise ValueError("renderer changed since recovery; rerun paired recovery first")
    baseline = Path(recovery["manifest"]["baseline_dir"])
    baseline_report, baseline_rows = baseline / "report.json", baseline / "rows.jsonl"
    baseline_hashes = {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)}
    if baseline_hashes != recovery["manifest"]["baseline_sha256"]:
        raise ValueError("frozen baseline changed since recovery")
    original = json.loads(baseline_report.read_text(encoding="utf-8"))
    verify_coverage(original, baseline_rows)
    cache = Path(recovery["manifest"]["cache_root"]).resolve()
    for root in (cache, source_dir, baseline.resolve(), Path(original["manifest"]["prepared_root"]).resolve()):
        if output == root or root in output.parents:
            raise ValueError("inventory output must be outside input artifact roots")
    output.mkdir(parents=True, exist_ok=False)
    catalog = {"manifest": {"schema": "extracted_template_inventory_v1", "renderer": "template-v3",
        "recovery_dir": str(source_dir), "recovery_sha256": source_hashes, "baseline_sha256": baseline_hashes,
        "renderer_sha256": fingerprints, "form_policy_sha256": file_sha256(Path(__file__).parents[1] / "atp_lean_gnn/template_catalog.py"),
        "inventory_runner_sha256": file_sha256(Path(__file__)), "cache_root": str(cache),
        "native_execution_measured": False, "training_labels_exported": False},
        "complete": False, "schema_registry": schema_registry(), "forms": {}, "splits": {}, "losses_vs_v2": []}
    _write(output / "catalog.json", catalog)
    totals, exact_totals, rendered_totals, parsed_totals = Counter(), Counter(), Counter(), Counter()
    counts_by_schema, counts_by_family = defaultdict(Counter), defaultdict(Counter)
    forms, train_choices, val_selection = {}, defaultdict(list), []
    sentinel = object()
    for processed, pair in enumerate(itertools.zip_longest(_rows(baseline_rows), _rows(rows_path), fillvalue=sentinel), 1):
        if any(value is sentinel for value in pair):
            raise ValueError("recovery/baseline population mismatch")
        base, recovered = pair[0][0], pair[1][0]
        identity = (base["split"], base["row_index"])
        if identity != (recovered["split"], recovered["row_index"]) or recovered.get("v3_evaluated") is not True:
            raise ValueError("recovery row identity/policy mismatch")
        split = base["split"]
        totals[split] += 1
        if recovered["v2_renderable"] and not recovered["v3_renderable"]:
            catalog["losses_vs_v2"].append({"split": split, "row_index": base["row_index"],
                "family": base["family"], "alignment_kind": base["alignment_kind"],
                "dataset_target_matches_source": base["dataset_target_matches_source"],
                "previous_rendered_tactic": recovered.get("rendered_tactic"),
                "error_code": recovered.get("v3_error_code"), "error": recovered.get("v3_error")})
        if processed % 10000 == 0:
            print(f"inventoried {processed} baseline rows", flush=True)
        if not recovered["v3_renderable"] and recovered.get("v3_schema") is None:
            continue
        if recovered["quarantined"]:
            raise ValueError("quarantined row cannot be renderable")
        raw, trace = read_snapshot(base, cache)
        rendered = tactic_templates.prepare_gold_rendering(raw["tactic"], trace, renderer="template-v3")
        if (rendered["rendered_tactic"] != recovered["v3_rendered_tactic"]
                or (rendered["template_target"] or {}).get("schema") != recovered.get("v3_schema")):
            raise ValueError(f"rendered action changed since recovery: {identity}")
        shape = canonical_form(rendered["template_target"], rendered["rendered_tactic"], trace["local_context"])
        key = form_id(shape)
        exact = base["alignment_kind"] == "exact" and base["dataset_target_matches_source"] is True
        renderable = rendered["rendered_tactic"] is not None
        parsed_totals[split] += rendered["template_target"] is not None
        if renderable:
            counts_by_schema[split][shape["schema"]] += 1
            counts_by_family[split][shape["family"]] += 1
            rendered_totals[split] += 1
            exact_totals[split] += exact
        entry = forms.setdefault(key, {"form": shape, "counts": {"train": 0, "val": 0},
            "renderable_counts": {"train": 0, "val": 0}, "unrenderable_counts": {"train": 0, "val": 0},
            "exact_counts": {"train": 0, "val": 0}, "example": {
                "split": split, "row_index": base["row_index"], "theorem": base["theorem"],
                "dataset_tactic": raw["tactic"], "rendered_tactic": rendered["rendered_tactic"],
                "typed_target": rendered["template_target"], "render_error": rendered["render_error"]}})
        entry["counts"][split] += 1
        entry["renderable_counts"][split] += renderable
        entry["unrenderable_counts"][split] += not renderable
        entry["exact_counts"][split] += exact and renderable
        if not renderable and "unrenderable_example" not in entry:
            entry["unrenderable_example"] = {"split": split, "row_index": base["row_index"],
                "typed_target": rendered["template_target"], "render_error": rendered["render_error"]}
        if exact and renderable:
            file_path = raw.get("file_path")
            if not isinstance(file_path, str) or not file_path or Path(file_path).is_absolute() or ".." in Path(file_path).parts:
                raise ValueError(f"invalid source coordinates for {identity}")
            selected = {"row_index": base["row_index"], "theorem": base["theorem"], "file_path": file_path,
                "form_id": key, "schema": shape["schema"], "family": shape["family"], "dataset": raw["dataset"],
                "cache_hashes": base["cache_sha256"],
                "rendered_sha256": hashlib.sha256(rendered["rendered_tactic"].encode()).hexdigest()}
            if split == "val":
                val_selection.append(selected)
            else:
                rank = hashlib.sha256(f"{split}:{base['row_index']}".encode()).hexdigest()
                choices = train_choices[key]
                if len(choices) < args.train_examples_per_form:
                    choices.append((rank, selected))
                    if len(choices) == args.train_examples_per_form:
                        choices.sort(key=lambda item: item[0])
                elif rank < choices[-1][0]:
                    choices.append((rank, selected))
                    choices.sort(key=lambda item: item[0])
                    choices.pop()
    for split in ("train", "val"):
        expected = recovery["splits"][split]["summary"]
        if (totals[split] != expected["selected_steps"] or rendered_totals[split] != expected["template_v3_renderable_steps"]
                or exact_totals[split] != expected["exact_aligned_v3_renderable_steps"]):
            raise ValueError(f"inventory denominator/support mismatch: {split}")
        catalog["splits"][split] = {"selected_steps": totals[split], "renderable_steps": rendered_totals[split],
            "parsed_target_steps": parsed_totals[split], "exact_aligned_renderable_steps": exact_totals[split], "by_schema": dict(counts_by_schema[split]),
            "by_family": dict(counts_by_family[split])}
    if ({"report": file_sha256(source_path), "rows": file_sha256(rows_path)} != source_hashes
            or {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)} != baseline_hashes):
        raise ValueError("input reports changed during inventory")
    catalog.update(complete=True, forms=forms, observed_forms=len(forms),
        forms_without_exact_candidates=sum(not sum(entry["exact_counts"].values()) for entry in forms.values()),
        forms_without_renderable_candidates=sum(not sum(entry["renderable_counts"].values()) for entry in forms.values()),
        loss_counts_vs_v2={split: sum(row["split"] == split for row in catalog["losses_vs_v2"]) for split in ("train", "val")})
    _write(output / "catalog.json", catalog)
    catalog_hash = file_sha256(output / "catalog.json")
    plans = {"val_all_exact": ("val", val_selection), "train_forms": (
        "train", [selected for choices in train_choices.values() for _, selected in choices])}
    for name, (split, selected) in plans.items():
        selected.sort(key=lambda row: row["row_index"])
        plan = {"manifest": {"schema": "template_validation_plan_v1", "renderer": "template-v3", "split": split,
            "catalog_sha256": catalog_hash, "renderer_sha256": fingerprints,
            "form_policy_sha256": catalog["manifest"]["form_policy_sha256"], "selection_policy": name,
            "selected_steps": len(selected), "cache_root": str(cache),
            "train_examples_per_form": args.train_examples_per_form if split == "train" else None},
            "complete": True, "rows": selected, "selected_forms": sorted({row["form_id"] for row in selected})}
        _write(output / f"plan_{name}.json", plan)
    print(json.dumps({"splits": catalog["splits"], "observed_forms": len(forms),
        "forms_without_exact_candidates": catalog["forms_without_exact_candidates"],
        "loss_counts_vs_v2": catalog["loss_counts_vs_v2"],
        "validation_steps": len(val_selection), "train_form_steps": len(plans["train_forms"][1])}, indent=2), flush=True)
    print(f"Catalog: {output / 'catalog.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
