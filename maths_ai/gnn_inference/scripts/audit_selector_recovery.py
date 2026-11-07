"""Measure syntax-only V7 method selectors against saved V6 across all frozen rows."""

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import prepare_gold_rendering
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import renderer_fingerprints
from maths_ai.gnn_inference.scripts.audit_scoped_recovery import _kinds
from maths_ai.gnn_inference.scripts.audit_template_recovery import select_native_pilot
from maths_ai.gnn_inference.scripts.audit_unsupported_actions import _rows, read_snapshot, verify_coverage


def _stats():
    return Counter({key: 0 for key in (
        "selected_steps", "quarantined_steps", "baseline_v6_renderable_steps", "template_v7_renderable_steps",
        "v7_rescues_vs_v6", "v7_losses_vs_v6", "exact_aligned_baseline_v6_steps", "exact_aligned_v7_steps",
        "exact_aligned_v7_rescues", "exact_aligned_v7_losses", "method_selector_steps", "method_selector_rescues",
        "other_source_repair_rescues", "method_selector_nodes",
    )})


def _add(stats, row):
    before, after = row["v6_renderable"], row["v7_renderable"]
    exact = row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True
    rescue, loss, selectors = after and not before, before and not after, bool(row["method_selector_count"])
    stats.update(selected_steps=1, quarantined_steps=row["quarantined"], baseline_v6_renderable_steps=before,
        template_v7_renderable_steps=after, v7_rescues_vs_v6=rescue, v7_losses_vs_v6=loss,
        exact_aligned_baseline_v6_steps=exact and before, exact_aligned_v7_steps=exact and after,
        exact_aligned_v7_rescues=exact and rescue, exact_aligned_v7_losses=exact and loss,
        method_selector_steps=after and selectors, method_selector_rescues=rescue and selectors,
        other_source_repair_rescues=rescue and not selectors,
        method_selector_nodes=row["method_selector_count"] if after else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-recovery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-actions", type=int, default=24)
    args = parser.parse_args(argv)
    if args.pilot_actions < 1:
        raise ValueError("pilot action count must be positive")
    previous, output = args.previous_recovery_dir.resolve(), args.output_dir.resolve()
    report_path, previous_rows = previous / "report.json", previous / "rows.jsonl"
    source = json.loads(report_path.read_text())
    previous_hashes = {"report": file_sha256(report_path), "rows": file_sha256(previous_rows)}
    if (source.get("complete") is not True or source["manifest"].get("schema") != "scoped_recovery_v1"
            or source["manifest"].get("renderer") != "template-v6" or previous_hashes["rows"] != source["row_records_sha256"]):
        raise ValueError("complete immutable V6 recovery report required")
    baseline = Path(source["manifest"]["baseline_dir"]).resolve()
    baseline_report, baseline_rows = baseline / "report.json", baseline / "rows.jsonl"
    baseline_hashes = {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)}
    if baseline_hashes != source["manifest"]["baseline_sha256"]:
        raise ValueError("original baseline changed since V6 audit")
    original = json.loads(baseline_report.read_text())
    verify_coverage(original, baseline_rows)
    cache = Path(source["manifest"]["cache_root"]).resolve()
    for root in (previous, baseline, cache, Path(original["manifest"]["prepared_root"]).resolve()):
        if output == root or root in output.parents:
            raise ValueError("output must be outside frozen source artifacts")
    report = {"manifest": {"schema": "selector_recovery_v1", "renderer": "template-v7",
        "previous_recovery_dir": str(previous), "previous_sha256": previous_hashes,
        "baseline_dir": str(baseline), "baseline_sha256": baseline_hashes, "cache_root": str(cache),
        "renderer_sha256": renderer_fingerprints(), "runner_sha256": file_sha256(Path(__file__)),
        "semantic_cache_used": False, "native_execution_measured": False, "model_training_measured": False,
        "pilot_actions": args.pilot_actions}, "complete": False, "splits": {}}
    output.mkdir(parents=True, exist_ok=False)
    totals, families = defaultdict(_stats), defaultdict(_stats)
    errors, family_errors, groups = defaultdict(Counter), defaultdict(Counter), defaultdict(list)

    def checkpoint():
        report["splits"] = {split: {"summary": {**dict(stats),
            "coverage_percent": 100 * stats["template_v7_renderable_steps"] / stats["selected_steps"]},
            "v7_error_counts": dict(errors[split]),
            "by_family": {family: {"summary": dict(counts), "error_counts": dict(family_errors[(split, family)])}
                          for (s, family), counts in families.items() if s == split}} for split, stats in totals.items()}
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        temporary.replace(output / "report.json")

    checkpoint()
    digest, sentinel = hashlib.sha256(), object()
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        pairs = itertools.zip_longest(_rows(baseline_rows), _rows(previous_rows), fillvalue=sentinel)
        for processed, pair in enumerate(pairs, 1):
            if any(value is sentinel for value in pair):
                raise ValueError("V6/original row population mismatch")
            original_row, previous_row = pair[0][0], pair[1][0]
            identity = (original_row["split"], original_row["row_index"])
            if identity != (previous_row["split"], previous_row["row_index"]):
                raise ValueError("V6 row identity mismatch")
            for key in ("alignment_kind", "dataset_target_matches_source"):
                if previous_row[key] != original_row[key]:
                    raise ValueError("V6 row alignment changed from original snapshot")
            raw, trace = read_snapshot(original_row, cache)
            row = {key: previous_row[key] for key in ("split", "row_index", "family", "alignment_kind",
                "dataset_target_matches_source", "quarantined", "v6_renderable")}
            if row["quarantined"] != (original_row["status"] in {"cache_invalid", "source_trace_invalid"}):
                raise ValueError("quarantine population changed")
            row.update(v7_renderable=False, v7_schema=None, method_selector_count=0)
            if not row["quarantined"]:
                result = prepare_gold_rendering(raw["tactic"], trace, renderer="template-v7")
                if result.get("semantic_reconstructions", 0):
                    raise ValueError("V7 syntax-only policy unexpectedly used semantic reconstruction")
                target = result["template_target"] or {}
                row.update(v7_renderable=isinstance(result["rendered_tactic"], str), v7_schema=target.get("schema"),
                    v7_rendered_tactic=result["rendered_tactic"], v7_error=result["render_error"],
                    v7_error_code=result["template_error_code"], method_selector_count=result.get("method_selector_count", 0))
                if row["v7_error"]:
                    error = row["v7_error_code"] or row["v7_error"]
                    errors[row["split"]][error] += 1
                    family_errors[(row["split"], row["family"])][error] += 1
                if (row["split"] == "val" and row["v7_renderable"] and not row["v6_renderable"]
                        and row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True):
                    features = (bool(row["method_selector_count"]), tuple(sorted(_kinds(target))))
                    groups[(row["family"], features)].append((hashlib.sha256(str(row["row_index"]).encode()).hexdigest(), row["row_index"]))
            _add(totals[row["split"]], row)
            _add(families[(row["split"], row["family"])], row)
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode())
            if processed % 10000 == 0:
                handle.flush()
                checkpoint()
    if {"report": file_sha256(report_path), "rows": file_sha256(previous_rows)} != previous_hashes:
        raise ValueError("V6 recovery changed during comparison")
    if {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)} != baseline_hashes:
        raise ValueError("original baseline changed during comparison")
    for split in ("train", "val"):
        expected = source["splits"][split]["summary"]
        if (totals[split]["selected_steps"] != expected["selected_steps"]
                or totals[split]["baseline_v6_renderable_steps"] != expected["template_v6_renderable_steps"]
                or totals[split]["quarantined_steps"] != expected["quarantined_steps"]):
            raise ValueError("V7 comparison changed frozen V6 population/support/quarantine")
    selected = select_native_pilot(groups, args.pilot_actions)
    (output / "new_val_pilot_rows.json").write_text(json.dumps(selected))
    report.update(complete=True, row_records_sha256=digest.hexdigest(), native_pilot={"selected_steps": len(selected),
        "split": "val", "policy": "round_robin_family_then_features_lowest_sha_v2", "candidate_groups": len(groups)})
    checkpoint()
    print("split  old_support  new_support  gained  lost  exact_gained  coverage", flush=True)
    for split, data in report["splits"].items():
        stats = data["summary"]
        print(f"{split:5}  {stats['baseline_v6_renderable_steps']:11}  {stats['template_v7_renderable_steps']:11}"
              f"  {stats['v7_rescues_vs_v6']:6}  {stats['v7_losses_vs_v6']:4}  {stats['exact_aligned_v7_rescues']:12}"
              f"  {stats['coverage_percent']:.2f}%", flush=True)
    print(f"Native pilot: {len(selected)} rows; report: {output / 'report.json'}", flush=True)
    return 2 if any(stats["v7_losses_vs_v6"] for stats in totals.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
