"""Measure V5 against frozen V4 results without recomputing older renderer stages."""

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn import tactic_templates
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import renderer_fingerprints
from maths_ai.gnn_inference.scripts.audit_template_recovery import select_native_pilot
from maths_ai.gnn_inference.scripts.audit_unsupported_actions import _rows, read_snapshot, verify_coverage


def _new_stats():
    return Counter({key: 0 for key in (
        "selected_steps", "quarantined_steps", "baseline_v4_renderable_steps", "template_v5_renderable_steps",
        "v5_rescues_vs_v4", "v5_losses_vs_v4", "exact_aligned_baseline_v4_steps", "exact_aligned_v5_steps",
        "exact_aligned_v5_rescues", "exact_aligned_v5_losses", "declaration_rescues", "simplifier_wrapper_rescues",
        "rewrite_wrapper_rescues", "existing_rule_syntax_rescues", "term_syntax_rescues", "other_wrapper_rescues",
    )})


def _add(stats, row):
    before, after = row["v4_renderable"], row["v5_renderable"]
    exact = row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True
    rescue = after and not before
    family = row["family"]
    category = ("declaration" if family in {"have", "let"} else
                "simplifier_wrapper" if family in {"simpa", "simp_all", "dsimp", "simp_rw"} else
                "rewrite_wrapper" if family in {"rewrite", "rwa"} else
                "existing_rule_syntax" if family in {"rw", "simp"} else
                "term_syntax" if family in {"exact", "apply", "refine", "refine'"} else "other_wrapper")
    stats.update({"selected_steps": 1, "quarantined_steps": row["quarantined"],
                  "baseline_v4_renderable_steps": before, "template_v5_renderable_steps": after,
                  "v5_rescues_vs_v4": rescue, "v5_losses_vs_v4": before and not after,
                  "exact_aligned_baseline_v4_steps": exact and before, "exact_aligned_v5_steps": exact and after,
                  "exact_aligned_v5_rescues": exact and rescue, "exact_aligned_v5_losses": exact and before and not after,
                  f"{category}_rescues": rescue})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-recovery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-actions", type=int, default=24)
    args = parser.parse_args(argv)
    if args.pilot_actions < 1:
        raise ValueError("pilot action count must be positive")
    previous, output = args.previous_recovery_dir.resolve(), args.output_dir.resolve()
    report_path, rows_path = previous / "report.json", previous / "rows.jsonl"
    source = json.loads(report_path.read_text(encoding="utf-8"))
    hashes = {"report": file_sha256(report_path), "rows": file_sha256(rows_path)}
    if (source.get("complete") is not True or source["manifest"].get("schema") != "template_recovery_v3"
            or source["manifest"].get("renderer") != "template-v4" or hashes["rows"] != source["row_records_sha256"]):
        raise ValueError("complete immutable V4 recovery report required")
    baseline = Path(source["manifest"]["baseline_dir"])
    baseline_report, baseline_rows = baseline / "report.json", baseline / "rows.jsonl"
    if {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)} != source["manifest"]["baseline_sha256"]:
        raise ValueError("original baseline changed since V4 audit")
    original = json.loads(baseline_report.read_text(encoding="utf-8"))
    verify_coverage(original, baseline_rows)
    cache = Path(source["manifest"]["cache_root"]).resolve()
    for root in (previous, baseline.resolve(), cache, Path(original["manifest"]["prepared_root"]).resolve()):
        if output == root or root in output.parents:
            raise ValueError("output must be outside source artifact roots")
    report = {"manifest": {"schema": "wrapper_recovery_v1", "renderer": "template-v5",
        "previous_recovery_dir": str(previous), "previous_sha256": hashes,
        "frozen_baseline_sha256": source["manifest"]["baseline_sha256"], "cache_root": str(cache),
        "renderer_sha256": renderer_fingerprints(), "runner_sha256": file_sha256(Path(__file__)),
        "pilot_actions": args.pilot_actions, "native_execution_measured": False, "model_training_measured": False},
        "complete": False, "splits": {}}
    output.mkdir(parents=True, exist_ok=False)
    totals, families, errors, family_errors, groups = defaultdict(_new_stats), defaultdict(_new_stats), defaultdict(Counter), defaultdict(Counter), defaultdict(list)

    def checkpoint():
        report["splits"] = {split: {"summary": dict(stats), "v5_error_counts": dict(errors[split]),
            "by_family": {family: {"summary": dict(counts), "error_counts": dict(family_errors[(split, family)])}
                          for (s, family), counts in families.items() if s == split}} for split, stats in totals.items()}
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(output / "report.json")

    checkpoint()
    digest, sentinel = hashlib.sha256(), object()
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        pairs = itertools.zip_longest(_rows(baseline_rows), _rows(rows_path), fillvalue=sentinel)
        for processed, pair in enumerate(pairs, 1):
            if any(value is sentinel for value in pair):
                raise ValueError("V4/baseline row population mismatch")
            original_row, previous_row = pair[0][0], pair[1][0]
            identity = (original_row["split"], original_row["row_index"])
            if identity != (previous_row["split"], previous_row["row_index"]) or previous_row.get("v4_evaluated") is not True:
                raise ValueError("V4 row identity/policy mismatch")
            raw, trace = read_snapshot(original_row, cache)
            row = {key: previous_row[key] for key in ("split", "row_index", "family", "alignment_kind", "dataset_target_matches_source", "quarantined", "v4_renderable")}
            if row["quarantined"] != (original_row["status"] in {"cache_invalid", "source_trace_invalid"}):
                raise ValueError("quarantine population changed")
            row.update(v5_renderable=False, v5_schema=None)
            if not row["quarantined"]:
                result = tactic_templates.prepare_gold_rendering(raw["tactic"], trace, renderer="template-v5")
                target = result["template_target"] or {}
                row.update(v5_renderable=isinstance(result["rendered_tactic"], str), v5_schema=target.get("schema"),
                           v5_rendered_tactic=result["rendered_tactic"], v5_error=result["render_error"], v5_error_code=result["template_error_code"])
                if row["v5_error"]:
                    error = row["v5_error_code"] or row["v5_error"]
                    errors[row["split"]][error] += 1
                    family_errors[(row["split"], row["family"])][error] += 1
                if (row["split"] == "val" and row["v5_renderable"] and not row["v4_renderable"]
                        and row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True):
                    features = (target.get("location", {}).get("kind", "goal"), bool(target.get("using")),
                                bool(target.get("declaration")), tuple(rule.get("kind") for rule in target.get("rules", [])))
                    groups[(row["family"], features)].append((hashlib.sha256(str(row["row_index"]).encode()).hexdigest(), row["row_index"]))
            _add(totals[row["split"]], row)
            _add(families[(row["split"], row["family"])], row)
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode("utf-8"))
            if processed % 10000 == 0:
                handle.flush()
                checkpoint()
                print(f"checked {processed} frozen V4 rows", flush=True)
    if {"report": file_sha256(report_path), "rows": file_sha256(rows_path)} != hashes:
        raise ValueError("V4 recovery changed during comparison")
    if {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)} != source["manifest"]["baseline_sha256"]:
        raise ValueError("original baseline changed during comparison")
    for split in ("train", "val"):
        expected = source["splits"][split]["summary"]
        if (totals[split]["selected_steps"] != expected["selected_steps"]
                or totals[split]["baseline_v4_renderable_steps"] != expected["template_v4_renderable_steps"]):
            raise ValueError("V5 audit changed frozen V4 denominator/support")
    chosen = select_native_pilot(groups, args.pilot_actions)
    (output / "new_val_pilot_rows.json").write_text(json.dumps(chosen), encoding="utf-8")
    report.update(complete=True, row_records_sha256=digest.hexdigest(), native_pilot={"selected_steps": len(chosen),
        "split": "val", "policy": "round_robin_family_then_features_lowest_sha_v2", "candidate_groups": len(groups)})
    checkpoint()
    print(json.dumps({split: dict(stats) for split, stats in totals.items()}, indent=2), flush=True)
    print("Native pilot:", json.dumps(report["native_pilot"]), flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)
    return 2 if any(counts["v5_losses_vs_v4"] for counts in totals.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
