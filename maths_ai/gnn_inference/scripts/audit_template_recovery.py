"""Paired static recovery versus a frozen V1 coverage report, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn import tactic_templates
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.scripts.audit_unsupported_actions import _rows, read_snapshot, verify_coverage


def _new_stats():
    return Counter({key: 0 for key in (
        "selected_steps", "quarantined_steps", "baseline_renderable_steps", "lookup_fixed_v1_renderable_steps",
        "template_v2_renderable_steps", "template_v2_parsed_steps", "applied_rule_parsed_steps",
        "lookup_rescues", "lookup_losses", "v2_grammar_rescues", "v2_grammar_losses",
        "applied_rule_rescues", "bare_rule_rescues",
        "combined_rescues", "combined_losses", "exact_aligned_baseline_renderable_steps",
        "exact_aligned_template_v2_renderable_steps", "exact_aligned_combined_rescues",
        "exact_aligned_combined_losses",
        "exact_aligned_lookup_rescues", "exact_aligned_applied_rule_rescues",
    )})


def _add(stats, row):
    before, lookup, final = (row[key] for key in ("baseline_renderable", "lookup_fixed_v1_renderable", "v2_renderable"))
    exact = row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True
    stats.update({"selected_steps": 1, "quarantined_steps": row["quarantined"],
                  "baseline_renderable_steps": before, "lookup_fixed_v1_renderable_steps": lookup,
                  "template_v2_renderable_steps": final, "template_v2_parsed_steps": row["v2_parsed"],
                  "applied_rule_parsed_steps": row["has_applied_rule"],
                  "lookup_rescues": lookup and not before, "lookup_losses": before and not lookup,
                  "v2_grammar_rescues": final and not lookup, "v2_grammar_losses": lookup and not final,
                  "applied_rule_rescues": final and not lookup and row["has_applied_rule"],
                  "bare_rule_rescues": final and not lookup and not row["has_applied_rule"],
                  "combined_rescues": final and not before, "combined_losses": before and not final,
                  "exact_aligned_baseline_renderable_steps": exact and before,
                  "exact_aligned_template_v2_renderable_steps": exact and final,
                  "exact_aligned_combined_rescues": exact and final and not before,
                  "exact_aligned_combined_losses": exact and before and not final,
                  "exact_aligned_lookup_rescues": exact and lookup and not before,
                  "exact_aligned_applied_rule_rescues": exact and final and not lookup and row["has_applied_rule"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-coverage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline, output = args.baseline_coverage_dir.resolve(), args.output_dir.resolve()
    source_path, rows_path = baseline / "report.json", baseline / "rows.jsonl"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    hashes = {"report": file_sha256(source_path), "rows": file_sha256(rows_path)}
    verify_coverage(source, rows_path)
    if source["manifest"]["template_schema"] != tactic_templates.TEMPLATE_SCHEMA:
        raise ValueError("frozen V1 coverage baseline required")
    cache = Path(source["manifest"]["cache_root"]).resolve()
    for root in (baseline, cache, Path(source["manifest"]["prepared_root"]).resolve()):
        if output == root or root in output.parents:
            raise ValueError("output must be outside input artifact roots")
    code_root = Path(__file__).parents[1]
    paths = [Path(__file__), code_root / "scripts/audit_unsupported_actions.py",
             *(code_root / "atp_lean_gnn" / name for name in (
                 "tactic_templates.py", "gold_replay.py", "preparation.py", "labels.py", "argument_relabeling.py")),
             *sorted((code_root / "atp_lean_gnn/action_targets").glob("*.py"))]
    report = {"manifest": {"schema": "template_recovery_v1", "baseline_dir": str(baseline),
                           "baseline_sha256": hashes, "cache_root": str(cache),
                           "template_schema": tactic_templates.APPLICATION_TEMPLATE_SCHEMA,
                           "max_rule_arguments": tactic_templates.MAX_RULE_ARGUMENTS,
                           "reference_lookup": "validated_source_or_exact_global_identity_v1",
                           "code_sha256": {str(path.relative_to(code_root)): file_sha256(path) for path in paths},
                           "native_execution_measured": False, "proof_success_measured": False},
              "complete": False, "splits": {}}
    output.mkdir(parents=True, exist_ok=False)
    totals, families, previous, errors = defaultdict(_new_stats), defaultdict(_new_stats), defaultdict(_new_stats), defaultdict(Counter)

    def checkpoint():
        report["splits"] = {split: {"summary": dict(stats),
            "by_family": {name: dict(value) for (s, name), value in families.items() if s == split},
            "by_previous_error": {name: dict(value) for (s, name), value in previous.items() if s == split},
            "v2_error_counts": dict(errors[split])} for split, stats in totals.items()}
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(output / "report.json")

    checkpoint()
    rows_digest = hashlib.sha256()
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for processed, (original, _) in enumerate(_rows(rows_path), 1):
            raw, trace = read_snapshot(original, cache)
            row = {key: original[key] for key in (
                "split", "row_index", "family", "alignment_kind", "dataset_target_matches_source")}
            row.update(baseline_renderable=original["template_mode_renderable"],
                       quarantined=original["status"] in {"cache_invalid", "source_trace_invalid"},
                       lookup_fixed_v1_renderable=False, v2_renderable=False, v2_parsed=False, has_applied_rule=False)
            if not row["quarantined"]:
                fixed = tactic_templates.prepare_gold_rendering(raw["tactic"], trace, renderer="template-v1")
                extended = tactic_templates.prepare_gold_rendering(raw["tactic"], trace, renderer="template-v2")
                target = extended["template_target"]
                row.update(lookup_fixed_v1_renderable=isinstance(fixed["rendered_tactic"], str),
                           v2_renderable=isinstance(extended["rendered_tactic"], str), v2_parsed=target is not None,
                           has_applied_rule=target is not None and any(rule.get("arguments") for rule in target["rules"]),
                           rendered_tactic=extended["rendered_tactic"], v2_error=extended["render_error"],
                           v2_error_code=extended["template_error_code"])
                if row["v2_error"]:
                    errors[row["split"]][row["v2_error_code"] or row["v2_error"]] += 1
                if target is not None:
                    row["rule_argument_counts"] = [len(rule["arguments"]) for rule in target["rules"]]
            prior_error = "supported" if original["status"] == "renderable" else (
                original.get("template_error_code") or original.get("render_error") or original.get("error") or "unknown")
            _add(totals[row["split"]], row)
            _add(families[(row["split"], row["family"])], row)
            _add(previous[(row["split"], prior_error)], row)
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(encoded)
            rows_digest.update(encoded.encode("utf-8"))
            if processed % 10000 == 0:
                handle.flush()
                checkpoint()
                print(f"checked {processed} baseline rows", flush=True)
    if file_sha256(source_path) != hashes["report"] or file_sha256(rows_path) != hashes["rows"]:
        raise ValueError("baseline coverage changed during recovery audit")
    for split in ("train", "val"):
        if (totals[split]["selected_steps"] != source["splits"][split]["summary"]["selected_steps"]
                or totals[split]["baseline_renderable_steps"] != source["splits"][split]["summary"]["template_mode_renderable_steps"]):
            raise ValueError("recovery cohort differs from baseline")
    report.update(complete=True, row_records_sha256=rows_digest.hexdigest())
    checkpoint()
    print(json.dumps({split: dict(stats) for split, stats in totals.items()}, indent=2), flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)
    return 2 if any(stats["combined_losses"] or stats["lookup_losses"] or stats["v2_grammar_losses"]
                    for stats in totals.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
