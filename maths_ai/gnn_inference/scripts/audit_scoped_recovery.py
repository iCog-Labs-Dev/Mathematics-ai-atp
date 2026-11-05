"""Pair V6 scope/pattern/semantic support against immutable saved V5 outcomes."""

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.semantic_templates import load_semantic_trace
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import prepare_gold_rendering
from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import renderer_fingerprints
from maths_ai.gnn_inference.scripts.audit_template_recovery import select_native_pilot
from maths_ai.gnn_inference.scripts.audit_unsupported_actions import _rows, read_snapshot, verify_coverage


def _stats():
    return Counter({key: 0 for key in (
        "selected_steps", "quarantined_steps", "baseline_v5_renderable_steps", "v6_syntax_renderable_steps",
        "template_v6_renderable_steps", "v6_rescues_vs_v5", "v6_losses_vs_v5", "exact_aligned_baseline_v5_steps",
        "exact_aligned_v6_steps", "exact_aligned_v6_rescues", "exact_aligned_v6_losses", "semantic_reconstructed_steps",
        "semantic_rescues", "pattern_rescues", "scoped_term_rescues", "other_syntax_rescues",
    )})


def _kinds(value):
    kinds, pending = set(), [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            if "kind" in current:
                kinds.add(str(current["kind"]))
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
    return kinds


def _add(stats, row):
    before, after = row["v5_renderable"], row["v6_renderable"]
    exact = row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True
    rescue, loss = after and not before, before and not after
    category = ("semantic" if row.get("semantic_reconstructions") else
                "pattern" if row["v6_schema"] == "scoped_pattern_tactic_v1" else
                "scoped_term" if {"lambda", "forall", "bound_reference"} & set(row["term_kinds"]) else "other_syntax")
    stats.update(selected_steps=1, quarantined_steps=row["quarantined"], baseline_v5_renderable_steps=before,
                 v6_syntax_renderable_steps=row["v6_syntax_renderable"], template_v6_renderable_steps=after,
                 v6_rescues_vs_v5=rescue, v6_losses_vs_v5=loss, exact_aligned_baseline_v5_steps=exact and before,
                 exact_aligned_v6_steps=exact and after, exact_aligned_v6_rescues=exact and rescue,
                 exact_aligned_v6_losses=exact and loss, semantic_reconstructed_steps=bool(row.get("semantic_reconstructions")))
    stats[f"{category}_rescues"] += rescue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-recovery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-actions", type=int, default=24)
    args = parser.parse_args(argv)
    if args.pilot_actions < 1:
        raise ValueError("pilot action count must be positive")
    previous, output = args.previous_recovery_dir.resolve(), args.output_dir.resolve()
    source_path, previous_rows = previous / "report.json", previous / "rows.jsonl"
    source = json.loads(source_path.read_text())
    previous_hashes = {"report": file_sha256(source_path), "rows": file_sha256(previous_rows)}
    if (source.get("complete") is not True or source["manifest"].get("schema") != "wrapper_recovery_v1"
            or source["manifest"].get("renderer") != "template-v5" or previous_hashes["rows"] != source["row_records_sha256"]):
        raise ValueError("complete immutable V5 recovery report required")
    v4 = Path(source["manifest"]["previous_recovery_dir"])
    if {"report": file_sha256(v4 / "report.json"), "rows": file_sha256(v4 / "rows.jsonl")} != source["manifest"]["previous_sha256"]:
        raise ValueError("V4 predecessor changed since V5 audit")
    v4_report = json.loads((v4 / "report.json").read_text())
    baseline = Path(v4_report["manifest"]["baseline_dir"])
    baseline_report, baseline_rows = baseline / "report.json", baseline / "rows.jsonl"
    baseline_hashes = {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)}
    if baseline_hashes != source["manifest"]["frozen_baseline_sha256"] or baseline_hashes != v4_report["manifest"]["baseline_sha256"]:
        raise ValueError("original baseline changed since previous audits")
    original = json.loads(baseline_report.read_text())
    verify_coverage(original, baseline_rows)
    cache = Path(source["manifest"]["cache_root"]).resolve()
    if str(cache) != v4_report["manifest"]["cache_root"]:
        raise ValueError("predecessor cache roots differ")
    for root in (previous, v4.resolve(), baseline.resolve(), cache, Path(original["manifest"]["prepared_root"]).resolve()):
        if output == root or root in output.parents:
            raise ValueError("output must be outside source artifact roots")
    report = {"manifest": {"schema": "scoped_recovery_v1", "renderer": "template-v6",
        "previous_recovery_dir": str(previous), "previous_sha256": previous_hashes,
        "baseline_dir": str(baseline), "baseline_sha256": baseline_hashes, "cache_root": str(cache),
        "renderer_sha256": renderer_fingerprints(), "runner_sha256": file_sha256(Path(__file__)),
        "semantic_policy": "optional_v2_digest_context_pin_owned_range_bound_v1", "pilot_actions": args.pilot_actions,
        "native_execution_measured": False, "model_training_measured": False}, "complete": False, "splits": {}}
    output.mkdir(parents=True, exist_ok=False)
    totals, families = defaultdict(_stats), defaultdict(_stats)
    errors, family_errors, semantic_status, groups = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), defaultdict(list)

    def checkpoint():
        report["splits"] = {split: {"summary": dict(stats), "v6_error_counts": dict(errors[split]),
            "semantic_trace_status_counts": dict(semantic_status[split]),
            "by_family": {family: {"summary": dict(counts), "error_counts": dict(family_errors[(split, family)])}
                          for (s, family), counts in families.items() if s == split}} for split, stats in totals.items()}
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        temporary.replace(output / "report.json")

    checkpoint()
    digest, sentinel = hashlib.sha256(), object()
    semantic_digest = hashlib.sha256()
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for processed, pair in enumerate(itertools.zip_longest(_rows(baseline_rows), _rows(previous_rows), fillvalue=sentinel), 1):
            if any(value is sentinel for value in pair):
                raise ValueError("V5/baseline row population mismatch")
            original_row, previous_row = pair[0][0], pair[1][0]
            identity = (original_row["split"], original_row["row_index"])
            if identity != (previous_row["split"], previous_row["row_index"]):
                raise ValueError("V5 row identity mismatch")
            raw, trace = read_snapshot(original_row, cache)
            row = {key: previous_row[key] for key in ("split", "row_index", "family", "alignment_kind",
                    "dataset_target_matches_source", "quarantined", "v5_renderable")}
            if row["quarantined"] != (original_row["status"] in {"cache_invalid", "source_trace_invalid"}):
                raise ValueError("quarantine population changed")
            row.update(v6_renderable=False, v6_syntax_renderable=False, v6_schema=None, semantic_reconstructions=0,
                       semantic_trace_status="quarantined", semantic_trace_sha256=None, term_kinds=[])
            if not row["quarantined"]:
                semantic = load_semantic_trace(cache, row["split"], row["row_index"], raw, trace)
                row.update(semantic_trace_status=semantic["status"], semantic_trace_sha256=semantic["sha256"])
                if "error" in semantic:
                    row["semantic_trace_error"] = semantic["error"]
                result = prepare_gold_rendering(raw["tactic"], trace, renderer="template-v6")
                row["v6_syntax_renderable"] = isinstance(result["rendered_tactic"], str)
                row["v6_syntax_error"] = result["render_error"]
                row["v6_syntax_error_code"] = result["template_error_code"]
                if not row["v6_syntax_renderable"] and semantic["status"] == "bound":
                    result = prepare_gold_rendering(raw["tactic"], trace, renderer="template-v6", semantic_trace=semantic)
                target = result["template_target"] or {}
                row.update(v6_renderable=isinstance(result["rendered_tactic"], str), v6_schema=target.get("schema"),
                           v6_rendered_tactic=result["rendered_tactic"], v6_error=result["render_error"],
                           v6_error_code=result["template_error_code"], semantic_reconstructions=result.get("semantic_reconstructions", 0),
                           term_kinds=sorted(_kinds(target)))
                if row["v6_error"]:
                    error = row["v6_error_code"] or row["v6_error"]
                    errors[row["split"]][error] += 1
                    family_errors[(row["split"], row["family"])][error] += 1
                if (row["split"] == "val" and row["v6_renderable"] and not row["v5_renderable"]
                        and row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True):
                    features = (bool(row["semantic_reconstructions"]), tuple(row["term_kinds"]))
                    groups[(row["family"], features)].append((hashlib.sha256(str(row["row_index"]).encode()).hexdigest(), row["row_index"]))
            semantic_status[row["split"]][row["semantic_trace_status"]] += 1
            semantic_digest.update(json.dumps([*identity, row["semantic_trace_status"], row["semantic_trace_sha256"]]).encode() + b"\n")
            _add(totals[row["split"]], row)
            _add(families[(row["split"], row["family"])], row)
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode())
            if processed % 10000 == 0:
                handle.flush()
                checkpoint()
                print(f"checked {processed} frozen V5 rows", flush=True)
    if {"report": file_sha256(source_path), "rows": file_sha256(previous_rows)} != previous_hashes:
        raise ValueError("V5 recovery changed during comparison")
    if {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)} != baseline_hashes:
        raise ValueError("original baseline changed during comparison")
    for split in ("train", "val"):
        expected = source["splits"][split]["summary"]
        if (totals[split]["selected_steps"] != expected["selected_steps"]
                or totals[split]["baseline_v5_renderable_steps"] != expected["template_v5_renderable_steps"]
                or totals[split]["quarantined_steps"] != expected["quarantined_steps"]):
            raise ValueError("V6 audit changed frozen V5 denominator/support/quarantine")
    chosen = select_native_pilot(groups, args.pilot_actions)
    (output / "new_val_pilot_rows.json").write_text(json.dumps(chosen))
    report.update(complete=True, row_records_sha256=digest.hexdigest(), semantic_inputs_sha256=semantic_digest.hexdigest(),
                  native_pilot={"selected_steps": len(chosen), "split": "val",
                    "policy": "round_robin_family_then_features_lowest_sha_v2", "candidate_groups": len(groups)})
    checkpoint()
    print(json.dumps({split: dict(stats) for split, stats in totals.items()}, indent=2), flush=True)
    print("Native pilot:", json.dumps(report["native_pilot"]), flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)
    return 2 if any(counts["v6_losses_vs_v5"] for counts in totals.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
