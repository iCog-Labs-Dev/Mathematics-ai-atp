"""Measure cached-syntax V7/V8/V9 against a completed saved coverage stage."""

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


def _stats(before="v6", after="v7"):
    return Counter({key: 0 for key in (
        "selected_steps", "quarantined_steps", f"baseline_{before}_renderable_steps", f"template_{after}_renderable_steps",
        f"{after}_rescues_vs_{before}", f"{after}_losses_vs_{before}", f"exact_aligned_baseline_{before}_steps", f"exact_aligned_{after}_steps",
        f"exact_aligned_{after}_rescues", f"exact_aligned_{after}_losses", "method_selector_steps", "method_selector_rescues",
        "other_source_repair_rescues", "method_selector_nodes", "math_notation_steps", "math_notation_nodes", "math_notation_rescues",
        "source_program_steps", "source_program_rescues", "deferred_identifier_steps", "deferred_identifier_rescues",
        "no_deferred_identifier_program_rescues",
    )})


def _add(stats, row, previous="v6", version="v7"):
    before, after = row[f"{previous}_renderable"], row[f"{version}_renderable"]
    exact = row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True
    rescue, loss, selectors = after and not before, before and not after, bool(row["method_selector_count"])
    math = bool(row.get("math_notation_count"))
    program = bool(row.get("source_program_used"))
    deferred = bool(row.get("deferred_name_count"))
    stats.update({"selected_steps": 1, "quarantined_steps": row["quarantined"], f"baseline_{previous}_renderable_steps": before,
        f"template_{version}_renderable_steps": after, f"{version}_rescues_vs_{previous}": rescue, f"{version}_losses_vs_{previous}": loss,
        f"exact_aligned_baseline_{previous}_steps": exact and before, f"exact_aligned_{version}_steps": exact and after,
        f"exact_aligned_{version}_rescues": exact and rescue, f"exact_aligned_{version}_losses": exact and loss,
        "method_selector_steps": after and selectors, "method_selector_rescues": rescue and selectors,
        "other_source_repair_rescues": rescue and not (program if version == "v9" else math if version == "v8" else selectors),
        "method_selector_nodes": row["method_selector_count"] if after else 0,
        "math_notation_steps": after and math, "math_notation_nodes": row.get("math_notation_count", 0) if after else 0,
        "math_notation_rescues": rescue and math, "source_program_steps": after and program,
        "source_program_rescues": rescue and program, "deferred_identifier_steps": after and deferred,
        "deferred_identifier_rescues": rescue and deferred,
        "no_deferred_identifier_program_rescues": rescue and program and not deferred})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-recovery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-actions", type=int, default=24)
    parser.add_argument("--renderer", choices=("template-v7", "template-v8", "template-v9"), default="template-v7")
    args = parser.parse_args(argv)
    if args.pilot_actions < 1:
        raise ValueError("pilot action count must be positive")
    previous, output = args.previous_recovery_dir.resolve(), args.output_dir.resolve()
    version = args.renderer.split("-")[-1]
    report_path, previous_rows = previous / "report.json", previous / "rows.jsonl"
    source = json.loads(report_path.read_text())
    prior = source["manifest"].get("renderer", "").split("-")[-1]
    allowed = {"v7": {"v6"}, "v8": {"v7"}, "v9": {"v7", "v8"}}[version]
    if prior not in allowed:
        expected = " or ".join(sorted(value.upper() for value in allowed))
        raise ValueError(f"complete immutable {expected} recovery report required")
    source_schema = {"v6": "scoped_recovery_v1", "v7": "selector_recovery_v1", "v8": "math_recovery_v1"}[prior]
    previous_hashes = {"report": file_sha256(report_path), "rows": file_sha256(previous_rows)}
    if (source.get("complete") is not True or source["manifest"].get("schema") != source_schema
            or source["manifest"].get("renderer") != f"template-{prior}" or previous_hashes["rows"] != source["row_records_sha256"]):
        raise ValueError(f"complete immutable {prior.upper()} recovery report required")
    baseline = Path(source["manifest"]["baseline_dir"]).resolve()
    baseline_report, baseline_rows = baseline / "report.json", baseline / "rows.jsonl"
    baseline_hashes = {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)}
    if baseline_hashes != source["manifest"]["baseline_sha256"]:
        raise ValueError(f"original baseline changed since {prior.upper()} audit")
    original = json.loads(baseline_report.read_text())
    verify_coverage(original, baseline_rows)
    cache = Path(source["manifest"]["cache_root"]).resolve()
    for root in (previous, baseline, cache, Path(original["manifest"]["prepared_root"]).resolve()):
        if output == root or root in output.parents:
            raise ValueError("output must be outside frozen source artifacts")
    report = {"manifest": {"schema": {"v7": "selector_recovery_v1", "v8": "math_recovery_v1", "v9": "program_recovery_v1"}[version], "renderer": args.renderer,
        "previous_recovery_dir": str(previous), "previous_sha256": previous_hashes,
        "baseline_dir": str(baseline), "baseline_sha256": baseline_hashes, "cache_root": str(cache),
        "renderer_sha256": renderer_fingerprints(), "runner_sha256": file_sha256(Path(__file__)),
        "semantic_cache_used": False, "native_execution_measured": False, "model_training_measured": False,
        "pilot_actions": args.pilot_actions}, "complete": False, "splits": {}}
    output.mkdir(parents=True, exist_ok=False)
    totals, families = defaultdict(lambda: _stats(prior, version)), defaultdict(lambda: _stats(prior, version))
    errors, family_errors, groups = defaultdict(Counter), defaultdict(Counter), defaultdict(list)

    def checkpoint():
        report["splits"] = {split: {"summary": {**dict(stats),
            "coverage_percent": 100 * stats[f"template_{version}_renderable_steps"] / stats["selected_steps"]},
            f"{version}_error_counts": dict(errors[split]),
            "by_family": {family: {"summary": dict(counts), "error_counts": dict(family_errors[(split, family)])}
                          for (s, family), counts in families.items() if s == split}} for split, stats in totals.items()}
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        temporary.replace(output / "report.json")

    checkpoint()
    print(f"Scanning frozen train/val caches with {args.renderer}; no Lean compilation", flush=True)
    digest, sentinel = hashlib.sha256(), object()
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        pairs = itertools.zip_longest(_rows(baseline_rows), _rows(previous_rows), fillvalue=sentinel)
        for processed, pair in enumerate(pairs, 1):
            if any(value is sentinel for value in pair):
                raise ValueError(f"{prior.upper()}/original row population mismatch")
            original_row, previous_row = pair[0][0], pair[1][0]
            identity = (original_row["split"], original_row["row_index"])
            if identity != (previous_row["split"], previous_row["row_index"]):
                raise ValueError(f"{prior.upper()} row identity mismatch")
            for key in ("alignment_kind", "dataset_target_matches_source"):
                if previous_row[key] != original_row[key]:
                    raise ValueError(f"{prior.upper()} row alignment changed from original snapshot")
            raw, trace = read_snapshot(original_row, cache)
            row = {key: previous_row[key] for key in ("split", "row_index", "family", "alignment_kind",
                "dataset_target_matches_source", "quarantined", f"{prior}_renderable")}
            if row["quarantined"] != (original_row["status"] in {"cache_invalid", "source_trace_invalid"}):
                raise ValueError("quarantine population changed")
            row.update({f"{version}_renderable": False, f"{version}_schema": None, "method_selector_count": 0, "math_notation_count": 0,
                        "source_program_used": False, "deferred_name_count": 0})
            if not row["quarantined"]:
                result = prepare_gold_rendering(raw["tactic"], trace, renderer=args.renderer)
                if result.get("semantic_reconstructions", 0):
                    raise ValueError(f"{version.upper()} syntax-only policy unexpectedly used semantic reconstruction")
                target = result["template_target"] or {}
                row.update({f"{version}_renderable": isinstance(result["rendered_tactic"], str), f"{version}_schema": target.get("schema"),
                    f"{version}_rendered_tactic": result["rendered_tactic"], f"{version}_error": result["render_error"],
                    f"{version}_error_code": result["template_error_code"], "method_selector_count": result.get("method_selector_count", 0),
                    "math_notation_count": result.get("math_notation_count", 0)})
                for key in ("source_program_used", "deferred_name_count", "scoped_name_count", "source_program_nodes", "syntax_choice_count",
                            "base_render_error", "base_template_error_code"):
                    if key in result:
                        row[key] = result[key]
                if row[f"{version}_error"]:
                    error = row[f"{version}_error_code"] or row[f"{version}_error"]
                    errors[row["split"]][error] += 1
                    family_errors[(row["split"], row["family"])][error] += 1
                if (row["split"] == "val" and row[f"{version}_renderable"] and not row[f"{prior}_renderable"]
                        and row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True):
                    features = (bool(row["method_selector_count"]), bool(row["math_notation_count"]),
                                bool(row["source_program_used"]), bool(row["deferred_name_count"]), tuple(sorted(_kinds(target))))
                    groups[(row["family"], features)].append((hashlib.sha256(str(row["row_index"]).encode()).hexdigest(), row["row_index"]))
            _add(totals[row["split"]], row, prior, version)
            _add(families[(row["split"], row["family"])], row, prior, version)
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode())
            if processed % 10000 == 0:
                handle.flush()
                checkpoint()
            if processed % 50000 == 0:
                print(f"checked {processed:,} cached rows", flush=True)
    if {"report": file_sha256(report_path), "rows": file_sha256(previous_rows)} != previous_hashes:
        raise ValueError(f"{prior.upper()} recovery changed during comparison")
    if {"report": file_sha256(baseline_report), "rows": file_sha256(baseline_rows)} != baseline_hashes:
        raise ValueError("original baseline changed during comparison")
    for split in ("train", "val"):
        expected = source["splits"][split]["summary"]
        if (totals[split]["selected_steps"] != expected["selected_steps"]
                or totals[split][f"baseline_{prior}_renderable_steps"] != expected[f"template_{prior}_renderable_steps"]
                or totals[split]["quarantined_steps"] != expected["quarantined_steps"]):
            raise ValueError(f"{version.upper()} comparison changed frozen {prior.upper()} population/support/quarantine")
    selected = select_native_pilot(groups, args.pilot_actions)
    (output / "new_val_pilot_rows.json").write_text(json.dumps(selected))
    report.update(complete=True, row_records_sha256=digest.hexdigest(), native_pilot={"selected_steps": len(selected),
        "split": "val", "policy": "round_robin_family_then_features_lowest_sha_v2", "candidate_groups": len(groups)})
    checkpoint()
    extra_header = "  program_gains  deferred_names" if version == "v9" else ""
    print("split  old_support  new_support  gained  lost  exact_gained  coverage" + extra_header, flush=True)
    for split, data in report["splits"].items():
        stats = data["summary"]
        extra = f"  {stats['source_program_rescues']:13}  {stats['deferred_identifier_rescues']:14}" if version == "v9" else ""
        print(f"{split:5}  {stats[f'baseline_{prior}_renderable_steps']:11}  {stats[f'template_{version}_renderable_steps']:11}"
              f"  {stats[f'{version}_rescues_vs_{prior}']:6}  {stats[f'{version}_losses_vs_{prior}']:4}  {stats[f'exact_aligned_{version}_rescues']:12}"
              f"  {stats['coverage_percent']:.2f}%" + extra, flush=True)
    print(f"Native pilot: {len(selected)} rows; report: {output / 'report.json'}", flush=True)
    return 2 if any(stats[f"{version}_losses_vs_{prior}"] for stats in totals.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
