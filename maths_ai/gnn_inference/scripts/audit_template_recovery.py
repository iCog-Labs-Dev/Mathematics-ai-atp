"""Paired V2/V3 static recovery versus frozen V1 coverage, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn import tactic_templates
from maths_ai.gnn_inference.atp_lean_gnn.application_templates import APPLICATION_TERM_SCHEMA
from maths_ai.gnn_inference.atp_lean_gnn.intro_templates import INTRO_TEMPLATE_SCHEMA, MAX_INTRO_BINDERS
from maths_ai.gnn_inference.atp_lean_gnn import recursive_templates
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.scripts.audit_unsupported_actions import _rows, read_snapshot, verify_coverage


def select_native_pilot(pilot_groups, limit):
    """Round-robin families first so numerous application shapes cannot starve rules."""
    by_family = defaultdict(list)
    for (family, _), values in sorted(pilot_groups.items()):
        if values:
            by_family[family].append(deque(sorted(values)))
    families = deque(deque(groups) for _, groups in sorted(by_family.items()))
    chosen = []
    while families and len(chosen) < limit:
        groups = families.popleft()
        group = groups.popleft()
        chosen.append(group.popleft()[1])
        if group:
            groups.append(group)
        if groups:
            families.append(groups)
    return sorted(chosen)


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
        "template_v3_renderable_steps", "intro_parsed_steps", "named_rule_parsed_steps",
        "v3_rescues_vs_v2", "v3_losses_vs_v2", "intro_rescues", "named_rule_rescues", "application_term_rescues", "other_v3_rescues",
        "application_term_parsed_steps",
        "rule_list_rescues", "long_rule_list_parsed_steps",
        "exact_aligned_v3_renderable_steps", "exact_aligned_v3_rescues_vs_v2", "exact_aligned_v3_losses_vs_v2",
        "overall_v3_rescues_vs_original", "overall_v3_losses_vs_original",
        "template_v4_renderable_steps", "v4_rescues_vs_v3", "v4_losses_vs_v3",
        "recursive_rule_rescues", "recursive_application_rescues", "other_v4_rescues",
        "exact_aligned_v4_renderable_steps", "exact_aligned_v4_rescues_vs_v3", "exact_aligned_v4_losses_vs_v3",
        "v4_constructor_rescues", "v4_hole_rescues", "v4_numeral_rescues", "v4_projection_rescues",
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
    if row.get("v3_evaluated"):
        v3 = row["v3_renderable"]
        rescue = v3 and not final
        intro = row.get("v3_schema") == "intro_binders_v1"
        named = bool(row.get("named_argument_count")) and row.get("v3_schema") == "rw_simp_template_v3"
        term = row.get("v3_schema") == "reference_application_v1"
        longer_list = row.get("v3_schema") == "rw_simp_template_v3" and row.get("v3_rule_count", 0) > tactic_templates.MAX_RULES
        stats.update({"template_v3_renderable_steps": v3, "intro_parsed_steps": intro,
                      "named_rule_parsed_steps": named, "v3_rescues_vs_v2": rescue,
                      "v3_losses_vs_v2": final and not v3, "intro_rescues": rescue and intro,
                      "named_rule_rescues": rescue and named, "application_term_rescues": rescue and term,
                      "application_term_parsed_steps": term,
                      "rule_list_rescues": rescue and longer_list and not named,
                      "long_rule_list_parsed_steps": longer_list,
                      "other_v3_rescues": rescue and not intro and not named and not term and not longer_list,
                      "exact_aligned_v3_renderable_steps": exact and v3,
                      "exact_aligned_v3_rescues_vs_v2": exact and rescue,
                      "exact_aligned_v3_losses_vs_v2": exact and final and not v3,
                      "overall_v3_rescues_vs_original": v3 and not before,
                      "overall_v3_losses_vs_original": before and not v3})
    if row.get("v4_evaluated"):
        v4, v3 = row["v4_renderable"], row["v3_renderable"]
        rescue = v4 and not v3
        rule = row.get("v4_schema") == recursive_templates.RECURSIVE_RULE_SCHEMA
        term = row.get("v4_schema") == recursive_templates.RECURSIVE_APPLICATION_SCHEMA
        features = row.get("v4_term_features", [])
        stats.update({"template_v4_renderable_steps": v4, "v4_rescues_vs_v3": rescue,
                      "v4_losses_vs_v3": v3 and not v4, "recursive_rule_rescues": rescue and rule,
                      "recursive_application_rescues": rescue and term, "other_v4_rescues": rescue and not rule and not term,
                      "exact_aligned_v4_renderable_steps": exact and v4,
                      "exact_aligned_v4_rescues_vs_v3": exact and rescue,
                      "exact_aligned_v4_losses_vs_v3": exact and v3 and not v4,
                      **{f"v4_{name}_rescues": rescue and name in features for name in ("constructor", "hole", "numeral", "projection")}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-coverage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--renderer", choices=("template-v2", "template-v3", "template-v4"), default="template-v2")
    parser.add_argument("--pilot-actions", type=int, default=24, help="Maximum new exact val actions in focused native row-ID file")
    args = parser.parse_args(argv)
    if args.pilot_actions < 1:
        raise ValueError("pilot action count must be positive")
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
    from maths_ai.gnn_inference.atp_lean_gnn.template_catalog import renderer_fingerprints

    paths = [Path(__file__), code_root / "scripts/audit_unsupported_actions.py"]
    report = {"manifest": {"schema": {"template-v2": "template_recovery_v1", "template-v3": "template_recovery_v2",
                                     "template-v4": "template_recovery_v3"}[args.renderer],
                           "renderer": args.renderer, "baseline_dir": str(baseline),
                           "baseline_sha256": hashes, "cache_root": str(cache),
                           "template_schema": {"template-v2": tactic_templates.APPLICATION_TEMPLATE_SCHEMA,
                                               "template-v3": tactic_templates.NAMED_APPLICATION_TEMPLATE_SCHEMA,
                                               "template-v4": recursive_templates.RECURSIVE_RULE_SCHEMA}[args.renderer],
                           "max_rule_arguments": (recursive_templates.MAX_TERM_CHILDREN if args.renderer == "template-v4"
                                                  else tactic_templates.MAX_RULE_ARGUMENTS),
                           "max_rules": tactic_templates.MAX_RULES if args.renderer == "template-v2" else tactic_templates.MAX_V3_RULES,
                           "intro_template_schema": INTRO_TEMPLATE_SCHEMA if args.renderer in {"template-v3", "template-v4"} else None,
                           "application_term_schema": APPLICATION_TERM_SCHEMA if args.renderer == "template-v3" else None,
                           "max_intro_binders": MAX_INTRO_BINDERS if args.renderer in {"template-v3", "template-v4"} else None,
                           "parameter_label_policy": "named_argument_position_v1" if args.renderer in {"template-v3", "template-v4"} else None,
                           "recursive_application_schema": recursive_templates.RECURSIVE_APPLICATION_SCHEMA if args.renderer == "template-v4" else None,
                           "reference_lookup": "validated_source_or_exact_global_identity_v1",
                           "code_sha256": {**renderer_fingerprints(), **{str(path.relative_to(code_root)): file_sha256(path) for path in paths}},
                           "recursive_limits": {"nodes": recursive_templates.MAX_TERM_NODES, "depth": recursive_templates.MAX_TERM_DEPTH,
                                                 "children": recursive_templates.MAX_TERM_CHILDREN,
                                                 "literal_chars": recursive_templates.MAX_LITERAL_CHARS} if args.renderer == "template-v4" else None,
                           "pilot_actions": args.pilot_actions,
                           "native_execution_measured": False, "proof_success_measured": False},
              "complete": False, "splits": {}}
    output.mkdir(parents=True, exist_ok=False)
    totals, families, previous, errors = defaultdict(_new_stats), defaultdict(_new_stats), defaultdict(_new_stats), defaultdict(Counter)
    v3_errors = defaultdict(Counter)
    v4_errors, pilot_groups = defaultdict(Counter), defaultdict(list)

    def checkpoint():
        report["splits"] = {split: {"summary": dict(stats),
            "by_family": {name: dict(value) for (s, name), value in families.items() if s == split},
            "by_previous_error": {name: dict(value) for (s, name), value in previous.items() if s == split},
            "v2_error_counts": dict(errors[split]), "v3_error_counts": dict(v3_errors[split]),
            "v4_error_counts": dict(v4_errors[split])} for split, stats in totals.items()}
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
            if args.renderer in {"template-v3", "template-v4"}:
                row.update(v3_evaluated=True, v3_renderable=False, v3_schema=None)
            if args.renderer == "template-v4":
                row.update(v4_evaluated=True, v4_renderable=False, v4_schema=None)
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
                if args.renderer in {"template-v3", "template-v4"}:
                    projected = tactic_templates.prepare_gold_rendering(raw["tactic"], trace, renderer="template-v3")
                    v3_target = projected["template_target"] or {}
                    v3_arguments = [argument for rule in v3_target.get("rules", []) for argument in rule.get("arguments", [])]
                    v3_arguments.extend(v3_target.get("arguments", []))
                    row.update(v3_renderable=isinstance(projected["rendered_tactic"], str),
                               v3_schema=v3_target.get("schema"), v3_rendered_tactic=projected["rendered_tactic"],
                               v3_rule_count=v3_target.get("rule_count", 0),
                               v3_error=projected["render_error"], v3_error_code=projected["template_error_code"],
                               named_argument_count=sum(argument.get("parameter") is not None for argument in v3_arguments))
                    if "binders" in v3_target:
                        row["intro_binder_count"] = v3_target["explicit_binder_count"]
                        row["intro_mode"] = v3_target["mode"]
                    if row["v3_error"]:
                        v3_errors[row["split"]][row["v3_error_code"] or row["v3_error"]] += 1
                if args.renderer == "template-v4":
                    projected = tactic_templates.prepare_gold_rendering(raw["tactic"], trace, renderer="template-v4")
                    target = projected["template_target"] or {}
                    pending = [target["term"]] if "term" in target else [rule["term"] for rule in target.get("rules", [])]
                    features = set()
                    while pending:
                        node = pending.pop()
                        features.add(node["kind"])
                        pending.extend(node.get("children", []))
                    row.update(v4_renderable=isinstance(projected["rendered_tactic"], str), v4_schema=target.get("schema"),
                               v4_rendered_tactic=projected["rendered_tactic"], v4_error=projected["render_error"],
                               v4_error_code=projected["template_error_code"], v4_term_features=sorted(features))
                    if row["v4_error"]:
                        v4_errors[row["split"]][row["v4_error_code"] or row["v4_error"]] += 1
                    if (row["split"] == "val" and row["v4_renderable"] and not row["v3_renderable"]
                            and row["alignment_kind"] == "exact" and row["dataset_target_matches_source"] is True):
                        key = (row["family"], tuple(sorted(features)))
                        rank = hashlib.sha256(str(row["row_index"]).encode()).hexdigest()
                        pilot_groups[key].append((rank, row["row_index"]))
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
    if args.renderer == "template-v4":
        chosen = select_native_pilot(pilot_groups, args.pilot_actions)
        (output / "new_val_pilot_rows.json").write_text(json.dumps(sorted(chosen)), encoding="utf-8")
        report["native_pilot"] = {"split": "val", "selected_steps": len(chosen), "candidate_groups": len(pilot_groups),
                                  "row_indices_file": "new_val_pilot_rows.json", "policy": "round_robin_family_then_features_lowest_sha_v2"}
    checkpoint()
    print(json.dumps({split: dict(stats) for split, stats in totals.items()}, indent=2), flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)
    return 2 if any(stats["combined_losses"] or stats["lookup_losses"] or stats["v2_grammar_losses"]
                    or stats["v3_losses_vs_v2"] or stats["overall_v3_losses_vs_original"]
                    or stats["v4_losses_vs_v3"]
                    for stats in totals.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
