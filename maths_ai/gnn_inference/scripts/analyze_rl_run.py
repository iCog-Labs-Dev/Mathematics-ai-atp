"""Summarize RL metrics and compare base/trained search evaluation reports.

Reads saved data only. No checkpoint, Lean process, or optimizer is executed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import EVAL_METRIC_VERSION, _searched
from maths_ai.hybrid_reasoner.joint_inference import SEARCH_EVIDENCE_VERSION


COUNTERS = (
    "attempted", "solved", "seeds_inadmissible", "searches_failed",
    "searches_resource_limited", "searches_hard_timed_out", "searches_cooperative_deadline",
    "onpolicy_optimizer_step", "htps_optimizer_steps", "parameter_updates_this_round",
    "raw_draws", "distinct_commands", "accepted_edges", "rejected_commands",
    "resource_limited_commands", "num_transitions", "num_failures",
    "imitation_samples_mined",
)
CURVES = (
    "total_loss", "actor_loss", "critic_loss", "entropy", "bc_loss", "mean_arg_logp",
    "mean_return", "bc_weight", "curriculum_size", "tactic_queue_len", "critic_queue_len",
    "tactic_imitation_loss", "critic_soft_loss", "htps_total_loss",
)


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def counter(row, key):
    value = number(row.get(key))
    if value is None or value < 0:
        raise ValueError(f"Missing, negative, or non-finite counter {key!r}.")
    return value


def outcome(row):
    attempted, solved = counter(row, "attempted"), counter(row, "solved")
    normalized = {"attempted": attempted}
    for key in ("seeds_inadmissible", "searches_failed", "searches_resource_limited"):
        normalized[key] = counter(row, key) if key in row else 0.0
    if sum(value for key, value in normalized.items() if key != "attempted") > attempted:
        raise ValueError("Excluded attempts exceed total attempts.")
    searched = _searched(normalized)
    if solved > searched:
        raise ValueError("Solved count exceeds searches with usable evidence.")
    return {
        "attempted": attempted, "searched": searched, "solved": solved,
        "proof_rate": solved / searched if searched else None,
        "coverage": searched / attempted if attempted else None,
        "solved_per_attempt": solved / attempted if attempted else None,
    }


def load_metrics(path):
    lines = path.read_text().splitlines(keepends=True)
    rows, notes = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                notes.append(f"Ignored incomplete final JSONL line {index + 1}; the run may still be writing.")
                continue
            raise ValueError(f"{path}:{index + 1}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index + 1}: expected a JSON object.")
        rows.append(row)
    return rows, notes


def evaluation(stats):
    if not isinstance(stats, dict):
        raise ValueError("Expected evaluation statistics.")
    contract = stats.get("evaluation")
    required = {"metric", "mode", "seed", "repeats", "seed_derivation", "pln_translation"}
    if (not isinstance(contract, dict) or not required <= contract.keys()
            or contract["metric"] != EVAL_METRIC_VERSION):
        raise ValueError("Unsupported or incomplete proof-rate evaluation contract.")
    if (contract["mode"] not in {"sample", "greedy"}
            or type(contract["seed"]) is not int or contract["seed"] < 0
            or type(contract["repeats"]) is not int or contract["repeats"] < 1
            or not contract["seed_derivation"]):
        raise ValueError("Invalid evaluation mode, seed, repeats, or seed derivation.")
    values = outcome(stats)
    if counter(stats, "searched") != values["searched"]:
        raise ValueError("Reported searched count disagrees with the evaluation denominator.")
    expected = values["proof_rate"] if values["proof_rate"] is not None else 0.0
    if number(stats.get("proof_rate")) is None or not math.isclose(stats["proof_rate"], expected, abs_tol=1e-10):
        raise ValueError("Reported proof rate disagrees with solved/searched counts.")
    return values


def total(rows, key):
    values = [number(row.get(key)) for row in rows]
    return sum(values) if values and all(value is not None for value in values) else None


def window_summary(rows):
    if not rows:
        return None
    outcomes = [outcome(row) for row in rows]
    attempted = sum(row["attempted"] for row in outcomes)
    searched = sum(row["searched"] for row in outcomes)
    solved = sum(row["solved"] for row in outcomes)
    means = {}
    for key in CURVES:
        values = [number(row.get(key)) for row in rows]
        finite = [value for value in values if value is not None]
        means[key] = mean(finite) if finite else None
    return {
        "first_round": rows[0]["round"], "last_round": rows[-1]["round"], "rounds": len(rows),
        "attempted": attempted, "searched": searched, "solved": solved,
        "proof_rate": solved / searched if searched else None,
        "means_of_reported_round_metrics": means,
    }


def summarize_training(rows, window):
    counters = {key: total(rows, key) for key in COUNTERS}
    outcomes = [outcome(row) for row in rows]
    attempted = sum(row["attempted"] for row in outcomes)
    searched = sum(row["searched"] for row in outcomes)
    solved = sum(row["solved"] for row in outcomes)
    wall = total(rows, "wall_clock_s")
    nonfinite = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite[key] = nonfinite.get(key, 0) + 1
    pln_keys = sorted({key for row in rows for key in row if key.startswith("pln_") and number(row[key]) is not None})
    return {
        "rounds": len(rows), "first_round": rows[0]["round"] if rows else None,
        "last_round": rows[-1]["round"] if rows else None,
        "attempted": attempted, "searched": searched, "solved": solved,
        "proof_rate": solved / searched if searched else None,
        "coverage": searched / attempted if attempted else None,
        "solved_per_attempt": solved / attempted if attempted else None,
        "reported_training_round_seconds": wall,
        "searched_per_reported_training_second": searched / wall if wall and wall > 0 else None,
        "counters": counters,
        # The driver creates PLN status keys only in rounds where they occur.
        "pln_counts": {key: total([{key: row.get(key, 0)} for row in rows], key) for key in pln_keys},
        "nonfinite_fields": nonfinite,
        "first_window": window_summary(rows[:window]),
        "last_window": window_summary(rows[-window:]),
    }


def load_eval_report(path):
    report = json.loads(path.read_text())
    if not isinstance(report, dict) or report.get("kind") != "rl-evaluation-report":
        raise ValueError(f"{path}: expected --eval-only --eval-report output, not supervised accuracy metrics.")
    report = dict(report)
    report["computed"] = evaluation(report.get("result"))
    report["path"] = str(path)
    return report


def comparison(base, trained):
    reasons = []
    left, right = base["result"], trained["result"]
    fields = {
        "held-out input identities": (left.get("eval_subset_sha256"), right.get("eval_subset_sha256")),
        "held-out input count": (left.get("eval_subset_size"), right.get("eval_subset_size")),
        "evaluation contract": (left.get("evaluation"), right.get("evaluation")),
        "search settings": (base.get("search_settings"), trained.get("search_settings")),
        "search budgets": (base.get("budgets"), trained.get("budgets")),
        "dataset settings": (base.get("dataset"), trained.get("dataset")),
        "worker count": (base.get("workers"), trained.get("workers")),
        "admission environment": (
            base["admission"].get("environment") if isinstance(base.get("admission"), dict) else None,
            trained["admission"].get("environment") if isinstance(trained.get("admission"), dict) else None,
        ),
    }
    for label, (one, two) in fields.items():
        if one is None or two is None or one == {} or two == {} or one == "" or two == "":
            reasons.append(f"Missing {label} provenance.")
        elif one != two:
            reasons.append(f"Different {label}.")
    repeats = []
    for report in (base, trained):
        stats = report["result"]
        items = stats.get("repeats")
        contract = stats["evaluation"]
        if (not isinstance(items, list) or not items or len(items) != contract["repeats"]
                or any(not isinstance(item, dict) or not item.get("proposal_seeds_sha256") for item in items)
                or [item.get("repeat") for item in items] != list(range(contract["repeats"]))):
            reasons.append(f"Missing or incomplete repeat proposal seeds in {report['path']}.")
            repeats.append(None)
        else:
            repeats.append([(item.get("repeat"), item["proposal_seeds_sha256"]) for item in items])
    if repeats[0] != repeats[1]:
        reasons.append("Different repeat proposal seeds.")
    settings = base.get("search_settings")
    use_pln = settings.get("use_pln") if isinstance(settings, dict) else None
    if use_pln and (left.get("pln_runtime") is None or right.get("pln_runtime") is None):
        reasons.append("Missing PLN runtime identity.")
    elif left.get("pln_runtime") != right.get("pln_runtime"):
        reasons.append("Different PLN runtime identities.")
    if base["computed"]["proof_rate"] is None or trained["computed"]["proof_rate"] is None:
        reasons.append("One evaluation has zero searches with usable evidence.")
    matched = not reasons
    difference = (trained["computed"]["proof_rate"] - base["computed"]["proof_rate"]) * 100 if matched else None
    return {
        "recorded_conditions_match": matched, "mismatches": reasons,
        "base": {"checkpoint": base.get("checkpoint"), **base["computed"]},
        "trained": {"checkpoint": trained.get("checkpoint"), **trained["computed"]},
        "proof_rate_difference_percentage_points": difference,
        "limitations": [
            "Conditions are checked from saved report fields, not reconstructed from checkpoint filenames.",
            "Current evaluation reports do not bind checkpoint bytes, device, retrieval artifacts, or every resource limit; keep these fixed externally.",
            "Aggregate reports cannot identify newly solved versus regressed individual states or provide paired statistical tests.",
            "Proof-rate change is descriptive; it is not a statistical-significance or causal claim.",
        ],
    }


def fmt(value, percent=False):
    if value is None:
        return "unavailable"
    return f"{100 * value:.2f}%" if percent else f"{value:.6g}" if isinstance(value, (int, float)) else str(value)


def markdown(report):
    training = report["training"]
    lines = ["# RL run metrics", "", f"Run: `{report['run_dir']}`", "", "| Training measure | Value |", "| --- | ---: |"]
    for key in ("rounds", "attempted", "searched", "solved", "proof_rate", "coverage", "solved_per_attempt", "reported_training_round_seconds"):
        lines.append(f"| {key} | {fmt(training[key], key in {'proof_rate', 'coverage', 'solved_per_attempt'})} |")
    for key, value in training["counters"].items():
        if key not in {"attempted", "solved"}:
            lines.append(f"| {key} | {fmt(value)} |")
    lines += ["", "Loss values below are means of finite per-round values, not sample-weighted losses.", "", "| Measure | First window | Last window |", "| --- | ---: | ---: |"]
    first, last = training["first_window"], training["last_window"]
    for key in CURVES:
        lines.append(f"| {key} | {fmt(first['means_of_reported_round_metrics'][key] if first else None)} | {fmt(last['means_of_reported_round_metrics'][key] if last else None)} |")
    lines += ["", "## PLN outcomes", ""]
    lines += [f"- {key}: {fmt(value)}" for key, value in training["pln_counts"].items()] or ["No numeric PLN counters were recorded."]
    lines += ["", "## Periodic held-out evaluation", "", "| Round | Solved / searched | Proof rate |", "| --- | ---: | ---: |"]
    for row in report["periodic_evaluations"]:
        lines.append(f"| {row['round']} | {fmt(row['solved'])} / {fmt(row['searched'])} | {fmt(row['proof_rate'], True)} |")
    compared = report.get("checkpoint_comparison")
    lines += ["", "## Baseline versus trained checkpoint", ""]
    if compared is None:
        lines.append("No paired evaluation reports supplied. The first training/evaluation round is not an untrained baseline.")
    else:
        lines += ["| Measure | Base | Trained |", "| --- | ---: | ---: |"]
        for key in ("attempted", "searched", "solved", "proof_rate", "coverage", "solved_per_attempt"):
            lines.append(f"| {key} | {fmt(compared['base'][key], key in {'proof_rate', 'coverage', 'solved_per_attempt'})} | {fmt(compared['trained'][key], key in {'proof_rate', 'coverage', 'solved_per_attempt'})} |")
        lines += ["", f"Recorded conditions match: {compared['recorded_conditions_match']}."]
        if compared["recorded_conditions_match"]:
            lines.append(f"Proof-rate difference: {compared['proof_rate_difference_percentage_points']:+.2f} percentage points.")
        else:
            lines.append("Proof-rate difference suppressed because comparison conditions are incomplete or differ.")
        lines += [f"- {reason}" for reason in compared["mismatches"]]
        lines += ["", *[f"- {note}" for note in compared["limitations"]]]
    lines += ["", "## Interpretation", "", *[f"- {note}" for note in report["notes"]]]
    return "\n".join(lines) + "\n"


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def write_csv(path, rows):
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean(rows))


def analyze(run_dir, output, *, window=20, baseline_eval=None, trained_eval=None):
    run_dir, output = Path(run_dir), Path(output)
    if window < 1:
        raise ValueError("Window must be positive.")
    if (baseline_eval is None) != (trained_eval is None):
        raise ValueError("Supply both --baseline-eval and --trained-eval.")
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("search_evidence") != SEARCH_EVIDENCE_VERSION:
        raise ValueError(f"Run must record current search evidence {SEARCH_EVIDENCE_VERSION!r}.")
    raw, notes = load_metrics(run_dir / "metrics.jsonl")
    training, periodic, seen = [], [], set()
    for row in raw:
        if "round" not in row:
            notes.append("Ignored a row without a round number.")
            continue
        index = row["round"]
        if type(index) is not int or index < 0:
            raise ValueError("Round numbers must be nonnegative integers.")
        kind = "eval" if "eval" in row else "training"
        if (kind, index) in seen:
            raise ValueError(f"Duplicate {kind} round {index}; inspect resumed/concatenated metrics before aggregation.")
        seen.add((kind, index))
        if kind == "eval":
            stats = row["eval"]
            values = evaluation(stats)
            periodic.append({
                **{key: value for key, value in stats.items() if not isinstance(value, (dict, list))},
                **values, "round": index, "eval_mode": stats["evaluation"]["mode"],
                "eval_seed": stats["evaluation"]["seed"], "eval_metric": stats["evaluation"]["metric"],
            })
        else:
            if row.get("search_evidence") != SEARCH_EVIDENCE_VERSION:
                raise ValueError(f"Training round {index} uses different search evidence.")
            values = outcome(row)
            training.append({**row, "training_searched": values["searched"], "training_proof_rate": values["proof_rate"]})
    training.sort(key=lambda row: row["round"])
    periodic.sort(key=lambda row: row["round"])
    notes += [
        "Training solve rates use a changing curriculum and changing weights; compare checkpoints with held-out evaluations.",
        "Actor/critic/BC losses are training objectives, not supervised accuracy or proof-rate measurements.",
        "Use explicit optimizer-step counters; a zero actor loss does not establish that no update occurred.",
        "Reported training-round time excludes periodic evaluation and does not measure total job duration.",
        "PLN fallback counts are not successful rule-derived STVs; missing counters are not evidence of zero calls.",
        "Action/observation counts need not equal distinct Lean requests because proposals can repeat or be unrenderable.",
        "Missing metrics are unavailable; non-finite values are counted and excluded from finite window means.",
    ]
    report = {
        "kind": "rl-run-analysis-v1", "run_dir": str(run_dir), "config": config,
        "window": window, "training": summarize_training(training, window),
        "periodic_evaluations": periodic, "notes": notes,
        "checkpoint_comparison": comparison(load_eval_report(Path(baseline_eval)), load_eval_report(Path(trained_eval))) if baseline_eval is not None else None,
    }
    if len(training) < 2 * window:
        notes.append("The first and last summary windows overlap or coincide because the run is short.")
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(markdown(report))
    write_csv(output / "training_rounds.csv", training)
    write_csv(output / "evaluations.csv", periodic)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Timestamped RL run containing config.json and metrics.jsonl.")
    parser.add_argument("--output", type=Path, required=True, help="New analysis directory; existing output is refused.")
    parser.add_argument("--window", type=int, default=20, help="First/last training-window size in rounds.")
    parser.add_argument("--baseline-eval", type=Path, help="Base model --eval-report JSON.")
    parser.add_argument("--trained-eval", type=Path, help="Trained model --eval-report JSON.")
    args = parser.parse_args(argv)
    try:
        result = analyze(args.run_dir, args.output, window=args.window, baseline_eval=args.baseline_eval, trained_eval=args.trained_eval)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Report: {args.output / 'report.md'}")
    print(f"Training proof rate: {fmt(result['training']['proof_rate'], True)}")
    compared = result["checkpoint_comparison"]
    if compared:
        if compared["recorded_conditions_match"]:
            print(f"Trained minus base: {compared['proof_rate_difference_percentage_points']:+.2f} percentage points (recorded conditions match).")
        else:
            print("Checkpoint difference suppressed: " + " ".join(compared["mismatches"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
