"""
Runner script generation, log parsing, and subgoal ranking.

Handles the post-generation workflow:
  - write_runner_script(): creates a bash script to run .metta files
  - extract_stv_scores(): parses (STV strength confidence) from logs
  - parse_and_rank_logs(): reads manifest, scores and ranks subgoals
  - print_ranked_results(): console output of ranked subgoals
"""

from __future__ import annotations

import json
import os
import random
import re
import shlex
from pathlib import Path
from typing import Any


def safe_name(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
    return cleaned or "test"


def shell_quote(path_or_text: str) -> str:
    return shlex.quote(path_or_text)


def write_runner_script(
    generated_items: list[dict[str, Any]],
    output_dir: str,
    *,
    runner_name: str = "run_all_generated.sh",
    stop_on_error: bool = False,
) -> str:
    """
    Create a shell script that runs each generated .metta file and captures
    one .log file per subgoal.

    If stop_on_error=False, every command is allowed to fail without stopping
    the whole run. The log still captures stdout/stderr.
    """
    runner_path = os.path.join(output_dir, runner_name)

    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        "",
        "echo 'Running generated PeTTaChainer subgoal files...'",
        "",
    ]

    if stop_on_error:
        lines.insert(1, "set -e")

    for item in generated_items:
        metta_path = os.path.abspath(item["metta_path"])
        log_path = os.path.abspath(item["log_path"])
        test_name = item["test_name"]
        goal_index = item["goal_index"]

        lines.append(f"echo '============================================================'")
        lines.append(f"echo 'Running {test_name} goal {goal_index}'")
        lines.append(f"echo 'Metta: {metta_path}'")
        lines.append(f"echo 'Log:   {log_path}'")

        if stop_on_error:
            lines.append(f"petta {shell_quote(metta_path)} > {shell_quote(log_path)} 2>&1")
        else:
            lines.append(
                f"petta {shell_quote(metta_path)} > {shell_quote(log_path)} 2>&1 "
                f"|| echo 'Command failed for {test_name} goal {goal_index}; see log.'"
            )
        lines.append("")

    lines.append("echo 'Done.'")

    with open(runner_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    os.chmod(runner_path, 0o755)
    return runner_path


def extract_stv_scores(log_text: str) -> list[tuple[float, float]]:
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    stv_pattern = re.compile(rf"\(STV\s+({number})\s+({number})\)")

    proof_scores: list[tuple[float, float]] = []

    for line in log_text.splitlines():
        if "rule-proof" not in line:
            continue

        for strength, confidence in stv_pattern.findall(line):
            proof_scores.append((float(strength), float(confidence)))

    return proof_scores


def score_from_stv(strength: float, confidence: float) -> float:
    """
    Default ranking score. You can change this if PeTTaChainer returns a more
    specific proof-quality metric.
    """
    return strength * confidence


def sample_fallback_score(
    rng: random.Random,
    *,
    distribution: str = "uniform",
    low: float = 0.0,
    high: float = 1.0,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> float:
    """
    Sample a random fallback score when PeTTaChainer returns no STVs for any
    generated subgoal.

    Supported distributions:
      uniform: random value in [low, high]
      beta:    beta(alpha, beta), then scaled to [low, high]

    The default uniform [0, 1] is intentionally simple. For Thompson-sampling-like
    exploration, beta is often useful because alpha/beta can later be updated
    from success/failure evidence.
    """
    if high < low:
        raise ValueError("fallback high must be greater than or equal to fallback low")

    distribution = distribution.lower().strip()

    if distribution == "uniform":
        return rng.uniform(low, high)

    if distribution == "beta":
        raw = rng.betavariate(alpha, beta)
        return low + (high - low) * raw

    raise ValueError("fallback distribution must be either 'uniform' or 'beta'")


def parse_and_rank_logs(
    manifest_path: str,
    *,
    ranking_output: str | None = None,
    random_fallback: bool = True,
    fallback_distribution: str = "uniform",
    fallback_low: float = 0.0,
    fallback_high: float = 1.0,
    fallback_alpha: float = 2.0,
    fallback_beta: float = 2.0,
    random_seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Read generated_manifest.json, parse every subgoal log, and rank subgoals.

    Normal case:
      score = max(strength * confidence) over all STVs found in that subgoal log.

    Fallback case:
      if no STV is found in any available subgoal log, assign each available
      subgoal a random score sampled from the requested distribution. This is
      useful for exploration when PeTTaChainer cannot prove any current subgoal.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    generated_items = manifest.get("generated_items", [])
    ranked: list[dict[str, Any]] = []

    for item in generated_items:
        log_path = item["log_path"]

        entry = {
            **item,
            "status": "unknown",
            "truth_values": [],
            "best_strength": 0.0,
            "best_confidence": 0.0,
            "score": 0.0,
            "random_fallback_used": False,
            "random_fallback_score": None,
            "log_excerpt": "",
        }

        if not os.path.exists(log_path):
            entry["status"] = "missing_log"
            ranked.append(entry)
            continue

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        entry["log_excerpt"] = text[:2000]
        truth_values = extract_stv_scores(text)
        entry["truth_values"] = [
            {"strength": s, "confidence": c, "score": score_from_stv(s, c)}
            for s, c in truth_values
        ]

        if truth_values:
            best_strength, best_confidence = max(
                truth_values,
                key=lambda tv: score_from_stv(tv[0], tv[1]),
            )
            entry["best_strength"] = best_strength
            entry["best_confidence"] = best_confidence
            entry["score"] = score_from_stv(best_strength, best_confidence)
            entry["status"] = "ok"
        else:
            lowered = text.lower()
            if "error" in lowered or "failed" in lowered or "exception" in lowered:
                entry["status"] = "log_error_no_stv"
            else:
                entry["status"] = "no_stv_found"

        ranked.append(entry)

    any_stv_found = any(item.get("truth_values") for item in ranked)
    any_log_available = any(item.get("status") != "missing_log" for item in ranked)

    if random_fallback and ranked and any_log_available and not any_stv_found:
        rng = random.Random(random_seed)

        for item in ranked:
            if item.get("status") == "missing_log":
                continue

            fallback_score = sample_fallback_score(
                rng,
                distribution=fallback_distribution,
                low=fallback_low,
                high=fallback_high,
                alpha=fallback_alpha,
                beta=fallback_beta,
            )

            item["score"] = fallback_score
            item["best_strength"] = fallback_score
            item["best_confidence"] = 1.0
            item["random_fallback_used"] = True
            item["random_fallback_score"] = fallback_score
            item["status"] = "random_fallback_no_stv_global"
            item["truth_values"] = [
                {
                    "strength": fallback_score,
                    "confidence": 1.0,
                    "score": fallback_score,
                    "source": "random_fallback",
                    "distribution": fallback_distribution,
                }
            ]

    ranked.sort(key=lambda x: x["score"], reverse=True)

    result = {
        "manifest_path": os.path.abspath(manifest_path),
        "ranking_method": (
            "max_strength_times_confidence"
            if any_stv_found
            else "random_fallback_when_no_stv_global"
        ),
        "random_fallback": {
            "enabled": random_fallback,
            "used": bool(random_fallback and ranked and any_log_available and not any_stv_found),
            "distribution": fallback_distribution,
            "low": fallback_low,
            "high": fallback_high,
            "alpha": fallback_alpha,
            "beta": fallback_beta,
            "seed": random_seed,
        },
        "ranked_subgoals": ranked,
    }

    if ranking_output:
        out_path = Path(ranking_output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return ranked


def print_ranked_results(ranked: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("Subgoal ranking")
    print("=" * 80)

    if not ranked:
        print("No ranked entries.")
        return

    for rank, item in enumerate(ranked, start=1):
        print(
            f"{rank}. {item.get('test_name')} goal {item.get('goal_index')} "
            f"| score={item.get('score', 0.0):.6f} "
            f"| strength={item.get('best_strength', 0.0):.6f} "
            f"| confidence={item.get('best_confidence', 0.0):.6f} "
            f"| status={item.get('status')}"
        )
        print(f"   metta: {item.get('metta_path')}")
        print(f"   log:   {item.get('log_path')}")
