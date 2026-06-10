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

    for strength, confidence in stv_pattern.findall(log_text):
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

    The default uniform [0, 1] is intentionally simple.
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


class DynamicThompsonSampler:
    """
    Per-subgoal Dynamic Thompson sampling state based on Gupta et al. / Kwon.

    Each subgoal maintains a Beta(alpha, beta) posterior. On first
    encounter, the prior is Beta(1.0, 1.0) (uniform). 
    
    To handle non-stationary environments, the parameter `C` bounds the 
    maximum sum of alpha + beta. If an update pushes the sum above C, 
    both parameters are scaled down, effectively discounting past observations.
    """

    def __init__(self, state: dict[str, dict[str, float]] | None = None, C: float = 100.0):
        self._state: dict[str, dict[str, float]] = state or {}
        self.C = C

    # -- public helpers ---------------------------------------------------

    @staticmethod
    def subgoal_key(item: dict) -> str:
        return f"{item.get('test_name', 'unknown')}_goal_{item.get('goal_index', 0)}"

    @staticmethod
    def load_from(path: str, C: float = 100.0) -> "DynamicThompsonSampler":
        """Load Thompson state from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return DynamicThompsonSampler(state=json.load(f), C=C)

    def save_to(self, path: str) -> None:
        """Persist Thompson state to a JSON file."""
        Path(path).write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- per-subgoal access ------------------------------------------------

    def _params(self, key: str) -> dict[str, float]:
        if key not in self._state:
            self._state[key] = {"alpha": 1.0, "beta": 1.0}
        return self._state[key]

    def alpha(self, key: str) -> float:
        return self._params(key)["alpha"]

    def beta(self, key: str) -> float:
        return self._params(key)["beta"]

    def sample(self, key: str, rng: random.Random) -> float:
        p = self._params(key)
        return rng.betavariate(p["alpha"], p["beta"])

    def record_success(self, key: str, reward: float = 1.0) -> None:
        p = self._params(key)
        p["alpha"] += reward
        self._apply_dts_discount(p)

    def record_failure(self, key: str, penalty: float = 1.0) -> None:
        p = self._params(key)
        p["beta"] += penalty
        self._apply_dts_discount(p)

    def record_observation(self, key: str, reward: float) -> None:
        """
        Update the posterior from a bounded proof score.

        A score near 1.0 behaves like success evidence; a score near 0.0
        behaves like failure evidence. Values outside [0, 1] are clamped so
        parser quirks cannot destabilize the sampler state.
        """
        reward = max(0.0, min(1.0, float(reward)))
        p = self._params(key)
        p["alpha"] += reward
        p["beta"] += 1.0 - reward
        self._apply_dts_discount(p)

    def _apply_dts_discount(self, p: dict[str, float]) -> None:
        """
        The core of Dynamic Thompson Sampling.
        If the total evidence exceeds C, scale it back down to C.
        """
        total = p["alpha"] + p["beta"]
        if total > self.C:
            scale_factor = self.C / total
            p["alpha"] *= scale_factor
            p["beta"] *= scale_factor

    # -- serialization -----------------------------------------------------

    @property
    def state_dict(self) -> dict[str, dict[str, float]]:
        return dict(self._state)

    def __repr__(self) -> str:
        return f"DynamicThompsonSampler({len(self._state)} subgoals, C={self.C})"


ThompsonSampler = DynamicThompsonSampler


def parse_and_rank_logs(
    manifest_path: str,
    *,
    ranking_output: str | None = None,
    random_fallback: bool = True,
    fallback_strategy: str = "random",
    fallback_distribution: str = "uniform",
    fallback_low: float = 0.0,
    fallback_high: float = 1.0,
    fallback_alpha: float = 2.0,
    fallback_beta: float = 2.0,
    random_seed: int | None = None,
    thompson_sampler: DynamicThompsonSampler | None = None,
    thompson_c: float = 100.0,
    thompson_state_output: str | None = None,
) -> list[dict[str, Any]]:
    """
    Read generated_manifest.json, parse every subgoal log, and rank subgoals.

    Normal case:
      score = max(strength * confidence) over all STVs found in that subgoal log.

    Fallback case (when no STV is found in *any* available log):
      - "random"  (default): assign a score sampled from the configured distribution.
      - "thompson":          assign a score sampled from each subgoal's
                             Beta(alpha, beta) posterior (Thompson sampling).

    Thompson-sampling state is carried via *thompson_sampler* (reused across calls)
    or loaded automatically from the ranking output's ``thompson_sampler_state``
    block when present.  State is saved to *thompson_state_output* if provided,
    otherwise embedded in the ranking output JSON.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    generated_items = manifest.get("generated_items", [])
    ranked: list[dict[str, Any]] = []

 
    # Always track state if a sampler is provided or requested
    ts: DynamicThompsonSampler | None = thompson_sampler
    if ts is None and (fallback_strategy == "thompson" or thompson_state_output):
        ts = DynamicThompsonSampler(C=thompson_c)
    elif ts is not None:
        ts.C = thompson_c

    for item in generated_items:
        log_path = item["log_path"]

        entry: dict[str, Any] = {
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

        if fallback_strategy == "thompson":
            entry["thompson_alpha"] = 1.0
            entry["thompson_beta"] = 1.0

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

            # --- dynamic Thompson update: learn from the observed score ---
            if ts is not None:
                key = ThompsonSampler.subgoal_key(entry)
                ts.record_observation(key, entry["score"])
                entry["thompson_alpha"] = ts.alpha(key)
                entry["thompson_beta"] = ts.beta(key)
        else:
            lowered = text.lower()
            if "error" in lowered or "failed" in lowered or "exception" in lowered:
                entry["status"] = "log_error_no_stv"
            else:
                entry["status"] = "no_stv_found"
                
            # CHANGE 2: Record the failure so the sampler learns
            if ts is not None:
                key = ThompsonSampler.subgoal_key(entry)
                ts.record_failure(key, penalty=1.0)
                entry["thompson_alpha"] = ts.alpha(key)
                entry["thompson_beta"] = ts.beta(key)

        ranked.append(entry)

    any_stv_found = any(item.get("truth_values") for item in ranked)
    any_log_available = any(item.get("status") != "missing_log" for item in ranked)

    # ------------------------------------------------------------------
    # Fallback: assign scores when no subgoal log contains an STV
    # ------------------------------------------------------------------
    if ranked and any_log_available and not any_stv_found:
        if fallback_strategy == "thompson":
            _apply_thompson_fallback(ranked, ts, random_seed)
        elif random_fallback:
            _apply_random_fallback(
                ranked,
                rng=random.Random(random_seed),
                distribution=fallback_distribution,
                low=fallback_low,
                high=fallback_high,
                alpha=fallback_alpha,
                beta=fallback_beta,
            )

    ranked.sort(key=lambda x: x["score"], reverse=True)

    # --- build result envelope -------------------------------------------
    if fallback_strategy == "thompson":
        if ts is None:
            ts = ThompsonSampler(C=thompson_c)
        for item in ranked:
            key = ThompsonSampler.subgoal_key(item)
            item["thompson_alpha"] = ts.alpha(key)
            item["thompson_beta"] = ts.beta(key)

    ranking_method: str
    if any_stv_found:
        ranking_method = "max_strength_times_confidence"
    elif fallback_strategy == "thompson":
        ranking_method = "thompson_sampling"
    elif random_fallback:
        ranking_method = "random_fallback_when_no_stv_global"
    else:
        ranking_method = "no_fallback_all_zero"

    result: dict[str, Any] = {
        "manifest_path": os.path.abspath(manifest_path),
        "ranking_method": ranking_method,
        "random_fallback": {
            "enabled": random_fallback,
            "strategy": fallback_strategy,
            "used": bool(ranked and any_log_available and not any_stv_found),
            "distribution": fallback_distribution,
            "low": fallback_low,
            "high": fallback_high,
            "alpha": fallback_alpha,
            "beta": fallback_beta,
            "seed": random_seed,
        },
        "ranked_subgoals": ranked,
    }

    if ts is not None:
        result["thompson_sampler_state"] = ts.state_dict
    if ranking_output:
        out_path = Path(ranking_output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if thompson_state_output and ts is not None:
        ts.save_to(thompson_state_output)

    return ranked


# ------------------------------------------------------------------
# Fallback helpers
# ------------------------------------------------------------------


def _apply_random_fallback(
    ranked: list[dict[str, Any]],
    rng: random.Random,
    distribution: str,
    low: float,
    high: float,
    alpha: float,
    beta: float,
) -> None:
    for item in ranked:
        if item.get("status") == "missing_log":
            continue

        fallback_score = sample_fallback_score(
            rng,
            distribution=distribution,
            low=low,
            high=high,
            alpha=alpha,
            beta=beta,
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
                "distribution": distribution,
            }
        ]


def _apply_thompson_fallback(
    ranked: list[dict[str, Any]],
    sampler: DynamicThompsonSampler | None,
    seed: int | None,
) -> None:
    if sampler is None:
        sampler = ThompsonSampler()

    rng = random.Random(seed)

    for item in ranked:
        if item.get("status") == "missing_log":
            continue

        key = ThompsonSampler.subgoal_key(item)
        thompson_score = sampler.sample(key, rng)

        item["score"] = thompson_score
        item["best_strength"] = thompson_score
        item["best_confidence"] = 1.0
        item["random_fallback_used"] = True
        item["random_fallback_score"] = thompson_score
        item["status"] = "thompson_fallback_no_stv_global"
        item["truth_values"] = [
            {
                "strength": thompson_score,
                "confidence": 1.0,
                "score": thompson_score,
                "source": "thompson_fallback",
            }
        ]


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
