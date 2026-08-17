#!/usr/bin/env python3
"""Benchmark a prover command against a miniF2F-style JSON or JSONL dataset.

The command is run once per theorem.  Use ``{goal}`` in ``--prover-command``
to receive the extracted Lean goal; otherwise the goal is appended as the last
argument.  A run is considered solved when its output contains ``Proof found!``
(customise this with ``--success-marker``).
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PROVER_COMMAND = (
    f"{shlex.quote(sys.executable)} -m maths_ai.hybrid_reasoner.joint_inference "
    "--goal_statement {goal}"
)
GOAL_FIELDS = ("formal_statement", "statement", "goal", "theorem")

# Canonical miniF2F JSONL (488 problems, valid+test splits combined), hosted on
# Hugging Face. Used to auto-download the dataset when it isn't present locally,
# so nobody has to manually source/copy the file to run the benchmark.
DEFAULT_DATASET_PATH = Path("data/minif2f.jsonl")
DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/hk/MiniF2F-Cleaned/resolve/main/minif2f_cleaned.jsonl"
)


def ensure_dataset(path: Path, url: str) -> Path:
    """Download the dataset to ``path`` if it isn't already present."""
    if path.exists():
        return path
    print(f"Dataset not found at {path}; downloading from {url} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except urllib.error.URLError as error:
        raise SystemExit(
            f"Could not download the dataset automatically ({error}). "
            f"Please download it manually and save it to {path}, "
            f"or pass an existing dataset file as the first argument."
        ) from error
    path.write_bytes(data)
    print(f"Saved dataset to {path} ({len(data) / 1024:.1f} KiB).")
    return path


def load_examples(dataset_path: Path) -> list[dict[str, Any]]:
    """Load a JSON array, a JSON object containing examples, or JSONL."""
    text = dataset_path.read_text(encoding="utf-8-sig")
    try:
        if dataset_path.suffix.lower() in {".jsonl", ".jsonlines", ".ndjson"}:
            raise json.JSONDecodeError("JSONL dataset", text, 0)
        parsed = json.loads(text)
    except json.JSONDecodeError:
        examples = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(parsed, list):
            examples = parsed
        elif isinstance(parsed, dict):
            examples = next(
                (value for key in ("examples", "data", "problems") if isinstance((value := parsed.get(key)), list)),
                None,
            )
            if examples is None:
                raise ValueError("JSON dataset must be a list or contain an examples/data/problems list")
        else:
            raise ValueError("Dataset must be a JSON list, object, or JSONL file")
    if not all(isinstance(example, dict) for example in examples):
        raise ValueError("Every dataset entry must be a JSON object")
    return examples


def extract_goal(example: dict[str, Any]) -> str:
    """Get a Lean target from common miniF2F fields or full declarations."""
    value = next((example[field] for field in GOAL_FIELDS if isinstance(example.get(field), str)), None)
    if value is None:
        raise ValueError(f"Missing a string goal field; expected one of {', '.join(GOAL_FIELDS)}")
    goal = value.strip()
    if goal.startswith(("theorem ", "example ")):
        depth = 0
        for index, character in enumerate(goal):
            if character in "([{":
                depth += 1
            elif character in ")]}":
                depth -= 1
            elif character == ":" and depth == 0:
                goal = goal[index + 1 :].strip()
                break
    if ":=" in goal:
        goal = goal.split(":=", 1)[0].strip()
    return goal


def example_id(example: dict[str, Any], position: int) -> str:
    for field in ("name", "id", "theorem_name", "uid"):
        if example.get(field) is not None:
            return str(example[field])
    return f"example_{position:05d}"


def run_prover(command_template: str, goal: str, timeout_seconds: float, success_marker: str) -> tuple[str, float, str]:
    command = command_template.format(goal=shlex.quote(goal))
    if "{goal}" not in command_template:
        command = f"{command} {shlex.quote(goal)}"
    started = time.perf_counter()
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    try:
        completed = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        output = _as_text(error.stdout) + _as_text(error.stderr)
        return "timeout", time.perf_counter() - started, output
    output = _as_text(completed.stdout) + _as_text(completed.stderr)
    if success_marker in output:
        return "solved", time.perf_counter() - started, output
    if completed.returncode != 0:
        return "error", time.perf_counter() - started, output
    return "unsolved", time.perf_counter() - started, output


def write_svg_chart(path: Path, counts: Counter[str], total: int) -> None:
    labels = ("solved", "unsolved", "timeout", "error")
    colours = {"solved": "#22c55e", "unsolved": "#64748b", "timeout": "#f59e0b", "error": "#ef4444"}
    bars = []
    for index, label in enumerate(labels):
        count = counts[label]
        height = 0 if not total else round(180 * count / total)
        x = 70 + index * 120
        bars.append(
            f'<rect x="{x}" y="{230 - height}" width="70" height="{height}" fill="{colours[label]}"/>'
            f'<text x="{x + 35}" y="255" text-anchor="middle">{label}</text>'
            f'<text x="{x + 35}" y="{220 - height}" text-anchor="middle">{count}</text>'
        )
    solved_pct = 0.0 if not total else 100 * counts["solved"] / total
    path.write_text(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"600\" height=\"290\" viewBox=\"0 0 600 290\">"
        "<style>text{font:14px sans-serif;fill:#1e293b}.title{font-size:20px;font-weight:bold}</style>"
        f'<text class="title" x="30" y="35">miniF2F benchmark — {solved_pct:.1f}% solved</text>'
        '<line x1="45" y1="230" x2="560" y2="230" stroke="#94a3b8"/>'
        + "".join(bars)
        + "</svg>",
        encoding="utf-8",
    )


def write_reports(output_dir: Path, results: Iterable[dict[str, Any]], command: str) -> dict[str, Any]:
    result_list = list(results)
    counts: Counter[str] = Counter(row["status"] for row in result_list)
    total = len(result_list)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prover_command": command,
        "total": total,
        "solved": counts["solved"],
        "unsolved": counts["unsolved"],
        "timeouts": counts["timeout"],
        "errors": counts["error"],
        "solve_rate_percent": 0.0 if not total else round(100 * counts["solved"] / total, 2),
        "average_runtime_seconds": 0.0 if not total else round(sum(row["runtime_seconds"] for row in result_list) / total, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("id", "status", "runtime_seconds", "goal", "output_file"))
        writer.writeheader()
        writer.writerows(result_list)
    (output_dir / "summary.md").write_text(
        "# miniF2F benchmark\n\n| Metric | Value |\n|---|---:|\n"
        + "\n".join(f"| {key.replace('_', ' ').title()} | {value} |" for key, value in summary.items() if key not in {"created_at", "prover_command"})
        + "\n\n![Result chart](results.svg)\n",
        encoding="utf-8",
    )
    write_svg_chart(output_dir / "results.svg", counts, total)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a miniF2F benchmark against the project prover.")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET_PATH,
        help=(
            "miniF2F JSON, JSONL, or JSON dataset file. Defaults to "
            f"{DEFAULT_DATASET_PATH}; if that file doesn't exist, it is "
            "downloaded automatically."
        ),
    )
    parser.add_argument(
        "--dataset-url",
        default=DEFAULT_DATASET_URL,
        help="URL to download the dataset from if it's missing locally.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Never auto-download the dataset; error out instead if it's missing.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--prover-command", default=DEFAULT_PROVER_COMMAND, help="Shell command; use {goal} for the Lean goal.")
    parser.add_argument("--success-marker", default="Proof found!", help="Output text that marks a solved theorem.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-theorem timeout in seconds.")
    parser.add_argument("--limit", type=int, default=None, help="Only benchmark the first N entries.")
    parser.add_argument("--split", default=None, help="Only benchmark entries whose split field matches this value (for example: test).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout <= 0 or args.limit is not None and args.limit < 1:
        raise SystemExit("--timeout must be positive and --limit must be at least 1")
    if args.no_download and not args.dataset.exists():
        raise SystemExit(f"Dataset file {args.dataset} does not exist and --no-download was set.")
    if not args.no_download:
        ensure_dataset(args.dataset, args.dataset_url)
    examples = load_examples(args.dataset)
    if args.split is not None:
        examples = [example for example in examples if str(example.get("split", "")).lower() == args.split.lower()]
        if not examples:
            raise SystemExit(f"No entries with split={args.split!r} were found in {args.dataset}")
    if args.limit is not None:
        examples = examples[:args.limit]
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    for position, example in enumerate(examples, start=1):
        identifier = example_id(example, position)
        try:
            goal = extract_goal(example)
            status, elapsed, output = run_prover(args.prover_command, goal, args.timeout, args.success_marker)
        except (ValueError, KeyError) as error:
            goal, status, elapsed, output = "", "error", 0.0, str(error)
        log_name = f"{position:05d}_{''.join(char if char.isalnum() or char in '-_' else '_' for char in identifier)}.log"
        (logs_dir / log_name).write_text(output, encoding="utf-8")
        results.append({"id": identifier, "status": status, "runtime_seconds": round(elapsed, 3), "goal": goal, "output_file": f"logs/{log_name}"})
        print(f"[{position}/{len(examples)}] {identifier}: {status} ({elapsed:.2f}s)")
    summary = write_reports(args.output_dir, results, args.prover_command)
    print(f"\nSolved {summary['solved']}/{summary['total']} ({summary['solve_rate_percent']:.2f}%). Reports: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
