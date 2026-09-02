"""CLI entry point for deterministic closed-loop proof search evaluation.

Usage:
    python -m maths_ai.gnn_inference.scripts.evaluate_closed_loop \\
        --output-dir evaluation_results \\
        --top-k 5 \\
        --max-depth 10 \\
        --max-nodes 100 \\
        --timeout 30 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from maths_ai.gnn_inference.atp_lean_gnn.closed_loop_evaluator import (
    ClosedLoopConfig,
    ClosedLoopEvaluator,
)


def build_default_demo_benchmark() -> list[dict]:
    """Default benchmark suite of representative Lean theorem goals."""
    return [
        {
            "id": "thm_rfl_id",
            "statement": "n : Nat\n⊢ n = n",
            "ground_truth_tactic": "rfl",
            "ground_truth_args": [],
        },
        {
            "id": "thm_exact_h",
            "statement": "h : True\n⊢ True",
            "ground_truth_tactic": "exact",
            "ground_truth_args": ["h"],
        },
        {
            "id": "thm_simp_true",
            "statement": "⊢ True ∧ True",
            "ground_truth_tactic": "simp",
            "ground_truth_args": [],
        },
        {
            "id": "thm_unknown_ident",
            "statement": "⊢ 1 + 1 = 2",
            "ground_truth_tactic": "decide",
            "ground_truth_args": [],
        },
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GNN predictions through Lean closed-loop proof search."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to JSON/JSONL file containing benchmark theorem goals.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory where summary.json, trace.jsonl, and report.md will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top tactic candidates per search expansion step.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum proof search depth.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=100,
        help="Maximum number of search nodes expanded per theorem.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Maximum time budget per theorem in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic evaluation.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume previous runs; overwrite existing traces.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = ClosedLoopConfig(
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        top_k_tactics=args.top_k,
        timeout_seconds=args.timeout,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )

    # Load benchmark suite
    if args.benchmark and Path(args.benchmark).exists():
        bench_path = Path(args.benchmark)
        if bench_path.suffix == ".jsonl":
            with open(bench_path, "r", encoding="utf-8") as f:
                benchmarks = [json.loads(line) for line in f if line.strip()]
        else:
            with open(bench_path, "r", encoding="utf-8") as f:
                benchmarks = json.load(f)
    else:
        benchmarks = build_default_demo_benchmark()

    print(f"Starting closed-loop evaluation on {len(benchmarks)} theorem(s)...")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print(f"Config: top_k={config.top_k_tactics}, max_depth={config.max_depth}, max_nodes={config.max_nodes}, seed={config.seed}\n")

    evaluator = ClosedLoopEvaluator(config)

    def on_progress(done: int, total: int, result):
        status = "[PROVED]" if result.proved else f"[{result.termination_reason}]"
        print(f"[{done}/{total}] {result.theorem_id}: {status} ({result.elapsed_seconds:.2f}s, {result.nodes_expanded} nodes)")

    summary = evaluator.evaluate_benchmark(benchmarks, progress_callback=on_progress)

    print("\n" + "=" * 60)
    print("CLOSED-LOOP EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Theorems Proved: {summary.get('proved_theorems', 0)} / {summary.get('total_theorems', 0)} ({summary.get('success_rate', 0.0):.1%})")
    print(f"Syntax Validity Rate: {summary.get('syntax_validity_rate', 0.0):.1%}")
    print(f"Lean Elaboration Success Rate: {summary.get('elaboration_success_rate', 0.0):.1%}")
    print(f"State Transition Success Rate: {summary.get('state_transition_success_rate', 0.0):.1%}")
    print(f"\nFull Report written to: {evaluator.report_file.resolve()}")
    print(f"Summary JSON written to: {evaluator.summary_file.resolve()}")
    print(f"Step-by-Step Trace: {evaluator.trace_file.resolve()}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
