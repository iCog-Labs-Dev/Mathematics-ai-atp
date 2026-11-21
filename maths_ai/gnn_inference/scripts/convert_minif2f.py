"""Convert pinned miniF2F Lean headers to structured validation/test JSONL."""

import argparse
import asyncio
from pathlib import Path

from pantograph_async import PantographEnv
from maths_ai.gnn_inference.atp_lean_gnn.minif2f_conversion import (
    ConversionOptions, PINNED_REVISION, convert_minif2f,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pantograph-repl", type=Path, required=True)
    parser.add_argument("--imports", nargs="+", default=["Init", "Mathlib"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--max-problems-per-split", type=int)
    parser.add_argument("--helper-timeout-s", type=float, default=300)
    parser.add_argument("--time-budget-s", type=float, default=3600)
    parser.add_argument("--request-timeout-s", type=float, default=120)
    parser.add_argument("--start-timeout-s", type=float, default=300)
    parser.add_argument("--max-output-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args(argv)
    env = PantographEnv(source_root=args.source_root.resolve(),
        pantograph_repl=args.pantograph_repl.resolve(), imports=tuple(args.imports),
        timeout=args.request_timeout_s, server_start_timeout_s=args.start_timeout_s)
    env.verify()
    options = ConversionOptions(benchmark_root=args.benchmark_root.resolve(),
        output=args.output, env=env, revision=args.revision, splits=tuple(args.splits),
        max_problems_per_split=args.max_problems_per_split,
        helper_timeout_s=args.helper_timeout_s, time_budget_s=args.time_budget_s,
        max_output_bytes=args.max_output_bytes)
    print(asyncio.run(convert_minif2f(options)))


if __name__ == "__main__":
    main()
