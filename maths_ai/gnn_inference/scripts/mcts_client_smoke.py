"""CPU-only MCTS search check over the installed client; explicit paths are required."""

import argparse
import asyncio
import json
from pathlib import Path

from maths_ai.hybrid_reasoner.mcts_diagnostics import verify_live_mcts
from pantograph_async import PantographEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pantograph-repl", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--searches", type=int, default=20)
    parser.add_argument(
        "--report",
        type=Path,
        help="Write evidence to a new JSON file; never overwrite.",
    )
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error(f"Report already exists: {args.report}")
    env = PantographEnv(
        source_root=args.source_root,
        pantograph_repl=args.pantograph_repl,
        imports=("Mathlib",),
        build_manifest=args.build_manifest,
    )
    report = json.dumps(
        asyncio.run(verify_live_mcts(env, searches=args.searches)), indent=2
    ) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
