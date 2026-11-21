"""CPU-only client and graph-contract check; explicit compiled project and REPL are required."""

import argparse
import asyncio
import json
from pathlib import Path

from maths_ai.hybrid_reasoner.pantograph_diagnostics import verify_live_application
from pantograph_async import PantographEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pantograph-repl", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="Write evidence to a new JSON file; never overwrite.",
    )
    parser.add_argument("--response-line-limit", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--stderr-tail-limit", type=int, default=64 * 1024)
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error(f"Report already exists: {args.report}")
    env = PantographEnv(
        source_root=args.source_root,
        pantograph_repl=args.pantograph_repl,
        imports=("Mathlib",),
        build_manifest=args.build_manifest,
        response_line_limit=args.response_line_limit,
        stderr_tail_limit=args.stderr_tail_limit,
    )
    report = json.dumps(asyncio.run(verify_live_application(env)), indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
