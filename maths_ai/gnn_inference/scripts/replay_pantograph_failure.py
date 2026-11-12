"""Replay an archived Pantograph failure bundle without a model.

    python -m maths_ai.gnn_inference.scripts.replay_pantograph_failure \
        runs/<run>/failures/attempts/<id>/failure.json \
        --source-root /path/to/mathlib4 --repl /path/to/Pantograph/.lake/build/bin/repl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pantograph_async import PantographEnv

from maths_ai.hybrid_reasoner.failure_replay import replay_failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repl", type=Path, required=True)
    parser.add_argument("--imports", default="Mathlib", help="Comma-separated imports, as in the original run.")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="Replay even if the installed companion differs from the recorded one.")
    args = parser.parse_args(argv)
    bundle = json.loads(args.bundle.read_text())
    env = PantographEnv(source_root=args.source_root, pantograph_repl=args.repl,
                        imports=tuple(args.imports.split(",")))
    result = asyncio.run(replay_failure(bundle, env, allow_mismatch=args.allow_mismatch))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "reproduced" else 1


if __name__ == "__main__":
    raise SystemExit(main())
