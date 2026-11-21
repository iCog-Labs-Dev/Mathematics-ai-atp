"""Audit PLN translation coverage on Lean-admitted local JSONL proof states."""

from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import time

from pantograph_async import PantographEnv, PantographError
from maths_ai.lean_replay import seed_from_text, ReplayInputError
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client,
    materialize_seed,
)
from .compiler import GoalInput, compile_goal, TranslationError, TRANSLATION_VERSION


async def audit(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = PantographEnv(
        source_root=args.source_root,
        pantograph_repl=args.repl,
        imports=("Mathlib",),
        timeout=10,
    )
    summary = {
        "translation_version": TRANSLATION_VERSION,
        "input": str(args.input.resolve()),
        "max_scan_rows": args.max_scan_rows,
        "time_budget_s": args.time_budget,
        "rows": [],
        "counts": {},
    }
    counts = Counter()
    started = time.monotonic()
    async with await create_model_sexpr_client(env) as client:
        with args.input.open() as stream:
            for index, line in enumerate(stream):
                if (
                    index >= args.max_scan_rows
                    or time.monotonic() - started >= args.time_budget
                ):
                    break
                row = json.loads(line)
                record = {"row": index, "theorem": row.get("theorem_name")}
                try:
                    if "input" in row:
                        text = row["input"].removeprefix("[GOAL]").strip()
                        seed = seed_from_text(state=text)
                    else:
                        seed = seed_from_text(
                            goal=row["initial_goal"],
                            hypotheses=row["initial_hypotheses"],
                        )
                    async with client.state_scope() as scope:
                        _, goal, _ = await materialize_seed(scope, seed)
                        counts["lean_admitted"] += 1
                        try:
                            compiled = compile_goal(GoalInput.from_goal(goal))
                            record.update(status="supported", facts=len(compiled.facts))
                            counts["supported"] += 1
                        except TranslationError as exc:
                            record.update(
                                status="unsupported", diagnostic=exc.as_dict()
                            )
                            counts["unsupported:" + exc.reason] += 1
                except (ReplayInputError, ValueError) as exc:
                    counts["admission_rejected"] += 1
                    record.update(status="admission_rejected", detail=str(exc)[:500])
                except PantographError as exc:
                    counts["engine_error"] += 1
                    record.update(status="engine_error", detail=str(exc)[:500])
                    if not client.is_usable:
                        summary["rows"].append(record)
                        break
                summary["rows"].append(record)
    summary["counts"] = dict(counts)
    summary["scanned"] = len(summary["rows"])
    summary["elapsed_s"] = time.monotonic() - started
    (output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--repl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-scan-rows", type=int, default=100)
    p.add_argument("--time-budget", type=float, default=120)
    args = p.parse_args()
    if args.max_scan_rows <= 0 or args.time_budget <= 0:
        p.error("limits must be positive")
    asyncio.run(audit(args))


if __name__ == "__main__":
    main()
