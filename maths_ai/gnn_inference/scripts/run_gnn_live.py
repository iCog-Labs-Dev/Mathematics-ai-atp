"""Day-one GNN-only live-state smoke runner, not a proof-search benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.live_policy import LiveJointPolicy
from maths_ai.gnn_inference.atp_lean_gnn.live_state import LiveLeanSession, TacticFailure


async def run_live_smoke(policy, session, expression, *, max_steps=8, top_k=8, declaration=False):
    """Follow the first executable ranked candidate, preserving all siblings.

    Failed candidates are tried from the same parent. This intentionally has
    no frontier/backtracking: it verifies the policy/runtime boundary before
    adding best-first search. Closure alone is not a verified theorem result.
    """
    snapshot = await (session.start_declaration(expression) if declaration else session.start(expression))
    report = {"format": "gnn_live_smoke_v1", "generator": "gnn_joint_source_flat_v2",
              "expression": expression, "max_steps": max_steps, "top_k": top_k,
              "input_kind": "declaration_header" if declaration else "closed_expression",
              "retrieval_text_policy": "live_pretty_fields_v1", "steps": [],
              "proof_verified": False}
    for _ in range(max_steps):
        prediction = policy.predict(snapshot.goals[0], top_k=top_k)
        entry = {"parent_state": snapshot.state_id, "pending_before": len(snapshot.goals),
                 "graph_nodes": prediction.graph.dag.num_nodes,
                 "local_count": len(prediction.graph.local_context),
                 "retrieval_text": prediction.graph.text,
                 "candidates": [candidate.to_dict() for candidate in prediction.candidates],
                 "rejected": prediction.rejected, "trials": []}
        report["steps"].append(entry)
        for candidate in prediction.candidates:
            try:
                child = await session.step(snapshot, candidate.code)
            except TacticFailure as exc:
                entry["trials"].append({"code": candidate.code, "status": "lean_error", "error": str(exc)})
                continue
            entry["trials"].append({"code": candidate.code, "status": "executed",
                                    "child_state": child.state_id, "pending_after": len(child.goals)})
            entry["accepted"] = candidate.code
            snapshot = child
            break
        else:
            report["status"] = "no_executable_candidate"
            break
        if snapshot.closed:
            report["status"] = "closed_unverified"
            break
    else:
        report["status"] = "step_budget_exhausted"
    report["pending_goals"] = len(snapshot.goals)
    report["accepted_path"] = [entry["accepted"] for entry in report["steps"] if "accepted" in entry]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Pinned Lean/Mathlib Lake project")
    parser.add_argument("--pantograph-repl", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--goal", help="Closed proposition; existing locals must be quantified")
    target.add_argument("--header", help="Trusted theorem/example header without := or proof")
    parser.add_argument("--imports", nargs="+", default=["Init", "Mathlib"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", choices=["base", "accepted_local_cues"], default="accepted_local_cues")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tactic-timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.max_steps, args.top_k, args.tactic_timeout) < 1:
        parser.error("step/candidate/time budgets must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"output parent must already exist: {args.output.parent}")
    policy = LiveJointPolicy.from_bundle(args.bundle, device=args.device, profile=args.profile)

    async def run():
        async with LiveLeanSession(source_root=args.source_root, pantograph_repl=args.pantograph_repl,
                                   imports=args.imports, timeout=args.tactic_timeout) as session:
            return await run_live_smoke(policy, session, args.header if args.header is not None else args.goal,
                max_steps=args.max_steps, top_k=args.top_k, declaration=args.header is not None)

    report = asyncio.run(run())
    report["policy_state_sha256"] = policy.system.manifest["policy_state_sha256"]
    report["retriever_state_sha256"] = policy.system.manifest["retriever_state_sha256"]
    report["profile"] = args.profile
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(report['accepted_path'])} accepted steps, "
          f"{report['pending_goals']} pending goals; report={args.output}")
    return 0 if report["status"] == "closed_unverified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
