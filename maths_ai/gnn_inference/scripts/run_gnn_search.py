"""Run GNN-only best-first search on one trusted Lean target."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn import gnn_search, live_policy, live_state, proof_verification
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pantograph-repl", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--goal", help="Closed proposition")
    target.add_argument("--header", help="Trusted theorem/lemma/example header without proof")
    parser.add_argument("--imports", nargs="+", default=["Init", "Mathlib"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", choices=["base", "accepted_local_cues"], default="accepted_local_cues")
    parser.add_argument("--model-calls", type=int, default=32)
    parser.add_argument("--tactic-trials", type=int, default=96)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--verification-seconds", type=float, default=30)
    parser.add_argument("--tactic-timeout", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        budget = gnn_search.GnnSearchBudget(args.model_calls, args.tactic_trials, args.nodes,
            args.depth, args.top_k, args.seconds, args.verification_seconds)
        header, _ = proof_verification.checked_header(
            args.header if args.header is not None else f"theorem GnnSearchChecked : {args.goal}")
    except ValueError as error:
        parser.error(str(error))
    if args.tactic_timeout < 1:
        parser.error("tactic timeout must be positive")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a new file inside an existing directory")
    print("Loading and validating accepted GNN/retriever bundle", flush=True)
    policy = live_policy.LiveJointPolicy.from_bundle(args.bundle, device=args.device, profile=args.profile)
    package = Path(gnn_search.__file__).parent
    sources = [Path(__file__), *sorted(package.rglob("*.py"))]
    manifest = {"policy_state_sha256": policy.system.manifest["policy_state_sha256"],
        "retriever_state_sha256": policy.system.manifest["retriever_state_sha256"],
        "bundle_sha256": file_sha256(policy.system.root / "bundle.json"),
        "profile": args.profile, "device": args.device, "imports": args.imports,
        "source_root": str(args.source_root.resolve()),
        "toolchain": (args.source_root / "lean-toolchain").read_text().strip(),
        "pantograph_repl_sha256": file_sha256(args.pantograph_repl),
        "implementation_sha256": {str(path.relative_to(package.parent)): file_sha256(path) for path in sources}}
    last_trials = -1

    def checkpoint(report):
        nonlocal last_trials
        report["manifest"] = manifest
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        trials = report["tactic_trials"]
        if trials != last_trials and trials % 10 == 0:
            print(f"calls={report['model_calls']} trials={trials} nodes={len(report['nodes'])}", flush=True)
            last_trials = trials

    async def run():
        async with live_state.LiveLeanSession(source_root=args.source_root, pantograph_repl=args.pantograph_repl,
                imports=args.imports, timeout=args.tactic_timeout) as session:
            verifier = proof_verification.LeanProofVerifier(source_root=args.source_root, imports=args.imports)
            return await gnn_search.search_gnn(policy, session, header, verifier, budget=budget, checkpoint=checkpoint)

    report = asyncio.run(run())
    print(f"{report['status']}: calls={report['model_calls']} trials={report['tactic_trials']} "
          f"nodes={len(report['nodes'])} seconds={report['seconds']:.2f}; report={args.output}", flush=True)
    return 0 if report["proof_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
