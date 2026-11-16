"""Submit mathematical statements to trained provers and replay their evidence.

Requires the matching persistent-atp v2 client. Proof bodies from the adjacent
Lean reference file are never sent to the model or used as search hints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

from mathproof.formal_v2 import FormalV2Error, FormalV2HTTPATP

EXAMPLES = Path(__file__).with_name("problems.json")
BACKENDS = {
    "mcts": ("MCTS", "mcts-source-aware", "mcts-actor-critic", "puct"),
    "gnn": ("Frozen GNN", "accepted-gnn-frozen", "accepted-gnn", "frozen-best-first"),
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def show(label, value):
    print(f"\n{label}\n{json.dumps(value, indent=2, ensure_ascii=False)}", flush=True)


def main(argv=None):
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:39743")
    parser.add_argument("--example", choices=tuple(examples), default="lagrange")
    parser.add_argument("--backend", choices=("both", *BACKENDS), default="both")
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--wall-clock-ms", type=int, default=60000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/formal-v2-examples"))
    args = parser.parse_args(argv)
    if args.max_steps < 1 or not 1 <= args.wall_clock_ms <= 120000:
        parser.error("max-steps must be positive; wall-clock-ms must be between 1 and 120000")
    selected = BACKENDS if args.backend == "both" else {args.backend: BACKENDS[args.backend]}
    client = FormalV2HTTPATP(args.url, timeout=120)
    print("Waiting for the selected trained backends...", flush=True)
    deadline = time.monotonic() + 300
    while True:
        try:
            capabilities = client.capabilities()
            if all(capabilities["deployments"].get(spec[1], {}).get("available") for spec in selected.values()):
                break
        except FormalV2Error as error:
            if error.status_code not in (None, 503):
                raise
            if "pinned v2 schemas" in str(error):
                raise
        if time.monotonic() >= deadline:
            raise RuntimeError("Selected deployments are not ready; inspect /ready and /formal-capabilities")
        time.sleep(2)

    example = examples[args.example]
    print(f"\n{example['title']}\n{example['description']}\n{example['model_search']}", flush=True)
    show("Input statement and ordered context", example["input"])
    output = args.output.resolve() / f"{args.example}-{uuid.uuid4().hex[:12]}"
    output.mkdir(parents=True)
    save(output / "example.json", example)
    verified = []
    for key, (label, deployment, backend, profile) in selected.items():
        config = capabilities["deployments"][deployment]
        proof_id = f"example-{key}-{uuid.uuid4().hex[:12]}"
        run_id = proof_id + "/fr-1"
        request = {
            "protocol_version": 2, "proof_id": proof_id, "claim_id": proof_id + "/c-1",
            "formal_declaration_id": proof_id + "/fd-1", "run_id": run_id,
            "base_revision": 0, "lease_id": "example-search-only", "fencing_token": 1,
            "environment_id": config["environment_id"], "environment_hash": config["environment_hash"],
            "deployment_id": deployment, "backend": {"name": backend, "profile": profile},
            "input": example["input"], "search_policy": {"deny_sorry": True},
            "budget": {"max_steps": args.max_steps, "wall_clock_ms": args.wall_clock_ms},
        }
        save(output / f"{key}-request.json", request)
        acknowledgement = client.formal_search_start(request)
        print(f"\n=== {label} ===\nRun: {run_id}\nAdmitted budget: {acknowledgement['budget']}", flush=True)
        until = time.monotonic() + args.wall_clock_ms / 1000 + 120
        while True:
            result = client.result(run_id)
            if result is not None:
                break
            if time.monotonic() >= until:
                raise RuntimeError(f"Observation timed out; reconcile run {run_id} before issuing more work")
            time.sleep(.2)
        save(output / f"{key}-result.json", result)
        save(output / f"{key}-binding.json", client.binding(run_id))
        show("Search outcome", {"disposition": result["disposition"], "termination": result["termination"],
                                 "budget_used": result["budget_used"]})
        if result["disposition"] != "proved-pending-replay":
            print("No certified proof within these bounds; this is not a refutation.", flush=True)
            continue
        certificate = json.loads(client.get_artifact(result["certificate"]["artifact_hash"]))
        save(output / f"{key}-certificate.json", certificate)
        show("Model-discovered proof trace", certificate.get("proof_trace"))
        print("Independent verification in a fresh Lean session...", flush=True)
        receipt = client.formal_replay(result["certificate"], result["environment_hash"],
                                       max_steps=128, wall_clock_ms=60000)
        save(output / f"{key}-replay.json", receipt)
        show("Replay receipt", receipt)
        if receipt["status"] == "verified" and receipt["verification_cost"].get("sorry_detected") is False:
            verified.append(key)
    summary = {"example": args.example, "selected_backends": list(selected), "verified_backends": verified,
               "claim_promotion": "not-requested", "purpose": "search-and-independent-replay"}
    save(output / "summary.json", summary)
    print(f"\nVerified {len(verified)}/{len(selected)} backend results.\nEvidence: {output}", flush=True)
    return 0 if len(verified) == len(selected) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FormalV2Error, RuntimeError) as error:
        print(f"Example failed: {error}", file=sys.stderr)
        raise SystemExit(1)
