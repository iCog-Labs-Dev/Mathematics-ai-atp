"""Verify a portable release and run a small live text-retrieval smoke."""

import argparse
import json
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import load_joint_source_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    system = load_joint_source_bundle(args.bundle_dir, device=args.device)
    ids, _, scores = system.retrieve(["n : Nat\n⊢ n = n"], k=min(5, len(system.lemma_index.lemma_ids)))
    print(json.dumps({"format": system.manifest["format"], "profile": system.profile,
        "policy_parameters": sum(parameter.numel() for parameter in system.policy.parameters()),
        "retriever_parameters": sum(parameter.numel() for parameter in system.retriever.parameters()),
        "indexed_lemmas": len(system.lemma_index.lemma_ids), "sample_ids": ids[0],
        "sample_scores": scores[0].tolist(), "offline_components_verified": True,
        "whole_proof_success_measured": False}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
