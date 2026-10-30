"""Explicitly migrate an RL config to source-aware artifacts and unit policy weights."""

import argparse
import json
from pathlib import Path
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import RetrievalConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "retrieval-config", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.input.read_text())
    config.pop("arg_loss_weight", None)
    retrieval = json.loads(args.retrieval_config.read_text())
    RetrievalConfig(**retrieval)
    config.update(
        warmstart_checkpoint=str(args.checkpoint),
        retrieval=retrieval,
        max_update_candidates=100000,
        max_update_candidate_bytes=268435456,
    )
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
