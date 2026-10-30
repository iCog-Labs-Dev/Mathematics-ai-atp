"""Run the shared source-aware policy against a materialized live Lean goal."""

import argparse
import asyncio
import json
from pathlib import Path
import torch
from pantograph_async import PantographEnv
from maths_ai.lean_replay import seed_from_text
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import (
    build_model_from_checkpoint,
)
from maths_ai.gnn_inference.atp_lean_gnn.training import load_prepared_metadata
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import (
    LiveLemmaProvider,
    RetrievalConfig,
)
from maths_ai.gnn_inference.atp_lean_gnn.inference import InferencePipeline
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client,
    materialize_seed,
)


async def run(args):
    metadata = load_prepared_metadata(args.prepared_root, splits=())
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    model, _, spec = build_model_from_checkpoint(
        checkpoint,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        graph_representation=metadata.graph_representation,
    )
    if not hasattr(model, "decoder"):
        raise ValueError("Inference requires a source-aware action checkpoint.")
    provider = LiveLemmaProvider(
        RetrievalConfig(**json.loads(args.retrieval_config.read_text()))
    )
    try:
        if spec.retrieval_binding != provider.library.fingerprint:
            raise ValueError("Model and retrieval artifacts differ.")
        pipeline = InferencePipeline(
            model.to(args.device),
            provider,
            metadata.node_vocab,
            metadata.tactic_vocab,
            torch.device(args.device),
        )
        env = PantographEnv(
            source_root=args.source_root,
            pantograph_repl=args.pantograph_repl,
            imports=("Init", "Mathlib"),
        )
        provider.verify_environment(env)
        async with await create_model_sexpr_client(env) as client:
            async with client.state_scope() as scope:
                _, goal, _ = await materialize_seed(
                    scope, seed_from_text(goal=args.state)
                )
                await pipeline.prepare_goal(goal)
                print(
                    pipeline.predict_from_goal(goal, top_k=args.top_k).predicted_tactic
                )
    finally:
        await provider.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "prepared-root",
        "checkpoint",
        "retrieval-config",
        "source-root",
        "pantograph-repl",
    ):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=3)
    asyncio.run(run(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
