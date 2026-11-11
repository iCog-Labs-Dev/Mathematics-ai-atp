"""Cache frozen features, then optimize only the new typed-action decoder."""

import argparse
import json
from pathlib import Path
import random

import torch

from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import load_joint_source_bundle
from maths_ai.gnn_inference.atp_lean_gnn.structured_decoder import StructuredDecoder
from maths_ai.gnn_inference.atp_lean_gnn.structured_features import build_feature_cache
from maths_ai.gnn_inference.atp_lean_gnn.structured_training import StructuredTrainingConfig, train_decoder
from maths_ai.gnn_inference.atp_lean_gnn.structured_vocabulary import StructuredVocabulary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compiled-dir", "bundle-dir", "candidate-cache", "feature-cache", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-updates", type=int, default=0, help="Global update budget; zero runs all requested epochs")
    parser.add_argument("--retrieval-k", type=int)
    parser.add_argument("--feature-shard-size", type=int, default=128)
    parser.add_argument("--graph-batch-size", type=int, default=16)
    parser.add_argument("--max-graph-nodes", type=int, default=30000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_root = args.run_dir.resolve()
    for root in (args.compiled_dir, args.bundle_dir, args.candidate_cache, args.feature_cache):
        resolved = root.resolve()
        if run_root == resolved or resolved in run_root.parents:
            parser.error("run-dir must be outside frozen data, bundle and feature-cache roots")
    if run_root in args.feature_cache.resolve().parents:
        parser.error("feature-cache must be outside run-dir")
    config = StructuredTrainingConfig(args.epochs, args.batch_size, args.max_batch_tokens,
        args.learning_rate, args.weight_decay, args.grad_clip, args.seed, args.checkpoint_every, args.max_updates)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; select --device cpu for CPU smoke checks")
    print("Validating accepted bundle and matched retrieval cache", flush=True)
    system = load_joint_source_bundle(args.bundle_dir, device=device, profile="base")
    features = build_feature_cache(args.compiled_dir, args.candidate_cache, system, args.feature_cache,
        k=args.retrieval_k, shard_size=args.feature_shard_size, graph_batch_size=args.graph_batch_size,
        max_graph_nodes=args.max_graph_nodes, resume=args.resume)
    vocabulary = StructuredVocabulary.from_dict(json.loads((args.compiled_dir / "vocabulary.json").read_text()))
    # Seed after feature extraction: cached/noncached startup must initialize
    # the same new decoder, without changing either accepted encoder.
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model = StructuredDecoder(system.policy, vocabulary).to(device)
    print(f"Training {sum(parameter.numel() for parameter in model.parameters()):,} decoder parameters; "
          "GNN/retriever frozen; validation is teacher-forced", flush=True)
    report = train_decoder(model, features, args.run_dir, config, resume=args.resume)
    print(f"{report['status']}: updates={report['global_updates']} epochs={report['epoch']}; "
          f"checkpoint={args.run_dir / 'last.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
