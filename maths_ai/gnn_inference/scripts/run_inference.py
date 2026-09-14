
"""Run tactic inference on a single proof state interactively."""

import argparse
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from atp_lean_gnn.cli import DEMO_STATE
from atp_lean_gnn.inference import InferencePipeline
from atp_lean_gnn.lemma_index import LemmaIndex
from atp_lean_gnn.training import load_pointer_config, load_prepared_metadata
from atp_lean_gnn.checkpointing import build_model_from_checkpoint
from atp_lean_gnn.premise_scoring import PremiseScorer, load_scorer_checkpoint
from atp_lean_gnn.lemma_corpus import load_lemma_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive Tactic Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json (e.g. from runs/baseline_gnn/run_*/config.json)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint (backbone + tactic model)")
    parser.add_argument("--scorer-checkpoint", type=str, default=None, help="Path to a premise scorer checkpoint (best.pt from a premise_selection run). If omitted, the scorer uses random weights.")
    parser.add_argument("--scorer-mode", type=str, default="dot", choices=["dot", "mlp"], help="Scorer mode")
    parser.add_argument("--index-path", type=str, help="Path to FAISS index. If missing, retrieval will return nothing.")
    parser.add_argument("--corpus-path", type=str, help="Path to lemmas.jsonl for decoding retrieved lemma IDs to names.")
    parser.add_argument("--k", type=int, default=500, help="Number of lemmas to retrieve")
    parser.add_argument("--state", type=str, default=DEMO_STATE, help="Raw Lean proof state string")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading on {device}...")

    # Load baseline config and metadata
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        return 1

    config = load_pointer_config(config_path)
    metadata = load_prepared_metadata(config.prepared_root)

    # Build and load the exact pointer model recorded by the checkpoint manifest.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, checkpoint_manifest, checkpoint_spec = build_model_from_checkpoint(
        ckpt,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        expected_model_kind="tactic_with_args",
    )
    model = model.to(device)

    # Build scorer and load trained weights
    scorer = PremiseScorer(hidden_dim=checkpoint_spec.hidden_dim, mode=args.scorer_mode).to(device)
    expected_fingerprint = str(checkpoint_manifest["encoder_fingerprint"])
    if args.scorer_checkpoint:
        scorer_ckpt_path = Path(args.scorer_checkpoint)
        if scorer_ckpt_path.exists():
            scorer_ckpt = torch.load(scorer_ckpt_path, map_location=device, weights_only=False)
            load_scorer_checkpoint(
                scorer,
                scorer_ckpt,
                expected_encoder_fingerprint=expected_fingerprint,
            )
            print(f"Loaded trained scorer weights from {scorer_ckpt_path}")
        else:
            print(f"WARNING: scorer checkpoint not found at {scorer_ckpt_path} — using random weights.")
    else:
        print("WARNING: No --scorer-checkpoint provided. PremiseScorer is using RANDOM weights.")
    
    # Load index if provided
    lemma_index = None
    if args.index_path:
        index_path = Path(args.index_path)
        if index_path.exists():
            lemma_index = LemmaIndex.load(index_path)
            lemma_index.validate_encoder_fingerprint(expected_fingerprint)
            print(f"Loaded index with {len(lemma_index.lemma_ids)} lemmas.")
        else:
            print(f"WARNING: index path {index_path} not found.")
            
    if lemma_index is None:
        # Create an empty index as fallback
        import faiss
        import numpy as np
        d = checkpoint_spec.hidden_dim
        lemma_index = LemmaIndex(
            index=faiss.IndexFlatL2(d),
            lemma_ids=[],
            lemma_vectors=np.empty((0, d), dtype=np.float32),
            encoder_fingerprint=expected_fingerprint,
        )

    lemma_corpus = None
    if args.corpus_path:
        corpus_path = Path(args.corpus_path)
        if corpus_path.exists():
            records = load_lemma_corpus(corpus_path)
            lemma_corpus = {record.lemma_id: record for record in records}
            print(f"Loaded corpus with {len(lemma_corpus)} lemmas.")
        else:
            print(f"WARNING: corpus path {corpus_path} not found.")

    # Initialize Pipeline
    pipeline = InferencePipeline(
        model=model,
        scorer=scorer,
        lemma_index=lemma_index,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        device=device,
        k=args.k,
        lemma_corpus=lemma_corpus,
    )

    print("\n--- Input State ---")
    print(args.state)
    print("-------------------\n")

    prediction = pipeline.predict_tactic(args.state)
    
    print(f"Predicted Tactic:  \033[1;32m{prediction}\033[0m")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
