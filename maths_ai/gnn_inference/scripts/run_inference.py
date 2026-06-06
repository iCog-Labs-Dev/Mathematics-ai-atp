
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
from atp_lean_gnn.training import load_prepared_metadata, load_baseline_config, build_model
from atp_lean_gnn.premise_scoring import PremiseScorer
from atp_lean_gnn.lemma_corpus import load_lemma_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive Tactic Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json (e.g. from runs/baseline_gnn/run_*/config.json)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint")
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

    config = load_baseline_config(config_path)
    metadata = load_prepared_metadata(config.prepared_root)

    # Build and load model
    from atp_lean_gnn.argument_selector import TacticWithArgsClassifier
    
    model = TacticWithArgsClassifier(
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        use_node_type=config.use_node_type,
        max_args=getattr(config, "max_args", 3),
    )
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    
    # Adjust state dict keys if they come from a pure baseline (GraphSAGEStateClassifier)
    # The pure baseline keys don't have the "backbone." prefix.
    adjusted_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("backbone.") and not k.startswith("tactic_embedding.") and not k.startswith("argument_selector."):
            adjusted_state_dict[f"backbone.{k}"] = v
        else:
            adjusted_state_dict[k] = v
            
    model.load_state_dict(adjusted_state_dict, strict=False)
    model = model.to(device)

    # Build scorer (using randomly initialized weights for demo if not loaded)
    scorer = PremiseScorer(hidden_dim=config.model.hidden_dim, mode=args.scorer_mode).to(device)
    
    # Load index if provided
    lemma_index = None
    if args.index_path:
        index_path = Path(args.index_path)
        if index_path.exists():
            lemma_index = LemmaIndex.load(index_path)
            print(f"Loaded index with {len(lemma_index.lemma_ids)} lemmas.")
        else:
            print(f"WARNING: index path {index_path} not found.")
            
    if lemma_index is None:
        # Create an empty index as fallback
        import faiss
        import numpy as np
        d = config.model.hidden_dim
        lemma_index = LemmaIndex(
            index=faiss.IndexFlatL2(d),
            lemma_ids=[],
            lemma_vectors=np.empty((0, d), dtype=np.float32)
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
