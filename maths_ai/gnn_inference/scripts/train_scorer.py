
"""Train the premise scoring head on top of a frozen or fine-tuned baseline model.

Usage::
    python scripts/train_scorer.py \\
        --config configs/pointer_graphsage_state.json \\
        --premise-config configs/premise_scoring.json \\
        --checkpoint runs/pointer_gnn/run_XXX/best.pt \\
        --index-path artifacts/lemmas/v1/index/lemma_index.faiss
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from atp_lean_gnn.lemma_index import LemmaIndex
from atp_lean_gnn.logger import TrainingLogger
from atp_lean_gnn.premise_scoring import PremiseScorer, PremiseScorerConfig
from atp_lean_gnn.premise_training import evaluate_model_with_premises, train_one_epoch_with_premises
from atp_lean_gnn.reporting import console_print
from atp_lean_gnn.training import build_dataloaders, load_pointer_config, load_prepared_metadata


def _create_run_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = run_root / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Premise Scorer")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline config")
    parser.add_argument("--premise-config", type=str, default="configs/premise_scoring.json", help="Path to premise scoring config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to baseline checkpoint (best.pt)")
    parser.add_argument("--index-path", type=str, required=True, help="Path to FAISS index built from the baseline")
    parser.add_argument("--run-root", type=str, default="runs/premise_gnn", help="Directory to save run logs and checkpoints")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    
    # Load configs
    config = load_pointer_config(Path(args.config))
    metadata = load_prepared_metadata(config.prepared_root)
    
    with open(args.premise_config, "r") as f:
        p_cfg_dict = json.load(f)
        p_config = PremiseScorerConfig(**p_cfg_dict)

    run_dir = _create_run_dir(Path(args.run_root))
    console_print(f"Saving run to {run_dir}")
    logger = TrainingLogger(run_dir)

    # Load Lemma Index
    console_print(f"Loading lemma index from {args.index_path}...")
    lemma_index = LemmaIndex.load(Path(args.index_path))

    # Build Dataloaders
    datasets, loaders = build_dataloaders(metadata, config)

    # Load baseline model and wrap it in TacticWithArgsClassifier
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
    adjusted_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("backbone.") and not k.startswith("tactic_embedding.") and not k.startswith("argument_selector."):
            adjusted_state_dict[f"backbone.{k}"] = v
        else:
            adjusted_state_dict[k] = v
            
    model.load_state_dict(adjusted_state_dict, strict=False)

    has_trained_tactic_embedding = any(
        k.startswith("tactic_embedding.") for k in adjusted_state_dict
    )
    if not has_trained_tactic_embedding:
        with torch.no_grad():
            model.tactic_embedding.weight.copy_(model.backbone.classifier.weight)

    # Freeze the GNN backbone (keeps embeddings compatible with FAISS index)
    for param in model.backbone.parameters():
        param.requires_grad = False

    model = model.to(device)

    # Build Premise Scorer
    scorer = PremiseScorer(hidden_dim=config.model.hidden_dim, mode=p_config.scoring_mode)
    scorer = scorer.to(device)

    # Only train: tactic_embedding, argument_selector, and scorer
    trainable_params = (
        list(model.tactic_embedding.parameters())
        + list(model.argument_selector.parameters())
        + list(scorer.parameters())
    )
    frozen_count = sum(p.numel() for p in model.backbone.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    console_print(
        f"Parameters — frozen backbone: {frozen_count:,}, "
        f"trainable (pointer + scorer + tactic_emb): {trainable_count:,}"
    )

    optimizer = AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_mrr = -1.0

    for epoch in range(1, config.training.epochs + 1):
        train_metrics = train_one_epoch_with_premises(
            model=model,
            scorer=scorer,
            loader=loaders["train"],
            lemma_index=lemma_index,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            device=device,
            grad_clip=config.training.grad_clip,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight if hasattr(config, "arg_loss_weight") else 0.5,
            premise_loss_weight=p_config.premise_loss_weight,
            k=p_config.k,
            epoch=epoch,
            total_epochs=config.training.epochs,
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
        )

        val_metrics = evaluate_model_with_premises(
            model=model,
            scorer=scorer,
            loader=loaders["val"],
            lemma_index=lemma_index,
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight if hasattr(config, "arg_loss_weight") else 0.5,
            premise_loss_weight=p_config.premise_loss_weight,
            k=p_config.k,
            split_name="val",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
        )

        console_print(
            f"Epoch {epoch} | Val MRR: {val_metrics['premise_mrr']:.4f} | "
            f"Hit@1: {val_metrics['premise_top1_accuracy']:.4f} | "
            f"Hit@5: {val_metrics['premise_top5_accuracy']:.4f} | "
            f"Recall: {val_metrics['premise_recall']:.4f}"
        )

        if val_metrics["premise_mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["premise_mrr"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "scorer_state_dict": scorer.state_dict(),
                "val_metrics": val_metrics,
            }, run_dir / "best.pt")

        logger.log_epoch(
            epoch,
            {
                "train_tactic_loss": float(train_metrics["tactic_loss"]),
                "train_arg_loss": float(train_metrics["arg_loss"]),
                "train_premise_loss": float(train_metrics["premise_loss"]),
                "train_combined_loss": float(train_metrics["combined_loss"]),
                "train_example_count": int(train_metrics["example_count"]),
                "val_tactic_loss": float(val_metrics["tactic_loss"]),
                "val_arg_loss": float(val_metrics["arg_loss"]),
                "val_premise_loss": float(val_metrics["premise_loss"]),
                "val_combined_loss": float(val_metrics["combined_loss"]),
                "val_premise_mrr": float(val_metrics["premise_mrr"]),
                "val_premise_top1_accuracy": float(val_metrics["premise_top1_accuracy"]),
                "val_premise_top5_accuracy": float(val_metrics["premise_top5_accuracy"]),
                "val_premise_recall": float(val_metrics["premise_recall"]),
                "val_known_label_count": int(val_metrics["known_label_count"]),
                "val_premise_target_present_count": int(val_metrics["premise_target_present_count"]),
                "val_premise_valid_count": int(val_metrics["premise_valid_count"]),
                "val_evaluated_count": int(val_metrics["evaluated_count"]),
                "best_val_mrr": float(best_val_mrr),
            },
        )

    console_print(f"Learning curves saved to {logger.jsonl_path} and {logger.csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())