
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

from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import build_model_from_checkpoint
from maths_ai.gnn_inference.atp_lean_gnn.logger import TrainingLogger
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer, PremiseScorerConfig
from maths_ai.gnn_inference.atp_lean_gnn.premise_training import evaluate_model_with_premises, train_one_epoch_with_premises
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print
from maths_ai.gnn_inference.atp_lean_gnn.training import build_dataloaders, load_pointer_config, load_prepared_metadata
from maths_ai.gnn_inference.atp_lean_gnn.training_safety import resolve_amp_dtype


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

    # Reconstruct the exact pointer encoder that produced the lemma index.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, checkpoint_manifest, checkpoint_spec = build_model_from_checkpoint(
        ckpt,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        expected_model_kind="tactic_with_args",
    )
    lemma_index.validate_encoder_fingerprint(str(checkpoint_manifest["encoder_fingerprint"]))
    amp_dtype = resolve_amp_dtype(
        architecture=checkpoint_spec.architecture,
        device=device,
        requested=config.training.use_amp,
    )
    use_amp = amp_dtype is not None

    # Freeze the encoder so the query space remains compatible with the lemma index.
    for param in model.encoder.parameters():
        param.requires_grad = False

    model = model.to(device)

    # Build Premise Scorer
    scorer = PremiseScorer(hidden_dim=checkpoint_spec.hidden_dim, mode=p_config.scoring_mode)
    scorer = scorer.to(device)

    # Only train: tactic_embedding, argument_selector, and scorer
    trainable_params = (
        list(model.tactic_embedding.parameters())
        + list(model.argument_selector.parameters())
        + list(scorer.parameters())
    )
    frozen_count = sum(p.numel() for p in model.encoder.parameters())
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
    grad_scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_dtype == torch.float16,
    )

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
            amp_dtype=amp_dtype,
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
            amp_dtype=amp_dtype,
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
                "encoder_fingerprint": checkpoint_manifest["encoder_fingerprint"],
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
