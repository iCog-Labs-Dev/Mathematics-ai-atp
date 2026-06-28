#!/usr/bin/env python3
"""Unified training pipeline for GNN-based tactic prediction.

Runs the full training loop: dataset preparation → baseline model → pointer model
→ premise scorer, with configurable options at every stage.

Usage:
    # Run full pipeline with default config
    python -m maths_ai.gnn_inference.scripts.run_training

    # Run with a custom config
    python -m maths_ai.gnn_inference.scripts.run_training --config my_config.json

    # Run only specific stages
    python -m maths_ai.gnn_inference.scripts.run_training --stages prepare,baseline

    # Resume from a checkpoint
    python -m maths_ai.gnn_inference.scripts.run_training --resume

    # Override individual parameters from CLI
    python -m maths_ai.gnn_inference.scripts.run_training \
        --model.hidden_dim 256 \
        --training.epochs 10 \
        --training.batch_size 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print


# ──────────────────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "default",
    "stages": ["prepare", "baseline", "pointer", "scorer"],
    "prepared_root": "maths_ai/gnn_inference/artifacts/prepared/v1",
    "run_root": "maths_ai/gnn_inference/runs",

    "prepare": {
        "dataset_name": "iCog-Labs-Dev/MathsATP_lean_4_wiki",
        "splits": "train,val,test",
        "sample_per_split": None,
        "force": False,
    },

    "baseline": {
        "hidden_dim": 512,
        "num_layers": 4,
        "dropout": 0.2,
        "batch_size": 256,
        "epochs": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "num_workers": 12,
        "use_amp": True,
    },

    "pointer": {
        "hidden_dim": 512,
        "num_layers": 4,
        "dropout": 0.2,
        "max_args": 3,
        "arg_loss_weight": 0.5,
        "batch_size": 256,
        "epochs": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "num_workers": 8,
        "use_amp": True,
    },

    "scorer": {
        "premise_config": "maths_ai/gnn_inference/configs/premise_scoring.json",
        "hidden_dim": 512,
        "num_layers": 4,
        "dropout": 0.2,
        "batch_size": 256,
        "epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "num_workers": 8,
        "use_amp": True,
        "k": 5,
        "premise_loss_weight": 1.0,
        "scoring_mode": "dot",
    },

    "seed": 42,
    "device": "auto",
}


# ──────────────────────────────────────────────────────────────────────────
# Pipeline state tracking
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """Tracks which stages have completed and where their outputs are."""
    completed_stages: list[str] = field(default_factory=list)
    stage_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    start_time: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["elapsed_seconds"] = time.time() - self.start_time if self.start_time else 0
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def mark_complete(self, stage: str, output: dict[str, Any]) -> None:
        self.completed_stages.append(stage)
        self.stage_outputs[stage] = output

    def is_complete(self, stage: str) -> bool:
        return stage in self.completed_stages

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            completed_stages=data.get("completed_stages", []),
            stage_outputs=data.get("stage_outputs", {}),
            start_time=data.get("start_time", 0.0),
        )


# ──────────────────────────────────────────────────────────────────────────
# Stage runners
# ──────────────────────────────────────────────────────────────────────────

def run_prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Stage 1: Prepare the dataset (build vocabularies, convert to PyG)."""
    from maths_ai.gnn_inference.atp_lean_gnn.preprocess import (
        PreprocessConfig,
        run_preprocessing,
    )

    prepare_cfg = config["prepare"]
    output_root = Path(config["prepared_root"])

    console_print("\n" + "=" * 60)
    console_print("  STAGE 1: PREPARE DATASET")
    console_print("=" * 60)
    console_print(f"  Dataset   : {prepare_cfg['dataset_name']}")
    console_print(f"  Output    : {output_root}")
    console_print(f"  Splits    : {prepare_cfg['splits']}")
    console_print(f"  Sample    : {prepare_cfg['sample_per_split'] or 'all'}")
    console_print(f"  Force     : {prepare_cfg['force']}")
    console_print("")

    cfg = PreprocessConfig(
        dataset_name=prepare_cfg["dataset_name"],
        splits=tuple(s.strip() for s in prepare_cfg["splits"].split(",")),
        output_root=output_root,
        sample_per_split=prepare_cfg["sample_per_split"],
        force=prepare_cfg["force"],
    )

    summary = run_preprocessing(cfg)
    return {"summary": summary, "output_root": str(output_root)}


def run_baseline(config: dict[str, Any], resume_run_dir: str | None = None) -> dict[str, Any]:
    """Stage 2: Train baseline GNN classifier."""
    from maths_ai.gnn_inference.atp_lean_gnn.training import (
        BaselineConfig,
        load_baseline_config,
        train_baseline,
        _create_run_dir,
        _write_json,
    )

    baseline_cfg = config["baseline"]
    prepared_root = Path(config["prepared_root"])
    run_root = Path(config["run_root"]) / "baseline_gnn"

    console_print("\n" + "=" * 60)
    console_print("  STAGE 2: TRAIN BASELINE MODEL")
    console_print("=" * 60)
    console_print(f"  Prepared  : {prepared_root}")
    console_print(f"  Run root  : {run_root}")
    console_print(f"  hidden    : {baseline_cfg['hidden_dim']}")
    console_print(f"  layers    : {baseline_cfg['num_layers']}")
    console_print(f"  dropout   : {baseline_cfg['dropout']}")
    console_print(f"  batch     : {baseline_cfg['batch_size']}")
    console_print(f"  epochs    : {baseline_cfg['epochs']}")
    console_print(f"  lr        : {baseline_cfg['learning_rate']}")
    console_print(f"  workers   : {baseline_cfg['num_workers']}")
    console_print("")

    cfg = BaselineConfig(
        prepared_root=prepared_root,
        run_root=run_root,
        seed=config["seed"],
        device=config["device"],
        edge_mode="bidirectional",
        use_node_type=True,
        model={
            "hidden_dim": baseline_cfg["hidden_dim"],
            "num_layers": baseline_cfg["num_layers"],
            "dropout": baseline_cfg["dropout"],
        },
        training={
            "batch_size": baseline_cfg["batch_size"],
            "epochs": baseline_cfg["epochs"],
            "learning_rate": baseline_cfg["learning_rate"],
            "weight_decay": baseline_cfg["weight_decay"],
            "grad_clip": baseline_cfg["grad_clip"],
            "log_every_batches": 100,
            "num_workers": baseline_cfg["num_workers"],
            "pin_memory": True,
            "persistent_workers": baseline_cfg["num_workers"] > 0,
            "prefetch_factor": 2,
            "use_amp": baseline_cfg["use_amp"],
        },
    )

    # Rebuild config from dict to ensure proper normalization
    cfg = BaselineConfig.from_dict({
        "prepared_root": str(prepared_root),
        "run_root": str(run_root),
        "seed": config["seed"],
        "device": config["device"],
        "edge_mode": "bidirectional",
        "use_node_type": True,
        "model": {
            "hidden_dim": baseline_cfg["hidden_dim"],
            "num_layers": baseline_cfg["num_layers"],
            "dropout": baseline_cfg["dropout"],
        },
        "training": {
            "batch_size": baseline_cfg["batch_size"],
            "epochs": baseline_cfg["epochs"],
            "learning_rate": baseline_cfg["learning_rate"],
            "weight_decay": baseline_cfg["weight_decay"],
            "grad_clip": baseline_cfg["grad_clip"],
            "log_every_batches": 100,
            "num_workers": baseline_cfg["num_workers"],
            "pin_memory": True,
            "persistent_workers": baseline_cfg["num_workers"] > 0,
            "prefetch_factor": 2,
            "use_amp": baseline_cfg["use_amp"],
        },
    })

    summary = train_baseline(cfg, resume_run_dir=resume_run_dir)
    best_checkpoint = summary.get("best_checkpoint", "")

    # Create best_run symlink
    best_run_link = run_root / "best_run"
    run_dir = Path(summary["run_dir"])
    if best_run_link.exists() or best_run_link.is_symlink():
        best_run_link.unlink()
    best_run_link.symlink_to(run_dir.name)

    return {
        "summary": summary,
        "run_dir": str(run_dir),
        "best_checkpoint": best_checkpoint,
        "best_run_link": str(best_run_link),
    }


def run_pointer(config: dict[str, Any], resume_run_dir: str | None = None) -> dict[str, Any]:
    """Stage 3: Train pointer-based argument selection model."""
    from maths_ai.gnn_inference.atp_lean_gnn.training import (
        PointerConfig,
        train_pointer,
    )

    pointer_cfg = config["pointer"]
    prepared_root = Path(config["prepared_root"])
    run_root = Path(config["run_root"]) / "pointer_gnn"

    console_print("\n" + "=" * 60)
    console_print("  STAGE 3: TRAIN POINTER MODEL")
    console_print("=" * 60)
    console_print(f"  Prepared  : {prepared_root}")
    console_print(f"  Run root  : {run_root}")
    console_print(f"  hidden    : {pointer_cfg['hidden_dim']}")
    console_print(f"  layers    : {pointer_cfg['num_layers']}")
    console_print(f"  dropout   : {pointer_cfg['dropout']}")
    console_print(f"  batch     : {pointer_cfg['batch_size']}")
    console_print(f"  epochs    : {pointer_cfg['epochs']}")
    console_print(f"  max_args  : {pointer_cfg['max_args']}")
    console_print(f"  arg_wt    : {pointer_cfg['arg_loss_weight']}")
    console_print("")

    cfg = PointerConfig.from_dict({
        "prepared_root": str(prepared_root),
        "run_root": str(run_root),
        "seed": config["seed"],
        "device": config["device"],
        "edge_mode": "bidirectional",
        "use_node_type": True,
        "max_args": pointer_cfg["max_args"],
        "arg_loss_weight": pointer_cfg["arg_loss_weight"],
        "model": {
            "hidden_dim": pointer_cfg["hidden_dim"],
            "num_layers": pointer_cfg["num_layers"],
            "dropout": pointer_cfg["dropout"],
        },
        "training": {
            "batch_size": pointer_cfg["batch_size"],
            "epochs": pointer_cfg["epochs"],
            "learning_rate": pointer_cfg["learning_rate"],
            "weight_decay": pointer_cfg["weight_decay"],
            "grad_clip": pointer_cfg["grad_clip"],
            "log_every_batches": 50,
            "num_workers": pointer_cfg["num_workers"],
            "pin_memory": True,
            "persistent_workers": pointer_cfg["num_workers"] > 0,
            "prefetch_factor": 2,
            "use_amp": pointer_cfg["use_amp"],
        },
    })

    summary = train_pointer(cfg, resume_run_dir=resume_run_dir)
    best_checkpoint = summary.get("best_checkpoint", "")

    # Create best_run symlink
    best_run_link = run_root / "best_run"
    run_dir = Path(summary["run_dir"])
    if best_run_link.exists() or best_run_link.is_symlink():
        best_run_link.unlink()
    best_run_link.symlink_to(run_dir.name)

    return {
        "summary": summary,
        "run_dir": str(run_dir),
        "best_checkpoint": best_checkpoint,
        "best_run_link": str(best_run_link),
    }


def run_scorer(config: dict[str, Any]) -> dict[str, Any]:
    """Stage 4: Train premise scorer on top of frozen pointer model."""
    scorer_cfg = config["scorer"]
    pointer_cfg = config["pointer"]
    prepared_root = Path(config["prepared_root"])
    run_root = Path(config["run_root"]) / "premise_gnn"

    console_print("\n" + "=" * 60)
    console_print("  STAGE 4: TRAIN PREMISE SCORER")
    console_print("=" * 60)

    # Find pointer checkpoint
    pointer_run_root = Path(config["run_root"]) / "pointer_gnn"
    best_run_link = pointer_run_root / "best_run"
    if best_run_link.is_symlink():
        pointer_run_dir = best_run_link.resolve()
    else:
        # Find latest run
        runs = sorted(pointer_run_root.glob("run_*"))
        if not runs:
            console_print("  ERROR: No pointer model runs found. Train pointer first.")
            return {"error": "no pointer checkpoint found"}
        pointer_run_dir = runs[-1]

    pointer_checkpoint = pointer_run_dir / "best.pt"
    if not pointer_checkpoint.exists():
        console_print(f"  ERROR: Pointer checkpoint not found at {pointer_checkpoint}")
        return {"error": "pointer checkpoint not found"}

    console_print(f"  Pointer ckpt: {pointer_checkpoint}")
    console_print(f"  Run root    : {run_root}")
    console_print(f"  hidden      : {scorer_cfg['hidden_dim']}")
    console_print(f"  scoring     : {scorer_cfg['scoring_mode']}")
    console_print(f"  k           : {scorer_cfg['k']}")
    console_print(f"  epochs      : {scorer_cfg['epochs']}")
    console_print("")

    from maths_ai.gnn_inference.scripts.train_scorer import _create_run_dir
    from maths_ai.gnn_inference.atp_lean_gnn.training import load_pointer_config, load_prepared_metadata, build_dataloaders
    from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer, PremiseScorerConfig
    from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
    from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
    from maths_ai.gnn_inference.atp_lean_gnn.logger import TrainingLogger
    from maths_ai.gnn_inference.atp_lean_gnn.premise_training import evaluate_model_with_premises, train_one_epoch_with_premises

    import torch
    from torch.optim import AdamW

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and scorer_cfg["use_amp"]

    # Load pointer config to get model architecture
    pointer_config_path = pointer_run_dir / "config.json"
    if not pointer_config_path.exists():
        pointer_config_path = Path("maths_ai/gnn_inference/configs/pointer_graphsage_state.json")
    p_config = load_pointer_config(pointer_config_path)
    metadata = load_prepared_metadata(prepared_root)

    # Load premise scoring config
    premise_config_path = Path(scorer_cfg["premise_config"])
    if premise_config_path.exists():
        with open(premise_config_path) as f:
            p_cfg_dict = json.load(f)
        # Override with our config values
        p_cfg_dict["scoring_mode"] = scorer_cfg["scoring_mode"]
        p_cfg_dict["k"] = scorer_cfg["k"]
        p_cfg_dict["premise_loss_weight"] = scorer_cfg["premise_loss_weight"]
    else:
        p_cfg_dict = {
            "scoring_mode": scorer_cfg["scoring_mode"],
            "k": scorer_cfg["k"],
            "premise_loss_weight": scorer_cfg["premise_loss_weight"],
        }
    p_config_obj = PremiseScorerConfig(**p_cfg_dict)

    run_dir = _create_run_dir(run_root)
    logger = TrainingLogger(run_dir)

    # Try to load lemma index
    lemma_index_path = Path(config["run_root"]) / "lemma_index_v1" / "best_run"
    if lemma_index_path.is_symlink():
        lemma_index_dir = lemma_index_path.resolve()
    else:
        lemma_index_dir = Path(config["run_root"]) / "lemma_index_v1"

    lemma_index_file = None
    for candidate in [
        lemma_index_dir / "lemma_index.faiss",
        lemma_index_dir / "index" / "lemma_index.faiss",
    ]:
        if candidate.exists():
            lemma_index_file = candidate
            break

    if lemma_index_file is None:
        console_print("  WARNING: No lemma index found. Scorer training will use local candidates only.")
        lemma_index = None
    else:
        console_print(f"  Lemma index: {lemma_index_file}")
        lemma_index = LemmaIndex.load(lemma_index_file)

    datasets, loaders = build_dataloaders(metadata, p_config)

    # Build model
    model = TacticWithArgsClassifier(
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
        hidden_dim=p_config.model.hidden_dim,
        num_layers=p_config.model.num_layers,
        dropout=p_config.model.dropout,
        use_node_type=p_config.use_node_type,
        max_args=getattr(p_config, "max_args", 3),
    )

    ckpt = torch.load(pointer_checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    adjusted_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("backbone.") and not k.startswith("tactic_embedding.") and not k.startswith("argument_selector."):
            adjusted_state_dict[f"backbone.{k}"] = v
        else:
            adjusted_state_dict[k] = v
    model.load_state_dict(adjusted_state_dict, strict=False)

    has_trained_tactic_embedding = any(k.startswith("tactic_embedding.") for k in adjusted_state_dict)
    if not has_trained_tactic_embedding:
        with torch.no_grad():
            model.tactic_embedding.weight.copy_(model.backbone.classifier.weight)

    for param in model.backbone.parameters():
        param.requires_grad = False
    model = model.to(device)

    scorer = PremiseScorer(hidden_dim=p_config.model.hidden_dim, mode=p_config_obj.scoring_mode).to(device)

    trainable_params = (
        list(model.tactic_embedding.parameters())
        + list(model.argument_selector.parameters())
        + list(scorer.parameters())
    )
    optimizer = AdamW(trainable_params, lr=p_config.training.learning_rate, weight_decay=p_config.training.weight_decay)
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_mrr = -1.0
    for epoch in range(1, scorer_cfg["epochs"] + 1):
        train_metrics = train_one_epoch_with_premises(
            model=model, scorer=scorer, loader=loaders["train"],
            lemma_index=lemma_index, optimizer=optimizer, grad_scaler=grad_scaler,
            device=device, grad_clip=p_config.training.grad_clip,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=p_config.arg_loss_weight if hasattr(p_config, "arg_loss_weight") else 0.5,
            premise_loss_weight=p_config_obj.premise_loss_weight,
            k=p_config_obj.k, epoch=epoch, total_epochs=scorer_cfg["epochs"],
            log_every_batches=p_config.training.log_every_batches,
            use_amp=use_amp, pin_memory=p_config.training.pin_memory,
        )

        val_metrics = evaluate_model_with_premises(
            model=model, scorer=scorer, loader=loaders["val"],
            lemma_index=lemma_index, device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=p_config.arg_loss_weight if hasattr(p_config, "arg_loss_weight") else 0.5,
            premise_loss_weight=p_config_obj.premise_loss_weight,
            k=p_config_obj.k, split_name="val",
            log_every_batches=p_config.training.log_every_batches,
            use_amp=use_amp, pin_memory=p_config.training.pin_memory,
        )

        console_print(
            f"  Epoch {epoch:02d}/{scorer_cfg['epochs']:02d} | "
            f"Val MRR: {val_metrics['premise_mrr']:.4f} | "
            f"Hit@1: {val_metrics['premise_top1_accuracy']:.4f} | "
            f"Hit@5: {val_metrics['premise_top5_accuracy']:.4f}"
        )

        if val_metrics["premise_mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["premise_mrr"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "scorer_state_dict": scorer.state_dict(),
                "val_metrics": val_metrics,
            }, run_dir / "best.pt")

        logger.log_epoch(epoch, {
            "train_tactic_loss": float(train_metrics["tactic_loss"]),
            "train_arg_loss": float(train_metrics["arg_loss"]),
            "train_premise_loss": float(train_metrics["premise_loss"]),
            "train_combined_loss": float(train_metrics["combined_loss"]),
            "val_premise_mrr": float(val_metrics["premise_mrr"]),
            "val_premise_top1_accuracy": float(val_metrics["premise_top1_accuracy"]),
            "val_premise_top5_accuracy": float(val_metrics["premise_top5_accuracy"]),
        })

    # Create best_run symlink
    best_run_link = run_root / "best_run"
    if best_run_link.exists() or best_run_link.is_symlink():
        best_run_link.unlink()
    best_run_link.symlink_to(run_dir.name)

    return {
        "run_dir": str(run_dir),
        "best_val_mrr": best_val_mrr,
        "best_run_link": str(best_run_link),
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────────

def _set_nested(d: dict, key: str, value: Any) -> None:
    """Set a nested config value from a dotted key like 'model.hidden_dim'."""
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    # Try to cast to the right type
    existing = d.get(parts[-1])
    if isinstance(existing, int):
        d[parts[-1]] = int(value)
    elif isinstance(existing, float):
        d[parts[-1]] = float(value)
    elif isinstance(existing, bool):
        d[parts[-1]] = value.lower() in ("true", "1", "yes")
    elif existing is None:
        # Try to infer from value
        if value.lower() in ("true", "false"):
            d[parts[-1]] = value.lower() == "true"
        else:
            try:
                d[parts[-1]] = int(value)
            except ValueError:
                try:
                    d[parts[-1]] = float(value)
                except ValueError:
                    d[parts[-1]] = value
    else:
        d[parts[-1]] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified training pipeline for GNN tactic prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with defaults
  python -m maths_ai.gnn_inference.scripts.run_training

  # Custom config file
  python -m maths_ai.gnn_inference.scripts.run_training --config my_config.json

  # Only run specific stages
  python -m maths_ai.gnn_inference.scripts.run_training --stages prepare,baseline

  # Override parameters
  python -m maths_ai.gnn_inference.scripts.run_training \\
      --baseline.hidden_dim 256 \\
      --baseline.epochs 10 \\
      --baseline.batch_size 128

  # Resume from last checkpoint
  python -m maths_ai.gnn_inference.scripts.run_training --resume

  # Force re-prepare dataset
  python -m maths_ai.gnn_inference.scripts.run_training --prepare.force true
        """,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (overrides defaults)")
    parser.add_argument("--stages", type=str, default=None, help="Comma-separated stages to run: prepare,baseline,pointer,scorer")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (requires pipeline_state.json)")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit without running")
    parser.add_argument("--experiment-name", type=str, default=None, help="Name for this experiment run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cpu, cuda")
    parser.add_argument("--prepared-root", type=str, default=None, help="Path to prepared dataset root")
    parser.add_argument("--run-root", type=str, default=None, help="Path to runs output directory")

    # Stage-specific overrides
    for stage in ["baseline", "pointer", "scorer", "prepare"]:
        group = parser.add_argument_group(f"{stage} overrides")
        for key, value in DEFAULT_CONFIG.get(stage, {}).items():
            if isinstance(value, dict):
                for k2, v2 in value.items():
                    group.add_argument(f"--{stage}.{k2}", type=str, default=None,
                                       help=f"{stage}.{k2} = {v2} (default)")
            else:
                group.add_argument(f"--{stage}.{key}", type=str, default=None,
                                   help=f"{stage}.{key} = {value} (default)")

    return parser


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load base config
    config = DEFAULT_CONFIG.copy()
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            console_print(f"  ERROR: Config file not found: {config_path}")
            return 1
        with open(config_path) as f:
            user_config = json.load(f)
        # Deep merge
        for key, value in user_config.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value

    # Apply CLI overrides
    if args.stages:
        config["stages"] = [s.strip() for s in args.stages.split(",")]
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    if args.prepared_root:
        config["prepared_root"] = args.prepared_root
    if args.run_root:
        config["run_root"] = args.run_root

    # Apply nested overrides
    for key, value in vars(args).items():
        if "." in key and value is not None:
            _set_nested(config, key, value)

    # Print config and exit if dry run
    if args.dry_run:
        console_print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0

    # Print banner
    console_print("=" * 60)
    console_print("  GNN TRAINING PIPELINE")
    console_print("=" * 60)
    console_print(f"  Experiment : {config['experiment_name']}")
    console_print(f"  Stages     : {', '.join(config['stages'])}")
    console_print(f"  Seed       : {config['seed']}")
    console_print(f"  Device     : {config['device']}")
    console_print(f"  Prepared   : {config['prepared_root']}")
    console_print(f"  Runs       : {config['run_root']}")
    console_print("")

    # Load or create pipeline state
    pipeline_dir = Path(config["run_root"]) / config["experiment_name"]
    pipeline_state_path = pipeline_dir / "pipeline_state.json"
    state = PipelineState.load(pipeline_state_path)
    state.start_time = time.time()

    # Determine which stages to run
    all_stages = ["prepare", "baseline", "pointer", "scorer"]
    stages_to_run = config["stages"]
    if args.resume:
        stages_to_run = [s for s in stages_to_run if not state.is_complete(s)]
        if not stages_to_run:
            console_print("  All stages already complete. Use --stages to force re-run.")
            return 0
        console_print(f"  Resuming — will run: {', '.join(stages_to_run)}")

    results = {}
    for stage in stages_to_run:
        if stage not in all_stages:
            console_print(f"  WARNING: Unknown stage '{stage}', skipping.")
            continue

        stage_start = time.time()
        try:
            if stage == "prepare":
                results[stage] = run_prepare(config)
            elif stage == "baseline":
                resume_dir = None
                if args.resume and state.stage_outputs.get("baseline", {}).get("run_dir"):
                    resume_dir = state.stage_outputs["baseline"]["run_dir"]
                results[stage] = run_baseline(config, resume_run_dir=resume_dir)
            elif stage == "pointer":
                resume_dir = None
                if args.resume and state.stage_outputs.get("pointer", {}).get("run_dir"):
                    resume_dir = state.stage_outputs["pointer"]["run_dir"]
                results[stage] = run_pointer(config, resume_run_dir=resume_dir)
            elif stage == "scorer":
                results[stage] = run_scorer(config)

            state.mark_complete(stage, results[stage])
            elapsed = time.time() - stage_start
            console_print(f"\n  [DONE] {stage} completed in {elapsed:.1f}s\n")
        except Exception as exc:
            elapsed = time.time() - stage_start
            console_print(f"\n  [FAILED] {stage} failed after {elapsed:.1f}s: {exc}\n")
            import traceback
            traceback.print_exc()
            # Save state and exit
            state.save(pipeline_state_path)
            return 1

        # Save state after each stage
        state.save(pipeline_state_path)

    # Print summary
    total_elapsed = time.time() - state.start_time
    console_print("\n" + "=" * 60)
    console_print("  PIPELINE COMPLETE")
    console_print("=" * 60)
    console_print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    console_print(f"  Stages: {', '.join(state.completed_stages)}")
    for stage, result in results.items():
        if "run_dir" in result:
            console_print(f"  {stage}: {result['run_dir']}")
        elif "summary" in result and "run_dir" in result["summary"]:
            console_print(f"  {stage}: {result['summary']['run_dir']}")
    console_print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
