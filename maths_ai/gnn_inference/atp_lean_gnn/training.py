from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from .argument_training import (
    evaluate_model_with_args,
    train_one_epoch_with_args,
)
from .argument_selector import TacticWithArgsClassifier, TacticWithArgsConfig
from .dataset import CANONICAL_SPLITS, canonicalize_split_name
from .labels import UNKNOWN_TACTIC, get_tactic_arity
from .model import GraphSAGEClassifierConfig, GraphSAGEStateClassifier
from .pyg import NODE_TYPE_TO_ID
from .reporting import console_print


DEFAULT_BASELINE_CONFIG_PATH = Path("configs") / "baseline_graphsage_state_rtx3050.json"
DEFAULT_POINTER_CONFIG_PATH = Path("configs") / "pointer_graphsage_state_rtx3050.json"
REQUIRED_DATA_FIELDS = ("x", "node_type", "edge_index", "y", "split", "row_index", "tactic_name")
REQUIRED_POINTER_DATA_FIELDS = REQUIRED_DATA_FIELDS + ("premise_mask", "arg_node_indices")


@dataclass(frozen=True)
class TrainingLoopConfig:
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every_batches: int = 100
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_amp: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "log_every_batches": self.log_every_batches,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "use_amp": self.use_amp,
        }


@dataclass(frozen=True)
class BaselineConfig:
    prepared_root: Path
    run_root: Path
    seed: int = 42
    device: str = "auto"
    edge_mode: str = "bidirectional"
    use_node_type: bool = True
    model: GraphSAGEClassifierConfig = field(default_factory=GraphSAGEClassifierConfig)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineConfig":
        if "prepared_root" not in payload:
            raise ValueError("Training config is missing the required 'prepared_root' field.")

        model_payload = payload.get("model", {})
        training_payload = payload.get("training", {})
        return cls(
            prepared_root=Path(payload["prepared_root"]),
            run_root=Path(payload.get("run_root", "runs/baseline_gnn")),
            seed=int(payload.get("seed", 42)),
            device=str(payload.get("device", "auto")),
            edge_mode=str(payload.get("edge_mode", "bidirectional")),
            use_node_type=bool(payload.get("use_node_type", True)),
            model=GraphSAGEClassifierConfig(
                hidden_dim=int(model_payload.get("hidden_dim", 128)),
                num_layers=int(model_payload.get("num_layers", 4)),
                dropout=float(model_payload.get("dropout", 0.2)),
            ),
            training=TrainingLoopConfig(
                batch_size=int(training_payload.get("batch_size", 32)),
                epochs=int(training_payload.get("epochs", 20)),
                learning_rate=float(training_payload.get("learning_rate", 1e-3)),
                weight_decay=float(training_payload.get("weight_decay", 1e-4)),
                grad_clip=float(training_payload.get("grad_clip", 1.0)),
                log_every_batches=int(training_payload.get("log_every_batches", 100)),
                num_workers=int(training_payload.get("num_workers", 2)),
                pin_memory=bool(training_payload.get("pin_memory", True)),
                persistent_workers=bool(training_payload.get("persistent_workers", True)),
                prefetch_factor=int(training_payload.get("prefetch_factor", 2)),
                use_amp=bool(training_payload.get("use_amp", True)),
            ),
        ).normalized()

    def normalized(self) -> "BaselineConfig":
        edge_mode = self.edge_mode.lower().strip()
        if edge_mode not in {"forward", "bidirectional"}:
            raise ValueError("Training config field 'edge_mode' must be either 'forward' or 'bidirectional'.")

        device = self.device.lower().strip()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Training config field 'device' must be one of: auto, cpu, cuda.")

        if self.model.hidden_dim < 1:
            raise ValueError("Training config field 'model.hidden_dim' must be positive.")
        if self.model.num_layers < 1:
            raise ValueError("Training config field 'model.num_layers' must be positive.")
        if self.training.batch_size < 1:
            raise ValueError("Training config field 'training.batch_size' must be positive.")
        if self.training.epochs < 1:
            raise ValueError("Training config field 'training.epochs' must be positive.")
        if self.training.learning_rate <= 0:
            raise ValueError("Training config field 'training.learning_rate' must be positive.")
        if self.training.weight_decay < 0:
            raise ValueError("Training config field 'training.weight_decay' cannot be negative.")
        if self.training.grad_clip <= 0:
            raise ValueError("Training config field 'training.grad_clip' must be positive.")
        if self.training.log_every_batches < 1:
            raise ValueError("Training config field 'training.log_every_batches' must be positive.")
        if self.training.num_workers < 0:
            raise ValueError("Training config field 'training.num_workers' cannot be negative.")
        if self.training.prefetch_factor < 1:
            raise ValueError("Training config field 'training.prefetch_factor' must be positive.")

        return BaselineConfig(
            prepared_root=self.prepared_root.resolve(),
            run_root=self.run_root.resolve(),
            seed=self.seed,
            device=device,
            edge_mode=edge_mode,
            use_node_type=self.use_node_type,
            model=self.model,
            training=self.training,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "prepared_root": str(self.prepared_root),
            "run_root": str(self.run_root),
            "seed": self.seed,
            "device": self.device,
            "edge_mode": self.edge_mode,
            "use_node_type": self.use_node_type,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }


@dataclass(frozen=True)
class PointerConfig:
    """Config for pointer-based argument selection model."""
    prepared_root: Path
    run_root: Path
    seed: int = 42
    device: str = "auto"
    edge_mode: str = "bidirectional"
    use_node_type: bool = True
    max_args: int = 3
    arg_loss_weight: float = 0.5
    model: TacticWithArgsConfig = field(default_factory=TacticWithArgsConfig)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PointerConfig":
        if "prepared_root" not in payload:
            raise ValueError("Training config is missing the required 'prepared_root' field.")

        model_payload = payload.get("model", {})
        training_payload = payload.get("training", {})
        return cls(
            prepared_root=Path(payload["prepared_root"]),
            run_root=Path(payload.get("run_root", "runs/pointer_gnn")),
            seed=int(payload.get("seed", 42)),
            device=str(payload.get("device", "auto")),
            edge_mode=str(payload.get("edge_mode", "bidirectional")),
            use_node_type=bool(payload.get("use_node_type", True)),
            max_args=int(payload.get("max_args", 3)),
            arg_loss_weight=float(payload.get("arg_loss_weight", 0.5)),
            model=TacticWithArgsConfig(
                hidden_dim=int(model_payload.get("hidden_dim", 128)),
                num_layers=int(model_payload.get("num_layers", 4)),
                dropout=float(model_payload.get("dropout", 0.2)),
                max_args=int(model_payload.get("max_args", 3)),
                arg_loss_weight=float(model_payload.get("arg_loss_weight", 0.5)),
            ),
            training=TrainingLoopConfig(
                batch_size=int(training_payload.get("batch_size", 32)),
                epochs=int(training_payload.get("epochs", 20)),
                learning_rate=float(training_payload.get("learning_rate", 1e-3)),
                weight_decay=float(training_payload.get("weight_decay", 1e-4)),
                grad_clip=float(training_payload.get("grad_clip", 1.0)),
                log_every_batches=int(training_payload.get("log_every_batches", 100)),
                num_workers=int(training_payload.get("num_workers", 2)),
                pin_memory=bool(training_payload.get("pin_memory", True)),
                persistent_workers=bool(training_payload.get("persistent_workers", True)),
                prefetch_factor=int(training_payload.get("prefetch_factor", 2)),
                use_amp=bool(training_payload.get("use_amp", True)),
            ),
        ).normalized()

    def normalized(self) -> "PointerConfig":
        edge_mode = self.edge_mode.lower().strip()
        if edge_mode not in {"forward", "bidirectional"}:
            raise ValueError("Training config field 'edge_mode' must be either 'forward' or 'bidirectional'.")

        device = self.device.lower().strip()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Training config field 'device' must be one of: auto, cpu, cuda.")

        if self.model.hidden_dim < 1:
            raise ValueError("Training config field 'model.hidden_dim' must be positive.")
        if self.model.num_layers < 1:
            raise ValueError("Training config field 'model.num_layers' must be positive.")
        if self.max_args < 1:
            raise ValueError("Training config field 'max_args' must be positive.")
        if self.arg_loss_weight < 0:
            raise ValueError("Training config field 'arg_loss_weight' cannot be negative.")
        if self.training.batch_size < 1:
            raise ValueError("Training config field 'training.batch_size' must be positive.")
        if self.training.epochs < 1:
            raise ValueError("Training config field 'training.epochs' must be positive.")
        if self.training.learning_rate <= 0:
            raise ValueError("Training config field 'training.learning_rate' must be positive.")
        if self.training.weight_decay < 0:
            raise ValueError("Training config field 'training.weight_decay' cannot be negative.")
        if self.training.grad_clip <= 0:
            raise ValueError("Training config field 'training.grad_clip' must be positive.")

        return PointerConfig(
            prepared_root=self.prepared_root.resolve(),
            run_root=self.run_root.resolve(),
            seed=self.seed,
            device=device,
            edge_mode=edge_mode,
            use_node_type=self.use_node_type,
            max_args=self.max_args,
            arg_loss_weight=self.arg_loss_weight,
            model=self.model,
            training=self.training,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "prepared_root": str(self.prepared_root),
            "run_root": str(self.run_root),
            "seed": self.seed,
            "device": self.device,
            "edge_mode": self.edge_mode,
            "use_node_type": self.use_node_type,
            "max_args": self.max_args,
            "arg_loss_weight": self.arg_loss_weight,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }


@dataclass(frozen=True)
class PreparedMetadata:
    root: Path
    node_vocab: dict[str, int]
    tactic_vocab: dict[str, int]
    manifests: dict[str, dict[str, object]]
    state_label_id: int
    unknown_tactic_id: int

    def split_manifest(self, split: str) -> dict[str, object]:
        canonical_split = canonicalize_split_name(split)
        return self.manifests[canonical_split]

    def split_pyg_dir(self, split: str) -> Path:
        manifest = self.split_manifest(split)
        artifact_paths = manifest.get("artifact_paths", {})
        pyg_dir_rel = artifact_paths.get("pyg_dir")
        if not pyg_dir_rel:
            raise ValueError(f"Manifest for split '{split}' is missing 'artifact_paths.pyg_dir'.")
        return self.root / str(pyg_dir_rel)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _append_jsonl(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return path


def load_baseline_config(
    config_path: str | Path = DEFAULT_BASELINE_CONFIG_PATH,
    *,
    prepared_root_override: str | Path | None = None,
    run_root_override: str | Path | None = None,
    epochs_override: int | None = None,
) -> BaselineConfig:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Training config file '{config_file}' does not exist.")

    payload = _read_json(config_file)
    if prepared_root_override is not None:
        payload["prepared_root"] = str(prepared_root_override)
    if run_root_override is not None:
        payload["run_root"] = str(run_root_override)
    if epochs_override is not None:
        payload.setdefault("training", {})["epochs"] = epochs_override
    return BaselineConfig.from_dict(payload)


def load_pointer_config(
    config_path: str | Path = DEFAULT_POINTER_CONFIG_PATH,
    *,
    prepared_root_override: str | Path | None = None,
    run_root_override: str | Path | None = None,
    epochs_override: int | None = None,
) -> PointerConfig:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Training config file '{config_file}' does not exist.")

    payload = _read_json(config_file)
    if prepared_root_override is not None:
        payload["prepared_root"] = str(prepared_root_override)
    if run_root_override is not None:
        payload["run_root"] = str(run_root_override)
    if epochs_override is not None:
        payload.setdefault("training", {})["epochs"] = epochs_override
    return PointerConfig.from_dict(payload)


def load_prepared_metadata(prepared_root: str | Path) -> PreparedMetadata:
    root = Path(prepared_root)
    if not root.exists():
        raise FileNotFoundError(f"Prepared dataset root '{root}' does not exist.")
    if not root.is_dir():
        raise FileNotFoundError(f"Prepared dataset root '{root}' is not a directory.")

    node_vocab_path = root / "vocab" / "node_vocab.json"
    tactic_vocab_path = root / "vocab" / "tactic_vocab.json"
    missing_paths = [path for path in (node_vocab_path, tactic_vocab_path) if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Prepared dataset is missing required vocab files: {missing_text}")

    node_vocab = {str(key): int(value) for key, value in _read_json(node_vocab_path).items()}
    tactic_vocab = {str(key): int(value) for key, value in _read_json(tactic_vocab_path).items()}

    if "State" not in node_vocab:
        raise ValueError(
            f"Prepared dataset node vocab '{node_vocab_path}' does not contain the required 'State' token."
        )
    if UNKNOWN_TACTIC not in tactic_vocab:
        raise ValueError(
            f"Prepared dataset tactic vocab '{tactic_vocab_path}' does not contain '{UNKNOWN_TACTIC}'."
        )

    manifests: dict[str, dict[str, object]] = {}
    for split in CANONICAL_SPLITS:
        manifest_path = root / "manifests" / f"{split}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Prepared dataset is missing manifest '{manifest_path}'.")
        manifest = _read_json(manifest_path)
        artifact_paths = manifest.get("artifact_paths", {})
        pyg_dir_rel = artifact_paths.get("pyg_dir")
        if not pyg_dir_rel:
            raise ValueError(f"Manifest '{manifest_path}' is missing 'artifact_paths.pyg_dir'.")
        pyg_dir = root / str(pyg_dir_rel)
        if not pyg_dir.exists():
            raise FileNotFoundError(
                f"Manifest '{manifest_path}' points to missing PyG artifact directory '{pyg_dir}'."
            )
        manifests[split] = manifest

    return PreparedMetadata(
        root=root.resolve(),
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        manifests=manifests,
        state_label_id=node_vocab["State"],
        unknown_tactic_id=tactic_vocab[UNKNOWN_TACTIC],
    )


def transform_edge_index(edge_index: torch.Tensor, *, edge_mode: str) -> torch.Tensor:
    if edge_mode == "forward":
        return edge_index.to(dtype=torch.long).contiguous()
    if edge_mode != "bidirectional":
        raise ValueError(f"Unsupported edge mode '{edge_mode}'.")
    if edge_index.numel() == 0:
        return edge_index.to(dtype=torch.long).contiguous()

    forward = edge_index.to(dtype=torch.long)
    reverse = forward[[1, 0], :]
    combined = torch.cat([forward, reverse], dim=1)
    return torch.unique(combined, dim=1).contiguous()


def validate_prepared_data(data, *, path: Path, split: str, required_fields: tuple[str, ...]) -> None:
    missing = [field for field in required_fields if not hasattr(data, field)]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Prepared example '{path}' is missing required fields: {missing_text}")

    if not torch.is_tensor(data.x) or data.x.dim() != 1:
        raise ValueError(f"Prepared example '{path}' has an invalid 'x' tensor shape.")
    if not torch.is_tensor(data.node_type) or data.node_type.dim() != 1:
        raise ValueError(f"Prepared example '{path}' has an invalid 'node_type' tensor shape.")
    if not torch.is_tensor(data.edge_index) or data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
        raise ValueError(f"Prepared example '{path}' has an invalid 'edge_index' tensor shape.")
    if not torch.is_tensor(data.y) or data.y.numel() != 1:
        raise ValueError(f"Prepared example '{path}' must store exactly one target label in 'y'.")
    if str(data.split) != split:
        raise ValueError(
            f"Prepared example '{path}' belongs to split '{data.split}', expected '{split}'."
        )


def infer_state_node_index(data, *, state_label_id: int, path: Path) -> torch.Tensor:
    state_matches = (data.x == state_label_id).nonzero(as_tuple=False).view(-1)
    if state_matches.numel() == 0:
        raise ValueError(
            f"Prepared example '{path}' does not contain the required 'State' node label."
        )

    source_nodes = {int(node_id) for node_id in data.edge_index[0].tolist()}
    root_candidates = [
        int(node_id)
        for node_id in state_matches.tolist()
        if int(node_id) not in source_nodes
    ]
    if len(root_candidates) == 1:
        return torch.tensor(root_candidates, dtype=torch.long)
    if state_matches.numel() == 1:
        return state_matches.to(dtype=torch.long)

    raise ValueError(
        f"Prepared example '{path}' must contain exactly one root 'State' node, "
        f"found {state_matches.numel()} 'State'-labeled nodes and {len(root_candidates)} root candidates."
    )


class PreparedGraphDataset(Dataset):
    def __init__(
        self,
        metadata: PreparedMetadata,
        *,
        split: str,
        edge_mode: str = "bidirectional",
        required_fields: tuple[str, ...] = REQUIRED_DATA_FIELDS,
    ) -> None:
        self.metadata = metadata
        self.split = canonicalize_split_name(split)
        self.edge_mode = edge_mode
        self.required_fields = required_fields
        self.pyg_dir = metadata.split_pyg_dir(self.split)
        self.files = sorted(self.pyg_dir.glob("*.pt"))
        if not self.files:
            raise RuntimeError(
                f"Prepared split '{self.split}' has no cached PyG examples under '{self.pyg_dir}'."
            )

        expected_count = int(metadata.split_manifest(self.split).get("success_count", len(self.files)))
        if expected_count != len(self.files):
            raise ValueError(
                f"Prepared split '{self.split}' manifest reports {expected_count} examples, "
                f"but '{self.pyg_dir}' contains {len(self.files)} '.pt' files."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        validate_prepared_data(data, path=path, split=self.split, required_fields=self.required_fields)

        data.x = data.x.to(dtype=torch.long)
        data.node_type = data.node_type.to(dtype=torch.long)
        data.state_node_index = infer_state_node_index(
            data,
            state_label_id=self.metadata.state_label_id,
            path=path,
        )
        data.edge_index = transform_edge_index(data.edge_index, edge_mode=self.edge_mode)
        data.y = data.y.view(-1).to(dtype=torch.long)
        return data


def build_dataloaders(
    metadata: PreparedMetadata,
    config: BaselineConfig | PointerConfig,
    required_fields: tuple[str, ...] = REQUIRED_DATA_FIELDS,
) -> tuple[dict[str, PreparedGraphDataset], dict[str, DataLoader]]:
    use_workers = config.training.num_workers > 0
    loader_kwargs: dict[str, object] = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
    }
    if use_workers:
        loader_kwargs["persistent_workers"] = config.training.persistent_workers
        loader_kwargs["prefetch_factor"] = config.training.prefetch_factor

    datasets = {
        split: PreparedGraphDataset(
            metadata,
            split=split,
            edge_mode=config.edge_mode,
            required_fields=required_fields,
        )
        for split in CANONICAL_SPLITS
    }
    loaders = {
        split: DataLoader(
            dataset,
            shuffle=(split == "train"),
            **loader_kwargs,
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders


def compute_eval_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    unknown_tactic_id: int,
) -> dict[str, float | int]:
    if logits.dim() != 2:
        raise ValueError("Expected logits to have shape [batch_size, num_classes].")
    if targets.dim() != 1:
        raise ValueError("Expected targets to have shape [batch_size].")
    if logits.size(0) != targets.size(0):
        raise ValueError("Logits and targets batch sizes do not match.")

    unknown_mask = targets == unknown_tactic_id
    known_mask = ~unknown_mask
    unknown_count = int(unknown_mask.sum().item())
    known_count = int(known_mask.sum().item())

    if known_count == 0:
        return {
            "known_label_count": 0,
            "unknown_label_excluded_count": unknown_count,
            "loss_sum": 0.0,
            "top1_correct": 0,
            "top5_correct": 0,
        }

    known_logits = logits[known_mask]
    known_targets = targets[known_mask]
    loss = F.cross_entropy(known_logits, known_targets)

    top1_predictions = known_logits.argmax(dim=1)
    top1_correct = int((top1_predictions == known_targets).sum().item())

    top_k = min(5, known_logits.size(1))
    topk_predictions = known_logits.topk(top_k, dim=1).indices
    top5_correct = int(
        (topk_predictions == known_targets.unsqueeze(1)).any(dim=1).sum().item()
    )

    return {
        "known_label_count": known_count,
        "unknown_label_excluded_count": unknown_count,
        "loss_sum": float(loss.item()) * known_count,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Training config requested CUDA, but no CUDA device is available.")
    return torch.device(device_name)


def build_baseline_model(metadata: PreparedMetadata, config: BaselineConfig) -> GraphSAGEStateClassifier:
    return GraphSAGEStateClassifier(
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
        num_node_types=len(NODE_TYPE_TO_ID),
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        use_node_type=config.use_node_type,
    )


def build_pointer_model(metadata: PreparedMetadata, config: PointerConfig) -> TacticWithArgsClassifier:
    return TacticWithArgsClassifier(
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
        num_node_types=len(NODE_TYPE_TO_ID),
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        use_node_type=config.use_node_type,
        max_args=config.max_args,
    )


def _use_cuda_amp(device: torch.device, config: BaselineConfig | PointerConfig) -> bool:
    return config.training.use_amp and device.type == "cuda"


def _should_log_batch(batch_index: int, total_batches: int, *, log_every_batches: int) -> bool:
    return (
        batch_index == 1
        or batch_index == total_batches
        or batch_index % log_every_batches == 0
    )


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def train_one_epoch(
    model: GraphSAGEStateClassifier,
    loader: DataLoader,
    *,
    optimizer: AdamW,
    grad_scaler,
    device: torch.device,
    grad_clip: float,
    unknown_tactic_id: int,
    epoch: int,
    total_epochs: int,
    log_every_batches: int,
    use_amp: bool,
    pin_memory: bool,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = len(loader)
    start_time = time.perf_counter()

    console_print(
        f"  Starting epoch {epoch:02d}/{total_epochs:02d} "
        f"with {total_batches} train batches..."
    )

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=(device.type == "cuda" and pin_memory))
        targets = batch.y.view(-1)
        if bool((targets == unknown_tactic_id).any()):
            raise ValueError("The train split contains '<UNK_TACTIC>' targets, which should never happen.")

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch)
            loss = F.cross_entropy(logits, targets)

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        batch_size = int(targets.numel())
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        if _should_log_batch(batch_index, total_batches, log_every_batches=log_every_batches):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            console_print(
                f"    train batch {batch_index:>5}/{total_batches} | "
                f"seen={total_examples} | "
                f"avg_loss={total_loss / max(total_examples, 1):.4f} | "
                f"elapsed={elapsed}"
            )

    return {
        "loss": total_loss / max(total_examples, 1),
        "example_count": total_examples,
    }


def evaluate_model(
    model: GraphSAGEStateClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    unknown_tactic_id: int,
    split_name: str | None = None,
    log_every_batches: int | None = None,
    use_amp: bool = False,
    pin_memory: bool = False,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    known_label_count = 0
    unknown_label_excluded_count = 0
    top1_correct = 0
    top5_correct = 0
    total_batches = len(loader)
    start_time = time.perf_counter()

    if split_name is not None:
        console_print(f"  Evaluating {split_name} split ({total_batches} batches)...")

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=(device.type == "cuda" and pin_memory))
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch)
            targets = batch.y.view(-1)
            batch_metrics = compute_eval_metrics_from_logits(
                logits,
                targets,
                unknown_tactic_id=unknown_tactic_id,
            )
            loss_sum += float(batch_metrics["loss_sum"])
            known_label_count += int(batch_metrics["known_label_count"])
            unknown_label_excluded_count += int(batch_metrics["unknown_label_excluded_count"])
            top1_correct += int(batch_metrics["top1_correct"])
            top5_correct += int(batch_metrics["top5_correct"])

            if (
                split_name is not None
                and log_every_batches is not None
                and _should_log_batch(batch_index, total_batches, log_every_batches=log_every_batches)
            ):
                elapsed = _format_elapsed(time.perf_counter() - start_time)
                console_print(
                    f"    {split_name} batch {batch_index:>5}/{total_batches} | "
                    f"known={known_label_count} | "
                    f"excluded={unknown_label_excluded_count} | "
                    f"elapsed={elapsed}"
                )

    top1 = top1_correct / known_label_count if known_label_count else 0.0
    top5 = top5_correct / known_label_count if known_label_count else 0.0
    loss = loss_sum / known_label_count if known_label_count else 0.0

    return {
        "loss": loss,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "known_label_count": known_label_count,
        "unknown_label_excluded_count": unknown_label_excluded_count,
        "evaluated_count": known_label_count + unknown_label_excluded_count,
    }


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


def _save_checkpoint(
    path: Path,
    *,
    model: GraphSAGEStateClassifier | TacticWithArgsClassifier,
    optimizer: AdamW,
    config: BaselineConfig | PointerConfig,
    epoch: int,
    val_metrics: dict[str, float | int],
) -> Path:
    torch.save(
        {
            "epoch": epoch,
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        },
        path,
    )
    return path


def _load_checkpoint(path: Path, *, device: torch.device) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
    return torch.load(path, map_location=device, weights_only=False)


def _write_eval_file(run_dir: Path, *, split: str, metrics: dict[str, object]) -> Path:
    return _write_json(run_dir / f"eval_{split}.json", metrics)


def train_baseline(
    config: BaselineConfig,
    *,
    resume_run_dir: str | Path | None = None,
) -> dict[str, object]:
    metadata = load_prepared_metadata(config.prepared_root)
    set_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = _use_cuda_amp(device, config)
    datasets, loaders = build_dataloaders(metadata, config, required_fields=REQUIRED_DATA_FIELDS)
    if resume_run_dir is None:
        run_dir = _create_run_dir(config.run_root)
        config_path = _write_json(run_dir / "config.json", config.to_dict())
        start_epoch = 1
        best_epoch = 0
        best_val_top1 = -1.0
    else:
        run_dir = Path(resume_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory '{run_dir}' does not exist.")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run path '{run_dir}' is not a directory.")
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Resume run directory '{run_dir}' is missing 'config.json'.")
        start_epoch = 1
        best_epoch = 0
        best_val_top1 = -1.0

    metrics_path = run_dir / "metrics.jsonl"
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    model = build_baseline_model(metadata, config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if resume_run_dir is not None:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume run directory '{run_dir}' is missing 'last.pt', so training cannot resume."
            )
        last_checkpoint = _load_checkpoint(last_checkpoint_path, device=device)
        model.load_state_dict(last_checkpoint["model_state_dict"])
        optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
        start_epoch = int(last_checkpoint["epoch"]) + 1
        if best_checkpoint_path.exists():
            best_checkpoint = _load_checkpoint(best_checkpoint_path, device=device)
            best_epoch = int(best_checkpoint["epoch"])
            best_val_top1 = float(
                dict(best_checkpoint.get("val_metrics", {})).get("top1_accuracy", -1.0)
            )

    console_print(f"\n  Training baseline run in: {run_dir}")
    console_print(f"  Prepared cache           : {config.prepared_root}")
    console_print(f"  Device                   : {device}")
    console_print(f"  AMP enabled              : {use_amp}")
    console_print(
        f"  Split sizes              : train={len(datasets['train'])}, "
        f"val={len(datasets['val'])}, test={len(datasets['test'])}"
    )
    console_print(
        f"  DataLoader settings      : batch_size={config.training.batch_size}, "
        f"workers={config.training.num_workers}, "
        f"pin_memory={config.training.pin_memory}, "
        f"persistent_workers={config.training.persistent_workers and config.training.num_workers > 0}, "
        f"prefetch_factor={config.training.prefetch_factor if config.training.num_workers > 0 else 'n/a'}"
    )
    if resume_run_dir is not None:
        console_print(
            f"  Resuming from checkpoint : {last_checkpoint_path} "
            f"(next epoch {start_epoch})"
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            device=device,
            grad_clip=config.training.grad_clip,
            unknown_tactic_id=metadata.unknown_tactic_id,
            epoch=epoch,
            total_epochs=config.training.epochs,
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
        )
        val_metrics = evaluate_model(
            model,
            loaders["val"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            split_name="val",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_example_count": int(train_metrics["example_count"]),
            "val_loss": float(val_metrics["loss"]),
            "val_top1": float(val_metrics["top1_accuracy"]),
            "val_top5": float(val_metrics["top5_accuracy"]),
            "known_label_eval_count": int(val_metrics["known_label_count"]),
            "unknown_label_excluded_count": int(val_metrics["unknown_label_excluded_count"]),
        }
        _append_jsonl(metrics_path, epoch_record)

        _save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            val_metrics=val_metrics,
        )
        if float(val_metrics["top1_accuracy"]) > best_val_top1:
            best_val_top1 = float(val_metrics["top1_accuracy"])
            best_epoch = epoch
            _save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                val_metrics=val_metrics,
            )

        console_print(
            f"  Epoch {epoch:02d}/{config.training.epochs:02d} | "
            f"train_loss={epoch_record['train_loss']:.4f} | "
            f"val_loss={epoch_record['val_loss']:.4f} | "
            f"val_top1={epoch_record['val_top1']:.4f} | "
            f"val_top5={epoch_record['val_top5']:.4f} | "
            f"known={epoch_record['known_label_eval_count']} | "
            f"excluded={epoch_record['unknown_label_excluded_count']}"
        )

    best_checkpoint = _load_checkpoint(best_checkpoint_path, device=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    eval_val = {
        "split": "val",
        "checkpoint": str(best_checkpoint_path),
        "epoch": int(best_checkpoint["epoch"]),
        **evaluate_model(
            model,
            loaders["val"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            split_name="val",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
        ),
    }
    eval_test = {
        "split": "test",
        "checkpoint": str(best_checkpoint_path),
        "epoch": int(best_checkpoint["epoch"]),
        **evaluate_model(
            model,
            loaders["test"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            split_name="test",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
        ),
    }
    _write_eval_file(run_dir, split="val", metrics=eval_val)
    _write_eval_file(run_dir, split="test", metrics=eval_test)

    summary = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "prepared_root": str(config.prepared_root),
        "device": str(device),
        "amp_enabled": use_amp,
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "start_epoch": start_epoch,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "resumed_from_checkpoint": resume_run_dir is not None,
        "best_validation": eval_val,
        "test_evaluation": eval_test,
    }
    _write_json(run_dir / "summary.json", summary)

    console_print(f"\n  Best checkpoint          : {best_checkpoint_path}")
    console_print(f"  Validation eval summary  : {run_dir / 'eval_val.json'}")
    console_print(f"  Test eval summary        : {run_dir / 'eval_test.json'}")
    console_print(f"  Training summary         : {run_dir / 'summary.json'}")

    return summary


def train_pointer(
    config: PointerConfig,
    *,
    resume_run_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train pointer-based argument selection model."""
    metadata = load_prepared_metadata(config.prepared_root)
    set_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = _use_cuda_amp(device, config)
    datasets, loaders = build_dataloaders(metadata, config, required_fields=REQUIRED_POINTER_DATA_FIELDS)
    
    if resume_run_dir is None:
        run_dir = _create_run_dir(config.run_root)
        config_path = _write_json(run_dir / "config.json", config.to_dict())
        start_epoch = 1
        best_epoch = 0
        best_val_loss = float("inf")
    else:
        run_dir = Path(resume_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory '{run_dir}' does not exist.")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run path '{run_dir}' is not a directory.")
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Resume run directory '{run_dir}' is missing 'config.json'.")
        start_epoch = 1
        best_epoch = 0
        best_val_loss = float("inf")

    metrics_path = run_dir / "metrics.jsonl"
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    model = build_pointer_model(metadata, config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if resume_run_dir is not None:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume run directory '{run_dir}' is missing 'last.pt', so training cannot resume."
            )
        last_checkpoint = _load_checkpoint(last_checkpoint_path, device=device)
        model.load_state_dict(last_checkpoint["model_state_dict"])
        optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
        start_epoch = int(last_checkpoint["epoch"]) + 1
        if best_checkpoint_path.exists():
            best_checkpoint = _load_checkpoint(best_checkpoint_path, device=device)
            best_epoch = int(best_checkpoint["epoch"])
            best_val_loss = float(
                dict(best_checkpoint.get("val_metrics", {})).get("combined_loss", float("inf"))
            )

    console_print(f"\n  Training pointer run in  : {run_dir}")
    console_print(f"  Prepared cache           : {config.prepared_root}")
    console_print(f"  Device                   : {device}")
    console_print(f"  AMP enabled              : {use_amp}")
    console_print(
        f"  Split sizes              : train={len(datasets['train'])}, "
        f"val={len(datasets['val'])}, test={len(datasets['test'])}"
    )
    console_print(
        f"  DataLoader settings      : batch_size={config.training.batch_size}, "
        f"workers={config.training.num_workers}, "
        f"pin_memory={config.training.pin_memory}, "
        f"persistent_workers={config.training.persistent_workers and config.training.num_workers > 0}, "
        f"prefetch_factor={config.training.prefetch_factor if config.training.num_workers > 0 else 'n/a'}"
    )
    console_print(f"  Max args per step        : {config.max_args}")
    console_print(f"  Argument loss weight     : {config.arg_loss_weight}")
    if resume_run_dir is not None:
        console_print(
            f"  Resuming from checkpoint : {last_checkpoint_path} "
            f"(next epoch {start_epoch})"
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        train_metrics = train_one_epoch_with_args(
            model,
            loaders["train"],
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            device=device,
            grad_clip=config.training.grad_clip,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight,
            epoch=epoch,
            total_epochs=config.training.epochs,
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
        )
        val_metrics = evaluate_model_with_args(
            model,
            loaders["val"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight,
            split_name="val",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
        )

        epoch_record = {
            "epoch": epoch,
            "train_tactic_loss": float(train_metrics["tactic_loss"]),
            "train_arg_loss": float(train_metrics["arg_loss"]),
            "train_combined_loss": float(train_metrics["combined_loss"]),
            "train_example_count": int(train_metrics["example_count"]),
            "val_tactic_loss": float(val_metrics["tactic_loss"]),
            "val_arg_loss": float(val_metrics["arg_loss"]),
            "val_combined_loss": float(val_metrics["combined_loss"]),
            "val_tactic_accuracy": float(val_metrics["tactic_top1_accuracy"]),
            "known_label_eval_count": int(val_metrics["known_label_count"]),
        }
        _append_jsonl(metrics_path, epoch_record)

        _save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            val_metrics=val_metrics,
        )
        if float(val_metrics["combined_loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["combined_loss"])
            best_epoch = epoch
            _save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                val_metrics=val_metrics,
            )

        console_print(
            f"  Epoch {epoch:02d}/{config.training.epochs:02d} | "
            f"train_loss={epoch_record['train_combined_loss']:.4f} | "
            f"val_loss={epoch_record['val_combined_loss']:.4f} | "
            f"val_tactic_acc={epoch_record['val_tactic_accuracy']:.4f} | "
            f"known={epoch_record['known_label_eval_count']}"
        )

    best_checkpoint = _load_checkpoint(best_checkpoint_path, device=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    eval_val = {
        "split": "val",
        "checkpoint": str(best_checkpoint_path),
        "epoch": int(best_checkpoint["epoch"]),
        **evaluate_model_with_args(
            model,
            loaders["val"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight,
            split_name="val",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
        ),
    }
    eval_test = {
        "split": "test",
        "checkpoint": str(best_checkpoint_path),
        "epoch": int(best_checkpoint["epoch"]),
        **evaluate_model_with_args(
            model,
            loaders["test"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight,
            split_name="test",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
        ),
    }
    _write_eval_file(run_dir, split="val", metrics=eval_val)
    _write_eval_file(run_dir, split="test", metrics=eval_test)

    summary = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "prepared_root": str(config.prepared_root),
        "device": str(device),
        "amp_enabled": use_amp,
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "start_epoch": start_epoch,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "resumed_from_checkpoint": resume_run_dir is not None,
        "best_validation": eval_val,
        "test_evaluation": eval_test,
    }
    _write_json(run_dir / "summary.json", summary)

    console_print(f"\n  Best checkpoint          : {best_checkpoint_path}")
    console_print(f"  Validation eval summary  : {run_dir / 'eval_val.json'}")
    console_print(f"  Test eval summary        : {run_dir / 'eval_test.json'}")
    console_print(f"  Training summary         : {run_dir / 'summary.json'}")

    return summary


def evaluate_baseline_run(run_dir: str | Path, *, split: str) -> dict[str, object]:
    run_directory = Path(run_dir)
    if not run_directory.exists():
        raise FileNotFoundError(f"Run directory '{run_directory}' does not exist.")
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Run path '{run_directory}' is not a directory.")

    config_path = run_directory / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run directory '{run_directory}' is missing '{config_path.name}'.")

    config = load_baseline_config(config_path)
    metadata = load_prepared_metadata(config.prepared_root)
    device = resolve_device(config.device)
    model = build_baseline_model(metadata, config).to(device)
    checkpoint_path = run_directory / "best.pt"
    checkpoint = _load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    canonical_split = canonicalize_split_name(split)
    if canonical_split not in {"val", "test"}:
        raise ValueError("Evaluation split must be either 'val' or 'test'.")

    dataset = PreparedGraphDataset(metadata, split=canonical_split, edge_mode=config.edge_mode)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    metrics = {
        "split": canonical_split,
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        **evaluate_model(
            model,
            loader,
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            split_name=canonical_split,
            log_every_batches=config.training.log_every_batches,
            use_amp=_use_cuda_amp(device, config),
            pin_memory=config.training.pin_memory,
        ),
    }
    _write_eval_file(run_directory, split=canonical_split, metrics=metrics)
    console_print(f"  Wrote evaluation summary : {run_directory / f'eval_{canonical_split}.json'}")
    return metrics


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GNN model from a prepared artifact cache")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["baseline", "pointer"],
        default="baseline",
        help="Which model type to train (baseline GraphSAGE or pointer argument selector)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the training JSON config (defaults to baseline or pointer config)",
    )
    parser.add_argument(
        "--prepared-root",
        type=str,
        default=None,
        help="Optional override for the prepared artifact root",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="Optional override for the run output root",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help="Resume an interrupted run from its existing run directory and last checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the number of training epochs",
    )
    return parser


def build_evaluate_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the saved baseline GNN checkpoint")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to a completed training run directory")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["val", "test"],
        help="Which split to evaluate with the best checkpoint",
    )
    return parser


def train_main(argv: list[str] | None = None) -> int:
    parser = build_train_arg_parser()
    args = parser.parse_args(argv)

    try:
        model_type = args.model_type.lower()
        
        if model_type == "baseline":
            config_path = args.config or DEFAULT_BASELINE_CONFIG_PATH
            if args.resume_run_dir:
                resume_config_path = Path(args.resume_run_dir) / "config.json"
                config = load_baseline_config(resume_config_path, epochs_override=args.epochs)
                train_baseline(config, resume_run_dir=args.resume_run_dir)
            else:
                config = load_baseline_config(
                    config_path,
                    prepared_root_override=args.prepared_root,
                    run_root_override=args.run_root,
                    epochs_override=args.epochs,
                )
                train_baseline(config)
        elif model_type == "pointer":
            config_path = args.config or DEFAULT_POINTER_CONFIG_PATH
            if args.resume_run_dir:
                resume_config_path = Path(args.resume_run_dir) / "config.json"
                config = load_pointer_config(resume_config_path, epochs_override=args.epochs)
                train_pointer(config, resume_run_dir=args.resume_run_dir)
            else:
                config = load_pointer_config(
                    config_path,
                    prepared_root_override=args.prepared_root,
                    run_root_override=args.run_root,
                    epochs_override=args.epochs,
                )
                train_pointer(config)
        else:
            console_print(f"  ERROR: Unknown model type '{model_type}'")
            return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    parser = build_evaluate_arg_parser()
    args = parser.parse_args(argv)

    try:
        evaluate_baseline_run(args.run_dir, split=args.split)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0
