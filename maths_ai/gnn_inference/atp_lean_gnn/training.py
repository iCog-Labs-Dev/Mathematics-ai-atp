from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from .batching import GraphBudgetBatchSampler, GraphSize
from .argument_training import (
    evaluate_model_with_args,
    train_one_epoch_with_args,
)
from .argument_selector import TacticWithArgsClassifier
from .dataset import CANONICAL_SPLITS, canonicalize_split_name
from .labels import UNKNOWN_TACTIC, get_tactic_arity
from .model import SupervisedTacticClassifier
from .model_factory import (
    build_actor_critic_model as create_actor_critic_model,
    build_pointer_model as create_pointer_model,
    build_supervised_tactic_model,
)
from .model_spec import ModelSpec
from .reporting import console_print
from .actor_critic import ActorCriticWithArgsClassifier, load_from_pointer_checkpoint
from .actor_critic_training import build_param_groups, train_one_epoch_actor_critic, evaluate_model_actor_critic
from .checkpointing import checkpoint_payload, validate_checkpoint_manifest
from .reward import MockRewardSource
from .training_safety import require_finite_loss, resolve_amp_dtype


DEFAULT_BASELINE_CONFIG_PATH = Path("configs") / "baseline_graphsage_state.json"
DEFAULT_POINTER_CONFIG_PATH = Path("configs") / "pointer_graphsage_state.json"
DEFAULT_ACTOR_CRITIC_CONFIG_PATH = Path("configs") / "actor_critic_graphsage_state.json"
REQUIRED_DATA_FIELDS = ("x", "node_type", "edge_index", "y", "split", "row_index", "tactic_name")
REQUIRED_POINTER_DATA_FIELDS = REQUIRED_DATA_FIELDS + ("premise_mask", "arg_node_indices")


def _default_model_spec() -> ModelSpec:
    return ModelSpec.from_dict(
        {
            "architecture": "graphsage",
            "hidden_dim": 128,
            "dropout": 0.2,
            "encoder": {"num_layers": 4},
            "use_node_type": True,
            "max_args": 3,
        }
    )


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
    max_batch_nodes: int = 0
    max_batch_edges: int = 0
    oversize_graph_policy: str = "error"
    cache_in_memory: bool = False

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
            "max_batch_nodes": self.max_batch_nodes,
            "max_batch_edges": self.max_batch_edges,
            "oversize_graph_policy": self.oversize_graph_policy,
            "cache_in_memory": self.cache_in_memory,
        }


def _validate_training_loop(training: TrainingLoopConfig) -> None:
    if training.batch_size < 1:
        raise ValueError("training.batch_size must be positive.")
    if training.epochs < 1:
        raise ValueError("training.epochs must be positive.")
    if training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive.")
    if training.weight_decay < 0:
        raise ValueError("training.weight_decay cannot be negative.")
    if training.grad_clip <= 0:
        raise ValueError("training.grad_clip must be positive.")
    if training.log_every_batches < 1:
        raise ValueError("training.log_every_batches must be positive.")
    if training.num_workers < 0:
        raise ValueError("training.num_workers cannot be negative.")
    if training.prefetch_factor < 1:
        raise ValueError("training.prefetch_factor must be positive.")
    if training.max_batch_nodes < 0 or training.max_batch_edges < 0:
        raise ValueError("training graph budgets cannot be negative.")
    if training.oversize_graph_policy not in {"error", "skip", "singleton"}:
        raise ValueError(
            "training.oversize_graph_policy must be one of: error, skip, singleton."
        )


@dataclass(frozen=True)
class BaselineConfig:
    prepared_root: Path
    run_root: Path
    seed: int = 42
    device: str = "auto"
    edge_mode: str = "bidirectional"
    model: ModelSpec = field(default_factory=_default_model_spec)
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
            model=ModelSpec.from_dict(model_payload),
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
                max_batch_nodes=int(training_payload.get("max_batch_nodes", 0)),
                max_batch_edges=int(training_payload.get("max_batch_edges", 0)),
                oversize_graph_policy=str(
                    training_payload.get("oversize_graph_policy", "error")
                ).lower().strip(),
                cache_in_memory=bool(training_payload.get("cache_in_memory", False)),
            ),
        ).normalized()

    def normalized(self) -> "BaselineConfig":
        edge_mode = self.edge_mode.lower().strip()
        if edge_mode not in {"forward", "bidirectional"}:
            raise ValueError("Training config field 'edge_mode' must be either 'forward' or 'bidirectional'.")

        device = self.device.lower().strip()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Training config field 'device' must be one of: auto, cpu, cuda.")

        _validate_training_loop(self.training)

        return BaselineConfig(
            prepared_root=self.prepared_root.resolve(),
            run_root=self.run_root.resolve(),
            seed=self.seed,
            device=device,
            edge_mode=edge_mode,
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
    arg_loss_weight: float = 0.5
    model: ModelSpec = field(default_factory=_default_model_spec)
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
            arg_loss_weight=float(payload.get("arg_loss_weight", 0.5)),
            model=ModelSpec.from_dict(model_payload),
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
                max_batch_nodes=int(training_payload.get("max_batch_nodes", 0)),
                max_batch_edges=int(training_payload.get("max_batch_edges", 0)),
                oversize_graph_policy=str(
                    training_payload.get("oversize_graph_policy", "error")
                ).lower().strip(),
                cache_in_memory=bool(training_payload.get("cache_in_memory", False)),
            ),
        ).normalized()

    def normalized(self) -> "PointerConfig":
        edge_mode = self.edge_mode.lower().strip()
        if edge_mode not in {"forward", "bidirectional"}:
            raise ValueError("Training config field 'edge_mode' must be either 'forward' or 'bidirectional'.")

        device = self.device.lower().strip()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Training config field 'device' must be one of: auto, cpu, cuda.")

        if self.arg_loss_weight < 0:
            raise ValueError("Training config field 'arg_loss_weight' cannot be negative.")
        _validate_training_loop(self.training)

        return PointerConfig(
            prepared_root=self.prepared_root.resolve(),
            run_root=self.run_root.resolve(),
            seed=self.seed,
            device=device,
            edge_mode=edge_mode,
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
            "arg_loss_weight": self.arg_loss_weight,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }


@dataclass(frozen=True)
class ActorCriticConfig:
    prepared_root: Path
    run_root: Path
    seed: int = 42
    device: str = "auto"
    edge_mode: str = "bidirectional"
    arg_loss_weight: float = 0.5
    critic_weight: float = 0.5
    entropy_weight: float = 0.01
    arg_lr_multiplier: float = 0.1
    pretrained_pointer_checkpoint: str | None = None
    model: ModelSpec = field(default_factory=_default_model_spec)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActorCriticConfig":
        if "prepared_root" not in payload:
            raise ValueError("Training config is missing the required 'prepared_root' field.")

        model_payload = payload.get("model", {})
        training_payload = payload.get("training", {})
        return cls(
            prepared_root=Path(payload["prepared_root"]),
            run_root=Path(payload.get("run_root", "runs/actor_critic_gnn")),
            seed=int(payload.get("seed", 42)),
            device=str(payload.get("device", "auto")),
            edge_mode=str(payload.get("edge_mode", "bidirectional")),
            arg_loss_weight=float(payload.get("arg_loss_weight", 0.5)),
            critic_weight=float(payload.get("critic_weight", 0.5)),
            entropy_weight=float(payload.get("entropy_weight", 0.01)),
            arg_lr_multiplier=float(payload.get("arg_lr_multiplier", 0.1)),
            pretrained_pointer_checkpoint=payload.get("pretrained_pointer_checkpoint"),
            model=ModelSpec.from_dict(model_payload),
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
                max_batch_nodes=int(training_payload.get("max_batch_nodes", 0)),
                max_batch_edges=int(training_payload.get("max_batch_edges", 0)),
                oversize_graph_policy=str(
                    training_payload.get("oversize_graph_policy", "error")
                ).lower().strip(),
                cache_in_memory=bool(training_payload.get("cache_in_memory", False)),
            ),
        ).normalized()

    def normalized(self) -> "ActorCriticConfig":
        edge_mode = self.edge_mode.lower().strip()
        if edge_mode not in {"forward", "bidirectional"}:
            raise ValueError("Training config field 'edge_mode' must be either 'forward' or 'bidirectional'.")

        device = self.device.lower().strip()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Training config field 'device' must be one of: auto, cpu, cuda.")

        if self.arg_loss_weight < 0:
            raise ValueError("Training config field 'arg_loss_weight' cannot be negative.")
        if self.critic_weight < 0:
            raise ValueError("Training config field 'critic_weight' cannot be negative.")
        if self.entropy_weight < 0:
            raise ValueError("Training config field 'entropy_weight' cannot be negative.")
        if self.arg_lr_multiplier < 0:
            raise ValueError("Training config field 'arg_lr_multiplier' cannot be negative.")
        _validate_training_loop(self.training)

        return ActorCriticConfig(
            prepared_root=self.prepared_root.resolve(),
            run_root=self.run_root.resolve(),
            seed=self.seed,
            device=device,
            edge_mode=edge_mode,
            arg_loss_weight=self.arg_loss_weight,
            critic_weight=self.critic_weight,
            entropy_weight=self.entropy_weight,
            arg_lr_multiplier=self.arg_lr_multiplier,
            pretrained_pointer_checkpoint=self.pretrained_pointer_checkpoint,
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
            "arg_loss_weight": self.arg_loss_weight,
            "critic_weight": self.critic_weight,
            "entropy_weight": self.entropy_weight,
            "arg_lr_multiplier": self.arg_lr_multiplier,
            "pretrained_pointer_checkpoint": self.pretrained_pointer_checkpoint,
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


def load_actor_critic_config(
    config_path: str | Path = DEFAULT_ACTOR_CRITIC_CONFIG_PATH,
    *,
    prepared_root_override: str | Path | None = None,
    run_root_override: str | Path | None = None,
    epochs_override: int | None = None,
) -> ActorCriticConfig:
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
    return ActorCriticConfig.from_dict(payload)


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
        io_threads: int = 0,
        cache_in_memory: bool = False,
    ) -> None:
        self.metadata = metadata
        self.split = canonicalize_split_name(split)
        self.edge_mode = edge_mode
        self.required_fields = required_fields
        self.io_threads = max(0, int(io_threads))
        self.cache_in_memory = bool(cache_in_memory)
        self._thread_pool: ThreadPoolExecutor | None = None
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
        self._cache = [None] * len(self.files) if self.cache_in_memory else None
        self.packed_cache_loaded = False
        self.graph_size_source = "unresolved"

    def _packed_manifest_path(self) -> Path:
        return self.metadata.root / "packed" / self.edge_mode / "manifest.json"

    def _load_packed_cache(self) -> bool:
        manifest_path = self._packed_manifest_path()
        if not manifest_path.exists():
            return False
        manifest = _read_json(manifest_path)
        if str(manifest.get("edge_mode", "")) != self.edge_mode:
            raise ValueError(
                f"Packed cache manifest '{manifest_path}' has edge_mode="
                f"'{manifest.get('edge_mode')}', expected '{self.edge_mode}'."
            )
        split_payload = dict(manifest.get("splits", {})).get(self.split)
        if not isinstance(split_payload, dict):
            return False
        if int(split_payload.get("count", -1)) != len(self):
            return False
        chunk_names = split_payload.get("chunks", [])
        if not isinstance(chunk_names, list) or not chunk_names:
            return False

        packed_root = manifest_path.parent / self.split
        offset = 0
        for chunk_name in chunk_names:
            chunk_path = packed_root / str(chunk_name)
            if not chunk_path.exists():
                return False
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
            if not isinstance(chunk, list):
                raise ValueError(f"Packed graph chunk '{chunk_path}' must contain a list.")
            end = offset + len(chunk)
            if end > len(self._cache):
                raise ValueError(f"Packed graph chunk '{chunk_path}' exceeds the split size.")
            for cache_index, data in enumerate(chunk, start=offset):
                source_path = self.files[cache_index]
                self._cache[cache_index] = self._normalize_data(data, path=source_path)
            offset = end
        if offset != len(self._cache):
            raise ValueError(
                f"Packed cache for split '{self.split}' loaded {offset} examples, "
                f"expected {len(self._cache)}."
            )
        return True

    def _normalize_data(self, data, *, path: Path):
        validate_prepared_data(
            data,
            path=path,
            split=self.split,
            required_fields=self.required_fields,
        )
        data.x = data.x.to(dtype=torch.long)
        data.node_type = data.node_type.to(dtype=torch.long)
        if not hasattr(data, "state_node_index"):
            data.state_node_index = infer_state_node_index(
                data,
                state_label_id=self.metadata.state_label_id,
                path=path,
            )
        data.edge_index = transform_edge_index(data.edge_index, edge_mode=self.edge_mode)
        data.y = data.y.view(-1).to(dtype=torch.long)
        return data

    def _load_file(self, index: int):
        path = self.files[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        return self._normalize_data(data, path=path)

    def _sidecar_graph_sizes(self) -> list[GraphSize] | None:
        sizes: list[GraphSize] = []
        edge_field = f"edges_{self.edge_mode}"
        for path in self.files:
            sidecar = path.with_suffix(".size.json")
            if not sidecar.exists():
                return None
            try:
                payload = _read_json(sidecar)
                nodes = int(payload["nodes"])
                edges = int(payload[edge_field])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return None
            if nodes < 0 or edges < 0:
                return None
            sizes.append(
                GraphSize(
                    dataset_id=str(payload.get("dataset_id", path.stem)),
                    nodes=nodes,
                    edges=edges,
                )
            )
        return sizes

    def graph_sizes(self) -> list[GraphSize]:
        sidecar_sizes = self._sidecar_graph_sizes()
        if sidecar_sizes is not None:
            self.graph_size_source = "sidecars"
            return sidecar_sizes

        if self._cache is not None and not self.packed_cache_loaded:
            self.packed_cache_loaded = self._load_packed_cache()
        if self._cache is not None and all(data is not None for data in self._cache):
            self.graph_size_source = "packed_cache" if self.packed_cache_loaded else "memory_cache"
            return [
                GraphSize(
                    dataset_id=self.files[index].stem,
                    nodes=int(data.num_nodes),
                    edges=int(data.edge_index.size(1)),
                )
                for index, data in enumerate(self._cache)
            ]

        console_print(
            f"  [warn] {self.split}: graph-size sidecars are unavailable; "
            "scanning individual PyG .pt files. Build the packed cache to avoid "
            "this startup scan on repeated runs."
        )
        sizes: list[GraphSize] = []
        for index, path in enumerate(self.files):
            try:
                data = self._load_file(index)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"Cannot derive graph size from prepared graph '{path}'. "
                    f"Build a packed cache at '{self._packed_manifest_path()}' or repair "
                    f"the graph artifact. Cause: {exc}"
                ) from exc
            if self._cache is not None:
                self._cache[index] = data
            sizes.append(
                GraphSize(
                    dataset_id=path.stem,
                    nodes=int(data.num_nodes),
                    edges=int(data.edge_index.size(1)),
                )
            )
        self.graph_size_source = "pt_scan"
        return sizes

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        if self._cache is not None and self._cache[index] is not None:
            return self._cache[index]
        data = self._load_file(index)
        if self._cache is not None:
            self._cache[index] = data
        return data

    def __getitems__(self, indices: list[int]):
        if self.io_threads <= 1 or len(indices) <= 1:
            return [self[index] for index in indices]
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.io_threads,
                thread_name_prefix=f"gnn-{self.split}-io",
            )
        return list(self._thread_pool.map(self.__getitem__, indices))


def build_dataloaders(
    metadata: PreparedMetadata,
    config: BaselineConfig | PointerConfig | ActorCriticConfig,
    required_fields: tuple[str, ...] = REQUIRED_DATA_FIELDS,
) -> tuple[dict[str, PreparedGraphDataset], dict[str, DataLoader]]:
    requested_workers = config.training.num_workers
    if config.training.cache_in_memory and requested_workers > 0:
        num_workers = 0
        io_threads = requested_workers
        console_print(
            f"  [info] cache_in_memory=true: using {io_threads} in-process I/O "
            "threads instead of DataLoader worker processes."
        )
    else:
        num_workers = requested_workers
        io_threads = 0
    use_workers = num_workers > 0
    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers,
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
            io_threads=io_threads,
            cache_in_memory=config.training.cache_in_memory,
        )
        for split in CANONICAL_SPLITS
    }
    if config.training.max_batch_nodes or config.training.max_batch_edges:
        samplers = {
            split: GraphBudgetBatchSampler(
                dataset.graph_sizes(),
                max_graphs=config.training.batch_size,
                max_nodes=config.training.max_batch_nodes,
                max_edges=config.training.max_batch_edges,
                oversize_policy=config.training.oversize_graph_policy,
                shuffle=(split == "train"),
                seed=config.seed,
            )
            for split, dataset in datasets.items()
        }
        for split, sampler in samplers.items():
            if sampler.oversize_indices:
                largest = max(
                    (sampler.graph_sizes[index] for index in sampler.oversize_indices),
                    key=lambda size: (size.nodes, size.edges),
                )
                console_print(
                    f"  [warn] {split}: {len(sampler.oversize_indices)} oversized graphs; "
                    f"policy={sampler.oversize_policy}, largest={largest.nodes} nodes/"
                    f"{largest.edges} edges."
                )
        loaders = {
            split: DataLoader(dataset, batch_sampler=samplers[split], **loader_kwargs)
            for split, dataset in datasets.items()
        }
    else:
        loaders = {
            split: DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=(split == "train"),
                **loader_kwargs,
            )
            for split, dataset in datasets.items()
        }
    return datasets, loaders


def _log_batching_settings(
    config: BaselineConfig | PointerConfig | ActorCriticConfig,
    datasets: dict[str, PreparedGraphDataset],
    loaders: dict[str, DataLoader],
) -> None:
    console_print(
        f"  Graph batching           : max_graphs={config.training.batch_size}, "
        f"max_nodes={config.training.max_batch_nodes or 'unlimited'}, "
        f"max_edges={config.training.max_batch_edges or 'unlimited'}, "
        f"oversize_policy={config.training.oversize_graph_policy}, "
        f"cache_in_memory={config.training.cache_in_memory}"
    )
    for split in CANONICAL_SPLITS:
        source = datasets[split].graph_size_source
        cache = datasets[split].packed_cache_loaded
        console_print(
            f"  {split} batching          : batches={len(loaders[split])}, "
            f"size_source={source}, packed_cache={cache}"
        )


def _batching_summary(
    config: BaselineConfig | PointerConfig | ActorCriticConfig,
    datasets: dict[str, PreparedGraphDataset],
    loaders: dict[str, DataLoader],
) -> dict[str, object]:
    return {
        "max_graphs": config.training.batch_size,
        "max_nodes": config.training.max_batch_nodes,
        "max_edges": config.training.max_batch_edges,
        "oversize_graph_policy": config.training.oversize_graph_policy,
        "cache_in_memory": config.training.cache_in_memory,
        "splits": {
            split: {
                "batch_count": len(loaders[split]),
                "graph_size_source": datasets[split].graph_size_source,
                "packed_cache_loaded": datasets[split].packed_cache_loaded,
                "oversize_graph_count": len(
                    getattr(loaders[split].batch_sampler, "oversize_indices", [])
                ),
            }
            for split in CANONICAL_SPLITS
        },
    }


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


def build_baseline_model(metadata: PreparedMetadata, config: BaselineConfig) -> SupervisedTacticClassifier:
    return build_supervised_tactic_model(
        model_spec=config.model,
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
    )


def build_pointer_model(metadata: PreparedMetadata, config: PointerConfig) -> TacticWithArgsClassifier:
    return create_pointer_model(
        model_spec=config.model,
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
    )


def build_actor_critic_model(metadata: PreparedMetadata, config: ActorCriticConfig) -> ActorCriticWithArgsClassifier:
    return create_actor_critic_model(
        model_spec=config.model,
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
    )


def _amp_dtype(
    device: torch.device,
    config: BaselineConfig | PointerConfig | ActorCriticConfig,
) -> torch.dtype | None:
    return resolve_amp_dtype(
        architecture=config.model.architecture,
        device=device,
        requested=config.training.use_amp,
    )


def _use_cuda_amp(
    device: torch.device,
    config: BaselineConfig | PointerConfig | ActorCriticConfig,
) -> bool:
    return _amp_dtype(device, config) is not None


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
    model: SupervisedTacticClassifier,
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
    amp_dtype: torch.dtype | None = None,
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
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            logits = model(batch)
            loss = F.cross_entropy(logits, targets)

        require_finite_loss(
            loss,
            architecture=model.model_spec.architecture,
            amp_dtype=amp_dtype,
            epoch=epoch,
            batch_index=batch_index,
        )

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
    model: SupervisedTacticClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    unknown_tactic_id: int,
    split_name: str | None = None,
    log_every_batches: int | None = None,
    use_amp: bool = False,
    pin_memory: bool = False,
    amp_dtype: torch.dtype | None = None,
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
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=amp_dtype,
            ):
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
    model: SupervisedTacticClassifier | TacticWithArgsClassifier | ActorCriticWithArgsClassifier,
    optimizer: AdamW,
    config: BaselineConfig | PointerConfig | ActorCriticConfig,
    metadata: PreparedMetadata,
    epoch: int,
    val_metrics: dict[str, float | int],
) -> Path:
    if isinstance(config, BaselineConfig):
        model_kind = "supervised_tactic"
    elif isinstance(config, PointerConfig):
        model_kind = "tactic_with_args"
    else:
        model_kind = "actor_critic_with_args"
    torch.save(
        checkpoint_payload(
            model_kind=model_kind,
            model_spec=config.model,
            node_vocab=metadata.node_vocab,
            tactic_vocab=metadata.tactic_vocab,
            model=model,
            epoch=epoch,
            config=config.to_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            val_metrics=val_metrics,
        ),
        path,
    )
    return path


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    metadata: PreparedMetadata,
    expected_model_kind: str,
    expected_model_spec: ModelSpec,
) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _, checkpoint_spec = validate_checkpoint_manifest(
        checkpoint,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        expected_model_kind=expected_model_kind,
    )
    if checkpoint_spec != expected_model_spec:
        raise ValueError("Checkpoint model specification does not match the run config.")
    return checkpoint


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
    amp_dtype = _amp_dtype(device, config)
    use_amp = amp_dtype is not None
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
    grad_scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_dtype == torch.float16,
    )

    if resume_run_dir is not None:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume run directory '{run_dir}' is missing 'last.pt', so training cannot resume."
            )
        last_checkpoint = _load_checkpoint(
            last_checkpoint_path,
            device=device,
            metadata=metadata,
            expected_model_kind="supervised_tactic",
            expected_model_spec=config.model,
        )
        model.load_state_dict(last_checkpoint["model_state_dict"])
        optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
        start_epoch = int(last_checkpoint["epoch"]) + 1
        if best_checkpoint_path.exists():
            best_checkpoint = _load_checkpoint(
                best_checkpoint_path,
                device=device,
                metadata=metadata,
                expected_model_kind="supervised_tactic",
                expected_model_spec=config.model,
            )
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
    _log_batching_settings(config, datasets, loaders)
    if resume_run_dir is not None:
        console_print(
            f"  Resuming from checkpoint : {last_checkpoint_path} "
            f"(next epoch {start_epoch})"
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        batch_sampler = getattr(loaders["train"], "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
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
            amp_dtype=amp_dtype,
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
            amp_dtype=amp_dtype,
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
            metadata=metadata,
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
                metadata=metadata,
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

    best_checkpoint = _load_checkpoint(
        best_checkpoint_path,
        device=device,
        metadata=metadata,
        expected_model_kind="supervised_tactic",
        expected_model_spec=config.model,
    )
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
            amp_dtype=amp_dtype,
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
            amp_dtype=amp_dtype,
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
        "batching": _batching_summary(config, datasets, loaders),
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
    amp_dtype = _amp_dtype(device, config)
    use_amp = amp_dtype is not None
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
    grad_scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_dtype == torch.float16,
    )

    if resume_run_dir is not None:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume run directory '{run_dir}' is missing 'last.pt', so training cannot resume."
            )
        last_checkpoint = _load_checkpoint(
            last_checkpoint_path,
            device=device,
            metadata=metadata,
            expected_model_kind="tactic_with_args",
            expected_model_spec=config.model,
        )
        model.load_state_dict(last_checkpoint["model_state_dict"])
        optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
        start_epoch = int(last_checkpoint["epoch"]) + 1
        if best_checkpoint_path.exists():
            best_checkpoint = _load_checkpoint(
                best_checkpoint_path,
                device=device,
                metadata=metadata,
                expected_model_kind="tactic_with_args",
                expected_model_spec=config.model,
            )
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
    _log_batching_settings(config, datasets, loaders)
    console_print(f"  Max args per step        : {config.model.max_args}")
    console_print(f"  Argument loss weight     : {config.arg_loss_weight}")
    if resume_run_dir is not None:
        console_print(
            f"  Resuming from checkpoint : {last_checkpoint_path} "
            f"(next epoch {start_epoch})"
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        batch_sampler = getattr(loaders["train"], "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
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
            amp_dtype=amp_dtype,
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
            amp_dtype=amp_dtype,
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
            metadata=metadata,
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
                metadata=metadata,
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

    best_checkpoint = _load_checkpoint(
        best_checkpoint_path,
        device=device,
        metadata=metadata,
        expected_model_kind="tactic_with_args",
        expected_model_spec=config.model,
    )
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
            amp_dtype=amp_dtype,
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
            amp_dtype=amp_dtype,
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
        "batching": _batching_summary(config, datasets, loaders),
        "best_validation": eval_val,
        "test_evaluation": eval_test,
    }
    _write_json(run_dir / "summary.json", summary)

    console_print(f"\n  Best checkpoint          : {best_checkpoint_path}")
    console_print(f"  Validation eval summary  : {run_dir / 'eval_val.json'}")
    console_print(f"  Test eval summary        : {run_dir / 'eval_test.json'}")
    console_print(f"  Training summary         : {run_dir / 'summary.json'}")

    return summary


def train_actor_critic(
    config: ActorCriticConfig,
    *,
    resume_run_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train actor-critic GNN tactic predictor via Actor-Critic RL."""
    metadata = load_prepared_metadata(config.prepared_root)
    set_seed(config.seed)
    device = resolve_device(config.device)
    amp_dtype = _amp_dtype(device, config)
    use_amp = amp_dtype is not None
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

    model = build_actor_critic_model(metadata, config).to(device)

    if config.pretrained_pointer_checkpoint is not None:
        console_print(f"  Loading pretrained pointer checkpoint: {config.pretrained_pointer_checkpoint}")
        load_from_pointer_checkpoint(
            model,
            config.pretrained_pointer_checkpoint,
            device,
            node_vocab=metadata.node_vocab,
            tactic_vocab=metadata.tactic_vocab,
        )

    param_groups = build_param_groups(model, config.training.learning_rate, config.arg_lr_multiplier)
    optimizer = AdamW(
        param_groups,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    grad_scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_dtype == torch.float16,
    )

    if resume_run_dir is not None:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume run directory '{run_dir}' is missing 'last.pt', so training cannot resume."
            )
        last_checkpoint = _load_checkpoint(
            last_checkpoint_path,
            device=device,
            metadata=metadata,
            expected_model_kind="actor_critic_with_args",
            expected_model_spec=config.model,
        )
        model.load_state_dict(last_checkpoint["model_state_dict"])
        optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
        start_epoch = int(last_checkpoint["epoch"]) + 1
        if best_checkpoint_path.exists():
            best_checkpoint = _load_checkpoint(
                best_checkpoint_path,
                device=device,
                metadata=metadata,
                expected_model_kind="actor_critic_with_args",
                expected_model_spec=config.model,
            )
            best_epoch = int(best_checkpoint["epoch"])
            best_val_loss = float(
                dict(best_checkpoint.get("val_metrics", {})).get("total_loss", float("inf"))
            )

    console_print(f"\n  Training Actor-Critic run in : {run_dir}")
    console_print(f"  Prepared cache              : {config.prepared_root}")
    console_print(f"  Device                      : {device}")
    console_print(f"  AMP enabled                 : {use_amp}")
    console_print(
        f"  Split sizes                 : train={len(datasets['train'])}, "
        f"val={len(datasets['val'])}, test={len(datasets['test'])}"
    )
    _log_batching_settings(config, datasets, loaders)

    reward_source = MockRewardSource()

    for epoch in range(start_epoch, config.training.epochs + 1):
        batch_sampler = getattr(loaders["train"], "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch_actor_critic(
            model,
            loaders["train"],
            reward_source=reward_source,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            device=device,
            grad_clip=config.training.grad_clip,
            epoch=epoch,
            total_epochs=config.training.epochs,
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
            amp_dtype=amp_dtype,
            critic_weight=config.critic_weight,
            entropy_weight=config.entropy_weight,
            arg_loss_weight=config.arg_loss_weight,
        )
        val_metrics = evaluate_model_actor_critic(
            model,
            loaders["val"],
            reward_source=reward_source,
            device=device,
            split_name="val",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
            amp_dtype=amp_dtype,
            critic_weight=config.critic_weight,
            entropy_weight=config.entropy_weight,
            arg_loss_weight=config.arg_loss_weight,
        )

        _append_jsonl(metrics_path, {"epoch": epoch, "train": train_metrics, "val": val_metrics})
        _save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
            metadata=metadata,
            epoch=epoch,
            val_metrics=val_metrics,
        )

        val_loss = val_metrics["total_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            _save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                metadata=metadata,
                epoch=epoch,
                val_metrics=val_metrics,
            )
            console_print(f"    --> Saved new best checkpoint (epoch={epoch}, total_loss={val_loss:.4f})")

    # Evaluate test split with best checkpoint
    console_print(f"\n  Training completed. Loading best checkpoint from epoch {best_epoch} for test evaluation...")
    best_checkpoint = _load_checkpoint(
        best_checkpoint_path,
        device=device,
        metadata=metadata,
        expected_model_kind="actor_critic_with_args",
        expected_model_spec=config.model,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate_model_actor_critic(
        model,
        loaders["test"],
        reward_source=reward_source,
        device=device,
        split_name="test",
        log_every_batches=config.training.log_every_batches,
        use_amp=use_amp,
        pin_memory=config.training.pin_memory,
        amp_dtype=amp_dtype,
        critic_weight=config.critic_weight,
        entropy_weight=config.entropy_weight,
        arg_loss_weight=config.arg_loss_weight,
    )
    _write_eval_file(run_dir, split="test", metrics={"split": "test", "epoch": best_epoch, **test_metrics})
    _write_eval_file(run_dir, split="val", metrics={"split": "val", "epoch": best_epoch, **val_metrics})

    summary = {
        "best_epoch": best_epoch,
        "batching": _batching_summary(config, datasets, loaders),
        "best_val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    _write_json(run_dir / "summary.json", summary)

    console_print(f"  Validation eval summary     : {run_dir / 'eval_val.json'}")
    console_print(f"  Test eval summary           : {run_dir / 'eval_test.json'}")
    console_print(f"  Training summary            : {run_dir / 'summary.json'}")
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
    amp_dtype = _amp_dtype(device, config)
    model = build_baseline_model(metadata, config).to(device)
    checkpoint_path = run_directory / "best.pt"
    checkpoint = _load_checkpoint(
        checkpoint_path,
        device=device,
        metadata=metadata,
        expected_model_kind="supervised_tactic",
        expected_model_spec=config.model,
    )
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
            use_amp=amp_dtype is not None,
            pin_memory=config.training.pin_memory,
            amp_dtype=amp_dtype,
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
        choices=["baseline", "pointer", "actor_critic"],
        default="baseline",
        help="Which model type to train (baseline, pointer argument selector, or actor critic)",
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
        elif model_type == "actor_critic":
            config_path = args.config or DEFAULT_ACTOR_CRITIC_CONFIG_PATH
            if args.resume_run_dir:
                resume_config_path = Path(args.resume_run_dir) / "config.json"
                config = load_actor_critic_config(resume_config_path, epochs_override=args.epochs)
                train_actor_critic(config, resume_run_dir=args.resume_run_dir)
            else:
                config = load_actor_critic_config(
                    config_path,
                    prepared_root_override=args.prepared_root,
                    run_root_override=args.run_root,
                    epochs_override=args.epochs,
                )
                train_actor_critic(config)
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
