"""Convert audited version-1 GNN checkpoints to the version-2 manifest format."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import checkpoint_payload
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import (
    build_actor_critic_model,
    build_pointer_model,
    build_supervised_tactic_model,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec


LAYOUT_MODEL_KINDS = {
    "graphsage_baseline": "supervised_tactic",
    "graphsage_pointer": "tactic_with_args",
    "ac_graphsage_actor_critic": "actor_critic_with_args",
    "gatv2_baseline": "supervised_tactic",
    "gatv2_pointer": "tactic_with_args",
}

MODEL_BUILDERS = {
    "supervised_tactic": build_supervised_tactic_model,
    "tactic_with_args": build_pointer_model,
    "actor_critic_with_args": build_actor_critic_model,
}

FEATURE_MODULES = {
    "label_embedding",
    "node_type_embedding",
    "is_bound_embedding",
    "binder_depth_embedding",
    "binder_kind_embedding",
}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file '{path}' must contain an object.")
    return payload


def _load_vocab(path: Path) -> dict[str, int]:
    payload = _read_json(path)
    return {str(key): int(value) for key, value in payload.items()}


def _model_spec_from_legacy_config(
    config: Mapping[str, object],
    *,
    layout: str,
) -> ModelSpec:
    raw_model = config.get("model", {})
    if not isinstance(raw_model, Mapping):
        raise ValueError("Legacy config field 'model' must be an object.")

    architecture = "gatv2" if layout.startswith("gatv2_") else "graphsage"
    hidden_dim = int(raw_model.get("hidden_dim", config.get("hidden_dim", 128)))
    dropout = float(raw_model.get("dropout", config.get("dropout", 0.2)))
    num_layers = int(raw_model.get("num_layers", config.get("num_layers", 4)))
    max_args = int(raw_model.get("max_args", config.get("max_args", 3)))
    use_node_type = bool(raw_model.get("use_node_type", config.get("use_node_type", True)))

    encoder: dict[str, object] = {"num_layers": num_layers}
    if architecture == "gatv2":
        encoder.update(
            heads=int(raw_model.get("heads", config.get("heads", 8))),
            readout=str(raw_model.get("readout", config.get("readout", "state"))),
        )
    return ModelSpec.from_dict(
        {
            "architecture": architecture,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "encoder": encoder,
            "use_node_type": use_node_type,
            "max_args": max_args,
        }
    )


def _representation_target(key: str) -> str | None:
    first = key.split(".", 1)[0]
    if first in FEATURE_MODULES:
        return f"encoder.node_features.{key}"
    if first in {"convs", "global_readout"}:
        return f"encoder.{key}"
    return None


def remap_legacy_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    layout: str,
) -> tuple[dict[str, Tensor], list[str]]:
    if layout not in LAYOUT_MODEL_KINDS:
        raise ValueError(f"Unsupported legacy checkpoint layout '{layout}'.")
    model_kind = LAYOUT_MODEL_KINDS[layout]
    remapped: dict[str, Tensor] = {}
    discarded: list[str] = []

    for old_key, value in state_dict.items():
        key = str(old_key)
        backbone_key = key.removeprefix("backbone.") if key.startswith("backbone.") else None
        representation_key = backbone_key if backbone_key is not None else key
        representation_target = _representation_target(representation_key)
        if representation_target is not None:
            new_key = representation_target
        elif representation_key.startswith("classifier."):
            if model_kind == "actor_critic_with_args":
                discarded.append(key)
                continue
            new_key = "tactic_classifier." + representation_key.removeprefix("classifier.")
        elif backbone_key is not None:
            raise ValueError(f"Unrecognized legacy backbone parameter '{key}'.")
        else:
            new_key = key

        if new_key in remapped:
            raise ValueError(f"Legacy parameters map to duplicate key '{new_key}'.")
        remapped[new_key] = value

    if model_kind == "actor_critic_with_args":
        for required in ("actor.base.weight", "actor.base.bias"):
            if required not in remapped:
                raise ValueError(
                    "AC actor-critic checkpoint is missing the trained actor base "
                    f"parameter '{required}'."
                )
    return remapped, discarded


def _verification_batch(num_node_labels: int) -> Batch:
    label_count = max(num_node_labels, 1)
    graphs: list[Data] = []
    for graph_index in range(2):
        data = Data(
            x=torch.tensor(
                [graph_index % label_count, (graph_index + 1) % label_count, 0],
                dtype=torch.long,
            ),
            node_type=torch.tensor([0, 1, 2], dtype=torch.long),
            edge_index=torch.tensor(
                [[0, 1, 1, 2], [1, 0, 2, 1]],
                dtype=torch.long,
            ),
            is_bound=torch.tensor([0, 1, 0], dtype=torch.long),
            binder_depth=torch.tensor([0, 1, 2], dtype=torch.long),
            binder_kind=torch.tensor([0, 1, 0], dtype=torch.long),
            premise_mask=torch.tensor([True, True, True], dtype=torch.bool),
        )
        graphs.append(data)
    batch = Batch.from_data_list(graphs)
    batch.state_node_index = batch.ptr[:-1].clone()
    return batch


def _assert_public_output_parity(model, *, model_kind: str, num_tactics: int) -> None:
    model.eval()
    batch = _verification_batch(model.encoder.node_features.label_embedding.num_embeddings)
    tactic_ids = torch.zeros(2, dtype=torch.long).clamp(max=max(num_tactics - 1, 0))

    with torch.no_grad():
        encoded = model.encoder(batch)
        if model_kind == "supervised_tactic":
            legacy_output = model.tactic_classifier(model.dropout(encoded.state_embeddings))
            current_output = model(batch)
            torch.testing.assert_close(current_output, legacy_output)
            return

        if model_kind == "tactic_with_args":
            legacy_tactics = model.tactic_classifier(
                model.tactic_dropout(encoded.state_embeddings)
            )
            tactic_embeddings = model.tactic_embedding(tactic_ids)
            legacy_args, _ = model.argument_selector(
                encoded.state_embeddings,
                tactic_embeddings,
                encoded.node_embeddings,
                batch.premise_mask,
                batch.batch,
            )
            current_tactics, current_args = model(
                batch,
                teacher_tactic_ids=tactic_ids,
                tactic_names=["exact", "exact"],
            )
            torch.testing.assert_close(current_tactics, legacy_tactics)
            if len(current_args) != 1:
                raise AssertionError("Pointer parity batch must produce one argument step.")
            torch.testing.assert_close(current_args[0], legacy_args)
            return

        legacy_tactics = model.actor(encoded.state_embeddings)
        legacy_values = model.critic(encoded.state_embeddings)
        tactic_embeddings = model.tactic_embedding(tactic_ids)
        legacy_args, _ = model.argument_selector(
            encoded.state_embeddings,
            tactic_embeddings,
            encoded.node_embeddings,
            batch.premise_mask,
            batch.batch,
        )
        current_tactics, current_values, current_args = model(
            batch,
            teacher_tactic_ids=tactic_ids,
            tactic_names=["exact", "exact"],
        )
        torch.testing.assert_close(current_tactics, legacy_tactics)
        torch.testing.assert_close(current_values, legacy_values)
        if len(current_args) != 1:
            raise AssertionError("Actor-critic parity batch must produce one argument step.")
        torch.testing.assert_close(current_args[0], legacy_args)


def migrate_checkpoint(
    *,
    checkpoint: Mapping[str, object],
    legacy_config: Mapping[str, object],
    layout: str,
    node_vocab: Mapping[str, int],
    tactic_vocab: Mapping[str, int],
) -> dict[str, object]:
    if "manifest" in checkpoint:
        raise ValueError("Checkpoint already contains a manifest; migration is not required.")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Legacy checkpoint is missing 'model_state_dict'.")

    model_kind = LAYOUT_MODEL_KINDS.get(layout)
    if model_kind is None:
        raise ValueError(f"Unsupported legacy checkpoint layout '{layout}'.")
    model_spec = _model_spec_from_legacy_config(legacy_config, layout=layout)
    model = MODEL_BUILDERS[model_kind](
        model_spec=model_spec,
        num_node_labels=len(node_vocab),
        num_tactics=len(tactic_vocab),
    )
    remapped, discarded = remap_legacy_state_dict(state_dict, layout=layout)
    model.load_state_dict(remapped, strict=True)
    _assert_public_output_parity(
        model,
        model_kind=model_kind,
        num_tactics=len(tactic_vocab),
    )

    retained_state = {
        str(key): value
        for key, value in checkpoint.items()
        if key
        not in {
            "manifest",
            "model_state_dict",
            "optimizer_state_dict",
            "torch_rng_state",
            "config",
            "migration",
        }
    }
    return checkpoint_payload(
        model_kind=model_kind,
        model_spec=model_spec,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        model=model,
        **retained_state,
        migration={
            "source_format_version": 1,
            "source_layout": layout,
            "discarded_parameters": discarded,
            "dropped_training_state": ["optimizer_state_dict", "torch_rng_state"],
            "public_output_parity_verified": True,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate an audited version-1 model checkpoint to version 2."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--node-vocab", type=Path)
    parser.add_argument("--tactic-vocab", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=sorted(LAYOUT_MODEL_KINDS), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _read_json(args.config)
    prepared_root = args.prepared_root
    if prepared_root is None and "prepared_root" in config:
        prepared_root = Path(str(config["prepared_root"]))
    node_vocab_path = args.node_vocab
    tactic_vocab_path = args.tactic_vocab
    if prepared_root is not None:
        node_vocab_path = node_vocab_path or prepared_root / "vocab" / "node_vocab.json"
        tactic_vocab_path = tactic_vocab_path or prepared_root / "vocab" / "tactic_vocab.json"
    if node_vocab_path is None or tactic_vocab_path is None:
        raise ValueError(
            "Provide --prepared-root or both --node-vocab and --tactic-vocab."
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Legacy checkpoint must contain a checkpoint object.")
    migrated = migrate_checkpoint(
        checkpoint=checkpoint,
        legacy_config=config,
        layout=args.layout,
        node_vocab=_load_vocab(node_vocab_path),
        tactic_vocab=_load_vocab(tactic_vocab_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.output)
    print(f"Wrote version-2 checkpoint to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
