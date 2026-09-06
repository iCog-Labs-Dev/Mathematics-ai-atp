from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from ..pyg import NODE_TYPE_TO_ID
from .base import StateGraphEncoder
from .gatv2 import GATv2Encoder, VALID_GATV2_READOUTS
from .graphsage import GraphSAGEEncoder


@dataclass(frozen=True)
class ArchitectureDefinition:
    name: str
    version: int
    config_fields: frozenset[str]
    normalize_config: Callable[[Mapping[str, object]], dict[str, object]]
    build: Callable[..., StateGraphEncoder]


def _reject_unknown(config: Mapping[str, object], allowed: frozenset[str], name: str) -> None:
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {name} encoder fields: {', '.join(unknown)}.")


def _normalize_graphsage(config: Mapping[str, object]) -> dict[str, object]:
    allowed = frozenset({"num_layers"})
    _reject_unknown(config, allowed, "graphsage")
    num_layers = int(config.get("num_layers", 4))
    if num_layers < 1:
        raise ValueError("model.encoder.num_layers must be positive.")
    return {"num_layers": num_layers}


def _normalize_gatv2(config: Mapping[str, object]) -> dict[str, object]:
    allowed = frozenset({"num_layers", "heads", "readout"})
    _reject_unknown(config, allowed, "gatv2")
    num_layers = int(config.get("num_layers", 4))
    heads = int(config.get("heads", 8))
    readout = str(config.get("readout", "state")).lower().strip()
    if num_layers < 1:
        raise ValueError("model.encoder.num_layers must be positive.")
    if heads < 1:
        raise ValueError("model.encoder.heads must be positive.")
    if readout not in VALID_GATV2_READOUTS:
        raise ValueError(
            "model.encoder.readout must be one of: "
            f"{', '.join(VALID_GATV2_READOUTS)}."
        )
    return {"num_layers": num_layers, "heads": heads, "readout": readout}


ARCHITECTURES: dict[str, ArchitectureDefinition] = {
    "graphsage": ArchitectureDefinition(
        name="graphsage",
        version=GraphSAGEEncoder.architecture_version,
        config_fields=frozenset({"num_layers"}),
        normalize_config=_normalize_graphsage,
        build=GraphSAGEEncoder,
    ),
    "gatv2": ArchitectureDefinition(
        name="gatv2",
        version=GATv2Encoder.architecture_version,
        config_fields=frozenset({"num_layers", "heads", "readout"}),
        normalize_config=_normalize_gatv2,
        build=GATv2Encoder,
    ),
}


def architecture_definition(name: str) -> ArchitectureDefinition:
    normalized = name.lower().strip()
    try:
        return ARCHITECTURES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model architecture '{name}'. Supported architectures: "
            f"{', '.join(sorted(ARCHITECTURES))}."
        ) from exc


def normalize_encoder_config(
    architecture: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    return architecture_definition(architecture).normalize_config(config)


def build_encoder(
    *,
    architecture: str,
    encoder_config: Mapping[str, object],
    num_node_labels: int,
    hidden_dim: int,
    dropout: float,
    use_node_type: bool,
    num_node_types: int = len(NODE_TYPE_TO_ID),
    num_binder_kinds: int = 6,
    max_binder_depth: int = 10,
) -> StateGraphEncoder:
    definition = architecture_definition(architecture)
    normalized_config = definition.normalize_config(encoder_config)
    return definition.build(
        num_node_labels=num_node_labels,
        num_node_types=num_node_types,
        num_binder_kinds=num_binder_kinds,
        max_binder_depth=max_binder_depth,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_node_type=use_node_type,
        **normalized_config,
    )


def amp_dtype_for_architecture(
    architecture: str,
    device: torch.device,
) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if architecture == "gatv2":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    return torch.float16
