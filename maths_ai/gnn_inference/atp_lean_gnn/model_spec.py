from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .architectures.registry import architecture_definition, normalize_encoder_config


@dataclass(frozen=True)
class ModelSpec:
    """Normalized architecture and shared policy-head dimensions."""

    architecture: str
    hidden_dim: int
    dropout: float
    encoder: Mapping[str, object]
    use_node_type: bool
    max_args: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelSpec":
        allowed = {
            "architecture",
            "hidden_dim",
            "dropout",
            "encoder",
            "use_node_type",
            "max_args",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown model fields: {', '.join(unknown)}.")
        if "architecture" not in payload:
            raise ValueError("Model config is missing required field 'architecture'.")

        architecture = str(payload["architecture"]).lower().strip()
        architecture_definition(architecture)
        hidden_dim = int(payload.get("hidden_dim", 128))
        dropout = float(payload.get("dropout", 0.2))
        use_node_type = bool(payload.get("use_node_type", True))
        max_args = int(payload.get("max_args", 3))
        raw_encoder = payload.get("encoder", {})
        if not isinstance(raw_encoder, Mapping):
            raise ValueError("Model config field 'encoder' must be an object.")
        encoder = normalize_encoder_config(architecture, raw_encoder)

        if hidden_dim < 1:
            raise ValueError("model.hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1).")
        if max_args < 1:
            raise ValueError("model.max_args must be positive.")
        if architecture == "gatv2" and hidden_dim % int(encoder["heads"]) != 0:
            raise ValueError(
                "model.hidden_dim must be divisible by model.encoder.heads for GATv2."
            )
        return cls(
            architecture=architecture,
            hidden_dim=hidden_dim,
            dropout=dropout,
            encoder=encoder,
            use_node_type=use_node_type,
            max_args=max_args,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "encoder": dict(self.encoder),
            "use_node_type": self.use_node_type,
            "max_args": self.max_args,
        }
