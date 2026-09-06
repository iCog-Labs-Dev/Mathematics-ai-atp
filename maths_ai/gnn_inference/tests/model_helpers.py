from __future__ import annotations

from maths_ai.gnn_inference.atp_lean_gnn.model_factory import (
    build_actor_critic_model,
    build_pointer_model,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec


def spec(*, hidden_dim: int = 16, num_layers: int = 2, dropout: float = 0.1, max_args: int = 2):
    return ModelSpec.from_dict(
        {
            "architecture": "graphsage",
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "encoder": {"num_layers": num_layers},
            "use_node_type": True,
            "max_args": max_args,
        }
    )


def pointer(num_node_labels: int, num_tactics: int, **kwargs):
    return build_pointer_model(
        model_spec=spec(**kwargs),
        num_node_labels=num_node_labels,
        num_tactics=num_tactics,
    )


def actor_critic(num_node_labels: int, num_tactics: int, **kwargs):
    return build_actor_critic_model(
        model_spec=spec(**kwargs),
        num_node_labels=num_node_labels,
        num_tactics=num_tactics,
    )
