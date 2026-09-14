from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EncoderOutput:
    """Node and graph representations produced by a state-graph encoder."""

    node_embeddings: Tensor
    state_embeddings: Tensor
    details: Mapping[str, Tensor] = field(default_factory=dict)


class StateGraphEncoder(nn.Module, ABC):
    """Common contract for encoders consumed by tactic and argument heads."""

    output_dim: int

    @abstractmethod
    def forward(self, data) -> EncoderOutput:
        """Encode a canonical PyTorch Geometric proof-state batch."""


class NodeFeatureEmbedding(nn.Module):
    """Shared embedding of graph-node labels, types, and binder attributes."""

    def __init__(
        self,
        *,
        num_node_labels: int,
        hidden_dim: int,
        num_node_types: int,
        num_binder_kinds: int,
        max_binder_depth: int,
        use_node_type: bool,
    ) -> None:
        super().__init__()
        if max_binder_depth < 1:
            raise ValueError("max_binder_depth must be positive.")

        self.label_embedding = nn.Embedding(num_node_labels, hidden_dim)
        self.node_type_embedding = (
            nn.Embedding(num_node_types, hidden_dim) if use_node_type else None
        )
        self.is_bound_embedding = nn.Embedding(2, hidden_dim)
        self.binder_depth_embedding = nn.Embedding(max_binder_depth, hidden_dim)
        self.binder_kind_embedding = nn.Embedding(num_binder_kinds, hidden_dim)

    def forward(self, data) -> Tensor:
        features = self.label_embedding(data.x)
        if self.node_type_embedding is not None:
            features = features + self.node_type_embedding(data.node_type)
        if hasattr(data, "is_bound"):
            features = features + self.is_bound_embedding(data.is_bound)
        if hasattr(data, "binder_depth"):
            binder_depth = data.binder_depth.clamp(
                min=0,
                max=self.binder_depth_embedding.num_embeddings - 1,
            )
            features = features + self.binder_depth_embedding(binder_depth)
        if hasattr(data, "binder_kind"):
            features = features + self.binder_kind_embedding(data.binder_kind)
        return features


def state_node_embeddings(node_embeddings: Tensor, data) -> Tensor:
    """Select one state-root embedding per graph without changing node order."""

    if not hasattr(data, "state_node_index"):
        raise ValueError("Batched graph data is missing 'state_node_index'.")
    state_node_index = data.state_node_index
    if not torch.is_tensor(state_node_index):
        state_node_index = torch.tensor(
            [int(state_node_index)],
            device=node_embeddings.device,
            dtype=torch.long,
        )
    else:
        state_node_index = state_node_index.to(
            device=node_embeddings.device,
            dtype=torch.long,
        ).view(-1)
    return node_embeddings.index_select(0, state_node_index)
