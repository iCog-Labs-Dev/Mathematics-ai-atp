from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from .base import EncoderOutput, NodeFeatureEmbedding, StateGraphEncoder, state_node_embeddings


class GraphSAGEEncoder(StateGraphEncoder):
    """GraphSAGE proof-state encoder with state-root readout."""

    architecture_version = 1

    def __init__(
        self,
        *,
        num_node_labels: int,
        num_node_types: int,
        num_binder_kinds: int,
        max_binder_depth: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_node_type: bool,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("GraphSAGE requires at least one message-passing layer.")
        self.output_dim = hidden_dim
        self.node_features = NodeFeatureEmbedding(
            num_node_labels=num_node_labels,
            hidden_dim=hidden_dim,
            num_node_types=num_node_types,
            num_binder_kinds=num_binder_kinds,
            max_binder_depth=max_binder_depth,
            use_node_type=use_node_type,
        )
        self.convs = nn.ModuleList(
            SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data) -> EncoderOutput:
        node_embeddings = self.node_features(data)
        for index, conv in enumerate(self.convs):
            node_embeddings = F.relu(conv(node_embeddings, data.edge_index))
            if index < len(self.convs) - 1:
                node_embeddings = self.dropout(node_embeddings)
        return EncoderOutput(
            node_embeddings=node_embeddings,
            state_embeddings=state_node_embeddings(node_embeddings, data),
        )
