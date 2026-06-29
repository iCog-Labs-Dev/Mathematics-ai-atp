from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from .pyg import NODE_TYPE_TO_ID


@dataclass(frozen=True)
class GraphSAGEClassifierConfig:
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class GraphSAGEStateClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_node_labels: int,
        num_tactics: int,
        num_node_types: int = len(NODE_TYPE_TO_ID),
        num_binder_kinds: int = 6,
        max_binder_depth: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_node_type: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("GraphSAGEStateClassifier requires at least one message-passing layer.")

        self.label_embedding = nn.Embedding(num_node_labels, hidden_dim)
        self.node_type_embedding = (
            nn.Embedding(num_node_types, hidden_dim) if use_node_type else None
        )

        # Binder feature embeddings
        self.is_bound_embedding = nn.Embedding(2, hidden_dim)  # 0/1
        self.binder_depth_embedding = nn.Embedding(max_binder_depth, hidden_dim)
        self.binder_kind_embedding = nn.Embedding(num_binder_kinds, hidden_dim)

        self.convs = nn.ModuleList(
            SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_tactics)

    def encode_nodes(self, data) -> torch.Tensor:
        x = self.label_embedding(data.x)
        if self.node_type_embedding is not None:
            x = x + self.node_type_embedding(data.node_type)

        # Add binder features if present
        if hasattr(data, "is_bound"):
            x = x + self.is_bound_embedding(data.is_bound)
        if hasattr(data, "binder_depth"):
            x = x + self.binder_depth_embedding(data.binder_depth)
        if hasattr(data, "binder_kind"):
            x = x + self.binder_kind_embedding(data.binder_kind)

        for index, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            x = F.relu(x)
            if index < len(self.convs) - 1:
                x = self.dropout(x)
        return x

    def readout(self, node_embeddings: torch.Tensor, data) -> torch.Tensor:
        if not hasattr(data, "state_node_index"):
            raise ValueError("Batched graph data is missing the 'state_node_index' attribute.")

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

    def forward(self, data) -> torch.Tensor:
        node_embeddings = self.encode_nodes(data)
        graph_embeddings = self.readout(node_embeddings, data)
        return self.classifier(self.dropout(graph_embeddings))
