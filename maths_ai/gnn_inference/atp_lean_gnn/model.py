from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, SAGEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as graph_softmax

from .pyg import NODE_TYPE_TO_ID


def _embed_binder_depth(
    embedding: nn.Embedding, binder_depth: torch.Tensor
) -> torch.Tensor:
    """Embed ordinal depths, bucketing depths beyond the configured capacity."""
    bounded_depth = binder_depth.clamp(min=0, max=embedding.num_embeddings - 1)
    return embedding(bounded_depth)


def _state_node_embeddings(
    node_embeddings: torch.Tensor, data, device: torch.device | None = None
) -> torch.Tensor:
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


class PublishedAttentionReadout(nn.Module):
    """Reference ``StateMeanAttentionReadout`` equations for ``state_mean_attention``.

    Parity with the published checkpoint's training-time forward computation:

    - scores are ``attention_score(tanh(node_projection(nodes)
      + state_projection(state_per_node)))`` — every node attends with its
      own graph's state embedding as context;
    - weights are a per-graph (batch-aware) softmax;
    - the fused summary is ``[state, mean_pool, weighted_add_pool]`` (3
      summaries, matching the checkpoint's ``fusion.weight`` of
      ``[hidden, 3 * hidden]``), followed by LayerNorm and GELU.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.state_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        state_embeddings: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_index = batch_index.to(device=node_embeddings.device, dtype=torch.long)
        graph_count = state_embeddings.size(0)
        state_per_node = state_embeddings.index_select(0, batch_index)
        scores = self.attention_score(
            torch.tanh(
                self.node_projection(node_embeddings)
                + self.state_projection(state_per_node)
            )
        ).squeeze(-1)
        attention_weights = graph_softmax(
            scores,
            batch_index,
            num_nodes=graph_count,
        )
        weighted_embeddings = global_add_pool(
            attention_weights.unsqueeze(-1) * node_embeddings,
            batch_index,
            size=graph_count,
        )
        mean_embeddings = global_mean_pool(
            node_embeddings,
            batch_index,
            size=graph_count,
        )
        combined = torch.cat(
            [state_embeddings, mean_embeddings, weighted_embeddings], dim=-1
        )
        fused = self.fusion(combined)
        return F.gelu(self.normalization(fused)), attention_weights


class PublishedGATBackbone(nn.Module):
    def __init__(
        self,
        *,
        num_node_labels: int,
        num_tactics: int,
        num_node_types: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        use_node_type: bool,
    ) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(num_node_labels, hidden_dim)
        self.node_type_embedding = (
            nn.Embedding(num_node_types, hidden_dim) if use_node_type else None
        )
        self.is_bound_embedding = nn.Embedding(2, hidden_dim)
        self.binder_depth_embedding = nn.Embedding(10, hidden_dim)
        self.binder_kind_embedding = nn.Embedding(6, hidden_dim)
        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(num_layers)
        )
        self.global_readout = PublishedAttentionReadout(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_tactics)

    def encode_nodes(self, data) -> torch.Tensor:
        x = self.label_embedding(data.x)
        if self.node_type_embedding is not None:
            x = x + self.node_type_embedding(data.node_type)
        if hasattr(data, "is_bound"):
            x = x + self.is_bound_embedding(data.is_bound)
        if hasattr(data, "binder_depth"):
            x = x + _embed_binder_depth(self.binder_depth_embedding, data.binder_depth)
        if hasattr(data, "binder_kind"):
            x = x + self.binder_kind_embedding(data.binder_kind)
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return x

    def readout(self, node_embeddings: torch.Tensor, data) -> torch.Tensor:
        state_embeddings = _state_node_embeddings(node_embeddings, data)
        batch_index = getattr(data, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(
                node_embeddings.size(0),
                device=node_embeddings.device,
                dtype=torch.long,
            )
        fused, _ = self.global_readout(node_embeddings, state_embeddings, batch_index)
        return fused
