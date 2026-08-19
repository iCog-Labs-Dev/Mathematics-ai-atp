from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax as graph_softmax

from .base import EncoderOutput, NodeFeatureEmbedding, StateGraphEncoder, state_node_embeddings


VALID_GATV2_READOUTS = (
    "state",
    "state_mean_attention",
    "state_max_attention",
    "state_mean_max_attention",
)


class StateAttentionReadout(nn.Module):
    """Fuse the state root and attention summary with optional global summaries."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        include_mean: bool,
        include_max: bool,
    ) -> None:
        super().__init__()
        self.include_mean = include_mean
        self.include_max = include_max
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)
        summary_count = 2 + int(include_mean) + int(include_max)
        self.fusion = nn.Linear(hidden_dim * summary_count, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: Tensor,
        state_embeddings: Tensor,
        batch_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_index = batch_index.to(device=node_embeddings.device, dtype=torch.long)
        graph_count = state_embeddings.size(0)
        state_per_node = state_embeddings.index_select(0, batch_index)
        scores = self.attention_score(
            torch.tanh(
                self.node_projection(node_embeddings)
                + self.state_projection(state_per_node)
            )
        ).squeeze(-1)
        attention_weights = graph_softmax(scores, batch_index, num_nodes=graph_count)
        attention_summary = global_add_pool(
            attention_weights.unsqueeze(-1) * node_embeddings,
            batch_index,
            size=graph_count,
        )
        summaries = [state_embeddings]
        if self.include_mean:
            summaries.append(global_mean_pool(node_embeddings, batch_index, size=graph_count))
        if self.include_max:
            summaries.append(global_max_pool(node_embeddings, batch_index, size=graph_count))
        summaries.append(attention_summary)
        return (
            F.gelu(self.normalization(self.fusion(torch.cat(summaries, dim=-1)))),
            attention_weights,
        )


class GATv2Encoder(StateGraphEncoder):
    """GATv2 proof-state encoder with configurable state-conditioned readout."""

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
        heads: int,
        readout: str,
        use_node_type: bool,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("GATv2 requires at least one message-passing layer.")
        if heads < 1:
            raise ValueError("GATv2 requires at least one attention head.")
        if hidden_dim % heads != 0:
            raise ValueError(
                "GATv2 requires hidden_dim to be divisible by heads "
                f"(hidden_dim={hidden_dim}, heads={heads})."
            )
        if readout not in VALID_GATV2_READOUTS:
            raise ValueError(
                f"GATv2 readout must be one of: {', '.join(VALID_GATV2_READOUTS)}."
            )

        self.output_dim = hidden_dim
        self.readout_mode = readout
        self.node_features = NodeFeatureEmbedding(
            num_node_labels=num_node_labels,
            hidden_dim=hidden_dim,
            num_node_types=num_node_types,
            num_binder_kinds=num_binder_kinds,
            max_binder_depth=max_binder_depth,
            use_node_type=use_node_type,
        )
        head_dim = hidden_dim // heads
        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden_dim,
                head_dim,
                heads=heads,
                dropout=dropout,
                concat=True,
            )
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.global_readout = (
            None
            if readout == "state"
            else StateAttentionReadout(
                hidden_dim,
                include_mean="mean" in readout,
                include_max="max" in readout,
            )
        )

    def forward(self, data) -> EncoderOutput:
        node_embeddings = self.node_features(data)
        for conv in self.convs:
            node_embeddings = self.dropout(F.relu(conv(node_embeddings, data.edge_index)))

        state_embeddings = state_node_embeddings(node_embeddings, data)
        if self.global_readout is None:
            return EncoderOutput(node_embeddings, state_embeddings)

        batch_index = getattr(data, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(
                node_embeddings.size(0),
                device=node_embeddings.device,
                dtype=torch.long,
            )
        state_embeddings, attention_weights = self.global_readout(
            node_embeddings,
            state_embeddings,
            batch_index,
        )
        return EncoderOutput(
            node_embeddings=node_embeddings,
            state_embeddings=state_embeddings,
            details={"attention_weights": attention_weights},
        )
