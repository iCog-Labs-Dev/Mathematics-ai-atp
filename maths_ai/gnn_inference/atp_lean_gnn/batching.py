from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Sampler


@dataclass(frozen=True)
class GraphSize:
    dataset_id: str
    nodes: int
    edges: int


class GraphBudgetBatchSampler(Sampler[list[int]]):
    """Greedily batch graphs under graph-count, node-count, and edge-count limits."""

    def __init__(
        self,
        graph_sizes: list[GraphSize],
        *,
        max_graphs: int,
        max_nodes: int = 0,
        max_edges: int = 0,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive.")
        if max_nodes < 0 or max_edges < 0:
            raise ValueError("Graph node and edge budgets cannot be negative.")
        self.graph_sizes = graph_sizes
        self.max_graphs = int(max_graphs)
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        for size in graph_sizes:
            if (self.max_nodes and size.nodes > self.max_nodes) or (
                self.max_edges and size.edges > self.max_edges
            ):
                raise ValueError(
                    f"Prepared graph '{size.dataset_id}' exceeds the batch budget: "
                    f"nodes={size.nodes}, edges={size.edges}, "
                    f"max_nodes={self.max_nodes}, max_edges={self.max_edges}."
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        indices = list(range(len(self.graph_sizes)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batches: list[list[int]] = []
        batch: list[int] = []
        node_count = 0
        edge_count = 0
        for index in indices:
            size = self.graph_sizes[index]
            would_exceed = bool(batch) and (
                len(batch) >= self.max_graphs
                or (self.max_nodes and node_count + size.nodes > self.max_nodes)
                or (self.max_edges and edge_count + size.edges > self.max_edges)
            )
            if would_exceed:
                batches.append(batch)
                batch = []
                node_count = 0
                edge_count = 0
            batch.append(index)
            node_count += size.nodes
            edge_count += size.edges
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
