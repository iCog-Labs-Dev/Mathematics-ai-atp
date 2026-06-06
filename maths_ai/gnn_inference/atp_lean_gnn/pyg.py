from __future__ import annotations

from typing import Iterable

from .graph import DAGBuilder


NODE_TYPE_TO_ID = {
    "var": 0,
    "type": 1,
    "predicate": 2,
    "operator": 3,
    "app": 4,
    "meta": 5,
}


def build_vocab_from_labels(labels: Iterable[str], *, unk_token: str = "<UNK>") -> dict[str, int]:
    vocab = {unk_token: 0}
    for index, label in enumerate(sorted(set(labels)), start=1):
        vocab[label] = index
    return vocab


def build_vocab(dags: Iterable[DAGBuilder]) -> dict[str, int]:
    all_labels: set[str] = set()
    for dag in dags:
        for node in dag.nodes:
            all_labels.add(node.label)
    return build_vocab_from_labels(all_labels)


def _dedupe_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return list(dict.fromkeys(edges))


def dag_to_pyg(
    dag: DAGBuilder,
    vocab: dict[str, int],
    *,
    add_reverse_edges: bool = False,
    add_self_loops: bool = False,
):
    """
    Convert a ``DAGBuilder`` to a ``torch_geometric.data.Data`` object.

    ``x`` stores label ids and ``node_type`` stores a coarse node-type id.
    """
    import torch
    from torch_geometric.data import Data

    x = torch.tensor([vocab.get(node.label, 0) for node in dag.nodes], dtype=torch.long)
    node_type = torch.tensor([NODE_TYPE_TO_ID.get(node.node_type, NODE_TYPE_TO_ID["meta"]) for node in dag.nodes], dtype=torch.long)

    edge_pairs = list(dag.edges)
    if add_reverse_edges:
        edge_pairs.extend((target, source) for (source, target) in dag.edges)
    if add_self_loops:
        edge_pairs.extend((node.id, node.id) for node in dag.nodes)
    edge_pairs = _dedupe_edges(edge_pairs)

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, node_type=node_type, num_nodes=dag.num_nodes)


# Node types eligible for pointer-based argument selection
_PREMISE_SELECTABLE_TYPES = {"var", "predicate", "type"}
_PREMISE_SELECTABLE_META_LABELS = {"Hyp"}


def build_premise_mask(dag: DAGBuilder) -> list[bool]:
    """Return a per-node boolean list where ``True`` marks a valid argument candidate.

    Valid candidates are:
    - Leaf-like nodes with type ``var``, ``predicate``, or ``type``
    - ``Hyp`` nodes (entire hypotheses)

    Excluded: ``App``, ``Arrow``, ``Forall``, ``Explicit``, ``State``,
    ``Goal``, operators, and other structural syntax nodes.
    """
    mask: list[bool] = []
    for node in dag.nodes:
        if node.label in _PREMISE_SELECTABLE_META_LABELS:
            mask.append(True)
        elif node.node_type in _PREMISE_SELECTABLE_TYPES:
            mask.append(True)
        else:
            mask.append(False)
    return mask
