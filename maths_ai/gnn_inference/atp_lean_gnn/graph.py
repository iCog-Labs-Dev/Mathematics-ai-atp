from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .parser import ExprParser
from .state import ProofState, parse_state


@dataclass(frozen=True)
class GraphNode:
    id: int
    label: str
    node_type: str
    children: tuple[int, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type,
            "children": list(self.children),
        }


@dataclass(frozen=True)
class GraphStats:
    num_nodes: int
    num_edges: int
    num_roots: int
    num_leaves: int
    num_reused_nodes: int
    sharing_ratio: float
    max_children: int
    max_parent_uses: int

    def as_dict(self) -> dict[str, object]:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_roots": self.num_roots,
            "num_leaves": self.num_leaves,
            "num_reused_nodes": self.num_reused_nodes,
            "sharing_ratio": self.sharing_ratio,
            "max_children": self.max_children,
            "max_parent_uses": self.max_parent_uses,
        }


def _classify_label(label: str) -> str:
    if not label:
        return "var"
    if label in ("App", "Arrow", "Forall", "Explicit"):
        return "app"
    if label in ("Hyp", "Goal", "State"):
        return "meta"
    if label == "\u2115" or (label[0].isupper() and len(label) <= 2):
        return "type"
    if label[0].isupper():
        return "predicate"
    if label in ("+", "-", "*", "/", "=", "\u2264", "\u2265", "<", ">", "\u2227", "\u2228", "\u00ac"):
        return "operator"
    return "var"


@dataclass
class DAGBuilder:
    """
    Build a DAG via hash-consing.

    Edges are stored as ``(child_id, parent_id)`` pairs.
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    _memo: dict[tuple[str, tuple[int, ...]], int] = field(default_factory=dict)

    def get_or_create(self, label: str, children: tuple[int, ...]) -> int:
        key = (label, children)
        if key in self._memo:
            return self._memo[key]

        node_id = len(self.nodes)
        self.nodes.append(GraphNode(node_id, label, _classify_label(label), children))
        for child_id in children:
            self.edges.append((child_id, node_id))
        self._memo[key] = node_id
        return node_id

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def sharing_ratio(self) -> float:
        return self.num_edges / max(self.num_nodes, 1)

    def incoming_counts(self) -> Counter[int]:
        return Counter(parent_id for (_, parent_id) in self.edges)

    def outgoing_counts(self) -> Counter[int]:
        return Counter(child_id for (child_id, _) in self.edges)

    def reused_nodes(self) -> list[GraphNode]:
        parent_uses = self.outgoing_counts()
        return [node for node in self.nodes if parent_uses[node.id] > 1]

    def shared_nodes(self) -> list[GraphNode]:
        return self.reused_nodes()

    def root_nodes(self) -> list[GraphNode]:
        parent_uses = self.outgoing_counts()
        return [node for node in self.nodes if parent_uses[node.id] == 0]

    def leaf_nodes(self) -> list[GraphNode]:
        child_counts = self.incoming_counts()
        return [node for node in self.nodes if child_counts[node.id] == 0]

    def stats(self) -> GraphStats:
        return graph_stats(self)


def graph_stats(dag: DAGBuilder) -> GraphStats:
    child_counts = dag.incoming_counts()
    parent_uses = dag.outgoing_counts()
    reused = [node for node in dag.nodes if parent_uses[node.id] > 1]
    return GraphStats(
        num_nodes=dag.num_nodes,
        num_edges=dag.num_edges,
        num_roots=len([node for node in dag.nodes if parent_uses[node.id] == 0]),
        num_leaves=len([node for node in dag.nodes if child_counts[node.id] == 0]),
        num_reused_nodes=len(reused),
        sharing_ratio=dag.sharing_ratio(),
        max_children=max((child_counts[node.id] for node in dag.nodes), default=0),
        max_parent_uses=max((parent_uses[node.id] for node in dag.nodes), default=0),
    )


def proof_state_to_dag(state: str | ProofState) -> DAGBuilder:
    parsed = state if isinstance(state, ProofState) else parse_state(state)
    dag = DAGBuilder()
    parser = ExprParser(dag)
    root_ids: list[int] = []

    for hypothesis in parsed.hypotheses:
        name_node = dag.get_or_create(hypothesis.name, ())
        type_node = parser.parse(hypothesis.type_expr) if hypothesis.type_expr else dag.get_or_create("?", ())
        hyp_node = dag.get_or_create("Hyp", (name_node, type_node))
        root_ids.append(hyp_node)

    goal_expr_node = parser.parse(parsed.goal)
    goal_node = dag.get_or_create("Goal", (goal_expr_node,))
    root_ids.append(goal_node)
    dag.get_or_create("State", tuple(root_ids))
    return dag


def lemma_statement_to_dag(statement: str) -> DAGBuilder:
    """Build a DAG for a lemma statement treated as a goal-only proof state."""
    dag = DAGBuilder()
    parser = ExprParser(dag)

    goal_expr_node = parser.parse(statement)
    goal_node = dag.get_or_create("Goal", (goal_expr_node,))
    dag.get_or_create("State", (goal_node,))
    return dag


def dag_to_dict(dag: DAGBuilder, metadata: dict[str, object] | None = None) -> dict[str, object]:
    child_counts = dag.incoming_counts()
    parent_uses = dag.outgoing_counts()
    root_ids = {node.id for node in dag.root_nodes()}
    leaf_ids = {node.id for node in dag.leaf_nodes()}

    return {
        "metadata": metadata or {},
        "stats": dag.stats().as_dict(),
        "nodes": [
            {
                **node.as_dict(),
                "num_children": child_counts[node.id],
                "num_parent_uses": parent_uses[node.id],
                "is_reused": parent_uses[node.id] > 1,
                "is_root": node.id in root_ids,
                "is_leaf": node.id in leaf_ids,
            }
            for node in dag.nodes
        ],
        "edges": [{"source": source, "target": target} for (source, target) in dag.edges],
    }


def write_dag_json(
    dag: DAGBuilder,
    output_path: str | Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dag_to_dict(dag, metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    return output
