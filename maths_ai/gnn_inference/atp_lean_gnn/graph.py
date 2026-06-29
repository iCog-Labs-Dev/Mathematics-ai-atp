from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .parser import ExprParser
from .state import ProofState, parse_state


BINDER_KIND_UNKNOWN = -1
BINDER_KIND_NONE = 0      # context variable (not bound in this goal)
BINDER_KIND_FORALL = 1    # ∀ binder
BINDER_KIND_EXISTS = 2    # ∃ binder
BINDER_KIND_LAMBDA = 3    # λ binder
BINDER_KIND_LET = 4       # let binder
BINDER_KIND_OTHER = 5     # other binder types

_BINDER_LABEL_TO_KIND: dict[str, int] = {
    "∀": BINDER_KIND_FORALL,
    "forall": BINDER_KIND_FORALL,
    "∃": BINDER_KIND_EXISTS,
    "exists": BINDER_KIND_EXISTS,
    "λ": BINDER_KIND_LAMBDA,
    "fun": BINDER_KIND_LAMBDA,
    "let": BINDER_KIND_LET,
}


@dataclass(frozen=True)
class GraphNode:
    id: int
    label: str
    node_type: str
    children: tuple[int, ...] = field(default_factory=tuple)
    is_bound: int = BINDER_KIND_NONE     # 1 if bound by a quantifier, 0 otherwise
    binder_depth: int = 0                 # nesting level (0 = context var)
    binder_kind: int = BINDER_KIND_UNKNOWN  # which binder (∀, ∃, λ, etc.)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type,
            "children": list(self.children),
            "is_bound": self.is_bound,
            "binder_depth": self.binder_depth,
            "binder_kind": self.binder_kind,
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

    def annotate_binders(self) -> None:
        """Post-process the DAG to annotate each node with binder information.

        Walks the DAG from root to leaves. When an App node wrapping a binder
        (∀/∃/λ) is found, the variable being bound gets annotated with
        ``is_bound=1``, ``binder_depth``, and ``binder_kind``.

        Variables not inside a binder keep ``is_bound=0``.

        DAG structure for ``∀ (q : Prop), body``:
            App(∀, App(App(q, :), Prop))
            └─ node 6: App children=(0, 5)
               ├─ node 0: ∀ (the binder symbol)
               └─ node 5: App children=(3, 4)
                  ├─ node 3: App children=(1, 2)
                  │  ├─ node 1: q (the variable being bound)
                  │  └─ node 2: :
                  └─ node 4: Prop

        Note: edges are stored as (child_id, parent_id) pairs.
        """
        # Build adjacency: parent → children
        # Edges are stored as (child_id, parent_id), so we reverse them
        children_of: dict[int, list[int]] = {}
        for child_id, parent_id in self.edges:
            children_of.setdefault(parent_id, []).append(child_id)

        def _walk(node_id: int, depth: int) -> None:
            node = self.nodes[node_id]
            kids = children_of.get(node_id, [])

            # Detect App nodes wrapping a binder: App(binder_symbol, body)
            if node.label == "App" and len(kids) >= 2:
                first_child = self.nodes[kids[0]]
                binder_kind = _BINDER_LABEL_TO_KIND.get(first_child.label, BINDER_KIND_UNKNOWN)

                if binder_kind != BINDER_KIND_UNKNOWN:
                    # This is App(∀/∃/λ, body)
                    # The variable is in the second child's subtree
                    # e.g., App(∀, App(App(q, :), Prop))
                    #   kids[0] = ∀ (binder symbol)
                    #   kids[1] = App(App(q, :), Prop) (variable + type + body)
                    # Variable is at depth+1 inside the binder
                    self._annotate_binder_var(kids[1], depth + 1, binder_kind, children_of)
                    # Continue into the body (skip the variable subtree entirely)
                    # The body is the second child, which we just annotated
                    return

            # Regular node: recurse into children
            for kid in kids:
                _walk(kid, depth)

        # Find root nodes (no parents) and start from there
        # Edges are (child_id, parent_id), so children are the first element
        has_parent: set[int] = {child for child, _ in self.edges}
        roots = [n.id for n in self.nodes if n.id not in has_parent]

        for root_id in roots:
            _walk(root_id, 0)

    def _annotate_binder_var(
        self, node_id: int, depth: int, binder_kind: int,
        children_of: dict[int, list[int]]
    ) -> None:
        """Annotate the variable node inside a binder (∀/∃/λ).

        For ``∀ (q : Prop)``, the structure is:
            App(∀, App(App(q, :), Prop))
            └─ ∀ is the first child of the App node
               └─ App(App(q, :), Prop) is the second child
                  └─ App(q, :) is the first child
                     └─ q is the variable being bound (leaf node)

        We need to find the leaf variable node and annotate it.
        """
        node = self.nodes[node_id]
        kids = children_of.get(node_id, [])

        # If this is a leaf var node (the variable name), annotate it
        # Exclude ':' and structural labels
        _EXCLUDE = {":", "App", "Arrow", "∀", "∃", "λ", "let", ","}
        if (node.node_type == "var" and not kids
                and node.label not in _EXCLUDE):
            self.nodes[node_id] = GraphNode(
                id=node.id,
                label=node.label,
                node_type=node.node_type,
                children=node.children,
                is_bound=1,
                binder_depth=depth,
                binder_kind=binder_kind,
            )
            return

        # If this is an App wrapping the variable (e.g., App(q, :)), recurse
        for kid in kids:
            kid_node = self.nodes[kid]
            kid_kids = children_of.get(kid, [])
            if kid_node.node_type == "var" and kid_node.label not in _EXCLUDE:
                # This is likely the variable name - check if it's a leaf
                if not kid_kids:
                    # Leaf variable - annotate it
                    self.nodes[kid] = GraphNode(
                        id=kid_node.id,
                        label=kid_node.label,
                        node_type=kid_node.node_type,
                        children=kid_node.children,
                        is_bound=1,
                        binder_depth=depth,
                        binder_kind=binder_kind,
                    )
                else:
                    # Non-leaf var - recurse to find the actual variable
                    self._annotate_binder_var(kid, depth, binder_kind, children_of)
            elif kid_node.label == "App":
                # App node - recurse to find the variable inside
                self._annotate_binder_var(kid, depth, binder_kind, children_of)

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

    # Annotate binder relationships (∀, ∃, λ)
    dag.annotate_binders()

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
