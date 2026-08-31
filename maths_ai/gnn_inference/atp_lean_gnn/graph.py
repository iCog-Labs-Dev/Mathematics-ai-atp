from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .state import ProofState, parse_state
from maths_ai.pln_inference.metta.translator.translator_modules.parser import (
    parse_sexp_string,
)


def patch_pantograph_for_sexp() -> None:
    """Monkey-patch Pantograph to return S-expressions instead of pretty-printed strings.

    After calling this, ``Goal.target`` and ``Variable.t`` will contain the
    Lean S-expression (e.g. ``((:c Eq) (:c Nat) ...)``) instead of the
    human-readable ``n = n`` form.

    Must be called BEFORE creating a Server instance.
    """
    import pantograph.expr as expr_mod
    import pantograph.server as server_mod

    def _parse_expr_sexp(payload: dict) -> str:
        return payload.get("sexp") or payload["pp"]

    expr_mod.parse_expr = _parse_expr_sexp
    server_mod.parse_expr = _parse_expr_sexp


def goal_state_to_proof_state(goal_state) -> tuple[str, list[tuple[str, str | None]], str | None]:
    """Extract proof state components from a Pantograph GoalState.

    Returns (text_state, hyp_sexps, goal_sexp) where:

    - ``text_state``: human-readable text for the proof state (backward compat)
    - ``hyp_sexps``: list of ``(name, type_sexp)`` for each hypothesis
    - ``goal_sexp``: S-expression of the goal type, or None

    Requires ``patch_pantograph_for_sexp()`` to have been called first.
    """
    if not goal_state.goals:
        return "", [], None

    goal = goal_state.goals[0]
    goal_sexp = goal.target  # Already an S-expression after patching
    hyp_sexps = [(v.name or "_", v.t) for v in goal.variables]

    # Build text representation for backward compatibility
    lines = []
    for v in goal.variables:
        lines.append(f"{v.name or '_'} : {v.t}")
    text_state = "\n".join(lines) + f"\n⊢ {goal_sexp}" if lines else f"⊢ {goal_sexp}"

    return text_state, hyp_sexps, goal_sexp


BINDER_KIND_UNKNOWN = -1
BINDER_KIND_NONE = 0      # context variable (not bound in this goal)
BINDER_KIND_FORALL = 1    # ∀ binder
BINDER_KIND_EXISTS = 2    # ∃ binder
BINDER_KIND_LAMBDA = 3    # λ binder
BINDER_KIND_LET = 4       # let binder
BINDER_KIND_OTHER = 5     # other binder types


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

# ---------------------------------------------------------------------------
# S-expression → DAG conversion
# ---------------------------------------------------------------------------

def sexp_to_dag(sexp: str) -> DAGBuilder:
    """Convert a Lean 4 S-expression string (from pantograph) to a DAG.

    Binder annotations (is_bound, binder_depth, binder_kind) are set during
    conversion — no post-processing needed.
    """
    dag = DAGBuilder()
    parsed = parse_sexp_string(sexp)
    _sexp_walk(parsed, [], dag)
    return dag


def get_node_labels(dag: DAGBuilder) -> list[str]:
    """Return labels of all nodes in order (debug helper)."""
    return [n.label for n in dag.nodes]


def _sexp_walk(sexp, ctx: list[str], dag: DAGBuilder) -> int:
    """Walk a parsed S-expression and build DAG nodes.

    Args:
        sexp: Nested list from parse_sexp_string
        ctx: Context stack of bound variable names (for de Bruijn resolution)
        dag: DAGBuilder to populate

    Returns:
        Node ID of the created node
    """
    if not isinstance(sexp, list):
        return _sexp_leaf(sexp, ctx, dag)

    if len(sexp) < 2:
        return dag.get_or_create("()", ())

    head = sexp[0]

    # Binder: (:forall name type body) or (:lambda name type body)
    if head in (":forall", ":lambda"):
        name = sexp[1]
        ty = sexp[2]
        body = sexp[3]
        binder_kind = BINDER_KIND_FORALL if head == ":forall" else BINDER_KIND_LAMBDA

        # Variable node (leaf, annotated inline)
        var_id = dag.get_or_create(name, ())
        dag.nodes[var_id] = GraphNode(
            id=var_id,
            label=name,
            node_type="var",
            children=(),
            is_bound=1,
            binder_depth=len(ctx) + 1,
            binder_kind=binder_kind,
        )

        # Type and body — both may reference this binder via de Bruijn
        ctx_with_var = ctx + [name]
        ty_id = _sexp_walk(ty, ctx_with_var, dag)
        body_id = _sexp_walk(body, ctx_with_var, dag)

        # (:forall name type body) — 3 children
        return dag.get_or_create(head, (var_id, ty_id, body_id))

    # Constant: (:c Name)
    if head == ":c" and len(sexp) == 2:
        return dag.get_or_create(str(sexp[1]), ())

    # Sort: (:sort N)
    if head == ":sort" and len(sexp) == 2:
        n = sexp[1]
        label = "Prop" if n == "0" else "Type" if n == "1" else f"Sort-{n}"
        return dag.get_or_create(label, ())

    # Free variable: (:fv Name)
    if head == ":fv" and len(sexp) == 2:
        return dag.get_or_create(str(sexp[1]), ())

    # Application: (f a b ...) — first is function, rest are args
    if len(sexp) >= 2:
        fn_id = _sexp_walk(sexp[0], ctx, dag)
        children = [fn_id]
        for arg in sexp[1:]:
            children.append(_sexp_walk(arg, ctx, dag))
        return dag.get_or_create("App", tuple(children))

    return dag.get_or_create(str(sexp), ())


def _sexp_leaf(token: str, ctx: list[str], dag: DAGBuilder) -> int:
    """Handle a bare token (not a list)."""
    # De Bruijn index (bare number)
    if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
        idx = int(token)
        if 0 <= idx < len(ctx):
            return dag.get_or_create(ctx[idx], ())
        return dag.get_or_create(f"?db-{idx}", ())

    # Named constant
    return dag.get_or_create(token, ())


# ---------------------------------------------------------------------------
# Proof state → DAG (supports both old text parser and new S-expression path)
# ---------------------------------------------------------------------------

def proof_state_to_dag(
    state: str | ProofState,
    *,
    sexp: str | None = None,
    goal_sexp: str | None = None,
    hyp_sexps: list[tuple[str, str | None]] | None = None,
) -> DAGBuilder:
    """Build a DAG from a proof state.

    Three calling conventions:

    1. ``state`` is a text string → parse with ExprParser (old path).
    2. ``sexp`` is provided → goal type parsed via ``_sexp_walk``.
    3. ``goal_sexp`` + ``hyp_sexps`` are provided → both goal and hypothesis
       types parsed via ``_sexp_walk`` (preferred path when Pantograph is
       available with ``printExprAST: true``).
    """
    parsed = state if isinstance(state, ProofState) else parse_state(state)

    if goal_sexp is not None and hyp_sexps is not None:
        # Best path: S-expressions for both goal and hypothesis types
        dag = sexp_to_dag(goal_sexp)
        goal_expr_id = dag.num_nodes - 1

        root_ids: list[int] = []
        for hyp, (hyp_name, hyp_sexp) in zip(parsed.hypotheses, hyp_sexps):
            name_node = dag.get_or_create(hyp_name or hyp.name, ())
            if hyp_sexp:
                type_node = _sexp_walk(parse_sexp_string(hyp_sexp), [], dag)
            elif hyp.type_expr:
                from .parser import ExprParser
                _hyp_parser = ExprParser(dag)
                type_node = _hyp_parser.parse(hyp.type_expr)
            else:
                type_node = dag.get_or_create("?", ())
            if hyp.is_local_definition:
                value_node = _hyp_parser.parse(hyp.value_expr)
                hyp_node = dag.get_or_create("Let", (name_node, type_node, value_node))
            else:
                hyp_node = dag.get_or_create("Hyp", (name_node, type_node))
            root_ids.append(hyp_node)

        goal_node = dag.get_or_create("Goal", (goal_expr_id,))
        root_ids.append(goal_node)
        dag.get_or_create("State", tuple(root_ids))
        return dag

    if sexp is not None:
        # Goal has S-expression, hypothesis types use text parser
        dag = sexp_to_dag(sexp)
        goal_expr_id = dag.num_nodes - 1

        from .parser import ExprParser
        _hyp_parser = ExprParser(dag)

        root_ids: list[int] = []
        for hypothesis in parsed.hypotheses:
            name_node = dag.get_or_create(hypothesis.name, ())
            type_node = _hyp_parser.parse(hypothesis.type_expr) if hypothesis.type_expr else dag.get_or_create("?", ())
            if hypothesis.is_local_definition:
                value_node = _hyp_parser.parse(hypothesis.value_expr)
                hyp_node = dag.get_or_create("Let", (name_node, type_node, value_node))
            else:
                hyp_node = dag.get_or_create("Hyp", (name_node, type_node))
            root_ids.append(hyp_node)

        goal_node = dag.get_or_create("Goal", (goal_expr_id,))
        root_ids.append(goal_node)
        dag.get_or_create("State", tuple(root_ids))
        return dag

    # Old path: text-based parser (offline, backward compatible)
    from .parser import ExprParser

    dag = DAGBuilder()
    parser = ExprParser(dag)
    root_ids = []

    for hypothesis in parsed.hypotheses:
        name_node = dag.get_or_create(hypothesis.name, ())
        type_node = parser.parse(hypothesis.type_expr) if hypothesis.type_expr else dag.get_or_create("?", ())
        if hypothesis.is_local_definition:
            value_node = parser.parse(hypothesis.value_expr)
            hyp_node = dag.get_or_create("Let", (name_node, type_node, value_node))
        else:
            hyp_node = dag.get_or_create("Hyp", (name_node, type_node))
        root_ids.append(hyp_node)

    goal_expr_node = parser.parse(parsed.goal)
    goal_node = dag.get_or_create("Goal", (goal_expr_node,))
    root_ids.append(goal_node)
    dag.get_or_create("State", tuple(root_ids))

    return dag


def lemma_statement_to_dag(statement: str, *, sexp: str | None = None) -> DAGBuilder:
    """Build a DAG for a lemma statement treated as a goal-only proof state.

    If *sexp* is provided, uses the new Lean AST parser.
    Otherwise falls back to the old text-based parser.
    """
    if sexp is not None:
        dag = sexp_to_dag(sexp)
        goal_node = dag.get_or_create("Goal", (dag.num_nodes - 1,))
        dag.get_or_create("State", (goal_node,))
        return dag

    from .parser import ExprParser

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
