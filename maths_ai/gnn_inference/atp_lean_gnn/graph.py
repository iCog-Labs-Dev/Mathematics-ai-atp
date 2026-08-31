from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .state import ProofState, parse_state
from maths_ai.pln_inference.metta.translator.translator_modules.parser import (
    parse_sexp_string,
)


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
    # Text parser labels
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
    # Pantograph S-expression labels
    if label.startswith(":"):
        if label in (":forall", ":lambda", ":let"):
            return "sbinder"
        if label in (":c", ":fv", ":sort", ":lit", ":app"):
            return "sconst"
        return "sconst"
    # Pantograph application node
    if label == "App":
        return "sapp"
    return "var"


@dataclass
class DAGBuilder:
    """
    Build a DAG via hash-consing.

    Edges are stored as ``(child_id, parent_id)`` pairs.
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    expression_root_id: int | None = None
    state_root_id: int | None = None
    _memo: dict[tuple[str, str, tuple[int, ...]], int] = field(default_factory=dict)

    def add_node(
        self,
        label: str,
        children: tuple[int, ...],
        *,
        node_type: str | None = None,
        is_bound: int = BINDER_KIND_NONE,
        binder_depth: int = 0,
        binder_kind: int = BINDER_KIND_UNKNOWN,
        hash_cons: bool = True,
    ) -> int:
        """Add a node, optionally hash-consing semantically identical nodes.

        Bound variables use ``hash_cons=False`` because their identity is their
        lexical scope, not their display name. References to a bound variable
        reuse its node id directly from the De Bruijn context.
        """
        resolved_node_type = node_type or _classify_label(label)
        key = (label, resolved_node_type, children)
        if hash_cons and key in self._memo:
            return self._memo[key]

        node_id = len(self.nodes)
        self.nodes.append(
            GraphNode(
                node_id,
                label,
                resolved_node_type,
                children,
                is_bound=is_bound,
                binder_depth=binder_depth,
                binder_kind=binder_kind,
            )
        )
        for child_id in children:
            self.edges.append((child_id, node_id))
        if hash_cons:
            self._memo[key] = node_id
        return node_id

    def get_or_create(
        self,
        label: str,
        children: tuple[int, ...],
        *,
        node_type: str | None = None,
    ) -> int:
        return self.add_node(label, children, node_type=node_type)

    def create_bound_variable(
        self,
        label: str,
        *,
        binder_depth: int,
        binder_kind: int,
    ) -> int:
        return self.add_node(
            label,
            (),
            node_type="var",
            is_bound=1,
            binder_depth=binder_depth,
            binder_kind=binder_kind,
            hash_cons=False,
        )

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
    dag.expression_root_id = _sexp_walk(parsed, [], dag)
    return dag


def get_node_labels(dag: DAGBuilder) -> list[str]:
    """Return labels of all nodes in order (debug helper)."""
    return [n.label for n in dag.nodes]


def _sexp_walk(sexp, ctx: list[int], dag: DAGBuilder) -> int:
    """Walk a parsed S-expression and build DAG nodes.

    Args:
        sexp: Nested list from parse_sexp_string
        ctx: Bound-variable node ids, newest first (De Bruijn index order)
        dag: DAGBuilder to populate

    Returns:
        Node ID of the created node
    """
    if not isinstance(sexp, list):
        return _sexp_leaf(sexp, ctx, dag)

    if not sexp:
        return dag.get_or_create("()", (), node_type="sconst")

    head = sexp[0]

    # Binder: raw Pantograph uses ``(:forall name type body [role])`` while
    # model S-expressions use ``(:forall name role type body)``.
    # The binder is not in scope in its own type, but is index 0 in its body.
    if head in (":forall", ":lambda"):
        if len(sexp) not in (4, 5):
            raise ValueError(
                f"Malformed {head} S-expression: expected 4 or 5 fields, got {len(sexp)}"
            )
        name = sexp[1]
        role = None
        if len(sexp) == 5 and str(sexp[2]).startswith(":"):
            role, ty, body = sexp[2:]
        else:
            ty, body = sexp[2:4]
            if len(sexp) == 5:
                role = sexp[4]
        binder_kind = BINDER_KIND_FORALL if head == ":forall" else BINDER_KIND_LAMBDA

        var_id = dag.create_bound_variable(
            str(name),
            binder_depth=len(ctx) + 1,
            binder_kind=binder_kind,
        )
        ty_id = _sexp_walk(ty, ctx, dag)
        body_id = _sexp_walk(body, [var_id, *ctx], dag)

        children = [var_id, ty_id, body_id]
        if role is not None:
            children.append(
                dag.get_or_create(f"BinderRole:{role}", (), node_type="sconst")
            )
        return dag.get_or_create(head, tuple(children), node_type="sbinder")

    # Let binder: (:let name type value body). The name is in scope only in
    # the body; neither its type nor defining value may refer to itself.
    if head == ":let":
        if len(sexp) != 5:
            raise ValueError(f"Malformed :let S-expression: expected 5 fields, got {len(sexp)}")
        name, ty, value, body = sexp[1:]
        var_id = dag.create_bound_variable(
            str(name),
            binder_depth=len(ctx) + 1,
            binder_kind=BINDER_KIND_LET,
        )
        ty_id = _sexp_walk(ty, ctx, dag)
        value_id = _sexp_walk(value, ctx, dag)
        body_id = _sexp_walk(body, [var_id, *ctx], dag)
        return dag.get_or_create(
            ":let",
            (var_id, ty_id, value_id, body_id),
            node_type="sbinder",
        )

    # Constant: (:c Name)
    if head == ":c":
        if len(sexp) != 2:
            raise ValueError(f"Malformed :c S-expression: expected 2 fields, got {len(sexp)}")
        return dag.get_or_create(str(sexp[1]), (), node_type="const")

    # Sort: (:sort N)
    if head == ":sort":
        if len(sexp) != 2:
            raise ValueError(f"Malformed :sort S-expression: expected 2 fields, got {len(sexp)}")
        n = sexp[1]
        label = (
            "Prop"
            if n in ("0", "Prop")
            else "Type"
            if n in ("1", "Type")
            else f"Sort-{n}"
        )
        return dag.get_or_create(label, (), node_type="type")

    # Free variable: (:fv Name)
    if head == ":fv":
        if len(sexp) != 2:
            raise ValueError(f"Malformed :fv S-expression: expected 2 fields, got {len(sexp)}")
        return dag.get_or_create(str(sexp[1]), (), node_type="var")

    # Literal: (:lit value). Keep the payload in the visible label while its
    # semantic node type records that it is a constant rather than a variable.
    if head == ":lit":
        if len(sexp) != 2:
            raise ValueError(f"Malformed :lit S-expression: expected 2 fields, got {len(sexp)}")
        return dag.get_or_create(f"Lit:{_sexp_payload_text(sexp[1])}", (), node_type="const")

    # Metadata: (:mdata metadata expression). Metadata is retained as a
    # structural node instead of being mistaken for a function application.
    if head == ":mdata":
        if len(sexp) != 3:
            raise ValueError(f"Malformed :mdata S-expression: expected 3 fields, got {len(sexp)}")
        metadata_id = dag.get_or_create(
            f"Metadata:{_sexp_payload_text(sexp[1])}",
            (),
            node_type="sconst",
        )
        expression_id = _sexp_walk(sexp[2], ctx, dag)
        return dag.get_or_create(
            ":mdata",
            (metadata_id, expression_id),
            node_type="meta",
        )

    # Projection: (:proj Structure index expression).
    if head == ":proj":
        if len(sexp) != 4:
            raise ValueError(f"Malformed :proj S-expression: expected 4 fields, got {len(sexp)}")
        structure_id = dag.get_or_create(str(sexp[1]), (), node_type="const")
        index_id = dag.get_or_create(f"Field:{sexp[2]}", (), node_type="sconst")
        expression_id = _sexp_walk(sexp[3], ctx, dag)
        return dag.get_or_create(
            ":proj",
            (structure_id, index_id, expression_id),
            node_type="sapp",
        )

    # Lean-native model application. Each argument wrapper records both its
    # semantic role and original application position.
    if head == ":app":
        if len(sexp) < 2:
            raise ValueError("Malformed :app S-expression: expected a function.")
        children = tuple(_sexp_walk(item, ctx, dag) for item in sexp[1:])
        return dag.get_or_create(":app", children, node_type="sapp")

    if head == ":arg":
        if len(sexp) not in (3, 4):
            raise ValueError(
                f"Malformed :arg S-expression: expected 3 or 4 fields, got {len(sexp)}"
            )
        role, position = sexp[1:3]
        children = [
            dag.get_or_create(f"ArgRole:{role}", (), node_type="sconst"),
            dag.get_or_create(f"ArgPosition:{position}", (), node_type="sconst"),
        ]
        if len(sexp) == 4:
            children.append(_sexp_walk(sexp[3], ctx, dag))
        return dag.get_or_create(":arg", tuple(children), node_type="sapp")

    if head in (":instance-of", ":proof-of"):
        if len(sexp) != 2:
            raise ValueError(
                f"Malformed {head} S-expression: expected 2 fields, got {len(sexp)}"
            )
        child = _sexp_walk(sexp[1], ctx, dag)
        return dag.get_or_create(head, (child,), node_type="meta")

    if head == ":metavar":
        if len(sexp) != 1:
            raise ValueError(
                f"Malformed :metavar S-expression: expected 1 field, got {len(sexp)}"
            )
        return dag.get_or_create(":metavar", (), node_type="meta")

    # Preserve unknown tagged Lean expression forms as tagged structural
    # nodes. This is safer than treating the tag itself as a callable term.
    if isinstance(head, str) and head.startswith(":"):
        children = tuple(_sexp_walk(item, ctx, dag) for item in sexp[1:])
        return dag.get_or_create(head, children, node_type="sconst")

    # Application: (f a b ...) — first is function, rest are args
    if len(sexp) >= 2:
        fn_id = _sexp_walk(sexp[0], ctx, dag)
        children = [fn_id]
        for arg in sexp[1:]:
            children.append(_sexp_walk(arg, ctx, dag))
        return dag.get_or_create("App", tuple(children), node_type="sapp")

    return dag.get_or_create(str(sexp), ())


def _sexp_payload_text(value) -> str:
    if isinstance(value, list):
        return "(" + " ".join(_sexp_payload_text(item) for item in value) + ")"
    return str(value)


def _sexp_leaf(token: str, ctx: list[int], dag: DAGBuilder) -> int:
    """Handle a bare token (not a list)."""
    # De Bruijn index (bare number)
    token_text = str(token)
    if token_text.isdigit() or (token_text.startswith("-") and token_text[1:].isdigit()):
        idx = int(token_text)
        if 0 <= idx < len(ctx):
            return ctx[idx]
        return dag.get_or_create(f"?db-{idx}", (), node_type="var")

    # Named constant
    return dag.get_or_create(token_text, (), node_type="sconst")


# ---------------------------------------------------------------------------
# Proof state → DAG (supports both old text parser and new S-expression path)
# ---------------------------------------------------------------------------

def proof_state_to_dag(
    state: str | ProofState,
    *,
    sexp: str | None = None,
    goal_sexp: str | None = None,
    hyp_sexps: list[tuple[str, str | None] | dict[str, object]] | None = None,
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
        if dag.expression_root_id is None:
            raise ValueError("Goal S-expression did not produce an expression root.")
        goal_expr_id = dag.expression_root_id

        root_ids: list[int] = []
        # Pantograph's local context is authoritative here. Text states may
        # contain branch labels such as ``case a.mk`` which are not hypotheses;
        # zipping the two sources shifted every following type.
        for hypothesis in hyp_sexps:
            if isinstance(hypothesis, dict):
                hyp_name = str(hypothesis.get("name", "_"))
                hyp_sexp = hypothesis.get("sexp")
                context_index = hypothesis.get("context_index")
                binder_role = str(hypothesis.get("binder_role", ":explicit"))
                is_instance = bool(hypothesis.get("is_instance", False))
                is_let = bool(hypothesis.get("is_let", False))
            else:
                hyp_name, hyp_sexp = hypothesis
                context_index = None
                binder_role = ":explicit"
                is_instance = is_let = False

            name_node = dag.get_or_create(
                hyp_name or "_",
                (),
                node_type="sconst" if isinstance(context_index, int) else "var",
            )
            if hyp_sexp:
                type_node = _sexp_walk(parse_sexp_string(str(hyp_sexp)), [], dag)
            else:
                type_node = dag.get_or_create("?", ())
            if isinstance(context_index, int):
                context_node = dag.get_or_create(
                    f"FV{context_index}", (), node_type="var"
                )
                role = (
                    "instance"
                    if is_instance
                    else "let"
                    if is_let
                    else binder_role.removeprefix(":")
                )
                role_node = dag.get_or_create(
                    f"HypRole:{role}", (), node_type="sconst"
                )
                hyp_children = (context_node, name_node, role_node, type_node)
            else:
                hyp_children = (name_node, type_node)
            hyp_node = dag.get_or_create("Hyp", hyp_children)
            root_ids.append(hyp_node)

        goal_node = dag.get_or_create("Goal", (goal_expr_id,))
        root_ids.append(goal_node)
        dag.state_root_id = dag.get_or_create("State", tuple(root_ids))
        return dag

    if sexp is not None:
        # Goal has S-expression, hypothesis types use text parser
        dag = sexp_to_dag(sexp)
        if dag.expression_root_id is None:
            raise ValueError("Goal S-expression did not produce an expression root.")
        goal_expr_id = dag.expression_root_id

        from .parser import ExprParser
        _hyp_parser = ExprParser(dag)

        root_ids: list[int] = []
        for hypothesis in parsed.hypotheses:
            name_node = dag.get_or_create(hypothesis.name, ())
            type_node = _hyp_parser.parse(hypothesis.type_expr) if hypothesis.type_expr else dag.get_or_create("?", ())
            hyp_node = dag.get_or_create("Hyp", (name_node, type_node))
            root_ids.append(hyp_node)

        goal_node = dag.get_or_create("Goal", (goal_expr_id,))
        root_ids.append(goal_node)
        dag.state_root_id = dag.get_or_create("State", tuple(root_ids))
        return dag

    # Old path: text-based parser (offline, backward compatible)
    from .parser import ExprParser

    dag = DAGBuilder()
    parser = ExprParser(dag)
    root_ids = []

    for hypothesis in parsed.hypotheses:
        name_node = dag.get_or_create(hypothesis.name, ())
        type_node = parser.parse(hypothesis.type_expr) if hypothesis.type_expr else dag.get_or_create("?", ())
        hyp_node = dag.get_or_create("Hyp", (name_node, type_node))
        root_ids.append(hyp_node)

    goal_expr_node = parser.parse(parsed.goal)
    goal_node = dag.get_or_create("Goal", (goal_expr_node,))
    root_ids.append(goal_node)
    dag.state_root_id = dag.get_or_create("State", tuple(root_ids))

    return dag


def model_goal_to_dag(goal) -> DAGBuilder:
    """Build the normalized training graph for one canonical structured Goal."""
    goal.require_model_state()
    dag = proof_state_to_dag(
        goal.expression,
        goal_sexp=goal.goal_model_sexp,
        hyp_sexps=goal.graph_locals(),
    )
    if dag.state_root_id is None:
        raise ValueError("Structured goal did not produce a State root.")
    return dag


def lemma_statement_to_dag(statement: str, *, sexp: str | None = None) -> DAGBuilder:
    """Build a DAG for a lemma statement treated as a goal-only proof state.

    If *sexp* is provided, uses the new Lean AST parser.
    Otherwise falls back to the old text-based parser.
    """
    if sexp is not None:
        dag = sexp_to_dag(sexp)
        if dag.expression_root_id is None:
            raise ValueError("Lemma S-expression did not produce an expression root.")
        goal_node = dag.get_or_create("Goal", (dag.expression_root_id,))
        dag.state_root_id = dag.get_or_create("State", (goal_node,))
        return dag

    from .parser import ExprParser

    dag = DAGBuilder()
    parser = ExprParser(dag)
    goal_expr_node = parser.parse(statement)
    goal_node = dag.get_or_create("Goal", (goal_expr_node,))
    dag.state_root_id = dag.get_or_create("State", (goal_node,))
    return dag


def dag_to_dict(dag: DAGBuilder, metadata: dict[str, object] | None = None) -> dict[str, object]:
    child_counts = dag.incoming_counts()
    parent_uses = dag.outgoing_counts()
    root_ids = {node.id for node in dag.root_nodes()}
    leaf_ids = {node.id for node in dag.leaf_nodes()}

    return {
        "metadata": metadata or {},
        "stats": dag.stats().as_dict(),
        "expression_root_id": dag.expression_root_id,
        "state_root_id": dag.state_root_id,
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


def dag_fingerprint(dag: DAGBuilder) -> str:
    payload = {
        "nodes": [node.as_dict() for node in dag.nodes],
        "edges": dag.edges,
        "expression_root_id": dag.expression_root_id,
        "state_root_id": dag.state_root_id,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_dag_json(
    dag: DAGBuilder,
    output_path: str | Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dag_to_dict(dag, metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    return output
