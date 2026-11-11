from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from maths_ai.pln_inference.metta.translator.translator_modules.parser import (
    parse_sexp_string,
)

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
    if label in ("App", "SApp", "Arrow", "Forall", "Explicit", ":app"):
        return "app"
    if label in ("Binder", "SBinder", ":binder", ":forall", ":lambda", ":let"):
        return "binder"
    if label in ("Hyp", "Goal", "State", "Meta"):
        return "meta"
    if label in ("Type", "Sort", ":sort", ":type"):
        return "type"
    if label in ("Const", "SConst", ":c"):
        return "const"
    if label == "\u2115" or (label[0].isupper() and len(label) <= 2):
        return "type"
    if label[0].isupper():
        return "predicate"
    if label in ("+", "-", "*", "/", "=", "\u2264", "\u2265", "<", ">", "\u2227", "\u2228", "\u00ac"):
        return "operator"
    # Pantograph S-expression labels — the canonical lean-model-sexp-v2 tags.
    if label.startswith(":"):
        if label in (":forall", ":lambda", ":let"):
            return "sbinder"
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


def root_state_node_id(dag: DAGBuilder) -> int:
    """Return structural proof-state root, ignoring constants named ``State``."""
    root_ids = {node.id for node in dag.root_nodes()}
    candidates = [
        node.id for node in dag.nodes if node.label == "State" and node.id in root_ids
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one root State node, found {len(candidates)}")
    return candidates[0]


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
# S-expression → DAG conversion (canonical lean-model-sexp-v2 builder)
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
        dag.get_or_create("State", tuple(root_ids))
        return dag

    if sexp is not None:
        # Goal has S-expression, hypothesis types use text parser
        dag = sexp_to_dag(sexp)
        if dag.expression_root_id is None:
            raise ValueError("Goal S-expression did not produce an expression root.")
        goal_expr_id = dag.expression_root_id

        _hyp_parser = ExprParser(dag)

        root_ids = []
        for hypothesis in parsed.hypotheses:
            name_node = dag.get_or_create(hypothesis.name, ())
            type_node = _hyp_parser.parse(hypothesis.type_expr) if hypothesis.type_expr else dag.get_or_create("?", ())
            hyp_node = dag.get_or_create("Hyp", (name_node, type_node))
            root_ids.append(hyp_node)

        goal_node = dag.get_or_create("Goal", (goal_expr_id,))
        root_ids.append(goal_node)
        dag.get_or_create("State", tuple(root_ids))
        return dag

    # Old path: text-based parser (offline, backward compatible)
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
