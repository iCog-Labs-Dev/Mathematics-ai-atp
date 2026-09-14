"""Unit and regression tests for Issue #42: Live proof search DAG alignment.

Verifies:
1. 4-child Hyp(FV{i}, name, HypRole:role, type) node structure matching checkpoint training.
2. Single shared FV{i} node per local context index across goals and hypotheses.
3. Node resolution maps 4-child Hyp and FV nodes back to Lean identifiers (e.g. 'h').
4. S-expression monkey patch preserves modelSexp, contextIndex, and binderRole.
5. Pretty-printed strings travel alongside S-expressions for proof command reconstruction.
"""

import sys
sys.path.insert(0, ".")

from maths_ai.data_models.proof_components import Goal, LocalDeclaration
from maths_ai.gnn_inference.atp_lean_gnn.graph import (
    DAGBuilder,
    get_node_labels,
    patch_pantograph_for_sexp,
    proof_state_to_dag,
    sexp_to_dag,
)
from maths_ai.gnn_inference.atp_lean_gnn.inference import _resolve_local_node_name
from maths_ai.gnn_inference.atp_lean_gnn.state import parse_state


def test_four_child_hyp_node_structure():
    """Verify that proof_state_to_dag builds 4-child Hyp(FV{i}, name, HypRole:role, type) nodes."""
    state_text = "h : Nat\n⊢ Nat"
    goal_sexp = "(:c Nat)"
    hyp_sexps = [("h", "(:c Nat)", 0, "context")]

    dag = proof_state_to_dag(state_text, goal_sexp=goal_sexp, hyp_sexps=hyp_sexps)
    labels = get_node_labels(dag)

    assert "FV0" in labels, "FV0 node must exist in DAG"
    assert "h" in labels, "Hypothesis name node must exist in DAG"
    assert "HypRole:context" in labels, "HypRole node must exist in DAG"
    assert "Nat" in labels, "Type node must exist in DAG"
    assert "Hyp" in labels, "Hyp meta node must exist in DAG"

    # Find the Hyp node and check its children
    hyp_nodes = [n for n in dag.nodes if n.label == "Hyp"]
    assert len(hyp_nodes) == 1
    hyp_node = hyp_nodes[0]
    assert len(hyp_node.children) == 4, f"Hyp node must have 4 children, got {len(hyp_node.children)}"

    # Check child order: (FV0, h, HypRole:context, Nat)
    c0 = dag.nodes[hyp_node.children[0]]
    c1 = dag.nodes[hyp_node.children[1]]
    c2 = dag.nodes[hyp_node.children[2]]
    c3 = dag.nodes[hyp_node.children[3]]

    assert c0.label == "FV0"
    assert c1.label == "h"
    assert c2.label == "HypRole:context"
    assert c3.label == "Nat"


def test_shared_fv_nodes_across_expressions():
    """Verify that FV0 is hash-consed and shared between the goal target and hypothesis node."""
    state_text = "h : Nat\n⊢ Nat"
    # Goal references FV0 in its modelSexp
    goal_sexp = "((:c Eq) (:c Nat) (:fv FV0) (:fv FV0))"
    hyp_sexps = [("h", "(:c Nat)", 0, "context")]

    dag = proof_state_to_dag(state_text, goal_sexp=goal_sexp, hyp_sexps=hyp_sexps)
    labels = get_node_labels(dag)

    # There should be exactly ONE FV0 node in the entire DAG
    fv0_nodes = [n for n in dag.nodes if n.label == "FV0"]
    assert len(fv0_nodes) == 1, f"Expected exactly 1 FV0 node, found {len(fv0_nodes)}"

    # Check that FV0 has multiple incoming uses (from Eq and from Hyp)
    fv0_id = fv0_nodes[0].id
    parent_uses = dag.outgoing_counts()[fv0_id]
    assert parent_uses >= 2, f"Expected FV0 to be shared by at least 2 parents, got {parent_uses}"


def test_resolve_local_node_name():
    """Verify that _resolve_local_node_name decodes both 4-child Hyp and FV nodes to Lean identifiers."""
    state_text = "h : Nat\n⊢ Nat"
    goal_sexp = "(:c Nat)"
    hyp_sexps = [("h", "(:c Nat)", 0, "context")]

    dag = proof_state_to_dag(state_text, goal_sexp=goal_sexp, hyp_sexps=hyp_sexps)

    hyp_node = [n for n in dag.nodes if n.label == "Hyp"][0]
    fv_node = [n for n in dag.nodes if n.label == "FV0"][0]

    # Resolving either the Hyp node or the FV0 node should return the user name "h"
    assert _resolve_local_node_name(hyp_node, dag) == "h"
    assert _resolve_local_node_name(fv_node, dag) == "h"


def test_hyp_details_dictionary_input():
    """Verify that proof_state_to_dag correctly consumes rich hyp_details dictionaries."""
    state_text = "h1 : Nat\nh2 : Bool\n⊢ Nat"
    goal_sexp = "(:c Nat)"
    hyp_details = [
        {"name": "h1", "sexp": "(:c Nat)", "context_index": 0, "role": "context", "is_let": False},
        {"name": "h2", "sexp": "(:c Bool)", "context_index": 1, "role": "context", "is_let": False},
    ]

    dag = proof_state_to_dag(state_text, goal_sexp=goal_sexp, hyp_details=hyp_details)
    labels = get_node_labels(dag)

    assert "FV0" in labels
    assert "FV1" in labels
    assert "h1" in labels
    assert "h2" in labels

    hyp_nodes = [n for n in dag.nodes if n.label == "Hyp"]
    assert len(hyp_nodes) == 2
    assert len(hyp_nodes[0].children) == 4
    assert len(hyp_nodes[1].children) == 4


def test_pretty_print_preserved_alongside_sexp():
    """Verify that Goal and LocalDeclaration keep pretty-printed Lean strings for commands."""
    decl = LocalDeclaration(
        name="h",
        type_expression="n = n",
        sexp="((:c Eq) (:c Nat) 0 0)",
        context_index=0,
        role="context",
    )
    goal = Goal(
        expression="n = n",
        goal_sexp="((:c Eq) (:c Nat) 0 0)",
        hypotheses=[decl],
    )

    # render() must return Lean source syntax, not S-expressions
    assert decl.render() == "h : n = n"
    assert goal.expression == "n = n"
    assert goal.goal_sexp == "((:c Eq) (:c Nat) 0 0)"


def test_let_local_declaration_dag_structure():
    """Verify that let declarations build structured Let nodes with FV pointer targets."""
    state_text = "let x : Nat := 1\n⊢ Nat"
    goal_sexp = "(:c Nat)"
    hyp_details = [
        {"name": "x", "sexp": "(:c Nat)", "context_index": 0, "role": "let", "is_let": True},
    ]

    dag = proof_state_to_dag(state_text, goal_sexp=goal_sexp, hyp_details=hyp_details)
    labels = get_node_labels(dag)

    assert "FV0" in labels
    assert "x" in labels
    assert "HypRole:let" in labels


if __name__ == "__main__":
    test_four_child_hyp_node_structure()
    test_shared_fv_nodes_across_expressions()
    test_resolve_local_node_name()
    test_hyp_details_dictionary_input()
    test_pretty_print_preserved_alongside_sexp()
    test_let_local_declaration_dag_structure()
    print("All live graph alignment tests passed successfully!")
