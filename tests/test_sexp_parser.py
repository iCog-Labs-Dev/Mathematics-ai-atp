"""Comprehensive tests for the S-expression → DAG parser."""
import sys
sys.path.insert(0, ".")

from maths_ai.gnn_inference.atp_lean_gnn.graph import (
    BINDER_KIND_FORALL,
    BINDER_KIND_LAMBDA,
    BINDER_KIND_NONE,
    DAGBuilder,
    get_node_labels,
    proof_state_to_dag,
    sexp_to_dag,
)


# ---------------------------------------------------------------------------
# S-expression → DAG conversion
# ---------------------------------------------------------------------------

def test_simple_forall():
    dag = sexp_to_dag("(:forall q (:sort 0) 0)")
    labels = get_node_labels(dag)
    assert ":forall" in labels
    assert "Prop" in labels
    assert "q" in labels


def test_nested_forall():
    dag = sexp_to_dag("(:forall p (:sort 0) (:forall q (:sort 0) ((:c Or) 1 0)))")
    labels = get_node_labels(dag)
    assert labels.count(":forall") == 2
    assert "Or" in labels
    assert "p" in labels
    assert "q" in labels


def test_arrow():
    dag = sexp_to_dag("(:forall a (:c Nat) (:forall a (:c Nat) (:c Nat)))")
    labels = get_node_labels(dag)
    assert labels.count(":forall") == 2
    assert "Nat" in labels


def test_constants():
    dag = sexp_to_dag("((:c Eq) (:c Nat) 0 0)")
    labels = get_node_labels(dag)
    assert "Eq" in labels
    assert "Nat" in labels


def test_lambda():
    dag = sexp_to_dag("(:lambda a 3 ((:c And) (3 0) ((:c Not) (2 0))))")
    labels = get_node_labels(dag)
    assert ":lambda" in labels
    assert "And" in labels
    assert "Not" in labels


def test_complex_expression():
    sexp = "(:forall U (:sort 1) (:forall P (:forall a 0 (:sort 0)) (:forall Q (:forall a 1 (:sort 0)) (:forall a (:forall x 2 (:forall a (2 0) (2 1))) ((:c Not) ((:c Exists) 3 (:lambda a 3 ((:c And) (3 0) ((:c Not) (2 0))))))))))"
    dag = sexp_to_dag(sexp)
    labels = get_node_labels(dag)
    assert "Not" in labels
    assert "Exists" in labels
    assert "And" in labels
    assert dag.num_nodes > 5
    assert dag.num_edges > 0


def test_hash_consing():
    dag = sexp_to_dag("((:c Or) 1 0)")
    dag2 = sexp_to_dag("((:c Or) 1 0)")
    assert dag.num_nodes == dag2.num_nodes


def test_node_count():
    dag = sexp_to_dag("(:forall n (:c Nat) ((:c Eq) (:c Nat) 0 0))")
    assert dag.num_nodes == 5


def test_node_count_no_sharing():
    dag = sexp_to_dag("(:forall n (:c Nat) ((:c Eq) (:c Nat) (:c Bool) 0))")
    assert dag.num_nodes == 6


# ---------------------------------------------------------------------------
# Binder annotations
# ---------------------------------------------------------------------------

def test_forall_binder_annotation():
    dag = sexp_to_dag("(:forall q (:sort 0) 0)")
    bound = [n for n in dag.nodes if n.is_bound == 1]
    assert len(bound) >= 1
    assert bound[0].binder_kind == BINDER_KIND_FORALL
    assert bound[0].binder_depth == 1


def test_nested_binder_depth():
    dag = sexp_to_dag("(:forall p (:sort 0) (:forall q (:sort 0) ((:c Or) 1 0)))")
    bound = [n for n in dag.nodes if n.is_bound == 1]
    depths = {n.label: n.binder_depth for n in bound}
    assert depths.get("p") == 1
    assert depths.get("q") == 2


def test_lambda_binder_annotation():
    dag = sexp_to_dag("(:lambda a 3 ((:c And) (3 0) ((:c Not) (2 0))))")
    bound = [n for n in dag.nodes if n.is_bound == 1]
    assert any(n.binder_kind == BINDER_KIND_LAMBDA for n in bound)


def test_context_variables_not_bound():
    dag = sexp_to_dag("((:c Or) ((:c Nat)) ((:c Bool)))")
    for n in dag.nodes:
        assert n.is_bound == BINDER_KIND_NONE


# ---------------------------------------------------------------------------
# De Bruijn index resolution
# ---------------------------------------------------------------------------

def test_debruijn_resolves_to_binder():
    dag = sexp_to_dag("(:forall n (:c Nat) ((:c Eq) (:c Nat) 0 0))")
    labels = get_node_labels(dag)
    # de Bruijn 0 should resolve to "n", not appear as a number
    assert "0" not in labels


def test_debruijn_multiple_indices():
    dag = sexp_to_dag("(:forall p (:sort 0) (:forall q (:sort 0) ((:c Or) 1 0)))")
    labels = get_node_labels(dag)
    assert "0" not in labels
    assert "1" not in labels
    assert "p" in labels
    assert "q" in labels


# ---------------------------------------------------------------------------
# Free variables and sorts
# ---------------------------------------------------------------------------

def test_free_variable():
    dag = sexp_to_dag("(:fv _uniq.28)")
    labels = get_node_labels(dag)
    assert "_uniq.28" in labels


def test_sort_prop():
    dag = sexp_to_dag("(:sort 0)")
    labels = get_node_labels(dag)
    assert "Prop" in labels


def test_sort_type():
    dag = sexp_to_dag("(:sort 1)")
    labels = get_node_labels(dag)
    assert "Type" in labels


# ---------------------------------------------------------------------------
# Edge structure
# ---------------------------------------------------------------------------

def test_edges_exist():
    dag = sexp_to_dag("((:c Or) 1 0)")
    assert len(dag.edges) > 0
    for child_id, parent_id in dag.edges:
        assert child_id < len(dag.nodes)
        assert parent_id < len(dag.nodes)


def test_root_nodes():
    dag = sexp_to_dag("(:forall q (:sort 0) 0)")
    roots = dag.root_nodes()
    assert len(roots) >= 1


def test_leaf_nodes():
    dag = sexp_to_dag("(:forall q (:sort 0) 0)")
    leaves = dag.leaf_nodes()
    assert len(leaves) >= 1


# ---------------------------------------------------------------------------
# Proof state with sexp param
# ---------------------------------------------------------------------------

def test_proof_state_with_sexp():
    state = "h : Nat\n⊢ Nat"
    sexp = "(:forall a (:c Nat) (:c Nat))"
    dag = proof_state_to_dag(state, sexp=sexp)
    assert dag.num_nodes > 0
    labels = get_node_labels(dag)
    assert "Nat" in labels
    assert "h" in labels
    assert "Goal" in labels
    assert "Hyp" in labels
    assert "State" in labels


def test_proof_state_without_sexp_fallback():
    state = "h : Nat\n⊢ Nat"
    dag = proof_state_to_dag(state)
    assert dag.num_nodes > 0
    labels = get_node_labels(dag)
    assert "Nat" in labels


# ---------------------------------------------------------------------------
# DAGBuilder interface
# ---------------------------------------------------------------------------

def test_dag_builder_hash_consing():
    dag = DAGBuilder()
    id1 = dag.get_or_create("Nat", ())
    id2 = dag.get_or_create("Nat", ())
    assert id1 == id2
    assert dag.num_nodes == 1


def test_dag_builder_different_children():
    dag = DAGBuilder()
    nat = dag.get_or_create("Nat", ())
    bool_ = dag.get_or_create("Bool", ())
    id1 = dag.get_or_create("App", (nat, bool_))
    id2 = dag.get_or_create("App", (nat, nat))
    assert id1 != id2
    assert dag.num_nodes == 4  # Nat, Bool, App(nat,bool), App(nat,nat)


def test_dag_builder_stats():
    dag = sexp_to_dag("(:forall q (:sort 0) 0)")
    stats = dag.stats()
    assert stats.num_nodes > 0
    assert stats.num_edges > 0


if __name__ == "__main__":
    test_simple_forall()
    test_nested_forall()
    test_arrow()
    test_constants()
    test_lambda()
    test_complex_expression()
    test_hash_consing()
    test_node_count()
    test_node_count_no_sharing()
    test_forall_binder_annotation()
    test_nested_binder_depth()
    test_lambda_binder_annotation()
    test_context_variables_not_bound()
    test_debruijn_resolves_to_binder()
    test_debruijn_multiple_indices()
    test_free_variable()
    test_sort_prop()
    test_sort_type()
    test_edges_exist()
    test_root_nodes()
    test_leaf_nodes()
    test_proof_state_with_sexp()
    test_proof_state_without_sexp_fallback()
    test_dag_builder_hash_consing()
    test_dag_builder_different_children()
    test_dag_builder_stats()
    print("All tests passed!")
