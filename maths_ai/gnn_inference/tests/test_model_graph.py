import pytest

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.gnn_inference.atp_lean_gnn.model_graph import (
    ModelSExprGraphError,
    goal_to_model_dag,
)


def _goal() -> Goal:
    return Goal(
        expression="P",
        goal_model_sexp="(:app (:c Eq) (:fv FV0) (:fv FV0))",
        locals=[
            GoalLocal(
                user_name="h",
                internal_name="h",
                context_index=0,
                binder_role="default",
                type_pp="P",
                type_model_sexp="(:c Prop)",
            )
        ],
        model_sexp_version=1,
    )


def test_goal_to_model_dag_preserves_state_and_hypothesis_roots() -> None:
    dag = goal_to_model_dag(_goal())
    labels = [node.label for node in dag.nodes]

    assert "State" in labels
    assert "Hyp" in labels
    assert "Goal" in labels
    assert dag.num_edges > 0


def test_goal_to_model_dag_rejects_unstructured_goal() -> None:
    with pytest.raises(ValueError, match="normalized"):
        goal_to_model_dag(Goal(expression="P"))


def test_model_graph_rejects_malformed_sexp() -> None:
    goal = _goal().model_copy(update={"goal_model_sexp": "(:app (:c Eq)"})

    with pytest.raises(ModelSExprGraphError, match="Unclosed"):
        goal_to_model_dag(goal)


# Canonical-builder parity (lean-model-sexp-v2, fv-name-role-type-v1) ---------


def test_goal_to_model_dag_uses_canonical_constant_nodes() -> None:
    dag = goal_to_model_dag(_goal())

    # (:c Eq) and (:c Prop) are each ONE const-typed node — the bespoke
    # builder turned constants like Nat/True into two predicate nodes.
    for name in ("Eq", "Prop"):
        matches = [node for node in dag.nodes if node.label == name]
        assert len(matches) == 1
        assert matches[0].node_type == "const"


def test_goal_to_model_dag_builds_structured_hypotheses() -> None:
    dag = goal_to_model_dag(_goal())
    hyp = next(node for node in dag.nodes if node.label == "Hyp")

    # fv-name-role-type-v1: Hyp(FV{i}, name, HypRole:role, type)
    assert len(hyp.children) == 4
    context_node, name_node, role_node, type_node = (
        dag.nodes[child] for child in hyp.children
    )
    assert (context_node.label, context_node.node_type) == ("FV0", "var")
    assert name_node.label == "h"
    assert role_node.label == "HypRole:default"
    assert (type_node.label, type_node.node_type) == ("Prop", "const")


def test_goal_to_model_dag_maps_binder_roles() -> None:
    goal = _goal().model_copy(
        update={
            "locals": [
                GoalLocal(
                    user_name="inst",
                    context_index=0,
                    binder_role=":implicit",
                    type_pp="P",
                    type_model_sexp="(:c Prop)",
                    is_instance=True,
                ),
                GoalLocal(
                    user_name="x",
                    context_index=1,
                    binder_role="default",
                    type_pp="Nat",
                    type_model_sexp="(:c Nat)",
                    is_let=True,
                    value_pp="0",
                    value_model_sexp="(:lit 0)",
                ),
            ]
        }
    )
    dag = goal_to_model_dag(goal)

    hyps = [node for node in dag.nodes if node.label == "Hyp"]
    assert [dag.nodes[hyp.children[2]].label for hyp in hyps] == [
        "HypRole:instance",
        "HypRole:let",
    ]
    # Context order is preserved in the State root's children.
    state = next(node for node in dag.nodes if node.label == "State")
    assert [dag.nodes[root].label for root in state.children] == ["Hyp", "Hyp", "Goal"]
