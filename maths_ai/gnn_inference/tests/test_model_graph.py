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
