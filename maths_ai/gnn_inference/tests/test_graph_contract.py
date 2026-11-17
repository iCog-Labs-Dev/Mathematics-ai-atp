from __future__ import annotations

import pytest

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import (
    MODEL_SEXPR_GRAPH_SPEC,
)


def _local(index: int, *, is_let: bool = False) -> GoalLocal:
    return GoalLocal(
        user_name="x",
        context_index=index,
        binder_role="explicit",
        is_let=is_let,
        type_pp="Nat",
        type_model_sexp="(:c Nat)",
        value_pp="0" if is_let else None,
        value_model_sexp="(:lit 0)" if is_let else None,
    )


def test_structured_goal_derives_hypotheses_and_preserves_local_identity():
    goal = Goal(
        expression="x = x",
        goal_model_sexp="(:app (:c Eq) (:fv FV0) (:fv FV0))",
        locals=[_local(0)],
        model_sexp_version=1,
    )

    assert goal.has_model_state
    assert goal.hypotheses == ["x : Nat"]
    assert goal.locals[0].context_index == 0


def test_structured_goal_rejects_unordered_or_duplicate_context_indices():
    with pytest.raises(ValueError, match="unique ordered"):
        Goal(
            expression="True",
            goal_model_sexp="(:c True)",
            locals=[_local(1), _local(0)],
            model_sexp_version=1,
        )


def test_graph_contract_identifies_required_live_format():
    assert MODEL_SEXPR_GRAPH_SPEC.representation == "lean-model-sexp-v2"
    assert MODEL_SEXPR_GRAPH_SPEC.hypothesis_schema == "fv-name-role-type-v1"
