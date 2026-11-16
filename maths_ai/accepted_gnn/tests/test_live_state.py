from __future__ import annotations

import pytest

from maths_ai.accepted_gnn.live_state import (
    ModelSExprProtocolError,
    goal_payload_to_goal,
)


def _expression(pp: str, sexp: str) -> dict[str, object]:
    return {"pp": pp, "modelSexp": sexp, "modelSexpVersion": 1}


def test_goal_payload_preserves_shadowed_names_by_context_index():
    goal = goal_payload_to_goal(
        {
            "name": "case",
            "target": _expression("x = x", "(:app (:c Eq) (:fv FV1) (:fv FV1))"),
            "variables": [
                {
                    "name": "x✝",
                    "userName": "x",
                    "contextIndex": 0,
                    "binderRole": "explicit",
                    "type": _expression("Nat", "(:c Nat)"),
                },
                {
                    "name": "x",
                    "userName": "x",
                    "contextIndex": 1,
                    "binderRole": "explicit",
                    "type": _expression("Nat", "(:c Nat)"),
                },
            ],
        }
    )

    assert [local.context_index for local in goal.locals] == [0, 1]
    assert [local.internal_name for local in goal.locals] == ["x✝", "x"]
    assert goal.hypotheses == ["x : Nat", "x : Nat"]


def test_goal_payload_preserves_local_let_value():
    goal = goal_payload_to_goal(
        {
            "target": _expression("y = 0", "(:app (:c Eq) (:fv FV0) (:lit 0))"),
            "variables": [
                {
                    "name": "y",
                    "userName": "y",
                    "contextIndex": 0,
                    "binderRole": "let",
                    "isLet": True,
                    "type": _expression("Nat", "(:c Nat)"),
                    "value": _expression("0", "(:lit 0)"),
                }
            ],
        }
    )

    assert goal.locals[0].is_let is True
    assert goal.locals[0].value_model_sexp == "(:lit 0)"
    assert goal.hypotheses == ["y : Nat := 0"]


def test_goal_payload_rejects_missing_model_expression():
    with pytest.raises(ModelSExprProtocolError, match="modelSexp"):
        goal_payload_to_goal(
            {
                "target": {"pp": "True", "modelSexpVersion": 1},
                "variables": [],
            }
        )


def _raw(pp: str, model: str, raw: str) -> dict[str, object]:
    return {**_expression(pp, model), "sexp": raw}


def test_raw_expressions_from_the_wire_reach_pln_translation():
    from maths_ai.pln_inference.compiler import GoalInput

    goal = goal_payload_to_goal(
        {
            "target": _raw("p", "(:fv FV0)", "(:fv _uniq.1)"),
            "variables": [{
                "name": "_uniq.1", "userName": "p", "contextIndex": 0, "binderRole": "explicit",
                "type": _raw("Prop", "(:sort Prop)", "(:sort 0)"),
            }],
        }
    )
    assert goal.goal_sexp == "(:fv _uniq.1)"
    assert [local.type_sexp for local in goal.locals] == ["(:sort 0)"]
    structured = GoalInput.from_goal(goal)
    assert structured.target_sexp == "(:fv _uniq.1)"
    assert structured.locals[0].internal_name == "_uniq.1"


def test_raw_expressions_are_all_or_none():
    goal = goal_payload_to_goal(
        {
            "target": _raw("p", "(:fv FV0)", "(:fv _uniq.1)"),
            "variables": [{
                "name": "_uniq.1", "userName": "p", "contextIndex": 0, "binderRole": "explicit",
                "type": _expression("Prop", "(:sort Prop)"),
            }],
        }
    )
    assert goal.goal_sexp is None and goal.locals[0].type_sexp is None


def test_a_malformed_raw_expression_is_a_protocol_error():
    with pytest.raises(ModelSExprProtocolError):
        goal_payload_to_goal({"target": {**_expression("p", "(:c True)"), "sexp": ""}, "variables": []})
