from types import SimpleNamespace

import pytest

from maths_ai.hybrid_reasoner.pantograph_adapter import (
    model_state_options,
    pantograph_state_to_goals,
)


def _expr(pp: str, sexp: str) -> SimpleNamespace:
    return SimpleNamespace(pp=pp, modelSexp=sexp, modelSexpVersion=1)


def test_model_state_options_force_required_fields() -> None:
    assert model_state_options({"printExprAST": False, "custom": 1}) == {
        "printExprAST": True,
        "printExprModelAST": True,
        "custom": 1,
    }


def test_pantograph_state_conversion_preserves_local_identity() -> None:
    state = SimpleNamespace(
        goals=[
            SimpleNamespace(
                name="goal",
                target=_expr("p", "(sort Prop)"),
                variables=[
                    SimpleNamespace(
                        userName="inst",
                        name="inst✝",
                        contextIndex=0,
                        binderRole="default",
                        isInstance=True,
                        isLet=False,
                        type=_expr("P", "Prop"),
                        value=None,
                    )
                ],
            )
        ]
    )
    goals = pantograph_state_to_goals(state)
    assert goals[0].locals[0].internal_name == "inst✝"
    assert goals[0].locals[0].is_instance is True
    assert goals[0].hypotheses == ["inst : P"]


def test_pantograph_state_conversion_fails_closed_when_model_fields_missing() -> None:
    state = SimpleNamespace(
        goals=[SimpleNamespace(target=SimpleNamespace(pp="p"), variables=[])]
    )
    with pytest.raises(ValueError, match="modelSexp"):
        pantograph_state_to_goals(state)


def test_pantograph_state_conversion_accepts_raw_client_vars_name() -> None:
    state = {
        "goals": [
            {
                "target": {
                    "pp": "p",
                    "modelSexp": "(sort Prop)",
                    "modelSexpVersion": 1,
                },
                "vars": [],
            }
        ]
    }
    assert pantograph_state_to_goals(state)[0].goal_model_sexp == "(sort Prop)"
