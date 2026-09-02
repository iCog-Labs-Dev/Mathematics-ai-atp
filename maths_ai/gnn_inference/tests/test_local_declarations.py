"""Regression tests for lossless local Lean context storage."""

from maths_ai.data_models.proof_components import Goal
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import goal_to_state
from maths_ai.gnn_inference.atp_lean_gnn.state import parse_state


def test_goal_keeps_ordinary_variables_and_local_definitions_distinct():
    goal = Goal(expression="x = 1", hypotheses=["x : Nat", "let y : Nat := x + 1"])

    assert goal.hypotheses[0].kind == "variable"
    assert goal.hypotheses[1].kind == "let"
    assert goal.hypotheses[1].value_expression == "x + 1"
    assert goal_to_state(goal) == "x : Nat\nlet y : Nat := x + 1\n⊢ x = 1"


def test_state_parser_keeps_a_local_definition_and_its_value():
    state = parse_state("let x : Nat := 1\nh : x = 1\n⊢ x = 1")

    assert state.hypotheses[0].is_local_definition
    assert state.hypotheses[0].value_expr == "1"
    assert state.hypotheses[1].name == "h"
