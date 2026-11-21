"""Smoke test the typed translator facade without starting Lean."""
from maths_ai.pln_inference.compiler import parse_text, render, build_query
from maths_ai.pln_inference.answers import rule_answers
from maths_ai.pln_inference.metta.translator.translator_modules import compile_goal, GoalInput

formula = parse_text("P → Q", propositions={"P":"P", "Q":"Q"})
assert "Implication" in render(formula)
assert "query" in build_query(formula)
assert rule_answers("(: (rule-proof rule fact) (Q) (STV 0.8 0.9))", "(Q)") == [(0.8, 0.9)]
assert not rule_answers("(: fact (Q) (STV 1 1))", "(Q)")
