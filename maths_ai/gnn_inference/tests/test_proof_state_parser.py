"""Printed proof states and the shared S-expression reader."""

from __future__ import annotations

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.state import parse_state, parse_states
from maths_ai.lean_replay import ReplayFailure, ReplayInputError
from maths_ai.lean_replay.sexp import parse_sexp_string
from maths_ai.lean_replay.sexp import (
    parse_sexp_string as translator_reader,
)


def reason(call) -> ReplayFailure:
    with pytest.raises(ReplayInputError) as error:
        call()
    return error.value.reason


# Printed text ------------------------------------------------------------------


def test_issue_reproduction_keeps_a_multiline_hypothesis_whole():
    state = parse_state("n : Nat\nh : n =\n  n\n⊢ n = n")
    assert state.as_dict() == {
        "hypotheses": [{"name": "n", "type": "Nat"}, {"name": "h", "type": "n = n"}],
        "goal": "n = n",
    }


def test_unicode_and_ascii_turnstiles_and_multiline_targets():
    assert parse_state("n : Nat\n|- Even\n    n").goal == "Even n"
    assert parse_state("n : Nat\n⊢ Even n").goal == "Even n"


def test_case_headers_versus_a_variable_named_case():
    state = parse_state("case h.mk\ncase : Nat\n⊢ case = case")
    assert state.case_tag == "h.mk"
    assert [h.name for h in state.hypotheses] == ["case"]


def test_grouped_names_values_colons_strings_and_comments():
    state = parse_state(
        "n m : Nat\n"
        "f : ∀ (x : Nat), x = x\n"
        's : String := "a : b"  -- not a declaration\n'
        "⊢ n = m"
    )
    assert [(h.name, h.type_expr, h.value) for h in state.hypotheses] == [
        ("n m", "Nat", None),
        ("f", "∀ (x : Nat), x = x", None),
        ("s", "String", '"a : b"'),
    ]
    assert state.hypotheses[0].names == ("n", "m")


def test_quoted_names_keep_their_spaces_and_colons():
    state = parse_state("«h : x» : 1 = 1\n⊢ «h : x» = «h : x»")
    assert [(h.name, h.type_expr) for h in state.hypotheses] == [("«h : x»", "1 = 1")]
    assert state.goal == "«h : x» = «h : x»"


def test_several_goals_are_reported_not_merged():
    text = "case a\nn : Nat\n⊢ n = n\n\ncase b\n⊢ True"
    assert [s.case_tag for s in parse_states(text)] == ["a", "b"]
    assert reason(lambda: parse_state(text)) is ReplayFailure.MALFORMED_TEXT


@pytest.mark.parametrize(
    "text",
    [
        "n : Nat\nthis line has no colon\n⊢ True",  # no invented `Prop`
        "n : (Nat\n⊢ True",
        "n : Nat\n⊢ True\nm : Nat",  # declarations after the last target
        "h x : Nat := 1\n⊢ True",
    ],
)
def test_malformed_text_is_rejected_with_a_position(text):
    with pytest.raises(ReplayInputError) as error:
        parse_state(text)
    assert error.value.reason is ReplayFailure.MALFORMED_TEXT


def test_goal_only_text_is_a_bare_target():
    assert parse_state("1 + 1 = 2").as_dict() == {"hypotheses": [], "goal": "1 + 1 = 2"}


# Shared S-expression reader ----------------------------------------------------


def test_quoted_atoms_stay_whole():
    tree = parse_sexp_string('((:c "Foo.«a b»") (:lit "x ) y"))')
    assert tree == [[":c", '"Foo.«a b»"'], [":lit", '"x ) y"']]


def test_the_translator_uses_the_shared_reader():
    assert translator_reader is parse_sexp_string
    assert translator_reader("(a b c)") == ["a", "b", "c"]


@pytest.mark.parametrize("text", ['(:lit "a"b")', '(:lit "open', "(a b", "a b"])
def test_ambiguous_or_broken_input_is_rejected(text):
    with pytest.raises(ValueError):
        parse_sexp_string(text)
