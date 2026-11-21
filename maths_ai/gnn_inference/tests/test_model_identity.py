"""Model identity of goal locals versus their live execution identity."""

from __future__ import annotations

import pytest

from maths_ai.data_models.model_identity import (
    carry_model_identity,
    relabel_free_variables,
    with_model_identity,
)
from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.inference import (
    _resolve_local_node_name,
    local_declarations,
)


def local(name, internal, index, sexp="(:c Nat)", **extra) -> GoalLocal:
    return GoalLocal(
        user_name=name,
        internal_name=internal,
        context_index=index,
        binder_role=":explicit",
        type_pp="ℕ",
        type_model_sexp=sexp,
        type_sexp="(:c Nat)",
        **extra,
    )


def goal(locals_, target="(:app (:c Eq) (:fv FV0) (:fv FV1))") -> Goal:
    return Goal(
        expression="x = y",
        goal_model_sexp=target,
        goal_sexp="(:c True)",
        locals=locals_,
        model_sexp_version=1,
    )


# A replayed context is dense; the source had a hidden declaration at index 1.
REPLAYED = goal(
    [
        local("x", "_live.1", 0),
        local("inst_1✝", "_live.2", 1, sexp="(:app (:c Inhabited) (:fv FV0))"),
    ]
)
SOURCE = [("x", 0), ("inst✝¹", 2)]


def test_relabelling_changes_only_free_variable_labels():
    sexp = '(:app (:c "FV1") (:fv FV1) (:fv FV?) (:lit "(:fv FV1)"))'
    assert relabel_free_variables(sexp, lambda k: k + 5) == (
        '(:app (:c "FV1") (:fv FV6) (:fv FV?) (:lit "(:fv FV1)"))'
    )
    assert relabel_free_variables(sexp, lambda k: k) == sexp


def test_source_identity_replaces_live_names_and_indices_for_the_model():
    mapped = with_model_identity(REPLAYED, SOURCE)
    assert [(l.user_name, l.context_index) for l in mapped.locals] == SOURCE
    assert mapped.goal_model_sexp == "(:app (:c Eq) (:fv FV0) (:fv FV2))"
    assert mapped.locals[1].type_model_sexp == "(:app (:c Inhabited) (:fv FV0))"
    # Lean's identity is kept for execution and replay.
    assert [(l.execution_or_live_name, l.execution_or_live_index) for l in mapped.locals] == [
        ("x", 0),
        ("inst_1✝", 1),
    ]
    assert mapped.locals[0].execution_name is None and mapped.locals[0].execution_index is None
    assert local_declarations(mapped) == {"FV0": "x", "FV2": "inst_1✝"}
    assert [l.internal_name for l in mapped.locals] == ["_live.1", "_live.2"]


def test_model_identity_must_be_complete_and_ordered():
    with pytest.raises(ValueError):
        with_model_identity(REPLAYED, SOURCE[:1])
    with pytest.raises(ValueError):
        with_model_identity(REPLAYED, [("x", 2), ("inst✝¹", 0)])


def test_successors_keep_carried_identities_and_number_new_locals_after_them():
    parent = with_model_identity(REPLAYED, SOURCE)
    child = goal(
        [
            local("x", "_live.1", 0),
            local("inst_1✝", "_live.2", 1, sexp="(:app (:c Inhabited) (:fv FV0))"),
            local("h", "_live.9", 2, sexp="(:app (:c Eq) (:fv FV0) (:fv FV1))"),
        ],
        target="(:app (:c P) (:fv FV2))",
    )
    carried = carry_model_identity(parent, child)
    assert [(l.user_name, l.context_index) for l in carried.locals] == [
        ("x", 0),
        ("inst✝¹", 2),
        ("h", 3),
    ]
    assert carried.locals[2].type_model_sexp == "(:app (:c Eq) (:fv FV0) (:fv FV2))"
    assert carried.goal_model_sexp == "(:app (:c P) (:fv FV3))"
    assert carried.locals[2].execution_index == 2


def test_a_goal_without_a_mapping_keeps_leans_own_indices():
    parent = goal([local("x", "_live.1", 0), local("y", "_live.2", 3)])
    child = goal(
        [local("x", "_live.1", 0), local("y", "_live.2", 3), local("z", "_live.5", 7)],
        target="(:fv FV7)",
    )
    carried = carry_model_identity(parent, child)
    assert [l.context_index for l in carried.locals] == [0, 3, 7]
    assert carried.goal_model_sexp == "(:fv FV7)"
    assert all(l.execution_name is None and l.execution_index is None for l in carried.locals)


def test_renamed_and_removed_locals_follow_the_tactic():
    parent = with_model_identity(REPLAYED, SOURCE)
    # `rename_i h` gave the instance a new name; `x` was cleared.
    child = goal(
        [local("h", "_live.2", 0, sexp="(:c Inhabited)")], target="(:fv FV0)"
    )
    carried = carry_model_identity(parent, child)
    assert [(l.user_name, l.context_index) for l in carried.locals] == [("h", 2)]
    assert carried.goal_model_sexp == "(:fv FV2)"


def test_replay_seed_uses_live_identity_and_carries_the_model_one():
    seed = with_model_identity(REPLAYED, SOURCE).replay_seed()
    assert [(l.name, l.context_index) for l in seed.locals] == [("x", 0), ("inst_1✝", 1)]
    assert [l.model_identity for l in seed.locals] == SOURCE


# Argument resolution by declaration identity ------------------------------------


def outer_aliased() -> Goal:
    """The review case after replay: the outer local is model `x` at index 5,
    live `x_1` (a nested `∀ x` refers to it)."""
    return with_model_identity(goal([local("x_1", "_live.1", 0)], "(:c True)"), [("x", 5)])


def after_intro_x(parent: Goal) -> Goal:
    child = goal(
        [local("x_1", "_live.1", 0), local("x", "_live.7", 1)],
        target="(:app (:c Eq) (:fv FV0) (:fv FV0))",
    )
    return carry_model_identity(parent, child)


def resolve_all(goal_: Goal) -> dict[str, tuple[str, str]]:
    """Each structured local's Hyp and FV node, resolved to a live name."""
    dag = model_goal_to_dag(goal_)
    declarations = local_declarations(goal_)
    resolved = {}
    for node in dag.nodes:
        if node.label == "Hyp" and len(node.children) == 4:
            label = dag.nodes[node.children[0]].label
            fv = dag.nodes[node.children[0]]
            resolved[label] = (
                _resolve_local_node_name(node, dag, declarations),
                _resolve_local_node_name(fv, dag, declarations),
            )
    return resolved


def test_review_case_each_selected_local_resolves_to_its_own_declaration():
    child = after_intro_x(outer_aliased())
    assert [(l.user_name, l.context_index, l.execution_or_live_name) for l in child.locals] == [
        ("x✝", 5, "x_1"),
        ("x", 6, "x"),
    ]
    assert resolve_all(child) == {"FV5": ("x_1", "x_1"), "FV6": ("x", "x")}


def test_resolution_survives_rematerialization():
    child = after_intro_x(outer_aliased())
    seed = child.replay_seed()
    assert [(l.name, l.model_identity) for l in seed.locals] == [
        ("x_1", ("x✝", 5)),
        ("x", ("x", 6)),
    ]
    # Lean gives the re-materialized locals new IDs; the model identity returns.
    replayed = goal(
        [local("x_1", "_new.3", 0), local("x", "_new.4", 1)],
        target="(:app (:c Eq) (:fv FV0) (:fv FV0))",
    )
    again = with_model_identity(replayed, [l.model_identity for l in seed.locals])
    assert again.state_fingerprint() == child.state_fingerprint()
    assert resolve_all(again) == {"FV5": ("x_1", "x_1"), "FV6": ("x", "x")}
    assert [l.internal_name for l in again.locals] == ["_new.3", "_new.4"]


def test_shadowing_renaming_removal_and_alternative_branches():
    parent = outer_aliased()
    # Two branches from the same state are carried independently.
    with_y = carry_model_identity(
        parent, goal([local("x_1", "_live.1", 0), local("y", "_live.8", 1)], "(:c True)")
    )
    assert resolve_all(with_y) == {"FV5": ("x_1", "x_1"), "FV6": ("y", "y")}
    with_x = after_intro_x(parent)
    assert resolve_all(with_x)["FV6"] == ("x", "x")
    # A second shadowing marks the older locals x✝¹ and x✝, newest last.
    twice = carry_model_identity(
        with_x,
        goal(
            [local("x_1", "_live.1", 0), local("x", "_live.7", 1), local("x", "_live.9", 2)],
            "(:c True)",
        ),
    )
    assert [l.user_name for l in twice.locals] == ["x✝¹", "x✝", "x"]
    assert resolve_all(twice) == {
        "FV5": ("x_1", "x_1"),
        "FV6": ("x", "x"),
        "FV7": ("x", "x"),
    }
    # An explicit rename of the outer local is followed by both identities.
    renamed = carry_model_identity(
        with_x, goal([local("h", "_live.1", 0), local("x", "_live.7", 1)], "(:c True)")
    )
    assert [(l.user_name, l.context_index) for l in renamed.locals] == [("h", 5), ("x", 6)]
    assert resolve_all(renamed) == {"FV5": ("h", "h"), "FV6": ("x", "x")}
    # A removed local has no declaration left to resolve.
    cleared = carry_model_identity(with_x, goal([local("x", "_live.7", 1)], "(:c True)"))
    assert resolve_all(cleared) == {"FV6": ("x", "x")}
    dag = model_goal_to_dag(cleared)
    hyp = next(n for n in dag.nodes if n.label == "Hyp")
    with pytest.raises(ValueError):
        _resolve_local_node_name(hyp, dag, {"FV5": "x_1"})


def test_a_structured_local_is_never_resolved_by_its_display_name():
    child = after_intro_x(outer_aliased())
    dag = model_goal_to_dag(child)
    hyp = next(n for n in dag.nodes if n.label == "Hyp")
    with pytest.raises(ValueError):
        _resolve_local_node_name(hyp, dag, None)
