"""Engine-level tests for v2 candidate rendering and retrieval text policy."""

from types import SimpleNamespace

import pytest

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import (
    CandidateSource,
    DecodedAction,
    DecodedArgument,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_graph import goal_to_model_dag


def _structured_goal() -> Goal:
    return Goal(
        expression="n = n",
        goal_model_sexp="(sort Prop)",
        locals=[
            GoalLocal(
                user_name="h",
                context_index=0,
                binder_role="default",
                type_pp="Nat",
                type_model_sexp="(sort Type)",
            )
        ],
        model_sexp_version=1,
    )


def _render(bundle, action: DecodedAction) -> tuple[list[str], bool]:
    from maths_ai.gnn_inference.inference_engine import GNNModelEngine

    goal = _structured_goal()
    dag = goal_to_model_dag(goal)
    return GNNModelEngine._render_action(dag, list(dag.nodes), bundle, action)


def test_render_action_maps_local_hyp_and_library_lemma() -> None:
    dag = goal_to_model_dag(_structured_goal())
    hyp_row = next(
        index
        for index, node in enumerate(dag.nodes)
        if node.label == "Hyp" and node.children
    )
    bundle = SimpleNamespace(
        lemma_corpus={5: SimpleNamespace(name="Nat.add_assoc")},
    )
    rendered, complete = _render(
        bundle,
        DecodedAction(
            1,
            (
                DecodedArgument(CandidateSource.LOCAL, hyp_row, 0, 0.5),
                DecodedArgument(CandidateSource.LIBRARY, 5, 0, 0.4),
            ),
            True,
        ),
    )
    assert rendered == ["h", "Nat.add_assoc"]
    assert complete is True


def test_render_action_drops_unresolvable_identities_fail_closed() -> None:
    bundle = SimpleNamespace(lemma_corpus={})
    dag = goal_to_model_dag(_structured_goal())
    hyp_row = next(
        index
        for index, node in enumerate(dag.nodes)
        if node.label == "Hyp" and node.children
    )
    rendered, complete = _render(
        bundle,
        DecodedAction(1, (DecodedArgument(CandidateSource.LIBRARY, 99, 0, 0.4),), True),
    )
    assert rendered == [] and complete is False

    # A local identity that is not a nameable hypothesis cannot be stringified.
    structural_row = next(
        index
        for index, node in enumerate(dag.nodes)
        if node.label not in {"Hyp"} and not node.children
    )
    rendered, complete = _render(
        bundle,
        DecodedAction(1, (DecodedArgument(CandidateSource.LOCAL, structural_row, 0, 0.4),), True),
    )
    assert rendered == [] and complete is False

    # Out-of-range local identity never becomes a placeholder.
    rendered, complete = _render(
        bundle,
        DecodedAction(1, (DecodedArgument(CandidateSource.LOCAL, len(dag.nodes), 0, 0.4),), True),
    )
    assert rendered == [] and complete is False


def test_render_action_accepts_zero_argument_actions() -> None:
    bundle = SimpleNamespace(lemma_corpus={})
    assert _render(bundle, DecodedAction(1, (), True)) == ([], True)


def test_render_action_resolves_bare_fv_selections() -> None:
    from maths_ai.gnn_inference.inference_engine import GNNModelEngine

    bundle = SimpleNamespace(lemma_corpus={})
    dag = goal_to_model_dag(_structured_goal())
    fv_row = next(
        index for index, node in enumerate(dag.nodes) if node.label == "FV0"
    )

    # The pointer head can select the FV{i} node itself; rendering must map it
    # back to the Lean name instead of emitting the raw pointer label.
    rendered, complete = GNNModelEngine._render_action(
        dag, list(dag.nodes), bundle,
        DecodedAction(1, (DecodedArgument(CandidateSource.LOCAL, fv_row, 0, 0.5),), True),
    )
    assert rendered == ["h"]
    assert complete is True


def test_render_action_reads_structured_hypothesis_name() -> None:
    from maths_ai.gnn_inference.inference_engine import GNNModelEngine

    bundle = SimpleNamespace(lemma_corpus={})
    dag = goal_to_model_dag(_structured_goal())
    hyp_row = next(
        index
        for index, node in enumerate(dag.nodes)
        if node.label == "Hyp" and len(node.children) == 4
    )

    # Structured Hyp nodes carry the name at children[1] (children[0] is the
    # FV{i} pointer); reading children[0] would emit "FV0", which is not Lean.
    rendered, complete = GNNModelEngine._render_action(
        dag, list(dag.nodes), bundle,
        DecodedAction(1, (DecodedArgument(CandidateSource.LOCAL, hyp_row, 0, 0.5),), True),
    )
    assert rendered == ["h"]
    assert complete is True


def test_render_action_is_tactic_family_aware() -> None:
    from maths_ai.gnn_inference.inference_engine import GNNModelEngine

    bundle = SimpleNamespace(lemma_corpus={5: SimpleNamespace(name="Nat.add_zero")})
    dag = goal_to_model_dag(_structured_goal())

    action = DecodedAction(
        1, (DecodedArgument(CandidateSource.LIBRARY, 5, 0, 0.4),), True
    )
    rendered, complete = GNNModelEngine._render_action(
        dag, list(dag.nodes), bundle, action, tactic_name="rw"
    )
    # Rewriting families render one bracketed rule list: rw [Nat.add_zero].
    assert rendered == ["[Nat.add_zero]"]
    assert complete is True


    # Non-rewriting families keep bare references.
    rendered, complete = GNNModelEngine._render_action(
        dag, list(dag.nodes), bundle, action, tactic_name="exact"
    )
    assert rendered == ["Nat.add_zero"]
    assert complete is True

    # Zero-argument rewriting actions stay bare (rw with no rules).
    rendered, complete = GNNModelEngine._render_action(
        dag, list(dag.nodes), SimpleNamespace(lemma_corpus={}),
        DecodedAction(1, (), True), tactic_name="rw",
    )
    assert rendered == []
    assert complete is True


@pytest.mark.parametrize("family", [
    "rw", "rewrite", "rwa", "erw", "erewrite", "rw_mod_cast",
    "simp", "simp_rw", "simp_all", "simpa", "dsimp",
    "norm_num", "field_simp", "linarith", "nlinarith",
])
def test_render_all_trained_rule_list_families(family: str) -> None:
    from maths_ai.gnn_inference.inference_engine import GNNModelEngine

    dag = goal_to_model_dag(_structured_goal())
    bundle = SimpleNamespace(lemma_corpus={5: SimpleNamespace(name="Nat.add_zero")})
    action = DecodedAction(
        1, (DecodedArgument(CandidateSource.LIBRARY, 5, 0, 0.4),), True
    )

    arguments, complete = GNNModelEngine._render_action(
        dag, list(dag.nodes), bundle, action, tactic_name=family
    )

    assert complete
    assert arguments == ["[Nat.add_zero]"]
