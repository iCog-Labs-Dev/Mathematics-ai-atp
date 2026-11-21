"""Frozen dataset rows through the loader, the replay adapters and real Lean."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import _dataset_row_from_sample
from maths_ai.lean_replay import (
    ReplayFailure,
    ReplayInputError,
    render_replay,
    seed_from_text,
)
from maths_ai.hybrid_reasoner.proposal import ProposalSpec

SAMPLE = ProposalSpec(mode="sample", seed=0)


FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/dataset_replay/rows.json").read_text(encoding="utf-8")
)
ROWS = {row["theorem"] if row["origin"] == "synthetic" else row["position"]: row for row in FIXTURE["rows"]}
PI = "synthetic.printed_pi"


def dataset_row(key):
    row = ROWS[key]
    return _dataset_row_from_sample(
        row,
        split=row["split"],
        row_index=row.get("position", row["row_index"]),
        dataset_name=FIXTURE["dataset"],
        dataset_revision=FIXTURE["revision"],
    )


def prepared_graph(row):
    """The graph the prepared data holds for a row: its model columns, as stored."""
    from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag

    return proof_state_to_dag(
        row.state, goal_sexp=row.model_goal_sexp, hyp_sexps=list(row.model_hyp_sexps)
    )


def graph_view(dag):
    from maths_ai.gnn_inference.atp_lean_gnn.graph import dag_fingerprint

    return (
        dag_fingerprint(dag),
        [node.as_dict() for node in dag.nodes],
        [list(edge) for edge in dag.edges],
        dag.expression_root_id,
        dag.state_root_id,
    )


def failure(call) -> ReplayFailure | None:
    try:
        call()
    except ReplayInputError as exc:
        return exc.reason
    return None


def test_fixture_rows_are_marked_original_or_synthetic():
    assert {row["origin"] for row in FIXTURE["rows"]} == {"original", "synthetic"}
    assert FIXTURE["revision"] == "f44841f98193c09b20c6092198a22e98ff2cd7a8"


@pytest.mark.parametrize(
    "key,structured,text",
    [
        (0, ReplayFailure.UNRESOLVED_UNIVERSE, ReplayFailure.UNRESOLVED_UNIVERSE),
        (5, None, None),
        (7, None, None),
        (27, ReplayFailure.MISSING_CONTEXT, None),
        (41, None, None),
        (74, None, None),
        (468, None, None),
        (PI, None, None),
    ],
)
def test_offline_classification(key, structured, text):
    row = dataset_row(key)
    assert failure(lambda: render_replay(row.structured_seed())) is structured
    assert failure(lambda: render_replay(seed_from_text(row.state))) is text


def test_structured_and_text_contexts_agree_once_multiline_types_parse():
    row = dataset_row(7)
    structured = row.structured_seed()
    text = seed_from_text(row.state)
    assert [local.name for local in text.locals] == [local.name for local in structured.locals]


def test_a_renamed_local_keeps_its_source_name_for_the_model():
    # Row 74: a nested `∀ x` refers to the local `x`, so Lean gets an alias.
    assert render_replay(dataset_row(74).structured_seed()).name_map == (("x", "x_1"),)


def test_raw_constants_replace_environment_dependent_printed_names():
    assert "@Real.pi" in render_replay(dataset_row(PI).structured_seed()).expression
    assert "π" in render_replay(seed_from_text(dataset_row(PI).state)).expression
    assert "@Submodule.span" in render_replay(dataset_row(41).structured_seed()).expression


@pytest.fixture
def mathlib_env(request):
    from pantograph_async import PantographEnv

    source = os.environ.get("PANTOGRAPH_TEST_SOURCE_ROOT")
    repl = os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail(
                "Required live run needs PANTOGRAPH_TEST_SOURCE_ROOT and PANTOGRAPH_TEST_REPL."
            )
        pytest.skip("Explicit live paths not configured.")
    return PantographEnv(
        source_root=Path(source), pantograph_repl=Path(repl), imports=("Mathlib",)
    )


@pytest.mark.pantograph_live
def test_live_replay_repairs_supported_rows_and_rejects_the_rest(mathlib_env):
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
        create_model_sexpr_client,
        materialize_seed,
    )

    async def outcome(scope, seed):
        try:
            _, goal, validation = await materialize_seed(scope, seed)
        except ReplayInputError as exc:
            return exc.reason, None, None
        return None, goal, validation

    async def run():
        client = await create_model_sexpr_client(mathlib_env)
        async with client:
            baseline = (await client.stat()).n_goals
            async with client.state_scope() as scope:
                # Rows whose raw columns are complete replay exactly, names included.
                for key in (5, 7, 41, 74, PI):
                    seed = dataset_row(key).structured_seed()
                    reason, goal, validation = await outcome(scope, seed)
                    assert reason is None, (key, reason)
                    # Lean prints exactly the names the replay predicted, and the
                    # model keeps the source names and indices regardless.
                    assert validation.name_drift == ()
                    replay = render_replay(seed)
                    assert [l.execution_or_live_name for l in goal.locals] == list(
                        replay.expected_names
                    )
                    assert [(l.user_name, l.context_index) for l in goal.locals] == [
                        (s.name, s.context_index) for s in seed.locals
                    ]
                    # The graph given to the model is the prepared graph of the
                    # row: source names and gapped context indices, unchanged.
                    raw = ROWS[key]
                    if raw["origin"] == "original":
                        from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag

                        row = dataset_row(key)
                        assert graph_view(model_goal_to_dag(goal)) == graph_view(prepared_graph(row))
                        if key == 74:
                            # The model sees `x`; tactics name the live local `x_1`.
                            from maths_ai.gnn_inference.atp_lean_gnn.inference import (
                                local_declarations,
                            )

                            renamed = next(l for l in goal.locals if l.user_name == "x")
                            assert local_declarations(goal)[f"FV{renamed.context_index}"] == "x_1"
                        assert goal.goal_model_sexp == row.model_goal_sexp
                        assert [
                            {k: l.graph_record()[k] for k in ("name", "context_index", "sexp")}
                            for l in goal.locals
                        ] == [
                            {k: h[k] for k in ("name", "context_index", "sexp")}
                            for h in row.model_hyp_sexps
                        ]
                    # The canonical goal replays through the same rules.
                    reason, again, _ = await outcome(scope, goal.replay_seed())
                    assert reason is None and again.state_fingerprint() == goal.state_fingerprint()

                # Lean rebuilds row 468, but its model input would differ.
                reason, _, _ = await outcome(scope, dataset_row(468).structured_seed())
                assert reason is ReplayFailure.MODEL_MISMATCH

                # Printed names that need the source file's `open` fail in Lean.
                for key in (41, PI):
                    reason, _, _ = await outcome(scope, seed_from_text(dataset_row(key).state))
                    assert reason is ReplayFailure.LEAN_REJECTED, key

                # Named universes: shared and multiple levels, dependent binders.
                universes = seed_from_text(
                    "α : Type u_1\nβ : Type u_1\nγ : Sort (max u v)\nf : α → β\na : α\n⊢ f a = f a"
                )
                assert render_replay(universes).universe_levels == ("u_1", "u", "v")
                reason, goal, _ = await outcome(scope, universes)
                assert reason is None and [l.user_name for l in goal.locals][-1] == "a"

                # A canonical goal with a local definition and an instance.
                start = await scope.start_goal(
                    "∀ (α : Type) [inst : Inhabited α] (n : Nat), let x := n; x = x",
                    levels=(),
                )
                state = await scope.apply_tactic(start, "intro α inst n x")
                from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_goal

                canonical = execution_goal_to_goal(state.goals[0])
                reason, replayed, _ = await outcome(scope, canonical.replay_seed())
                assert reason is None
                assert replayed.state_fingerprint() == canonical.state_fingerprint()

                # Alternative branches from one state each replay to their own goals.
                start = await scope.start_goal("∀ (p q : Prop), p ∧ q → q ∧ p", levels=())
                parent = await scope.apply_tactic(start, "intro p q h")
                for tactic, count in (("constructor", 2), ("obtain ⟨hp, hq⟩ := h", 1)):
                    branch = await scope.apply_tactic(parent, tactic)
                    assert len(branch.goals) == count
                    for execution in branch.goals:
                        child = execution_goal_to_goal(execution)
                        reason, again, _ = await outcome(scope, child.replay_seed())
                        assert reason is None
                        assert again.state_fingerprint() == child.state_fingerprint()
            assert (await client.stat()).n_goals == baseline

            # Cancelling a replay leaves nothing allocated, or retires the session.
            async with client.state_scope() as scope:
                task = asyncio.create_task(
                    materialize_seed(scope, dataset_row(41).structured_seed())
                )
                await asyncio.sleep(0)  # let the replay start its request
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            if client.is_usable:
                assert (await client.stat()).n_goals == baseline
        assert client.phase.value == "CLOSED"

    asyncio.run(run())


async def _with_scope(env, body):
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client

    client = await create_model_sexpr_client(env)
    async with client:
        baseline = (await client.stat()).n_goals
        async with client.state_scope() as scope:
            await body(scope)
        assert (await client.stat()).n_goals == baseline
    assert client.phase.value == "CLOSED"


@pytest.mark.pantograph_live
def test_live_successors_and_rematerialized_goals_keep_the_source_identity(mathlib_env):
    from maths_ai.data_models.model_identity import carry_model_identity
    from maths_ai.hybrid_reasoner.pantograph_adapter import execution_state_to_goals
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import materialize_seed

    async def body(scope):
        # Row 5 has a context gap; row 41 has a renamed instance.
        for key in (5, 41):
            state, root, _ = await materialize_seed(scope, dataset_row(key).structured_seed())
            successor = await scope.apply_tactic(state, "have hnew : True := trivial")
            (child,) = execution_state_to_goals(successor)
            child = carry_model_identity(root, child)
            *carried, new = child.locals
            assert [(l.user_name, l.context_index, l.type_model_sexp) for l in carried] == [
                (l.user_name, l.context_index, l.type_model_sexp) for l in root.locals
            ]
            assert (new.user_name, new.context_index) == ("hnew", root.locals[-1].context_index + 1)
            # Replaying the successor (as search does for a later expansion)
            # gives Lean new IDs but the model the same goal.
            _, again, _ = await materialize_seed(scope, child.replay_seed())
            assert again.state_fingerprint() == child.state_fingerprint()
            assert [l.context_index for l in again.locals] == [l.context_index for l in child.locals]

    asyncio.run(_with_scope(mathlib_env, body))


@pytest.mark.pantograph_live
def test_live_replay_refers_to_the_outer_local_not_a_nested_binder(mathlib_env):
    from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_structured_locals
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
        materialize_replay,
        materialize_seed,
    )
    from maths_ai.lean_replay import StructuredLocal, seed_from_records, validate_replay

    def record(internal, name, index):
        return StructuredLocal(internal, name, index, ":explicit", False, False, "(:c Nat)")

    # (fun x_1 => x✝) 0 = x✝ with a later x: true by rfl only if the body is the
    # outer local.
    seed = seed_from_records(
        "((:c Eq) (:c Nat) ((:lambda x_1 (:c Nat) (:fv _a)) (:lit 0)) (:fv _a))",
        [record("_a", "x✝", 0), record("_b", "x", 1)],
    )
    replay = render_replay(seed)

    async def body(scope):
        state, _, _ = await materialize_seed(scope, seed)
        assert not (await scope.apply_tactic(state, "rfl")).goals
        # The reconstruction the review found: the nested binder captures it.
        captured = dataclasses.replace(
            replay, expression=replay.expression.replace("=> @«x✝»", "=> @x_1")
        )
        assert captured.expression != replay.expression
        wrong = await materialize_replay(scope, captured)
        goal = wrong.goals[0]
        with pytest.raises(ReplayInputError) as error:
            validate_replay(seed, captured, goal.target.sexp, execution_goal_to_structured_locals(goal))
        assert error.value.reason is ReplayFailure.SEMANTIC_MISMATCH
        from pantograph_async import PantographTacticRejected

        with pytest.raises(PantographTacticRejected):
            await scope.apply_tactic(wrong, "rfl")
        # Text is replayed verbatim, so its scoping is Lean's own.
        text, _, _ = await materialize_seed(
            scope, seed_from_text("x : Nat\n⊢ (fun x_1 : Nat => x) 0 = x")
        )
        assert not (await scope.apply_tactic(text, "rfl")).goals

    asyncio.run(_with_scope(mathlib_env, body))


@pytest.fixture
def prepared_artifacts():
    prepared = os.environ.get("PANTOGRAPH_TEST_PREPARED_ROOT")
    checkpoint = os.environ.get("PANTOGRAPH_TEST_CHECKPOINT")
    if not prepared or not checkpoint:
        pytest.skip("Set PANTOGRAPH_TEST_PREPARED_ROOT and PANTOGRAPH_TEST_CHECKPOINT.")
    return Path(prepared), Path(checkpoint)


@pytest.mark.pantograph_live
def test_live_replayed_graphs_fit_the_prepared_vocabulary_and_checkpoint(
    mathlib_env, prepared_artifacts
):
    """Run where the supervised artifacts exist (for example JupyterHub)."""
    import torch
    from torch_geometric.data import Batch

    from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import build_model_from_checkpoint
    from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import require_graph_representation
    from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import make_dag_featurizer
    from maths_ai.gnn_inference.atp_lean_gnn.training import load_prepared_metadata
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import materialize_seed

    prepared_root, checkpoint_path = prepared_artifacts
    metadata = load_prepared_metadata(prepared_root)
    require_graph_representation(metadata.graph_representation)
    model, _, _ = build_model_from_checkpoint(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        graph_representation=metadata.graph_representation,
        expected_model_kind="actor_critic_with_args",
    )
    model.eval()
    featurize = make_dag_featurizer(metadata.node_vocab)
    id_to_tactic = {index: name for name, index in metadata.tactic_vocab.items()}

    async def body(scope):
        for key in (5, 7, 41):
            _, goal, _ = await materialize_seed(scope, dataset_row(key).structured_seed())
            # Raises on any structural label the prepared vocabulary lacks.
            _, replayed = featurize(goal)
            assert torch.equal(replayed.x, _prepared_features(dataset_row(key), metadata.node_vocab))
            with torch.no_grad():
                sample = model.act(Batch.from_data_list([replayed]), id_to_tactic=id_to_tactic, greedy=True)
            assert torch.isfinite(sample.value).all()

    asyncio.run(_with_scope(mathlib_env, body))


def _prepared_features(row, node_vocab):
    from maths_ai.gnn_inference.atp_lean_gnn.pyg import dag_to_pyg

    return dag_to_pyg(prepared_graph(row), node_vocab, add_reverse_edges=True).x


@pytest.mark.pantograph_live
def test_live_search_nodes_see_the_source_identity(mathlib_env):
    from maths_ai.data_models.proof_components import TacticCandidate
    from maths_ai.gnn_inference.tests.model_helpers import scored
    from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner, PantographExecutor
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client

    class OneStep(HybridReasoner):
        def _build_gnn_engine(self, **_kwargs):
            return None

        def predict_next_tactic(self, goal):
            return [
                scored(TacticCandidate(
                    tactic_name="have hnew : True := trivial", arguments=[], probability=1.0
                ))
            ]

    seed = dataset_row(5).structured_seed()
    source = [(local.name, local.context_index) for local in seed.locals]

    async def run():
        client = await create_model_sexpr_client(mathlib_env)
        async with client:
            reasoner = OneStep(
                None, None, None, executor=PantographExecutor(client), use_pln=False,
                max_depth=1, max_nodes=4,
            )
            graph = await reasoner.prove(seed)
            root = graph.nodes[graph.root_id].goal
            assert [(l.user_name, l.context_index) for l in root.locals] == source
            children = [node.goal for node in graph.nodes.values() if node.id != graph.root_id]
            assert children
            for child in children:
                assert [(l.user_name, l.context_index) for l in child.locals][:-1] == source
                assert child.locals[-1].context_index == source[-1][1] + 1
        assert client.phase.value == "CLOSED"

    asyncio.run(run())


@pytest.mark.pantograph_live
def test_live_selected_locals_reach_their_own_declarations(mathlib_env):
    """Review case: an aliased outer `x` and a newly introduced `x`."""
    from maths_ai.data_models.model_identity import carry_model_identity
    from maths_ai.gnn_inference.atp_lean_gnn.inference import local_declarations
    from maths_ai.hybrid_reasoner.pantograph_adapter import execution_state_to_goals
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import materialize_seed
    from maths_ai.lean_replay import StructuredLocal, seed_from_records

    seed = seed_from_records(
        "(:forall x (:c Nat) ((:c Eq) (:c Nat) (:fv _x) (:fv _x)))",
        [StructuredLocal("_x", "x", 5, ":explicit", False, False, "(:c Nat)")],
    )
    assert render_replay(seed).expression == (
        "∀ (x_1 : @Nat), (∀ (x : @Nat), (@Eq @Nat @x_1 @x_1))"
    )

    async def check_reverts(scope, state, goal):
        """Reverting each resolved name removes exactly that declaration."""
        declarations = local_declarations(goal)
        for local in goal.locals:
            name = declarations[f"FV{local.context_index}"]
            (after,) = execution_state_to_goals(await scope.apply_tactic(state, f"revert {name}"))
            remaining = {l.internal_name for l in after.locals}
            assert remaining == {l.internal_name for l in goal.locals} - {local.internal_name}

    async def body(scope):
        state, root, _ = await materialize_seed(scope, seed)
        assert [(l.user_name, l.context_index, l.execution_or_live_name) for l in root.locals] == [
            ("x", 5, "x_1")
        ]
        intro = await scope.apply_tactic(state, "intro x")
        (child,) = execution_state_to_goals(intro)
        child = carry_model_identity(root, child)
        assert local_declarations(child) == {"FV5": "x_1", "FV6": "x"}
        await check_reverts(scope, intro, child)

        # After re-materialization Lean has new IDs; the same selections still
        # reach the right declarations.
        again_state, again, _ = await materialize_seed(scope, child.replay_seed())
        assert {l.internal_name for l in again.locals}.isdisjoint(
            {l.internal_name for l in child.locals}
        )
        assert again.state_fingerprint() == child.state_fingerprint()
        assert local_declarations(again) == {"FV5": "x_1", "FV6": "x"}
        await check_reverts(scope, again_state, again)

        # An alternative branch from the same state is resolved on its own.
        other = await scope.apply_tactic(state, "intro y")
        (branch,) = execution_state_to_goals(other)
        branch = carry_model_identity(root, branch)
        assert local_declarations(branch) == {"FV5": "x_1", "FV6": "y"}
        await check_reverts(scope, other, branch)

    asyncio.run(_with_scope(mathlib_env, body))
