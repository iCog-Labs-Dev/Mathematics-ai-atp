"""Replay combination against the actual snapshot companion REPL.

Branches are explored independently in retained snapshots and combined by
sequential replay; only a kernel certificate counts. Run with
``--run-pantograph-live`` and the usual ``PANTOGRAPH_TEST_*`` paths.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest

from maths_ai.hybrid_reasoner.execution_types import DeclarationKey
from maths_ai.hybrid_reasoner.proof_recipes import ArtifactArgument, ReplayArtifact
from maths_ai.hybrid_reasoner.proof_replay import (
    ArtifactMismatch,
    ReplayCoordinator,
    ReplayStatus,
    build_artifact,
    reverify,
)
from maths_ai.gnn_inference.tests.snapshot_explorer import Explorer
from maths_ai.lean_replay import render_replay, seed_from_text
from pantograph_async import PantographClient, PantographEnv


@pytest.fixture
def live_env(request):
    source = os.environ.get("PANTOGRAPH_TEST_SOURCE_ROOT")
    repl = os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail(
                "Required live run needs PANTOGRAPH_TEST_SOURCE_ROOT and PANTOGRAPH_TEST_REPL."
            )
        pytest.skip("Explicit live paths not configured.")
    imports = tuple(os.environ.get("PANTOGRAPH_TEST_IMPORTS", "Mathlib").split(","))
    return PantographEnv(source_root=Path(source), pantograph_repl=Path(repl), imports=imports)


def scoped(env, body):
    async def run():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                result = await body(scope)
            assert (await client.stat()).n_goals == 0
            return result

    return asyncio.run(run())


async def combine(explorer):
    coordinator = ReplayCoordinator(explorer.registry)
    return await coordinator.combine(explorer, explorer.root, 0)


@pytest.mark.pantograph_live
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_conjunction_combines_from_either_exploration_order(live_env, order):
    async def body(scope):
        explorer = await Explorer.start(scope, "True ∧ True")
        sides = await explorer.expand(0, "constructor")
        for index in order:
            assert await explorer.expand(sides[index], "trivial") == []
        result = await combine(explorer)
        assert result.status is ReplayStatus.CERTIFIED, result.conflicts
        assert [step.command for step in result.witness] == ["constructor", "trivial", "trivial"]
        assert all(step.observation_matched for step in result.witness)
        await explorer.registry.close()

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_conflicting_witness_branches_never_certify(live_env):
    async def body(scope):
        explorer = await Explorer.start(scope, "∃ n : Nat, n = 0 ∧ n = 1")
        w, a, b = await explorer.expand(0, "refine ⟨?w, ?a, ?b⟩")
        # Each equality succeeds on its own by choosing its own witness.
        assert await explorer.expand(a, "exact (rfl : 0 = 0)") == []
        assert await explorer.expand(b, "exact (rfl : 1 = 1)") == []
        assert await explorer.expand(w, "exact 0") == []
        assert await explorer.expand(w, "exact 1") == []
        result = await combine(explorer)
        assert result.status is ReplayStatus.CONFLICT
        assert result.certificate is None
        assert {conflict.kind for conflict in result.conflicts} == {"rejected"}
        # The changed equality is kept for a fresh search of its actual state.
        assert sorted((c.node, c.occurrence.goal.expression) for c in result.continuations) == [
            (a, "1 = 0"),
            (b, "0 = 1"),
        ]
        for continuation in result.continuations:
            await explorer.registry.release(continuation.occurrence.lease)
        await explorer.registry.close()

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_another_alternative_certifies_after_a_conflict(live_env):
    async def body(scope):
        explorer = await Explorer.start(scope, "(∃ n : Nat, n = 0 ∧ n = 1) ∨ True")
        (exists,) = await explorer.expand(0, "left")
        w, a, b = await explorer.expand(exists, "refine ⟨?_, ?_, ?_⟩")
        await explorer.expand(a, "exact (rfl : 0 = 0)")
        await explorer.expand(b, "exact (rfl : 1 = 1)")
        await explorer.expand(w, "exact 0")
        (truth,) = await explorer.expand(0, "right")
        await explorer.expand(truth, "trivial")
        result = await combine(explorer)
        assert result.status is ReplayStatus.CERTIFIED
        assert result.conflicts and result.witness[0].command == "right"
        await explorer.registry.close()

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_a_sibling_solved_by_a_shared_assignment_needs_no_step(live_env):
    async def body(scope):
        explorer = await Explorer.start(scope, "∃ n : Nat, n = 0")
        children = await explorer.expand(0, "constructor")
        names = [explorer.observation(node).expression for node in children]
        assert names == ["?w = 0", "ℕ"]
        proof, witness = children
        await explorer.expand(proof, "rfl")
        # Explored on its own, the witness would be 1; replay never runs that.
        await explorer.expand(witness, "exact 1")
        result = await combine(explorer)
        assert result.status is ReplayStatus.CERTIFIED
        assert [step.command for step in result.witness] == ["constructor", "rfl"]
        await explorer.registry.close()

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_identical_targets_keep_their_slots_in_any_order(live_env):
    async def body(scope):
        explorer = await Explorer.start(scope, "True ∧ True ∧ True")
        parts = await explorer.expand(0, "refine ⟨?_, ?_, ?_⟩")
        assert len(parts) == 3 and len({explorer.observation(p).expression for p in parts}) == 1
        for index in (2, 0, 1):
            await explorer.expand(parts[index], "trivial")
        result = await combine(explorer)
        assert result.status is ReplayStatus.CERTIFIED
        assert [step.obligation.slot for step in result.witness[1:]] == [0, 1, 2]
        await explorer.registry.close()

    scoped(live_env, body)


async def _introduced_proof(scope):
    explorer = await Explorer.start(scope, "∀ (p q : Prop), p → q → p ∧ q")
    (body,) = await explorer.expand(0, "intro p q hp hq")
    left, right = await explorer.expand(body, "constructor")
    await explorer.expand(left, "exact", "hp")
    await explorer.expand(right, "exact", "hq")
    return explorer


@pytest.mark.pantograph_live
def test_introduced_locals_are_rebound_through_their_keys(live_env):
    async def body(scope):
        explorer = await _introduced_proof(scope)
        result = await combine(explorer)
        assert result.status is ReplayStatus.CERTIFIED
        # The replay's intro created new locals; arguments follow their keys.
        exact = [step for step in result.witness if step.command.startswith("exact")]
        assert [step.command for step in exact] == ["exact hp", "exact hq"]
        keys = [step.step.arguments[0].declaration for step in exact]
        assert keys == [DeclarationKey(0, 0, 2), DeclarationKey(0, 0, 3)]
        await explorer.registry.close()
        return result

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_an_exported_artifact_reverifies_in_a_fresh_session(live_env):
    statement = "∀ (p q : Prop), p → q → p ∧ q"

    async def first(scope):
        explorer = await _introduced_proof(scope)
        result = await combine(explorer)
        await explorer.registry.close()
        spec = render_replay(seed_from_text(goal=statement))
        return build_artifact(
            {"expression": spec.expression, "local_names": list(spec.local_names),
             "universe_levels": list(spec.universe_levels)},
            result,
            {"lean": result.certificate.receipt.lean},
        )

    artifact = ReplayArtifact.from_dict(scoped(live_env, first).as_dict())

    async def again(scope):
        return await reverify(scope, artifact)

    assert scoped(live_env, again).receipt.proof == artifact.receipt["proof"]

    altered = replace(artifact, statement={**artifact.statement, "expression": "∀ (p q : Prop), p → q → q ∧ p"})
    swapped = list(artifact.steps)
    index = next(i for i, step in enumerate(swapped) if step.arguments)
    argument = swapped[index].arguments[0]
    swapped[index] = replace(
        swapped[index],
        arguments=(ArtifactArgument(argument.kind, argument.text, DeclarationKey(0, 0, 3)),),
    )
    foreign = replace(artifact, receipt={**artifact.receipt, "environment": "0" * 16})
    for broken in (altered, replace(artifact, steps=tuple(swapped)), foreign):
        async def check(scope, broken=broken):
            with pytest.raises(ArtifactMismatch):
                await reverify(scope, broken)

        scoped(live_env, check)


def _search(env, statement, proposals, **kwargs):
    from maths_ai.data_models.proof_components import TacticCandidate
    from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner, PantographExecutor
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client

    class Deterministic(HybridReasoner):
        def _build_gnn_engine(self, **_kwargs):
            return None

        def predict_next_tactic(self, goal):
            return [
                TacticCandidate(tactic_name=name, arguments=list(arguments), probability=0.9)
                for name, *arguments in proposals.get(goal.expression, [])
            ]

    async def run():
        async with await create_model_sexpr_client(env) as client:
            reasoner = Deterministic(
                config_path=None, tactic_model_path=None, argument_model_path=None,
                executor=PantographExecutor(client), use_pln=False, **kwargs,
            )
            graph = await reasoner.prove(seed_from_text(goal=statement))
            assert (await client.stat()).n_goals == 0
            return graph

    return asyncio.run(run())


@pytest.mark.pantograph_live
def test_the_search_certifies_through_retained_snapshots(live_env):
    graph = _search(
        live_env,
        "∀ (p q : Prop), p → q → p ∧ q",
        {
            "∀ (p q : Prop), p → q → p ∧ q": [("intro p q hp hq",)],
            "p ∧ q": [("constructor",)],
            "p": [("exact", "hp")],
            "q": [("exact", "hq")],
        },
    )
    assert graph.is_solved()
    assert graph.execution["root_admissions"] == 1
    assert graph.execution["intermediate_reconstructions"] == 0
    assert graph.execution["replay_tactic_calls"] == 4
    assert graph.certificate.receipt.lean.startswith("4.10.0-rc1")
    artifact = ReplayArtifact.from_dict(graph.artifact.as_dict())

    async def again(scope):
        return await reverify(scope, artifact)

    assert scoped(live_env, again).receipt.proof == graph.certificate.receipt.proof


@pytest.mark.pantograph_live
def test_the_search_never_reports_conflicting_witnesses_as_a_proof(live_env):
    graph = _search(
        live_env,
        "∃ n : Nat, n = 0 ∧ n = 1",
        {
            "∃ n, n = 0 ∧ n = 1": [("refine ⟨?_, ?_, ?_⟩",)],
            "ℕ": [("exact 0",), ("exact 1",)],
            "?refine_1 = 0": [("exact (rfl : 0 = 0)",)],
            "?refine_1 = 1": [("exact (rfl : 1 = 1)",)],
        },
        max_nodes=30,
    )
    assert not graph.is_solved()
    assert graph.execution["replay_attempts"] >= 1
    assert graph.execution["replay_conflicts"] >= 1


@pytest.mark.pantograph_live
def test_local_definitions_and_instances_certify_through_their_context(live_env):
    graph = _search(
        live_env,
        "∀ (α : Type) [inst : Inhabited α], let x : α := default; x = x",
        {
            "∀ (α : Type) [inst : Inhabited α],\n  let x := default;\n  x = x": [("intro α inst x",)],
            "x = x": [("rfl",)],
        },
    )
    assert graph.is_solved(), [(n.goal.expression, n.status) for n in graph.nodes.values()]


@pytest.mark.pantograph_live
def test_universe_parameters_reach_the_certificate(live_env):
    async def body(scope):
        explorer = await Explorer.start(scope, "∀ (α : Sort u_1) (a : α), a = a")
        (inner,) = await explorer.expand(0, "intro α a")
        await explorer.expand(inner, "rfl")
        result = await combine(explorer)
        await explorer.registry.close()
        return result

    result = scoped(live_env, body)
    assert result.is_certified
    assert result.certificate.receipt.levels == ("u_1",)


@pytest.mark.pantograph_live
def test_a_shadowed_local_binds_to_its_own_declaration(live_env):
    async def body(scope):
        explorer = await Explorer.start(scope, "∀ (h : 1 = 1) (h : 2 = 2), 2 = 2")
        (inner,) = await explorer.expand(0, "intro h h")
        locals_ = explorer.observation(inner).locals
        assert [local.execution_or_live_name for local in locals_][-1] == "h"
        await explorer.expand(inner, "exact", "h")
        result = await combine(explorer)
        assert result.is_certified
        assert result.witness[-1].step.arguments[0].declaration == DeclarationKey(0, 0, 1)
        await explorer.registry.close()

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_a_cancelled_search_retires_its_session_and_a_new_one_starts_clean(live_env):
    from maths_ai.data_models.proof_components import TacticCandidate
    from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner, PantographExecutor
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client

    proposed = asyncio.Event()

    class Slow(HybridReasoner):
        def _build_gnn_engine(self, **_kwargs):
            return None

        def predict_next_tactic(self, goal):
            proposed.set()
            return [TacticCandidate(tactic_name="constructor", arguments=[], probability=0.9)]

    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            reasoner = Slow(
                config_path=None, tactic_model_path=None, argument_model_path=None,
                executor=PantographExecutor(client), use_pln=False, max_nodes=10_000, max_depth=10_000,
            )
            task = asyncio.create_task(reasoner.prove(seed_from_text(goal="True ∧ (True ∧ (True ∧ True))")))
            await proposed.wait()
            # The search is now inside its first tactic request.
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            usable = client.is_usable
            if usable:
                assert (await client.stat()).n_goals == 0
        async with await create_model_sexpr_client(live_env) as fresh:
            reasoner = Slow(
                config_path=None, tactic_model_path=None, argument_model_path=None,
                executor=PantographExecutor(fresh), use_pln=False,
            )
            reasoner.predict_next_tactic = lambda goal: [
                TacticCandidate(tactic_name="trivial", arguments=[], probability=0.9)
            ]
            graph = await reasoner.prove(seed_from_text(goal="True"))
            assert graph.is_solved()
            assert (await fresh.stat()).n_goals == 0

    asyncio.run(run())


@pytest.mark.pantograph_live
@pytest.mark.parametrize("split", ["refine ⟨?w, ?h⟩", "constructor"])
def test_a_reflexive_existential_certifies_through_its_conditional_recipe(live_env, split):
    from maths_ai.hybrid_reasoner.hypergraph import EdgeStatus

    # rfl proves ?w = ?w only by referring to the witness. With refine the
    # witness comes first; with constructor the equality does, so the
    # dependency is later in the pending order.
    graph = _search(
        live_env,
        "∃ n : Nat, n = n",
        {"∃ n, n = n": [(split,)], "ℕ": [("exact 0",)], "?w = ?w": [("rfl",)]},
    )
    rfl = next(edge for edge in graph.edges.values() if edge.command == "rfl")
    assert rfl.status == EdgeStatus.READY and rfl.dependencies
    assert graph.execution["replay_attempts"] >= 1
    assert graph.is_solved(), [(n.goal.expression, n.status) for n in graph.nodes.values()]


@pytest.mark.pantograph_live
def test_conditional_recipes_with_contradictory_witnesses_never_certify(live_env):
    graph = _search(
        live_env,
        "∃ n : Nat, n = n ∧ n = 1 ∧ n = 2",
        {
            "∃ n, n = n ∧ n = 1 ∧ n = 2": [("refine ⟨?w, ?a, ?b, ?c⟩",)],
            "ℕ": [("exact 0",), ("exact 1",), ("exact 2",)],
            "?w = ?w": [("rfl",)],
            # rfl cannot assign the opaque witness; these assignments can.
            "?w = 1": [("exact (rfl : 1 = 1)",)],
            "?w = 2": [("exact (rfl : 2 = 2)",)],
        },
        max_nodes=40,
    )
    assert not graph.is_solved()
    assert graph.execution["replay_attempts"] >= 1
    assert graph.execution["replay_conflicts"] >= 1


@pytest.mark.pantograph_live
def test_replay_stays_within_a_small_snapshot_bound(live_env):
    from maths_ai.hybrid_reasoner.execution_limits import RetentionLimits

    graph = _search(
        live_env,
        "True ∧ True",
        {"True ∧ True": [("constructor",)], "True": [("trivial",)]},
        retention=RetentionLimits(max_retained_snapshots=3),
    )
    assert graph.is_solved()
    assert graph.execution["replay_tactic_calls"] == 3
    assert graph.execution["peak_snapshots"] <= 3


@pytest.mark.pantograph_live
def test_an_assigned_sibling_waits_for_its_later_dependency(live_env):
    graph = _search(
        live_env,
        "∃ w v : Nat, w = v",
        {
            "∃ w v, w = v": [("refine ⟨?w, ?v, ?h⟩\nrotate_left\nswap",)],
            "?w = ?v": [("exact Eq.refl ?v",)],
            "ℕ": [("exact 0",)],
        },
    )
    assert graph.has_candidate()
    assert graph.execution["replay_attempts"] >= 1
    assert graph.is_solved(), [(n.goal.expression, n.status) for n in graph.nodes.values()]
    artifact = ReplayArtifact.from_dict(graph.artifact.as_dict())

    async def again(scope):
        return await reverify(scope, artifact)

    assert scoped(live_env, again).is_certified
