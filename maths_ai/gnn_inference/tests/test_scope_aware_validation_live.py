"""Application execution, replay, and learning over validation-version-2 snapshots."""

import asyncio
from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch

from pantograph_async import (
    PantographClient, PantographEnv, PantographProtocolError,
    TRANSITION_VALIDATION_VERSION, TRANSITION_VALIDATION_SCOPE,
)
from maths_ai.api.results import graph_view
from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import (
    collect_round, evaluate_proof_rate, EvaluationSettings,
)
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import LocalCandidateProvider
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import TheoremItem
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor
from maths_ai.hybrid_reasoner.mcts_diagnostics import ScriptedSearchReasoner
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client, materialize_seed
from maths_ai.hybrid_reasoner.proposal import ProposalSpec
from maths_ai.lean_replay import seed_from_text


CONJUNCTION = seed_from_text(goal="p ∧ q → q ∧ p", hypotheses=["p : Prop", "q : Prop"])
DISJUNCTION = seed_from_text(goal="p ∨ q → q ∨ p", hypotheses=["p : Prop", "q : Prop"])
FINSET = seed_from_text(
    goal="a ∉ s → ((∀ a ∈ t, a ∉ s) ↔ ∀ ⦃a : α⦄, a ∈ s → a ∉ t)",
    hypotheses=[
        "α : Type u_1", "β : Type u_2", "γ : Type u_3", "inst : DecidableEq α",
        "s t u v : Finset α", "a b : α",
    ],
)
# Printed declarations omit binder roles. This supplied instance role is known.
FINSET = replace(FINSET, locals=tuple(
    replace(local, role=":instance-implicit") if local.name == "inst" else local
    for local in FINSET.locals
))


def settings(policy):
    return {
        "selection_policy": policy, "top_k_tactics": 1, "max_nodes": 16, "max_depth": 4,
        **({"num_simulations": 12, "sim_batch_size": 2} if policy == "puct" else {}),
    }


@pytest.mark.pantograph_live
@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_tauto_controls_certify_through_both_searches_and_replay(live_env, policy):
    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            for seed in (CONJUNCTION, DISJUNCTION, FINSET):
                async with client.state_scope() as scope:
                    _state, goal, _ = await materialize_seed(scope, seed)
                reasoner = ScriptedSearchReasoner(client, {goal.expression: (("tauto", 1.0),)}, **settings(policy))
                graph = await reasoner.prove(seed)
                assert graph.is_solved(), graph.execution
                assert graph.certificate.is_certified
                assert graph.proof_trace()["command"] == "tauto"
                assert graph.execution["intermediate_reconstructions"] == 0
                assert graph.execution["original_tactic_calls"] == 1
                assert graph.execution["replay_tactic_calls"] == 1
                assert (await client.stat()).n_goals == 0

            seed = seed_from_text(goal="(p ∧ q → q ∧ p) ∧ True", hypotheses=["p : Prop", "q : Prop"])
            async with client.state_scope() as scope:
                _state, root, _ = await materialize_seed(scope, seed)
            script = {
                root.expression: (("constructor", 1.0),),
                "p ∧ q → q ∧ p": (("tauto", 1.0),), "True": (("trivial", 1.0),),
            }
            graph = await ScriptedSearchReasoner(client, script, **settings(policy)).prove(seed)
            assert graph.is_solved(), graph.execution
            assert graph.execution["original_tactic_calls"] == 3
            assert graph.execution["replay_tactic_calls"] == 3
            assert graph.execution["intermediate_reconstructions"] == 0
            assert (await client.stat()).n_goals == 0
            view = graph_view(graph)
            assert view["certified"] and view["certificate"]["status"] == "certified"

    asyncio.run(run())


@pytest.mark.pantograph_live
@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_rl_collection_and_shared_evaluation_accept_valid_tauto(live_env, policy):
    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            async with client.state_scope() as scope:
                state, goal, _ = await materialize_seed(scope, CONJUNCTION)
                assert state.validation.version == TRANSITION_VALIDATION_VERSION == 2
                assert state.validation.scope == TRANSITION_VALIDATION_SCOPE
            vocab = build_vocab([model_goal_to_dag(goal)])
            model = build_actor_critic_model(
                model_spec=ModelSpec.from_dict({
                    "architecture": "graphsage", "hidden_dim": 8, "max_args": 1,
                    "encoder": {"num_layers": 1}, "dropout": .2,
                }), num_node_labels=len(vocab), num_tactics=1,
            )
            with torch.no_grad():
                model.decoder.arg_count_head.weight.zero_()
                model.decoder.arg_count_head.bias.copy_(torch.tensor([100., -100.]))
            reasoner = RLHybridReasoner(
                model, vocab, {"tauto": 0}, executor=PantographExecutor(client),
                candidate_provider=LocalCandidateProvider(), use_pln=False, **settings(policy),
            )
            item = TheoremItem(CONJUNCTION, tactic_label="", size=1)
            results, stats = await collect_round(
                reasoner, [item], proposals=[ProposalSpec(mode="greedy", seed=0)], timeout_s=60,
            )
            assert stats["solved"] == 1 and len(results) == 1
            assert results[0].certified and results[0].observations
            assert all(o.outcome.value == "accepted" for o in results[0].observations)
            report = await evaluate_proof_rate(
                reasoner, [item], settings=EvaluationSettings(mode="greedy", seed=0, repeats=1),
            )
            assert report["proof_rate"] == 1.0
            assert (await client.stat()).n_goals == 0

    asyncio.run(run())


@pytest.mark.pantograph_live
def test_real_integrity_failure_is_fatal_in_application_search(request):
    source, repl = os.environ.get("PANTOGRAPH_RESOURCE_SOURCE_ROOT"), os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail("Integrity control requires PANTOGRAPH_RESOURCE_SOURCE_ROOT and PANTOGRAPH_TEST_REPL")
        pytest.skip("Explicit native fixture paths not configured")

    async def run():
        env = PantographEnv(
            source_root=Path(source), pantograph_repl=Path(repl), imports=("Init", "Test.ContractFixtures"),
        )
        async with await PantographClient.create(env) as client:
            reasoner = ScriptedSearchReasoner(client, {"True": (("contract_invalid_mutation", 1.0),)})
            with pytest.raises(PantographProtocolError, match="owner="):
                await reasoner.prove(seed_from_text(goal="True"))
            assert not client.is_usable

    asyncio.run(run())
