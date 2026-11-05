"""Compiler-producing tactics through search, replay, learning and evaluation.

`unfold Nat.pow` generates the match splitter for `Nat.pow.match_1` on demand,
and the code generator stores its stage entries in the retained environment.
Validation version 2 ended the session on that transition; version 3 audits the
entries as compiler data and never lets a proof depend on them.
"""

import asyncio
import json
import os

import pytest
import torch

from pantograph_async import PantographClient, PantographProtocolError, TRANSITION_VALIDATION_VERSION
from maths_ai.api.results import graph_view
from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import (
    collect_round, evaluate_proof_rate, EvaluationSettings,
)
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import RewardConfig, train_step_onpolicy
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import LocalCandidateProvider
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import TheoremItem
from maths_ai.hybrid_reasoner.command_journal import JOURNAL_ENV
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor
from maths_ai.hybrid_reasoner.mcts_diagnostics import ScriptedSearchReasoner
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client, materialize_seed
from maths_ai.hybrid_reasoner.proposal import ProposalSpec
from maths_ai.lean_replay import seed_from_text

POW = seed_from_text(goal="Nat.pow m n = Nat.pow m n", hypotheses=["m : Nat", "n : Nat"])
ONE_STEP = "unfold Nat.pow; split <;> rfl"


class Fallback(dict):
    """Script keyed by goal text, with one command for every other goal."""

    def __init__(self, entries, default):
        super().__init__(entries)
        self.default = default

    def get(self, key, fallback=None):
        return super().get(key, self.default)


def settings(policy):
    return {
        "selection_policy": policy, "top_k_tactics": 1, "max_nodes": 16, "max_depth": 4,
        **({"num_simulations": 12, "sim_batch_size": 2} if policy == "puct" else {}),
    }


def assert_certified_without_compiler_entries(graph):
    assert graph.is_solved(), graph.execution
    assert graph.certificate.is_certified
    receipt = graph.certificate.receipt
    assert any("splitter" in name for name in receipt.auxiliary)
    assert not any("_cstage" in name for name in receipt.auxiliary)
    assert graph.execution["intermediate_reconstructions"] == 0


@pytest.mark.pantograph_live
@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_compiler_producing_tactics_certify_through_both_searches(live_env, policy):
    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            async with client.state_scope() as scope:
                _state, goal, _ = await materialize_seed(scope, POW)
            # One command that compiles the splitter and closes the goal.
            graph = await ScriptedSearchReasoner(client, {goal.expression: ((ONE_STEP, 1.0),)}, **settings(policy)).prove(POW)
            assert_certified_without_compiler_entries(graph)
            assert graph.execution["original_tactic_calls"] == graph.execution["replay_tactic_calls"] == 1
            assert (await client.stat()).n_goals == 0
            # The compiler-producing step leaves a retained goal that a later step continues.
            script = Fallback({goal.expression: (("unfold Nat.pow", 1.0),)}, (("split <;> rfl", 1.0),))
            graph = await ScriptedSearchReasoner(client, script, **settings(policy)).prove(POW)
            assert_certified_without_compiler_entries(graph)
            assert graph.execution["original_tactic_calls"] == graph.execution["replay_tactic_calls"] == 2
            view = graph_view(graph)
            assert view["certified"] and view["certificate"]["status"] == "certified"
            assert client.is_usable and (await client.stat()).n_goals == 0

    asyncio.run(run())


@pytest.mark.pantograph_live
def test_multi_goal_continuation_after_a_compiler_producing_step(live_env):
    seed = seed_from_text(goal="Nat.pow m n = Nat.pow m n ∧ True", hypotheses=["m : Nat", "n : Nat"])

    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            async with client.state_scope() as scope:
                _state, root, _ = await materialize_seed(scope, seed)
            script = {
                root.expression: (("constructor", 1.0),),
                "Nat.pow m n = Nat.pow m n": ((ONE_STEP, 1.0),),
                "m.pow n = m.pow n": ((ONE_STEP, 1.0),),
                "True": (("trivial", 1.0),),
            }
            for policy in ("legacy", "puct"):
                graph = await ScriptedSearchReasoner(client, script, **settings(policy)).prove(seed)
                assert_certified_without_compiler_entries(graph)
                assert graph.execution["original_tactic_calls"] == graph.execution["replay_tactic_calls"] == 3
            assert (await client.stat()).n_goals == 0

    asyncio.run(run())


@pytest.mark.pantograph_live
@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_learning_update_then_shared_evaluation_accepts_compiler_tactics(live_env, policy):
    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            async with client.state_scope() as scope:
                state, goal, _ = await materialize_seed(scope, POW)
                assert state.validation.version == TRANSITION_VALIDATION_VERSION == 3
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
                model, vocab, {ONE_STEP: 0}, executor=PantographExecutor(client),
                candidate_provider=LocalCandidateProvider(), use_pln=False, **settings(policy),
            )
            item = TheoremItem(POW, tactic_label="", size=1)
            results, stats = await collect_round(
                reasoner, [item], proposals=[ProposalSpec(mode="sample", seed=0)], timeout_s=60,
            )
            assert stats["solved"] == 1 and results[0].certified
            assert all(o.outcome.value == "accepted" for o in results[0].observations)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            before = [p.detach().clone() for p in model.parameters()]
            metrics = train_step_onpolicy(
                model, optimizer, results, reasoner.dag_featurize_data,
                reward_cfg=RewardConfig(step_penalty=0.0),
            )
            assert metrics["num_transitions"] > 0 and metrics["num_failures"] == 0
            assert any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))
            # The same evaluation function serves periodic and standalone evaluation.
            for mode in ("greedy", "sample"):
                report = await evaluate_proof_rate(
                    reasoner, [item], settings=EvaluationSettings(mode=mode, seed=0, repeats=1),
                )
                assert report["proof_rate"] == 1.0, report
            assert (await client.stat()).n_goals == 0

    asyncio.run(run())


@pytest.mark.pantograph_live
def test_command_journal_records_each_command_before_its_outcome(live_env, tmp_path, monkeypatch):
    path = tmp_path / "commands.jsonl"
    monkeypatch.setenv(JOURNAL_ENV, str(path))

    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            async with client.state_scope() as scope:
                _state, goal, _ = await materialize_seed(scope, POW)
            script = Fallback({goal.expression: (("unfold Nat.pow", 1.0),)}, (("split <;> rfl", 1.0),))
            graph = await ScriptedSearchReasoner(client, script, **settings("legacy")).prove(POW)
            assert graph.is_solved()

    asyncio.run(run())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    commands = [r for r in records if r["kind"] == "command"]
    outcomes = {r["seq"]: r for r in records if r["kind"] == "outcome"}
    assert {r["phase"] for r in commands} >= {"operation", "replay"}
    assert [r["command"] for r in commands if r["phase"] == "operation"] == ["unfold Nat.pow", "split <;> rfl"]
    for record in commands:
        assert records.index(record) < records.index(outcomes[record["seq"]])
        assert outcomes[record["seq"]]["outcome"] == "accepted"
        assert outcomes[record["seq"]]["validation"].startswith(f"{TRANSITION_VALIDATION_VERSION}:")


@pytest.mark.pantograph_live
def test_journal_keeps_the_last_command_of_a_terminal_failure(request, tmp_path, monkeypatch):
    from dataclasses import replace

    source, repl = os.environ.get("PANTOGRAPH_RESOURCE_SOURCE_ROOT"), os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail("Integrity control requires PANTOGRAPH_RESOURCE_SOURCE_ROOT and PANTOGRAPH_TEST_REPL")
        pytest.skip("Explicit native fixture paths not configured")
    from pathlib import Path
    from pantograph_async import PantographEnv

    path = tmp_path / "commands.jsonl"
    monkeypatch.setenv(JOURNAL_ENV, str(path))

    async def run():
        env = PantographEnv(source_root=Path(source), pantograph_repl=Path(repl), imports=("Init", "Test.ContractFixtures"))
        async with await PantographClient.create(env) as client:
            reasoner = ScriptedSearchReasoner(client, {"True": (("contract_invalid_mutation", 1.0),)})
            with pytest.raises(PantographProtocolError, match="owner="):
                await reasoner.prove(seed_from_text(goal="True"))
            assert not client.is_usable

    asyncio.run(run())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-2]["kind"] == "command" and records[-2]["command"] == "contract_invalid_mutation"
    assert records[-1]["outcome"] == "terminal" and records[-1]["error_type"] == "PantographProtocolError"
    assert "validation" in records[-1]["payload"]
