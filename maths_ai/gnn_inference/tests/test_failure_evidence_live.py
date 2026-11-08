"""Native failure evidence: aborts stay candidate failures; real failures replay."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from pantograph_async import PantographClient, PantographEnv, PantographRemoteInternalError

from maths_ai.hybrid_reasoner.failure_evidence import FailureArchive, RequestHistory, failure_record, runtime_identity
from maths_ai.hybrid_reasoner.failure_replay import replay_failure
from maths_ai.hybrid_reasoner.mcts_diagnostics import ScriptedSearchReasoner
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from maths_ai.lean_replay import seed_from_text

ABORTING = "exact (fun (x : Nat) => True.intro) _"


@pytest.mark.pantograph_live
@pytest.mark.parametrize("policy", ["legacy", "puct"])
def test_an_unsynthesizable_placeholder_is_a_rejected_candidate(live_env, policy):
    search = {"selection_policy": policy, "top_k_tactics": 2, "max_nodes": 8, "max_depth": 2,
              **({"num_simulations": 8, "sim_batch_size": 1} if policy == "puct" else {})}

    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            graph = await ScriptedSearchReasoner(
                client, {"True": ((ABORTING, 0.6), ("trivial", 0.4))}, **search,
            ).prove(seed_from_text(goal="True"))
            assert graph.is_solved() and graph.certificate.is_certified, graph.execution
            assert client.is_usable and client.last_failure is None
            outcomes = [e.get("outcome") for e in client.request_history.events()
                        if e["operation"] == "goal.tactic" and e["stage"] == "completed"]
            assert "TacticFailure" in outcomes

    asyncio.run(run())


def _fixture_env(request, *imports):
    source, repl = os.environ.get("PANTOGRAPH_RESOURCE_SOURCE_ROOT"), os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail("This control requires PANTOGRAPH_RESOURCE_SOURCE_ROOT and PANTOGRAPH_TEST_REPL")
        pytest.skip("Explicit native fixture paths not configured")
    return PantographEnv(source_root=Path(source), pantograph_repl=Path(repl),
                         imports=("Init", "Test.ContractFixtures", *imports))


async def _archived_failure(env, tmp_path, before=None):
    """Run a session that ends in an injected internal failure; return its bundle."""
    seed = seed_from_text(goal="True")
    history = RequestHistory()
    client = await PantographClient.create(env, observer=history)
    client.request_history = history
    if before is not None:
        await before(client)
    with pytest.raises(PantographRemoteInternalError):
        await ScriptedSearchReasoner(client, {"True": (("contract_internal_failure", 1.0),)}).prove(seed)
    await client.aclose()
    identity = await asyncio.to_thread(runtime_identity, client)
    record = failure_record(context={"test": True}, theorem=seed, error=RuntimeError("search failed"),
                            server=client, identity=identity)
    summary = FailureArchive(tmp_path / "failures").write(record)
    return json.loads((tmp_path / "failures" / summary["path"]).read_text())


@pytest.mark.pantograph_live
def test_a_real_terminal_failure_is_archived_and_replays_without_a_model(request, tmp_path):
    env = _fixture_env(request)

    async def run():
        bundle = await _archived_failure(env, tmp_path)
        assert bundle["replayable"], bundle["missing"]
        assert bundle["snapshot"]["association"] == "matched" and "execute" in bundle["summary"]
        replay = await replay_failure(bundle, env)
        assert replay["status"] == "reproduced", replay
        assert replay["observed"]["internal_exception_id"] == "pantographCommandResource"

    asyncio.run(run())


@pytest.mark.pantograph_live
def test_replay_continues_past_recorded_recoverable_failures(request, tmp_path):
    from pantograph_async import PantographGoalRejected

    env = _fixture_env(request)

    async def reject_first(client):
        async with client.state_scope() as scope:
            with pytest.raises(PantographGoalRejected):
                await scope.start_goal("ReviewUnknownIdentifier")

    async def run():
        bundle = await _archived_failure(env, tmp_path, before=reject_first)
        failed = [e for e in bundle["requests"] if e["stage"] == "failed"]
        assert failed[0]["error_type"] == "PantographGoalRejected"
        replay = await replay_failure(bundle, env)
        assert replay["status"] == "reproduced", replay
        assert replay["expected"]["operation"] == "goal.tactic"

    asyncio.run(run())


@pytest.mark.pantograph_live
def test_replay_refuses_a_different_lean_environment(request, tmp_path):
    env = _fixture_env(request)
    other = _fixture_env(request, "Lean")

    async def run():
        bundle = await _archived_failure(env, tmp_path)
        refused = await replay_failure(bundle, other)
        assert refused["status"] == "environment-mismatch" and "imports" in refused["mismatch"]
        allowed = await replay_failure(bundle, other, allow_mismatch=True)
        assert allowed["status"] == "reproduced-in-different-environment"
        assert "imports" in allowed["environment_mismatch"]

    asyncio.run(run())
