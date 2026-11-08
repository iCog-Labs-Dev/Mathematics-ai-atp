"""Native failure evidence: aborts stay candidate failures; real failures replay."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from pantograph_async import PantographClient, PantographEnv, PantographRemoteInternalError

from maths_ai.hybrid_reasoner.failure_evidence import FailureArchive, RequestHistory, failure_record
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


@pytest.mark.pantograph_live
def test_a_real_terminal_failure_is_archived_and_replays_without_a_model(request, tmp_path):
    source, repl = os.environ.get("PANTOGRAPH_RESOURCE_SOURCE_ROOT"), os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail("This control requires PANTOGRAPH_RESOURCE_SOURCE_ROOT and PANTOGRAPH_TEST_REPL")
        pytest.skip("Explicit native fixture paths not configured")
    env = PantographEnv(source_root=Path(source), pantograph_repl=Path(repl),
                        imports=("Init", "Test.ContractFixtures"))
    seed = seed_from_text(goal="True")

    async def run():
        history = RequestHistory()
        client = await PantographClient.create(env, observer=history)
        client.request_history = history
        with pytest.raises(PantographRemoteInternalError):
            await ScriptedSearchReasoner(client, {"True": (("contract_internal_failure", 1.0),)}).prove(seed)
        await client.aclose()
        record = failure_record(context={"test": True}, theorem=seed, error=RuntimeError("search failed"),
                                server=client)
        summary = FailureArchive(tmp_path / "failures").write(record)
        bundle = json.loads((tmp_path / "failures" / summary["path"]).read_text())
        assert bundle["replayable"], bundle["missing"]
        assert bundle["snapshot"]["association"] == "matched"
        assert "execute" in bundle["summary"]
        replay = await replay_failure(bundle, env)
        assert replay["status"] == "reproduced", replay
        assert replay["observed"]["internal_exception_id"] == "pantographCommandResource"

    asyncio.run(run())
