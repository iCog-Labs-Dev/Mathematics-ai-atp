"""Stage 5 live gate; no mocked Lean execution or model checkpoint."""

import asyncio

import pytest

from maths_ai.hybrid_reasoner.pantograph_diagnostics import verify_live_application
from maths_ai.lean_replay import seed_from_text


@pytest.mark.pantograph_live
def test_actual_executable_model_ownership_and_repeated_proofs(live_env):
    report = asyncio.run(verify_live_application(live_env))
    assert report["scoped_proofs"] == 100
    assert report["initial_states"] == report["final_states"] == 0
    assert report["phase"] == "CLOSED" and report["subprocess_reaped"]
    assert "positive-unsafe-rejection" in report["checks"]
    assert "constant-and-dependent-let-replay" in report["checks"]
    application = report["application"]
    assert application["initial_states"] == application["final_states"]
    assert application["phase"] == "CLOSED" and application["subprocess_reaped"]


@pytest.mark.pantograph_live
def test_actual_factory_executor_and_two_independent_workers(live_env):
    import torch
    from maths_ai.data_models.proof_components import TacticCandidate
    from maths_ai.gnn_inference.tests.model_helpers import scored
    from maths_ai.hybrid_reasoner.joint_inference import (
        HybridReasoner,
        PantographExecutor,
    )
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
        create_model_sexpr_client,
    )
    from maths_ai.gnn_inference.atp_lean_gnn.rl_live_collection import LiveReasonerPool
    from maths_ai.gnn_inference.tests.test_rl_live_collection import _plan

    class DeterministicReasoner(HybridReasoner):
        def _build_gnn_engine(self, **_kwargs):
            return None

        def predict_next_tactic(self, goal):
            goal.require_model_state()
            return [
                scored(TacticCandidate(tactic_name="trivial", arguments=[], probability=1.0))
            ]

    async def run():
        async def factory(_worker_id):
            return await create_model_sexpr_client(live_env)

        def reasoner_factory(worker_id, device, model, server):
            return DeterministicReasoner(
                None, None, None, executor=PantographExecutor(server), use_pln=False
            )

        async def search(reasoner, theorem, greedy, deadline):
            return await reasoner.prove(theorem, deadline=deadline)

        pool = await LiveReasonerPool.create(
            model=torch.nn.Linear(1, 1),
            plan=_plan(2),
            server_factory=factory,
            reasoner_factory=reasoner_factory,
            search_factory=search,
            is_solved=lambda graph: graph.is_solved(),
            is_cooperative_deadline=lambda graph: False,
            startup_timeout_s=120,
            shutdown_timeout_s=10,
        )
        clients = [worker.server for worker in pool.workers]
        async with pool:
            assert len({client.session_id for client in clients}) == 2
            assert pool.workers[0].model is pool.workers[1].model
            result = await pool.collect_round(
                [seed_from_text(goal="True")] * 4, timeout_s=30, model_generation=0,
                proposals=[None] * 4,
            )
            assert result.stats["solved"] == 4 and len(result.results) == 4
            for client in clients:
                assert (await client.stat()).n_goals == 0
        assert all(client.phase.value == "CLOSED" for client in clients)

    asyncio.run(run())
