"""Collection-pool lifecycle over real client child processes.

Each worker owns an actual ``PantographClient`` running the fake REPL, so these
tests check that every spawned process is reaped after partial startup,
cancellation, hard timeouts, and shutdown racing worker replacement.
"""

import asyncio

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn.rl_live_collection import (
    LiveReasonerPool,
    WorkerStartupError,
)
from maths_ai.gnn_inference.tests.pantograph_fake import (  # noqa: F401
    assert_reaped,
    records,
    run,
    setup,
    started_operation,
)
from maths_ai.gnn_inference.tests.test_rl_live_collection import _plan
from pantograph_async import PantographClient, PantographTransportError


class _Reasoner:
    def __init__(self, server):
        self.server = server


class _Result:
    def __init__(self, theorem):
        self.theorem = theorem

    def is_solved(self):
        return True


async def _scoped_search(reasoner, theorem, _greedy, _deadline):
    """A start/tactic/release round trip; ``slow`` waits on a delayed reply."""
    async with reasoner.server.state_scope() as scope:
        root = await scope.start_goal("∀ (p : Prop), p → p")
        if theorem == "slow":
            await scope.apply_tactic(root, "skip")
    return _Result(theorem)


async def _pool(env, *, workers=2, server_factory=None, search_factory=_scoped_search):
    async def create(_worker_id):
        return await PantographClient.create(env)

    return await LiveReasonerPool.create(
        model=torch.nn.Linear(2, 1),
        plan=_plan(workers),
        server_factory=server_factory or create,
        reasoner_factory=lambda _worker_id, _device, _model, server: _Reasoner(server),
        search_factory=search_factory,
        is_solved=lambda result: result.is_solved(),
        is_cooperative_deadline=lambda result: False,
        startup_timeout_s=env.server_start_timeout_s,
        shutdown_timeout_s=env.worker_shutdown_timeout_s,
        log=lambda message: None,
    )


def _repl_pids(report):
    return {
        entry["pid"]
        for entry in records(report)
        if entry.get("kind") == "spawn" and entry.get("executable") == "repl"
    }


def test_partial_startup_failure_reaps_the_started_children(setup):  # noqa: F811
    env, report = setup

    async def body():
        async def create(worker_id):
            if worker_id == 1:
                await asyncio.sleep(0.05)
                raise RuntimeError("second worker failed")
            return await PantographClient.create(env)

        with pytest.raises(WorkerStartupError) as error:
            await _pool(env, server_factory=create)
        assert error.value.worker_id == 1

    run(body)
    assert len(_repl_pids(report)) == 1
    assert_reaped(report)


def test_cancelled_startup_reaps_every_started_child(setup):  # noqa: F811
    env, report = setup

    async def body():
        started = asyncio.Event()

        async def create(worker_id):
            if worker_id == 1:
                await started.wait()
                await asyncio.Event().wait()
            client = await PantographClient.create(env)
            started.set()
            return client

        async with started_operation(_pool(env, server_factory=create), started) as task:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    run(body)
    assert _repl_pids(report)
    assert_reaped(report)


def test_hard_timeout_mid_request_retires_and_reaps_the_worker(setup, monkeypatch):  # noqa: F811
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "tactic_delay")

    async def body():
        pool = await _pool(env, workers=1)
        first = pool.workers[0].server
        async with pool:
            result = await pool.collect_round(
                ["slow", "fast"], timeout_s=0.08, model_generation=0,
                proposals=[None] * 2,
            )
            assert result.records[0].outcome == "hard_timeout"
            assert [item.theorem for item in result.results] == ["fast"]
            assert result.stats["workers_replaced"] == 1
            assert pool.workers[0].server is not first
            assert first.phase.value == "CLOSED"

    run(body)
    assert len(_repl_pids(report)) == 2
    assert_reaped(report)


def test_cancelled_round_mid_request_closes_and_reaps(setup, monkeypatch):  # noqa: F811
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "tactic_delay")

    async def body():
        pool = await _pool(env)
        async with pool:
            task = asyncio.create_task(
                pool.collect_round(["slow", "slow"], timeout_s=5, model_generation=0, proposals=[None] * 2)
            )
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not pool._collecting

    run(body)
    assert len(_repl_pids(report)) == 2
    assert_reaped(report)


def test_shutdown_racing_worker_replacement_reaps_every_child(setup):  # noqa: F811
    env, report = setup

    async def body():
        replacing = asyncio.Event()

        async def search(reasoner, theorem, greedy, deadline):
            if theorem == "crash":
                # Retire this worker the way a dead pipe would.
                await reasoner.server.aclose()
                raise PantographTransportError("process exited")
            return await _scoped_search(reasoner, theorem, greedy, deadline)

        pool = await _pool(env, workers=1, search_factory=search)

        async def replacement(_worker_id):
            client = await PantographClient.create(env)
            replacing.set()
            try:
                await asyncio.Event().wait()
            finally:
                await client.aclose()

        pool.server_factory = replacement

        async def collect():
            async with pool:
                await pool.collect_round(
                    ["crash", "fast"], timeout_s=5, model_generation=0,
                    proposals=[None] * 2,
                )

        async with started_operation(collect(), replacing) as task:
            await asyncio.wait_for(pool.close(), 5)
            with pytest.raises(asyncio.CancelledError):
                await task

    run(body)
    assert len(_repl_pids(report)) == 2
    assert_reaped(report)
