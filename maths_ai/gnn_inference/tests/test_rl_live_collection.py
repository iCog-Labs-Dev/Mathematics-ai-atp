from __future__ import annotations

import asyncio
import time

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn.rl_live_collection import (
    LiveReasonerPool,
    LiveReasonerWorker,
    WorkerStartupError,
)
from maths_ai.gnn_inference.atp_lean_gnn.rl_resources import ResourcePlan


class _Server:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.closed = False

    def _close(self) -> None:
        self.closed = True


class _Graph:
    def __init__(self, solved: bool):
        self._solved = solved

    def is_solved(self) -> bool:
        return self._solved


class _Result:
    def __init__(self, theorem: str):
        self.theorem = theorem
        self.graph = _Graph(theorem.startswith("solve"))


class _Reasoner:
    def __init__(self, worker_id: int, model: torch.nn.Module, server: _Server):
        self.worker_id = worker_id
        self.model = model
        self.server = server


def _plan(workers: int = 2) -> ResourcePlan:
    return ResourcePlan(
        reported_cpu_count=8,
        affinity_cpu_count=8,
        quota_cpu_count=None,
        usable_cpu_count=8,
        requested_worker_count=workers,
        worker_count=workers,
        collection_devices=(torch.device("cpu"),),
        update_device=torch.device("cpu"),
        gpu_resources=(),
    )


async def _create_pool(
    *,
    workers: int = 2,
    delays: dict[str, float] | None = None,
    failing_start_workers: set[int] | None = None,
    server_sink: list[_Server] | None = None,
):
    servers = server_sink if server_sink is not None else []
    delays = delays or {}
    failing_start_workers = failing_start_workers or set()

    async def server_factory(worker_id: int):
        if worker_id in failing_start_workers:
            raise RuntimeError("simulated startup failure")
        server = _Server(worker_id)
        servers.append(server)
        return server

    def reasoner_factory(worker_id, device, model, server):
        assert device == torch.device("cpu")
        return _Reasoner(worker_id, model, server)

    async def search_factory(reasoner, theorem, greedy):
        del reasoner, greedy
        await asyncio.sleep(delays.get(theorem, 0.0))
        return _Result(theorem)

    model = torch.nn.Linear(2, 1)
    pool = await LiveReasonerPool.create(
        model=model,
        plan=_plan(workers),
        server_factory=server_factory,
        reasoner_factory=reasoner_factory,
        search_factory=search_factory,
        is_solved=lambda result: result.graph.is_solved(),
        startup_timeout_s=1.0,
        shutdown_timeout_s=1.0,
        environment_description="test executable=/tmp/repl source=/tmp/mathlib imports=Init",
        log=lambda message: None,
    )
    return pool, model, servers


def test_pool_has_one_reasoner_and_server_per_worker_and_closes_idempotently():
    async def run():
        pool, _model, servers = await _create_pool(workers=3)
        assert len(pool.workers) == 3
        assert len({id(worker.reasoner) for worker in pool.workers}) == 3
        assert len({id(worker.server) for worker in pool.workers}) == 3
        await pool.close()
        await pool.close()
        return servers

    servers = asyncio.run(run())
    assert all(server.closed for server in servers)


def test_context_closes_servers_when_body_raises():
    async def run():
        pool, _model, servers = await _create_pool()
        with pytest.raises(RuntimeError, match="body failed"):
            async with pool:
                raise RuntimeError("body failed")
        return servers

    servers = asyncio.run(run())
    assert all(server.closed for server in servers)


def test_startup_failure_rolls_back_started_servers_with_worker_context():
    async def run():
        servers: list[_Server] = []
        with pytest.raises(WorkerStartupError) as error:
            await _create_pool(
                workers=2,
                failing_start_workers={1},
                server_sink=servers,
            )
        return error.value, servers

    error, servers = asyncio.run(run())
    assert error.worker_id == 1
    assert error.device == torch.device("cpu")
    assert "source=/tmp/mathlib" in str(error)
    assert "simulated startup failure" in str(error)
    assert servers and all(server.closed for server in servers)


def test_collection_overlaps_workers_and_restores_input_order():
    async def run():
        pool, _model, _servers = await _create_pool(
            workers=2,
            delays={"solve-a": 0.06, "solve-b": 0.06},
        )
        async with pool:
            started = time.perf_counter()
            round_result = await pool.collect_round(
                ["solve-a", "solve-b"],
                timeout_s=1.0,
                model_generation=0,
            )
            return round_result, time.perf_counter() - started

    round_result, elapsed = asyncio.run(run())
    assert [result.theorem for result in round_result.results] == ["solve-a", "solve-b"]
    assert [record.batch_index for record in round_result.records] == [0, 1]
    assert elapsed < 0.105
    assert round_result.stats["solved"] == 2.0


def test_hard_timeout_replaces_worker_before_next_theorem():
    async def run():
        starts = 0
        servers: list[_Server] = []

        async def server_factory(worker_id: int):
            nonlocal starts
            starts += 1
            server = _Server(worker_id)
            servers.append(server)
            return server

        def reasoner_factory(worker_id, device, model, server):
            return _Reasoner(worker_id, model, server)

        async def search_factory(reasoner, theorem, greedy):
            del reasoner, greedy
            if theorem == "hang":
                await asyncio.sleep(1.0)
            return _Result(theorem)

        pool = await LiveReasonerPool.create(
            model=torch.nn.Linear(2, 1),
            plan=_plan(1),
            server_factory=server_factory,
            reasoner_factory=reasoner_factory,
            search_factory=search_factory,
            is_solved=lambda result: result.graph.is_solved(),
            startup_timeout_s=1.0,
            shutdown_timeout_s=1.0,
            log=lambda message: None,
        )
        async with pool:
            result = await pool.collect_round(
                ["hang", "solve-after-replacement"],
                timeout_s=0.02,
                model_generation=0,
            )
        return result, starts, servers

    result, starts, servers = asyncio.run(run())
    assert starts == 2
    assert servers[0].closed
    assert [item.theorem for item in result.results] == ["solve-after-replacement"]
    assert result.stats["searches_timed_out"] == 1.0
    assert result.stats["workers_replaced"] == 1.0


def test_replacement_failure_removes_only_the_affected_worker():
    async def run():
        starts = {0: 0, 1: 0}

        async def server_factory(worker_id: int):
            starts[worker_id] += 1
            if worker_id == 0 and starts[worker_id] > 1:
                raise RuntimeError("replacement failed")
            return _Server(worker_id)

        def reasoner_factory(worker_id, device, model, server):
            return _Reasoner(worker_id, model, server)

        async def search_factory(reasoner, theorem, greedy):
            del greedy
            if theorem == "hang-on-worker-zero":
                assert reasoner.worker_id == 0
                await asyncio.sleep(1.0)
            return _Result(theorem)

        pool = await LiveReasonerPool.create(
            model=torch.nn.Linear(2, 1),
            plan=_plan(2),
            server_factory=server_factory,
            reasoner_factory=reasoner_factory,
            search_factory=search_factory,
            is_solved=lambda result: result.graph.is_solved(),
            startup_timeout_s=1.0,
            shutdown_timeout_s=1.0,
            log=lambda message: None,
        )
        async with pool:
            result = await pool.collect_round(
                ["hang-on-worker-zero", "solve-b", "solve-c"],
                timeout_s=0.02,
                model_generation=0,
            )
            active_ids = [worker.worker_id for worker in pool.active_workers]
        return result, active_ids

    result, active_ids = asyncio.run(run())
    assert active_ids == [1]
    assert [item.theorem for item in result.results] == ["solve-b", "solve-c"]
    assert result.stats["worker_replacement_failures"] == 1.0
    assert result.stats["worker_count"] == 1.0


def test_replica_sync_copies_final_state_and_advances_generation():
    canonical = torch.nn.Linear(2, 1)
    replica = torch.nn.Linear(2, 1)
    replica.load_state_dict(canonical.state_dict())
    workers = [
        LiveReasonerWorker(0, torch.device("cpu"), canonical, object(), _Server(0)),
        LiveReasonerWorker(1, torch.device("cpu"), replica, object(), _Server(1)),
    ]

    async def unused_server_factory(worker_id):
        return _Server(worker_id)

    pool = LiveReasonerPool(
        workers=workers,
        plan=_plan(2),
        canonical_model=canonical,
        server_factory=unused_server_factory,
        reasoner_factory=lambda *args: object(),
        search_factory=lambda *args: None,
        is_solved=lambda result: False,
        startup_timeout_s=1.0,
        shutdown_timeout_s=1.0,
        queue_size=0,
        environment_description="test",
        model_generation=0,
        log=lambda message: None,
    )
    with torch.no_grad():
        canonical.weight.add_(3.0)
        canonical.bias.sub_(2.0)

    pool.synchronize(canonical, 1)
    pool.assert_generation(1)
    for key, value in canonical.state_dict().items():
        assert torch.equal(value, replica.state_dict()[key])


def test_collection_rejects_mixed_generation_before_search():
    async def run():
        pool, _model, _servers = await _create_pool(workers=1)
        async with pool:
            with pytest.raises(RuntimeError, match="does not match canonical"):
                await pool.collect_round(
                    ["solve-a"], timeout_s=1.0, model_generation=1
                )

    asyncio.run(run())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_two_gpu_topology_assigns_inference_workers_to_both_devices():
    async def run():
        plan = ResourcePlan(
            reported_cpu_count=8,
            affinity_cpu_count=8,
            quota_cpu_count=None,
            usable_cpu_count=8,
            requested_worker_count=2,
            worker_count=2,
            collection_devices=(torch.device("cuda:0"), torch.device("cuda:1")),
            update_device=torch.device("cuda:0"),
            gpu_resources=(),
        )

        async def server_factory(worker_id):
            return _Server(worker_id)

        def reasoner_factory(worker_id, device, model, server):
            assert next(model.parameters()).device == device
            return _Reasoner(worker_id, model, server)

        async def search_factory(reasoner, theorem, greedy):
            del reasoner, greedy
            return _Result(theorem)

        model = torch.nn.Linear(2, 1).to("cuda:0")
        pool = await LiveReasonerPool.create(
            model=model,
            plan=plan,
            server_factory=server_factory,
            reasoner_factory=reasoner_factory,
            search_factory=search_factory,
            is_solved=lambda result: result.graph.is_solved(),
            startup_timeout_s=1.0,
            shutdown_timeout_s=1.0,
            log=lambda message: None,
        )
        async with pool:
            return {worker.device for worker in pool.workers}

    assert asyncio.run(run()) == {torch.device("cuda:0"), torch.device("cuda:1")}
