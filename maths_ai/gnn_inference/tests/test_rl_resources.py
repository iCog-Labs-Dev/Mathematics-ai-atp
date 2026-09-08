from __future__ import annotations

import asyncio
import time

import torch

from maths_ai.data_models.proof_components import LeanGoalSeed
from maths_ai.gnn_inference.atp_lean_gnn.rl_resources import plan_resources
from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import (
    LiveReasonerPool,
    TheoremItem,
    _LiveReasonerWorker,
)


class _Graph:
    def __init__(self, solved: bool = False):
        self._solved = solved

    def is_solved(self) -> bool:
        return self._solved


class _Result:
    def __init__(self, solved: bool = False):
        self.graph = _Graph(solved)


class _Reasoner:
    def __init__(self, delays: dict[str, float], failures: set[str] | None = None):
        self.delays = delays
        self.failures = failures or set()

    async def prove(self, expression: str, *, hypotheses=None, greedy=False):
        await asyncio.sleep(self.delays.get(expression, 0.0))
        if expression in self.failures:
            raise RuntimeError("simulated worker failure")
        return _Result(solved=expression == "fast")


def _item(expression: str) -> TheoremItem:
    return TheoremItem(LeanGoalSeed(expression=expression, hypotheses=[]), "", 1)


def test_cpu_resource_plan_uses_requested_worker_capacity():
    plan = plan_resources(
        collection_workers=2,
        server_pool_size=2,
        collection_devices=["cpu"],
        update_device="cpu",
        cpu_reserve=0,
    )
    assert plan.worker_count == 2
    assert plan.server_count == 2
    assert plan.collection_devices == (torch.device("cpu"),)
    assert plan.update_device == torch.device("cpu")


def test_warning_resource_policy_clamps_workers(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    monkeypatch.setattr("os.sched_getaffinity", lambda _: {0, 1})
    plan = plan_resources(
        collection_workers=8,
        server_pool_size=8,
        collection_devices=["cpu"],
        update_device="cpu",
        cpu_reserve=1,
        resource_check="warn",
    )
    assert plan.worker_count == 1
    assert plan.server_count == 1


def test_pool_returns_results_in_batch_order_and_isolates_failures():
    async def run():
        plan = plan_resources(
            collection_workers=2,
            server_pool_size=2,
            collection_devices=["cpu"],
            update_device="cpu",
            cpu_reserve=0,
        )
        workers = [
            _LiveReasonerWorker(
                0,
                torch.device("cpu"),
                _Reasoner({"slow": 0.03, "fast": 0.0}),
                object(),
            ),
            _LiveReasonerWorker(
                1,
                torch.device("cpu"),
                _Reasoner({"bad": 0.0}, failures={"bad"}),
                object(),
            ),
        ]
        pool = LiveReasonerPool(workers, plan)
        started = time.perf_counter()
        results, stats = await pool.collect_round(
            [_item("slow"), _item("fast"), _item("bad")], timeout_s=1.0
        )
        elapsed = time.perf_counter() - started
        return results, stats, elapsed

    results, stats, elapsed = asyncio.run(run())
    assert len(results) == 2
    assert stats["searches_failed"] == 1
    assert stats["worker_count"] == 2
    assert elapsed < 0.08
