"""Fair dispatch between interactive proofs and evaluation steps."""

from __future__ import annotations

import asyncio

from maths_ai.api.config import ServiceConfig
from maths_ai.api.service import ProvingService
from maths_ai.api.tests.helpers import write_config


class _Runtime:
    """A ready runtime that records the operations it ran."""

    ready = True
    state = "ready"
    error = None
    generation = 1
    provenance = {}
    status = {"usable_workers": 1}

    def __init__(self):
        self.ops: list[str] = []

    async def settled(self):
        return None

    async def run(self, op, payload, *, timeout):
        self.ops.append(op)
        await asyncio.sleep(0.01)
        if op == "prove":
            return {"outcome": "solved", "certified": True}
        if op == "admit":
            return {"input_records": [{"identity": "fake-0"}, {"identity": "fake-1"}], "proposal_records": [[]], "rows_seen": 2, "admitted": 2, "newly_admitted": 2, "done": True, "eval_items": 2,
                    "eval_subset_sha256": "s", "proposal_seeds_sha256": ["a"], "admission": {}}
        if op == "evaluate":
            count = payload["count"]
            return {"attempts": [{"index": payload["start"] + i} for i in range(count)],
                    "stats": {"attempted": float(count), "solved": float(count)}, "chunk_s": 0.0}
        return {"released": True}

    def start(self):
        return None

    async def stop(self):
        return None


def test_two_proofs_then_one_evaluation_step(tmp_path):
    async def run():
        path = write_config(tmp_path, limits={"max_pending_proofs": 8, "eval_chunk_size": 1})
        service = ProvingService(ServiceConfig.from_json(path), path)
        runtime = _Runtime()
        service.supervisor = runtime
        await service.start()
        try:
            await service.submit_evaluation({"dataset_profile": "smoke"})
            proofs = [asyncio.create_task(service.prove({"goal": "p"})) for _ in range(4)]
            await asyncio.gather(*proofs)
            while service._active is not None or service._jobs:
                await asyncio.sleep(0.01)
        finally:
            await service.stop()
        return runtime.ops

    ops = asyncio.run(run())
    # The job was queued first, yet at most two proofs run before each job step.
    job_ops = {"admit", "evaluate", "release"}
    streak = longest = 0
    for op in ops:
        streak = streak + 1 if op == "prove" else 0
        longest = max(longest, streak)
    assert longest <= 2, ops
    assert ops.count("prove") == 4 and {"admit", "evaluate", "release"} <= set(ops)
    assert ops.index("admit") < 3  # the waiting job is not starved by proofs
    assert set(ops) <= job_ops | {"prove"}
