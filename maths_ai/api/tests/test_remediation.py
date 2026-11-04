"""Regression cases for independent deadlines, durable capacity and ownership."""

import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from maths_ai.api.config import ServiceConfig
from maths_ai.api.job_store import JobStore
from maths_ai.api.process_guardian import GuardedRuntime
from maths_ai.api.service import DeadlineExceeded, ProvingService, QueueFull
from maths_ai.api.tests.helpers import write_config
from maths_ai.api.tests.test_scheduler import _Runtime


class BlockedRuntime(_Runtime):
    def __init__(self):
        super().__init__()
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def run(self, op, payload, *, timeout):
        if op == "prove":
            self.entered.set()
            await self.release.wait()
            return {"outcome": "unsolved", "certified": False}
        return await super().run(op, payload, timeout=timeout)


def test_queued_proof_expires_while_dispatcher_is_busy(tmp_path):
    async def run():
        path = write_config(tmp_path)
        service = ProvingService(ServiceConfig.from_json(path), path)
        service.supervisor = runtime = BlockedRuntime()
        await service.start()
        first = asyncio.create_task(service.prove({"goal": "p", "timeout_s": 5.0}))
        try:
            await runtime.entered.wait()
            started = time.monotonic()
            with pytest.raises(DeadlineExceeded):
                await service.prove(
                    {"goal": "q", "timeout_s": 0.3, "queue_timeout_s": 0.05}
                )
            assert time.monotonic() - started < 0.2
            assert not first.done()
            assert not service._proofs
        finally:
            runtime.release.set()
            await first
            await service.stop()

    asyncio.run(run())


def test_concurrent_job_submission_is_atomically_bounded(tmp_path):
    async def run():
        path = write_config(tmp_path, limits={"max_pending_jobs": 1})
        service = ProvingService(ServiceConfig.from_json(path), path)
        await service.store.open()
        results = await asyncio.gather(
            *[
                service.submit_evaluation({"dataset_profile": "smoke"})
                for _ in range(10)
            ],
            return_exceptions=True,
        )
        assert sum(isinstance(result, dict) for result in results) == 1
        assert sum(isinstance(result, QueueFull) for result in results) == 9
        assert await service.store.count_active() == 1
        for timer in service._job_timers.values():
            timer.cancel()

    asyncio.run(run())


def test_database_capacity_transaction_is_shared_across_store_instances(tmp_path):
    from maths_ai.api.job_store import JobCapacityError

    async def run():
        stores = [JobStore(tmp_path), JobStore(tmp_path)]
        for store in stores:
            await store.open()
        results = await asyncio.gather(
            *[stores[index % 2].create({}, {}, capacity=1) for index in range(10)],
            return_exceptions=True,
        )
        assert sum(isinstance(result, str) for result in results) == 1
        assert sum(isinstance(result, JobCapacityError) for result in results) == 9

    asyncio.run(run())


def test_disconnected_submission_still_has_a_scheduled_durable_job(tmp_path):
    async def run():
        path = write_config(tmp_path)
        service = ProvingService(ServiceConfig.from_json(path), path)
        await service.store.open()
        original = service.store.create
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        service.store.create = delayed
        request = asyncio.create_task(
            service.submit_evaluation({"dataset_profile": "smoke"})
        )
        await entered.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        release.set()
        await asyncio.gather(*service._submissions)
        assert len(service._jobs) == 1
        assert await service.store.count_active() == 1
        for timer in service._job_timers.values():
            timer.cancel()

    asyncio.run(run())


def test_queued_job_deadline_includes_waiting_time(tmp_path):
    async def run():
        path = write_config(tmp_path)
        service = ProvingService(ServiceConfig.from_json(path), path)
        service.supervisor = runtime = BlockedRuntime()
        await service.start()
        first = asyncio.create_task(service.prove({"goal": "p", "timeout_s": 5.0}))
        try:
            await runtime.entered.wait()
            job = await service.submit_evaluation(
                {"dataset_profile": "smoke", "deadline_s": 0.05}
            )
            await asyncio.sleep(0.15)
            saved = await service.job(job["id"])
            assert saved["state"] == "failed" and saved["error_category"] == "deadline"
            assert (await service.job_result(job["id"]))["partial"]
            assert not first.done()
        finally:
            runtime.release.set()
            await first
            await service.stop()

    asyncio.run(run())


def test_restart_publishes_committed_attempts_and_is_idempotent(tmp_path):
    async def run():
        path = write_config(tmp_path)
        service = ProvingService(ServiceConfig.from_json(path), path)
        await service.store.open()
        job = await service.submit_evaluation({"dataset_profile": "smoke"})
        job_id = job["id"]
        await service.store.transition(job_id, "admitting")
        await service.store.transition(job_id, "running")
        await service.store.record_chunk(
            job_id,
            0,
            [{"index": 0, "outcome": "solved"}],
            {"attempted": 1.0, "solved": 1.0},
            {"attempts_completed": 1},
        )
        await JobStore(service.config.state_dir).open()
        first = await service.job_result(job_id)
        assert first["state"] == "interrupted" and first["partial"]
        assert len(first["attempts"]) == 1
        await JobStore(service.config.state_dir).open()
        assert await service.job_result(job_id) == first
        for timer in service._job_timers.values():
            timer.cancel()

    asyncio.run(run())


def test_missing_committed_result_is_a_storage_error(tmp_path):
    from maths_ai.api.service import ServiceError

    async def run():
        path = write_config(tmp_path)
        service = ProvingService(ServiceConfig.from_json(path), path)
        await service.store.open()
        job = await service.submit_evaluation({"dataset_profile": "smoke"})
        await service.cancel_job(job["id"])
        saved = await service.job(job["id"])
        (service.store.results_dir / saved["result_file"]).unlink()
        with pytest.raises(ServiceError) as error:
            await service.job_result(job["id"])
        assert error.value.category == "result-storage" and error.value.status == 500

    asyncio.run(run())


@pytest.mark.parametrize("kill_parent", [True, False])
def test_guardian_reaps_separate_session_term_resistant_children(tmp_path, kill_parent):
    async def run():
        marker = tmp_path / "child.json"
        child_code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(300)"
        code = (
            "import subprocess,time,json; from pathlib import Path; "
            f"p=subprocess.Popen([{sys.executable!r},'-c',{child_code!r}],start_new_session=True); "
            f"Path({str(marker)!r}).write_text(json.dumps(p.pid)); time.sleep(300)"
        )
        runtime = await GuardedRuntime.start([sys.executable, "-c", code], 5)
        try:
            for _ in range(100):
                if marker.exists():
                    break
                await asyncio.sleep(0.02)
            child = json.loads(marker.read_text())
            await asyncio.sleep(0.1)
            if kill_parent:
                os.kill(runtime.pid, signal.SIGKILL)
                await asyncio.wait_for(runtime.wait(), 15)
            else:
                await runtime.close()
            assert not Path(f"/proc/{child}").exists()
            assert not Path(f"/proc/{runtime.pid}").exists()
        finally:
            await runtime.close()

    asyncio.run(run())
