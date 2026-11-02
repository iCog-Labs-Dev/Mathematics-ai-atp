"""The runtime process over the real IPC: handshake, results, cancellation, recovery."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from maths_ai.api.config import ServiceConfig
from maths_ai.api.runtime_supervisor import (
    FAILED,
    READY,
    OperationFailed,
    OperationInterrupted,
    OperationTimeout,
    RuntimeSupervisor,
    RuntimeUnavailable,
)
from maths_ai.api.tests.helpers import FAKE_BACKEND, write_config


def _payload(goal: str) -> dict:
    return {"goal": goal, "hypotheses": [], "proposal_mode": "sample", "seed": 1, "timeout_s": 5.0,
            "profile": {}}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as handle:
        return handle.read().rsplit(")", 1)[1].split()[0] != "Z"


async def _started(tmp_path, **overrides) -> RuntimeSupervisor:
    path = write_config(tmp_path, **overrides)
    supervisor = RuntimeSupervisor(ServiceConfig.from_json(path), path, backend=FAKE_BACKEND,
                                   restart_delay_s=0.1)
    supervisor.start()
    await supervisor.wait_ready(60)
    return supervisor


def test_ready_handshake_and_result(tmp_path):
    async def run():
        supervisor = await _started(tmp_path)
        try:
            assert supervisor.state == READY and supervisor.generation == 1
            assert supervisor.provenance == {"checkpoint": "fake", "workers": 1}
            result = await supervisor.run("prove", _payload("proved"), timeout=10)
            assert result["certified"] is True
            with pytest.raises(OperationFailed) as error:
                await supervisor.run("prove", _payload("fail"), timeout=10)
            assert error.value.category == "input:malformed-text"
        finally:
            await supervisor.stop()

    asyncio.run(run())


def test_startup_failure_is_a_readiness_failure_not_a_crash_loop(tmp_path):
    async def run():
        supervisor = await _started(tmp_path, checkpoint="missing.pt")
        try:
            assert supervisor.state == FAILED
            assert "startup:artifacts" in supervisor.error
            with pytest.raises(RuntimeUnavailable):
                await supervisor.run("prove", _payload("proved"), timeout=1)
        finally:
            await supervisor.stop()

    asyncio.run(run())


def test_a_cooperative_cancellation_settles_and_keeps_the_generation(tmp_path):
    async def run():
        supervisor = await _started(tmp_path)
        try:
            with pytest.raises(OperationTimeout):
                await supervisor.run("prove", _payload("sleep:30"), timeout=0.3)
            assert not supervisor.ready  # reserved until the cancellation settles
            await supervisor.settled()
            assert supervisor.ready and supervisor.generation == 1
            assert (await supervisor.run("prove", _payload("proved"), timeout=10))["certified"]
        finally:
            await supervisor.stop()

    asyncio.run(run())


def test_a_blocked_runtime_is_killed_with_its_lean_processes_and_restarted(tmp_path):
    async def run():
        supervisor = await _started(tmp_path)
        try:
            spawned = (await supervisor.run("prove", _payload("spawn"), timeout=10))["spawned"]
            runtime_pid = supervisor.process.pid
            started = time.monotonic()
            with pytest.raises(OperationTimeout):
                await supervisor.run("prove", _payload("block:60"), timeout=0.3)
            # The HTTP side stays responsive while the runtime is blocked.
            assert time.monotonic() - started < 2.0
            await supervisor.settled()
            await supervisor.wait_ready(60)
            assert supervisor.generation == 2 and supervisor.state == READY
            assert not _alive(runtime_pid)
            assert not _alive(spawned)  # its own session: killed by group, not orphaned
            assert (await supervisor.run("prove", _payload("proved"), timeout=10))["certified"]
        finally:
            await supervisor.stop()

    asyncio.run(run())


def test_a_dying_runtime_interrupts_in_flight_work_and_restarts(tmp_path):
    async def run():
        supervisor = await _started(tmp_path)
        try:
            task = asyncio.create_task(supervisor.run("prove", _payload("sleep:30"), timeout=60))
            await asyncio.sleep(0.3)
            os.kill(supervisor.process.pid, 9)
            with pytest.raises(OperationInterrupted):
                await task
            await asyncio.sleep(0.2)
            await supervisor.wait_ready(60)
            assert supervisor.generation == 2 and supervisor.state == READY
        finally:
            await supervisor.stop()

    asyncio.run(run())


def test_stop_leaves_no_runtime_or_lean_process(tmp_path):
    async def run():
        supervisor = await _started(tmp_path)
        spawned = (await supervisor.run("prove", _payload("spawn"), timeout=10))["spawned"]
        runtime_pid = supervisor.process.pid
        await supervisor.stop()
        return runtime_pid, spawned

    runtime_pid, spawned = asyncio.run(run())
    assert not _alive(runtime_pid) and not _alive(spawned)
