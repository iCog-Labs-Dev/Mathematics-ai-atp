"""Native query cancellation, bounded output and shared process capacity."""

import asyncio
from pathlib import Path
import time

from maths_ai.pln_inference.processes import QueryProcesses


def executable(tmp_path, body):
    script = tmp_path / "petta"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(0o755)
    return str(script)


def test_cancelled_query_is_reaped_before_releasing_capacity(tmp_path):
    marker = tmp_path / "pid"
    script = executable(tmp_path, f"echo $$ > {marker}\nsleep 300\n")

    async def run():
        owner = QueryProcesses(concurrency=1)
        task = asyncio.create_task(owner.run(script, "query", timeout=300))
        while not marker.exists():
            await asyncio.sleep(0.01)
        pid = int(marker.read_text())
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()  # A second cancellation must not interrupt process reaping.
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert owner.pending == 0 and not owner.active
        assert not Path(f"/proc/{pid}").exists()
        await owner.aclose()

    asyncio.run(run())


def test_process_slot_wait_counts_toward_query_deadline(tmp_path):
    script = executable(tmp_path, "sleep 0.2\nprintf 'completed'\n")

    async def run():
        owner = QueryProcesses(concurrency=1)
        first = asyncio.create_task(owner.run(script, "a", timeout=2))
        while not owner.active:
            await asyncio.sleep(0.005)
        started = time.monotonic()
        output, status = await owner.run(script, "b", timeout=0.03)
        assert status == "timeout" and time.monotonic() - started < 0.15
        assert len(owner.active) == 1
        assert (await first)[1] is None
        assert not owner.pending

    asyncio.run(run())


def test_oversized_output_has_no_successful_result(tmp_path):
    script = executable(tmp_path, "yes overflow\n")

    async def run():
        owner = QueryProcesses(output_limit=100)
        output, status = await owner.run(script, "a", timeout=2)
        assert status == "output_limit"
        assert not owner.active and owner.pending == 0

    asyncio.run(run())
