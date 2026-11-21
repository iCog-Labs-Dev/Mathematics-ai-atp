"""Fake-REPL fixtures for application tests that need a real client child.

Protocol and transport behaviour is tested in the client repository; these
helpers only let the model-session factory run against a diagnostic process.
"""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys

import pytest

from pantograph_async import PantographEnv


FAKE = Path(__file__).parent / "fixtures" / "pantograph" / "fake_repl.py"


def _write_executable(destination):
    # Diagnostic children must use pytest's interpreter, independently of PATH.
    _, _, body = FAKE.read_text().partition("\n")
    destination.write_text(f"#!{sys.executable}\n{body}")
    destination.chmod(0o755)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "mathlib"
    project = tmp_path / "Pantograph"
    for root in (source, project):
        root.mkdir()
        (root / "lakefile.lean").write_text("import Lake\n")
        (root / "lean-toolchain").write_text("leanprover/lean4:v4.10.0-rc1\n")
    repl = project / ".lake" / "build" / "bin" / "repl"
    repl.parent.mkdir(parents=True)
    _write_executable(repl)
    tools = tmp_path / "tools"
    tools.mkdir()
    lake = tools / "lake"
    _write_executable(lake)
    report = tmp_path / "events.jsonl"
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PANTOGRAPH_FAKE_REPORT", str(report))
    monkeypatch.setenv(
        "PANTOGRAPH_FAKE_CONTRACT", str(FAKE.with_name("future_contract.json"))
    )
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "normal")
    monkeypatch.setenv("PANTOGRAPH_FAKE_RESOLVER", "normal")
    env = PantographEnv(
        source_root=source,
        pantograph_repl=repl,
        imports=("Init",),
        timeout=1,
        server_start_timeout_s=2,
        worker_shutdown_timeout_s=0.4,
    )
    return env, report


def records(report):
    return (
        [json.loads(line) for line in report.read_text().splitlines()]
        if report.exists()
        else []
    )


def assert_reaped(report):
    for pid in {entry["pid"] for entry in records(report)}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@asynccontextmanager
async def started_operation(operation, entered, *, timeout=5.0):
    """Observe startup or propagate its failure, retaining task cleanup ownership."""
    task = asyncio.create_task(operation)
    notification = asyncio.create_task(entered.wait())
    try:
        done, _ = await asyncio.wait(
            (task, notification), timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            await task
            raise AssertionError("Operation completed before its startup notification.")
        if notification not in done:
            raise TimeoutError("Operation did not reach its startup notification.")
        yield task
    finally:
        notification.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(notification, task, return_exceptions=True)


def run(body):
    async def checked():
        errors = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        async with asyncio.timeout(15):
            await body()
        await asyncio.sleep(0)
        assert not errors
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name().startswith("pantograph-")
        ]

    asyncio.run(checked(), debug=True)
