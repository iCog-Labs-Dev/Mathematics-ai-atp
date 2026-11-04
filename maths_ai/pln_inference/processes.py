"""Bounded async PeTTa processes, with cancellation ownership until reaping."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import tempfile
import time


class OutputLimit(RuntimeError):
    pass


class QueryProcesses:
    def __init__(
        self, concurrency=2, pending_limit=64, output_limit=1024 * 1024, temp_dir=None
    ):
        if min(concurrency, pending_limit, output_limit) < 1:
            raise ValueError("PLN process bounds must be positive")
        self.concurrency = concurrency
        self.pending_limit = pending_limit
        self.output_limit = output_limit
        self.temp_dir = temp_dir
        self.pending = 0
        self.active: set[asyncio.subprocess.Process] = set()
        self._semaphore = None
        self._loop = None

    async def run(self, executable, source, *, timeout, deadline=None):
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self.pending:
                raise RuntimeError("A PLN process owner cannot span active event loops")
            self._loop, self._semaphore = loop, asyncio.Semaphore(self.concurrency)
        if self.pending >= self.pending_limit:
            return "PLN pending query limit reached", "resource_limit"
        self.pending += 1
        captured = []
        until = min(
            time.monotonic() + timeout,
            deadline if deadline is not None else float("inf"),
        )
        try:
            async with asyncio.timeout_at(until):
                async with self._semaphore:
                    return await self._execute(executable, source, captured)
        except TimeoutError:
            return "".join(captured) + "\nPLN query deadline expired", "timeout"
        except OutputLimit:
            return "PLN output exceeded its configured bound", "output_limit"
        except OSError as exc:
            return str(exc), "petta_unavailable"
        finally:
            self.pending -= 1

    async def _execute(self, executable, source, captured):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".metta", delete=False, dir=self.temp_dir, encoding="utf-8"
        )
        process = None
        readers = []
        try:
            handle.write(source)
            handle.close()
            # Shield launch: cancellation must still obtain the process handle.
            launch = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    executable,
                    handle.name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            )
            try:
                process = await asyncio.shield(launch)
            except asyncio.CancelledError:
                while not launch.done():
                    try:
                        await asyncio.shield(launch)
                    except asyncio.CancelledError:
                        continue
                process = launch.result()
                raise
            self.active.add(process)
            received = 0

            async def read(stream):
                nonlocal received
                chunks = []
                while data := await stream.read(65536):
                    received += len(data)
                    if received > self.output_limit:
                        raise OutputLimit()
                    chunks.append(data)
                    captured.append(data.decode(errors="replace"))
                return b"".join(chunks).decode(errors="replace")

            readers = [
                asyncio.create_task(read(process.stdout)),
                asyncio.create_task(read(process.stderr)),
            ]
            stdout, stderr = await asyncio.gather(*readers)
            code = await process.wait()
            output = stdout + stderr
            return (
                (output, None)
                if code == 0
                else (f"PeTTa exited with code {code}.\n{output}", "engine_error")
            )
        finally:

            async def finish():
                try:
                    for task in readers:
                        task.cancel()
                    await asyncio.gather(*readers, return_exceptions=True)
                    if process is not None:
                        await self._reap(process)
                        self.active.discard(process)
                finally:
                    handle.close()
                    Path(handle.name).unlink(missing_ok=True)

            cleanup = asyncio.create_task(finish())
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError

    @staticmethod
    async def _reap(process):
        # Signal the entire query group even if its initial process has exited.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            if sig == signal.SIGTERM:
                await asyncio.sleep(0.05)
        await process.wait()
        # Drain pipe EOF before this owner can outlive its event loop.
        await process.communicate()

    async def aclose(self):
        if self.pending:
            raise RuntimeError("Cannot close a PLN owner with unsettled queries")
