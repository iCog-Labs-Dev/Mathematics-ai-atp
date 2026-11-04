"""Ownership of the runtime process from the API process.

The supervisor starts the runtime in its own session, waits for its readiness
handshake and correlates operations with replies. An operation that times out
or whose caller goes away is cancelled in the runtime, and the runtime stays
reserved (``settled``) until the cancellation is acknowledged. If it is not
acknowledged within the cleanup grace period, or the runtime dies, the
supervisor kills the runtime's process group and every Lean process it owns,
fails its in-flight work, and starts a new runtime generation. Replies from an
earlier generation are dropped.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

from .process_guardian import GuardedRuntime
from .config import ServiceConfig
from .ipc import IPCError, read_message, send_message

logger = logging.getLogger("maths_ai.api.supervisor")

STARTING, READY, FAILED, RECOVERING, STOPPED = "starting", "ready", "failed", "recovering", "stopped"


class RuntimeUnavailable(RuntimeError):
    """The runtime is starting, recovering, failed or stopped."""


class OperationFailed(RuntimeError):
    def __init__(self, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


class OperationTimeout(RuntimeError):
    """The operation's deadline passed; its cleanup continues in the runtime."""


class OperationInterrupted(RuntimeError):
    """The runtime ended (or was killed) while the operation was in flight."""


class _Cancelled(RuntimeError):
    """The runtime acknowledged an operation's cancellation."""


def descendants(pid: int) -> list[int]:
    """Every live descendant of ``pid`` (Linux ``/proc``), parents first."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(entry.name))
    found, frontier = [], [pid]
    while frontier:
        current = frontier.pop()
        for child in children.get(current, []):
            found.append(child)
            frontier.append(child)
    return found




class RuntimeSupervisor:
    def __init__(
        self,
        config: ServiceConfig,
        config_path: Path,
        *,
        backend: str | None = None,
        on_restart: Callable[[int], Any] | None = None,
        restart_delay_s: float = 5.0,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path)
        self.backend = backend
        self.on_restart = on_restart
        self.restart_delay_s = restart_delay_s
        self.limits = config.limits
        self.limit = config.limits.max_message_bytes
        self.state = STOPPED
        self.error: str | None = None
        self.generation = 0
        self.provenance: dict[str, Any] | None = None
        self.status: dict[str, Any] = {}
        self.process: GuardedRuntime | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._connected: asyncio.Future | None = None
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task] = set()
        self._cleanup: asyncio.Task | None = None
        self._restarting = False
        self._stopping = False
        run_dir = Path(config.state_dir) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(run_dir, 0o700)
        self.socket_path = run_dir / "runtime.sock"

    # Lifecycle -----------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self.state == READY and self._cleanup is None

    def start(self) -> None:
        """Begin starting the runtime without waiting for its model to load."""
        self.state = STARTING
        self._spawn_task(self._start_generation())

    async def wait_ready(self, timeout: float) -> None:
        async with asyncio.timeout(timeout):
            while self.state in (STARTING, RECOVERING):
                await asyncio.sleep(0.05)

    async def _start_generation(self) -> None:
        self.generation += 1
        generation = self.generation
        self.state = STARTING if generation == 1 else RECOVERING
        self.provenance = None
        if self.socket_path.exists():
            self.socket_path.unlink()
        loop = asyncio.get_running_loop()
        self._connected = loop.create_future()
        self._server = await asyncio.start_unix_server(self._accept, path=str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        command = [
            sys.executable, "-m", "maths_ai.api.runtime_worker",
            "--config", str(self.config_path), "--socket", str(self.socket_path),
            "--generation", str(generation),
        ]
        if self.backend:
            command += ["--backend", self.backend]
        try:
            self.process = await GuardedRuntime.start(command, self.limits.runtime_start_timeout_s)
        except Exception as exc:
            self.state, self.error = FAILED, f"generation ownership failed: {exc}"
            self._server.close()
            return
        self._spawn_task(self._watch_exit(self.process, generation))
        try:
            async with asyncio.timeout(self.limits.runtime_start_timeout_s):
                reader, writer = await self._connected
                hello = await read_message(reader, limit=self.limit)
                if hello["type"] != "hello" or hello["generation"] != generation:
                    raise IPCError("Unexpected handshake from the runtime")
                self._writer = writer
                first = await read_message(reader, limit=self.limit)
        except (TimeoutError, IPCError, OSError) as exc:
            if generation == self.generation and not self._stopping:
                self.error = f"runtime did not become ready: {type(exc).__name__}: {exc}"
                logger.error(self.error)
                await self._kill(self.process)
                self.state = FAILED
            return
        if first["type"] == "failed":
            self.error = f"{first.get('category')}: {first.get('detail')}"
            logger.error("runtime startup failed: %s", self.error)
            self.state = FAILED
            await self._kill(self.process)
            return
        self.provenance = first.get("provenance")
        self.status = first.get("status", {})
        self.error = None
        self.state = READY
        self._spawn_task(self._read(reader, generation))

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._connected is not None and not self._connected.done():
            self._connected.set_result((reader, writer))
        else:
            writer.close()

    async def _watch_exit(self, process: asyncio.subprocess.Process, generation: int) -> None:
        await process.wait()
        if generation != self.generation or self._stopping:
            return
        self._interrupt_pending(f"runtime exited with status {process.returncode}")
        if self.state == READY:
            logger.error("runtime generation %d exited (%s); restarting", generation, process.returncode)
            self._schedule_restart()

    def _schedule_restart(self) -> None:
        """Start one recovery: further requests see RECOVERING until it is ready."""
        if self._restarting or self._stopping:
            return
        self._restarting = True
        self.state = RECOVERING
        self._spawn_task(self._restart())

    async def _restart(self) -> None:
        try:
            await self._kill(self.process)
            if self._server is not None:
                self._server.close()
            if self.on_restart is not None:
                await self.on_restart(self.generation)
            await asyncio.sleep(self.restart_delay_s)
            if not self._stopping:
                await self._start_generation()
        except Exception as exc:
            self.state, self.error = FAILED, f"generation cleanup failed: {exc}"
            logger.exception(self.error)
        finally:
            self._restarting = False

    async def stop(self) -> None:
        self._stopping = True
        self.state = STOPPED
        process = self.process
        if process is not None and process.returncode is None and self._writer is not None:
            try:
                await send_message(self._writer, {"type": "shutdown", "generation": self.generation},
                                   limit=self.limit)
                async with asyncio.timeout(self.limits.cleanup_grace_s):
                    await process.wait()
            except (TimeoutError, IPCError, OSError, ConnectionError):
                pass
        await self._kill(process)
        self._interrupt_pending("the service is shutting down")
        if self._server is not None:
            self._server.close()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _kill(self, process: GuardedRuntime | None) -> None:
        """The guardian owns descendants even after the compute parent dies."""
        if process is not None:
            await process.close()

    # Operations ----------------------------------------------------------

    async def settled(self) -> None:
        """Wait until no cancelled operation still holds the runtime."""
        while self._cleanup is not None:
            await asyncio.shield(self._cleanup)

    async def run(self, op: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """Run one operation; a timeout or caller cancellation starts its cleanup."""
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            async with asyncio.timeout_at(deadline):
                await self.settled()
        except TimeoutError as exc:
            raise OperationTimeout("Deadline expired waiting for runtime cleanup") from exc
        if self.state != READY or self._writer is None:
            raise RuntimeUnavailable(self.error or f"runtime is {self.state}")
        op_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[op_id] = future
        generation = self.generation
        try:
            async with asyncio.timeout_at(deadline):
                await send_message(
                    self._writer,
                    {"type": "run", "generation": generation, "op_id": op_id, "op": op, "payload": payload},
                    limit=self.limit,
                )
                return await asyncio.shield(future)
        except TimeoutError as exc:
            self._begin_cleanup(op_id, future, generation)
            raise OperationTimeout(f"{op} exceeded its {timeout:.1f}s allowance") from exc
        except asyncio.CancelledError:
            self._begin_cleanup(op_id, future, generation)
            raise
        except _Cancelled as exc:
            raise OperationInterrupted("the runtime cancelled the operation") from exc
        finally:
            if future.done():
                self._pending.pop(op_id, None)

    def _begin_cleanup(self, op_id: str, future: asyncio.Future, generation: int) -> None:
        if future.done():
            self._pending.pop(op_id, None)
            return
        task = asyncio.create_task(self._settle(op_id, future, generation))
        self._cleanup = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _settle(self, op_id: str, future: asyncio.Future, generation: int) -> None:
        try:
            if generation == self.generation and self._writer is not None:
                await send_message(self._writer, {"type": "cancel", "generation": generation, "op_id": op_id},
                                   limit=self.limit)
                async with asyncio.timeout(self.limits.cleanup_grace_s):
                    await asyncio.gather(future, return_exceptions=True)
        except (TimeoutError, IPCError, OSError, ConnectionError):
            if generation == self.generation:
                logger.error("runtime did not settle a cancelled operation; killing generation %d", generation)
                self._interrupt_pending("the runtime was killed after a cancellation did not settle")
                self._schedule_restart()
                # The kill is part of the cleanup: no operation runs before it is done.
                await self._kill(self.process)
        finally:
            self._pending.pop(op_id, None)
            if self._cleanup is asyncio.current_task():
                self._cleanup = None

    def _interrupt_pending(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(OperationInterrupted(reason))
        self._pending.clear()

    async def _read(self, reader: asyncio.StreamReader, generation: int) -> None:
        try:
            while True:
                message = await read_message(reader, limit=self.limit)
                if message["generation"] != generation or generation != self.generation:
                    continue  # stale: from a runtime lifetime that no longer owns work
                if "status" in message:
                    self.status = message["status"]
                future = self._pending.get(message.get("op_id", ""))
                if future is None or future.done():
                    continue
                kind = message["type"]
                if kind == "result":
                    future.set_result(message["result"])
                elif kind == "error":
                    future.set_exception(OperationFailed(message["category"], message.get("detail", "")))
                elif kind == "cancelled":
                    future.set_exception(_Cancelled())
        except (IPCError, OSError):
            return  # _watch_exit handles the runtime's end

    def _spawn_task(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
