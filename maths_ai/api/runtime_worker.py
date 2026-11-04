"""The long-lived runtime process: one loaded model, one operation at a time.

Started by the API process's supervisor, never by a request. It connects back
over the private socket, loads its backend once, reports ``ready`` (or a
typed ``failed``) and then serves ``run`` messages one at a time. ``cancel``
cancels the running operation and is acknowledged only after its cleanup
(worker replacement, scope release) has finished. It owns its event loop, the
GPU policy, the retrieval provider and every Lean client.

    python -m maths_ai.api.runtime_worker --config service.json \\
        --socket /state/run/runtime.sock --generation 3
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import sys
import traceback
from typing import Any

from .config import ServiceConfig
from .ipc import IPCError, read_message, send_message

logger = logging.getLogger("maths_ai.api.runtime")

DEFAULT_BACKEND = "maths_ai.api.runtime_backend:LiveBackend"
OPERATIONS = ("prove", "admit", "evaluate", "release")


def _load_backend(spec: str):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def _failure(exc: BaseException) -> tuple[str, str]:
    category = getattr(exc, "category", None)
    if isinstance(category, str):
        return category, getattr(exc, "detail", str(exc))
    return "internal", f"{type(exc).__name__}: {exc}"


class RuntimeWorker:
    def __init__(self, config: ServiceConfig, backend, *, socket_path: str, generation: int) -> None:
        self.config = config
        self.backend = backend
        self.socket_path = socket_path
        self.generation = generation
        self.limit = config.limits.max_message_bytes
        self.writer: asyncio.StreamWriter | None = None
        self.write_lock = asyncio.Lock()
        self.current: tuple[str, asyncio.Task] | None = None

    async def send(self, kind: str, **fields: Any) -> None:
        async with self.write_lock:
            await send_message(
                self.writer, {"type": kind, "generation": self.generation, **fields}, limit=self.limit
            )

    async def serve(self) -> int:
        reader, self.writer = await asyncio.open_unix_connection(self.socket_path)
        await self.send("hello", pid=os.getpid())
        try:
            provenance = await self.backend.start()
        except BaseException as exc:  # noqa: BLE001 — every startup failure is reported, then fatal
            category, detail = _failure(exc)
            logger.error("runtime startup failed: %s", traceback.format_exc())
            await self.send("failed", category=f"startup:{category}", detail=detail[:2000])
            await self.backend.close()
            return 1
        await self.send("ready", provenance=provenance, status=self.backend.status())
        try:
            while True:
                message = await read_message(reader, limit=self.limit)
                if message["generation"] != self.generation:
                    continue  # addressed to another runtime lifetime
                kind = message["type"]
                if kind == "run":
                    self._start(message)
                elif kind == "cancel":
                    await self._cancel(message["op_id"])
                elif kind == "status":
                    await self.send("status", status=self.backend.status())
                elif kind == "shutdown":
                    break
        except IPCError:
            pass  # the API process went away: shut down
        finally:
            if self.current is not None:
                self.current[1].cancel()
                await asyncio.gather(self.current[1], return_exceptions=True)
            await self.backend.close()
        return 0

    def _start(self, message: dict[str, Any]) -> None:
        op_id, operation = message["op_id"], message["op"]
        if self.current is not None:
            asyncio.create_task(self.send("error", op_id=op_id, category="busy",
                                          detail="An operation is already running"))
            return
        if operation not in OPERATIONS:
            asyncio.create_task(self.send("error", op_id=op_id, category="protocol",
                                          detail=f"Unknown operation {operation!r}"))
            return
        task = asyncio.create_task(self._run(op_id, operation, message.get("payload", {})))
        self.current = (op_id, task)

    async def _run(self, op_id: str, operation: str, payload: dict[str, Any]) -> None:
        try:
            result = await getattr(self.backend, operation)(payload)
        except asyncio.CancelledError:
            self.current = None
            await self.send("cancelled", op_id=op_id, status=self.backend.status())
            raise
        except BaseException as exc:  # noqa: BLE001 — reported to the API, the runtime stays up
            self.current = None
            category, detail = _failure(exc)
            if category == "internal":
                logger.error("operation %s failed: %s", operation, traceback.format_exc())
            await self.send("error", op_id=op_id, category=category, detail=detail[:2000])
            return
        self.current = None
        await self.send("result", op_id=op_id, result=result, status=self.backend.status())

    async def _cancel(self, op_id: str) -> None:
        current = self.current
        if current is None or current[0] != op_id:
            await self.send("cancelled", op_id=op_id, status=self.backend.status())
            return
        current[1].cancel()  # _run acknowledges once the operation's cleanup finished


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCTS proving runtime (started by the API)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s runtime %(levelname)s %(message)s")
    config = ServiceConfig.from_json(args.config)
    backend = _load_backend(args.backend)(config)
    worker = RuntimeWorker(config, backend, socket_path=args.socket, generation=args.generation)
    return asyncio.run(worker.serve())


if __name__ == "__main__":
    raise SystemExit(main())
