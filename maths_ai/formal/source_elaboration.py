"""Isolate a declaration with Lean syntax before elaborating its signature.

The helper executes only required prefix commands. The target proof and all
later commands are excluded before frontend elaboration.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from .admission import AdmissionRejected, source_seed

MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_SORRY_WARNING = "declaration uses 'sorry'"


def _local_record(variable: dict[str, Any]) -> dict[str, Any]:
    value = variable.get("value")
    return {
        "internal_id": variable["name"],
        "name": variable.get("userName") or "_",
        "context_index": variable["contextIndex"],
        "binder_role": variable["binderRole"],
        "is_instance": bool(variable.get("isInstance", False)),
        "is_let": bool(variable.get("isLet", False)),
        "type_sexp": variable["type"]["sexp"],
        "value_sexp": None if not value else value["sexp"],
        "model_name": variable.get("userName") or "_",
        "model_index": variable["contextIndex"],
        "model_type_sexp": variable["type"].get("modelSexp"),
    }


async def elaborate_declaration(send, prepared: dict[str, Any]) -> dict[str, Any]:
    start, rewritten = prepared["start"], prepared["rewritten"]
    second = await send("frontend.process", {"file": rewritten, "sorrys": True, "invocations": False})
    units = second.get("units")
    if not units and second.get("recovery") == "session-unusable":
        # Prefix declarations may be available to the helper but unavailable
        # to the deployment's goal serializer and search environment.
        raise AdmissionRejected(
            "input:unsupported-context",
            "The declaration's signature could not be exported in the deployment environment; "
            "it likely refers to declarations defined in the source file.",
        )
    if not units:
        raise AdmissionRejected("input:malformed-source", f"Lean could not process the declaration: {second}")
    target_unit = units[-1]
    unexpected = [
        m for m in target_unit.get("messages", []) if _SORRY_WARNING not in m and ": error:" in m
    ]
    if unexpected:
        raise AdmissionRejected("input:lean-rejected", f"The declaration does not elaborate: {unexpected[:3]}")
    goals = target_unit.get("goals") or []
    if len(goals) != 1:
        raise AdmissionRejected(
            "input:unsupported-declaration",
            f"Expected one goal at the declaration's sorry, Lean reported {len(goals)}.",
        )
    (goal,) = goals
    return {
        "target_sexp": goal["target"]["sexp"],
        "target_pp": goal["target"].get("pp"),
        "locals": [_local_record(variable) for variable in goal.get("vars", [])],
        "target_unit_start": start,
    }


async def check_statement(send, statement: dict[str, Any]) -> dict[str, Any]:
    payload = {"expr": statement["expression"]}
    if statement["universe_levels"]:
        payload["levels"] = list(statement["universe_levels"])
    response = await send("goal.start", payload)
    if "stateId" not in response:
        return {"ok": False, "error": response.get("desc") or response.get("error") or str(response)}
    await send("goal.delete", {"stateIds": [response["stateId"]]})
    return {"ok": True}


class ReplSession:
    """One REPL process with the deployment's executable, Lake project and imports."""

    def __init__(self, repl: Path, source_root: Path, imports: list[str], *, timeout_s: float) -> None:
        self.repl, self.source_root, self.imports = Path(repl), Path(source_root), list(imports)
        self.deadline = time.monotonic() + timeout_s
        self.children: set[asyncio.subprocess.Process] = set()
        self.process: Optional[asyncio.subprocess.Process] = None

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Source admission deadline expired")
        return remaining

    async def _process(self, *args, **kwargs):
        creation = asyncio.create_task(asyncio.create_subprocess_exec(*args, **kwargs))
        try:
            process = await asyncio.shield(creation)
        except asyncio.CancelledError:
            process = await creation
            self.children.add(process)
            raise
        self.children.add(process)
        return process

    async def __aenter__(self) -> "ReplSession":
        lake, stdbuf = shutil.which("lake"), shutil.which("stdbuf")
        if lake is None or stdbuf is None:
            raise RuntimeError("Source admission needs Lake and GNU stdbuf on PATH.")
        try:
            return await self._start(lake, stdbuf)
        except BaseException:
            await self.__aexit__()
            raise

    async def _start(self, lake, stdbuf):
        probe = await self._process(
            lake, "env", "printenv", "LEAN_PATH", cwd=self.source_root,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
        out, _ = await asyncio.wait_for(probe.communicate(), self.remaining())
        env = dict(os.environ, LEAN_PATH=out.decode().strip())
        self.process = await self._process(
            stdbuf, "-oL", str(self.repl), *self.imports,
            cwd=self.source_root, env=env, start_new_session=True,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=MAX_RESPONSE_BYTES,
        )
        ready = await asyncio.wait_for(self.process.stdout.readline(), self.remaining())
        if ready.strip() != b"ready.":
            raise RuntimeError(f"The admission REPL did not start: {ready[:200]!r}")
        await self.send(
            "options.set", {"printExprAST": True, "printExprPretty": True, "printExprModelAST": True}
        )
        return self

    async def send(self, cmd: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write((json.dumps({"cmd": cmd, "payload": payload}) + "\n").encode())
        await self.process.stdin.drain()
        line = await asyncio.wait_for(self.process.stdout.readline(), self.remaining())
        if not line:
            raise RuntimeError("The admission REPL exited.")
        return json.loads(line)

    async def prepare(self, body, declaration):
        with tempfile.TemporaryDirectory(prefix="formal-source-") as tmp:
            request = Path(tmp) / "request.json"
            request.write_text(json.dumps({"body": body, "declaration": declaration, "imports": self.imports}))
            process = await self._process(
                "lake", "env", "lean", "--run", str(Path(__file__).with_name("prepare_source.lean")),
                str(request), cwd=self.source_root, start_new_session=True,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            async def collect():
                async def bounded_read(stream):
                    chunks, total = [], 0
                    while chunk := await stream.read(65536):
                        total += len(chunk)
                        if total > MAX_RESPONSE_BYTES:
                            raise AdmissionRejected("input:malformed-source", "Source helper output exceeds limit")
                        chunks.append(chunk)
                    return b"".join(chunks)
                out, err = await asyncio.gather(bounded_read(process.stdout), bounded_read(process.stderr))
                await process.wait()
                return out, err
            out, err = await asyncio.wait_for(collect(), self.remaining())
            if process.returncode:
                raise AdmissionRejected("input:malformed-source", err.decode(errors="replace")[-4000:])
            return json.loads(out)

    async def __aexit__(self, *_exc) -> None:
        for process in self.children:
            # Kill the owned group even if its launcher already exited.
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
            await process.wait()


async def elaborate_source(session_factory, body: str, declaration: str, statement_of) -> dict[str, Any]:
    """Elaborate in one session and check the rendered statement in the same one."""
    async with session_factory() as session:
        prepared = await session.prepare(body, declaration)
        goal = await elaborate_declaration(session.send, prepared)
        statement = statement_of(goal)
        goal["check"] = await check_statement(session.send, statement)
        return goal



async def elaborate_in_environment(env, payload: dict[str, Any], *, timeout_s: float):
    """Use a dedicated source elaboration process in the deployment environment."""
    declaration = payload["declaration"]
    return await elaborate_source(
        lambda: ReplSession(env.pantograph_repl, env.source_root, list(env.imports), timeout_s=timeout_s),
        payload["body"],
        declaration,
        lambda goal: source_seed(goal, declaration)[1],
    )
