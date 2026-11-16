"""Elaborate a source declaration's signature in a dedicated Lean REPL session.

The owned Pantograph client deliberately does not expose ``frontend.process``
(it can leave a session unusable), so admission runs its own short-lived REPL
process from the same executable, Lake project and imports as the search
workers, and retires it afterwards. The steps are:

1. Parse the file body without elaborating any caller proof, using the
   deployment's imported syntax, and locate the target declaration's unit.
2. Process the body up to that unit with the target's proof replaced by
   ``by sorry``; later declarations are dropped. The goal Lean reports at that
   ``sorry`` is the signature with its outer binders introduced.
3. Check that the rendered closed statement elaborates with ``goal.start`` in
   the REPL's own environment, which is the one search runs in.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from .admission import AdmissionRejected, target_unit_signature

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


def _errors(messages: list[str]) -> list[str]:
    return [
        message for message in messages
        if ": error:" in message or (": warning:" not in message and "error" in message.lower())
    ]


async def elaborate_declaration(send, body: str, declaration: str, *, units) -> dict[str, Any]:
    """Run the three steps over ``send(cmd, payload) -> dict`` (one REPL session)."""
    body_bytes = body.encode("utf-8")
    start, assign = target_unit_signature(units, body_bytes, declaration)
    rewritten = (body_bytes[:assign] + b":= by sorry\n").decode("utf-8")
    second = await send("frontend.process", {"file": rewritten, "sorrys": True, "invocations": False})
    units = second.get("units")
    if not units and second.get("recovery") == "session-unusable":
        # The REPL cannot export a goal that mentions constants
        # outside its own environment, i.e. declarations of the source file.
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
        self.timeout_s = timeout_s
        self.process: Optional[asyncio.subprocess.Process] = None

    async def __aenter__(self) -> "ReplSession":
        lake, stdbuf = shutil.which("lake"), shutil.which("stdbuf")
        if lake is None or stdbuf is None:
            raise RuntimeError("Source admission needs Lake and GNU stdbuf on PATH.")
        probe = await asyncio.create_subprocess_exec(
            lake, "env", "printenv", "LEAN_PATH", cwd=self.source_root,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(probe.communicate(), self.timeout_s)
        env = dict(os.environ, LEAN_PATH=out.decode().strip())
        self.process = await asyncio.create_subprocess_exec(
            stdbuf, "-oL", str(self.repl), *self.imports,
            cwd=self.source_root, env=env, start_new_session=True,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=MAX_RESPONSE_BYTES,
        )
        ready = await asyncio.wait_for(self.process.stdout.readline(), self.timeout_s)
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
        line = await asyncio.wait_for(self.process.stdout.readline(), self.timeout_s)
        if not line:
            raise RuntimeError("The admission REPL exited.")
        return json.loads(line)

    async def __aexit__(self, *_exc) -> None:
        process = self.process
        if process is None or process.returncode is not None:
            return
        try:
            process.stdin.close()
            await asyncio.wait_for(process.wait(), 5)
        except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
            await process.wait()


async def elaborate_source(session_factory, body: str, declaration: str, statement_of, *, units) -> dict[str, Any]:
    """Elaborate in one session and check the rendered statement in the same one."""
    async with session_factory() as session:
        goal = await elaborate_declaration(session.send, body, declaration, units=units)
        statement = statement_of(goal)
        goal["check"] = await check_statement(session.send, statement)
        return goal
