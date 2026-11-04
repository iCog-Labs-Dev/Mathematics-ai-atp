"""Linux generation ownership, including adopted children in separate sessions.

Executed as a file so the guardian imports only the standard library. The API
keeps its stdin open; EOF or a stop command initiates bounded teardown.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import platform
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def signal_identity(pid, identity, sig):
    """Use Linux pidfds without depending on optional CPython os bindings."""
    # These numbers are shared by the supported x86-64 and AArch64 Linux ABIs.
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Generation ownership requires x86-64 or AArch64 Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    descriptor = libc.syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        if process_identity(pid) == identity:
            result = libc.syscall(
                424,
                ctypes.c_int(descriptor),
                ctypes.c_int(sig),
                ctypes.c_void_p(),
                ctypes.c_uint(0),
            )
            if result < 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
    finally:
        os.close(descriptor)


def process_identity(pid: int) -> tuple[int, str] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return pid, fields[19]
    except (OSError, IndexError):
        return None


def owned_processes() -> dict[int, tuple[int, str]]:
    parents: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            parents.setdefault(int(fields[1]), []).append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    pending, owned = [os.getpid()], {}
    while pending:
        for pid in parents.get(pending.pop(), []):
            identity = process_identity(pid)
            if identity is not None:
                owned[pid] = identity
                pending.append(pid)
    return owned


def cleanup() -> bool:
    for sig, allowance in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 10.0)):
        until = time.monotonic() + allowance
        while True:
            owned = owned_processes()
            for pid, identity in owned.items():
                if process_identity(pid) == identity:
                    try:
                        signal_identity(pid, identity, sig)
                    except ProcessLookupError:
                        pass
            while True:
                try:
                    pid, _status = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        break
                except ChildProcessError:
                    break
            if not owned_processes():
                return True
            if time.monotonic() >= until:
                break
            time.sleep(0.02)
    return False


def main(command: list[str]) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot establish generation subreaper")
    signal_identity(os.getpid(), process_identity(os.getpid()), 0)
    stopping = False

    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        child = subprocess.Popen(command, start_new_session=True, stdout=sys.stderr)
        print(json.dumps({"pid": child.pid}), flush=True)
        while child.poll() is None and not stopping:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable:
                # EOF also means that the API owner has disappeared.
                sys.stdin.buffer.readline()
                stopping = True
        code = child.poll()
    finally:
        cleaned = cleanup()
    return (0 if stopping or code == 0 else 1) if cleaned else 2


class GuardedRuntime:
    """API-side handle; wait completes only after the guardian reaps its tree."""

    def __init__(self, guardian: asyncio.subprocess.Process, pid: int):
        self.guardian = guardian
        self.pid = pid
        self._closing = asyncio.Lock()

    @classmethod
    async def start(cls, command: list[str], timeout: float):
        guardian = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            "--",
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            async with asyncio.timeout(timeout):
                line = await guardian.stdout.readline()
                pid = json.loads(line)["pid"]
            return cls(guardian, pid)
        except BaseException:
            guardian.stdin.close()
            wait = asyncio.create_task(guardian.wait())
            while not wait.done():
                try:
                    await asyncio.shield(wait)
                except asyncio.CancelledError:
                    continue
            wait.result()
            await guardian.communicate()
            raise

    @property
    def returncode(self):
        return self.guardian.returncode

    async def wait(self):
        return await self.guardian.wait()

    async def close(self):
        async with self._closing:
            self.guardian.stdin.close()
            async with asyncio.timeout(15):
                code = await self.guardian.wait()
                await self.guardian.communicate()
            if code not in (0, 1):
                raise RuntimeError(f"Generation guardian exited unexpectedly: {code}")


if __name__ == "__main__":
    try:
        code = main(sys.argv[2:])
    except Exception:
        import traceback

        traceback.print_exc()
        code = 2
    raise SystemExit(code)
