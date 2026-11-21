"""Theorem-context native snapshots, extracted from frozen GNN 1315dca."""

from dataclasses import dataclass
import asyncio
from functools import lru_cache
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

from .proof_tasks import checked_header, split_prefix
from .retrieval_artifacts import file_hash

LEAN_TOOLCHAIN = "leanprover/lean4:v4.10.0-rc1"
MATHLIB_COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"
PANTOGRAPH_COMMIT = "73781c2d58456e4bf369dadd4a501e1b78a0b177"
HELPER = Path(__file__).parents[1] / "lean/GnnSearch.lean"


@lru_cache(maxsize=8)
def lean_environment(source_root, pantograph_root):
    source_root, pantograph_root = Path(source_root).resolve(), Path(pantograph_root).resolve()
    paths = []
    for root in dict.fromkeys((pantograph_root, source_root)):
        value = subprocess.run(["lake", "env", "printenv", "LEAN_PATH"], cwd=root,
            check=True, capture_output=True, text=True, timeout=60).stdout.strip()
        paths.extend(str((root / part).resolve()) if not Path(part).is_absolute() else part
                     for part in value.split(os.pathsep) if part)
    lean = subprocess.run(["elan", "which", "lean"], cwd=source_root,
        check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    return lean, {**os.environ, "LEAN_PATH": os.pathsep.join(dict.fromkeys(paths))}


def environment_binding(source_root, pantograph_root, *, imports=("Init", "Mathlib")):
    roots = {"source": Path(source_root).resolve(), "pantograph": Path(pantograph_root).resolve()}
    revisions = {}
    for label, root in roots.items():
        revisions[label] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
            check=True, capture_output=True, text=True, timeout=30).stdout.strip()
        if dirty or (root / "lean-toolchain").read_text().strip() != LEAN_TOOLCHAIN:
            raise ValueError(f"unclean or wrong-toolchain {label} checkout")
    if revisions["pantograph"] != PANTOGRAPH_COMMIT or ("Mathlib" in imports and revisions["source"] != MATHLIB_COMMIT):
        raise ValueError("native Lean/Mathlib/Pantograph revision mismatch")
    lean, _ = lean_environment(str(roots["source"]), str(roots["pantograph"]))
    version = subprocess.run([lean, "--version"], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    return {"revisions": revisions, "toolchain": LEAN_TOOLCHAIN, "lean_version": version,
        "native_helper_sha256": file_hash(HELPER), "compiled_backend_sha256": {
            name: file_hash(roots["pantograph"] / ".lake/build/lib" / name)
            for name in ("Pantograph.olean", "Pantograph/Serial.olean")}}


class LiveStateError(ValueError):
    pass


class TacticFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeSnapshot:
    state_id: int
    goals: tuple[dict, ...]

    @property
    def closed(self):
        return not self.goals


class LiveLeanSession:
    def __init__(self, *, source_root, pantograph_root, imports=("Init", "Mathlib"), prefix="", timeout=20,
                 startup_timeout=120, heartbeats=1000000, rec_depth=2048):
        self.source_root, self.pantograph_root = Path(source_root).resolve(), Path(pantograph_root).resolve()
        prefix_imports, self.prefix = split_prefix(prefix)
        self.imports = tuple(dict.fromkeys((*imports, *prefix_imports)))
        self.timeout, self.startup_timeout = timeout, startup_timeout
        self.heartbeats, self.rec_depth = heartbeats, rec_depth
        self.proc, self.temporary, self.stderr_task = None, None, None
        self.logs = []
        self.had_errors = False
        self.root, self.header = None, None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def _stderr(self, process):
        while line := await process.stderr.readline():
            self.logs.append(line.decode("utf-8", errors="replace").rstrip())
            self.had_errors |= ": error:" in self.logs[-1]
            del self.logs[:-40]

    async def _response(self):
        while True:
            raw = await self.proc.stdout.readline()
            if not raw:
                raise RuntimeError("native Lean exited: " + "\n".join(self.logs))
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line.startswith("GNN_STATE "):
                return json.loads(line[len("GNN_STATE "):])
            self.logs.append(line)
            self.had_errors |= ": error:" in line
            del self.logs[:-40]

    @staticmethod
    def _snapshot(response):
        if response.get("status") == "lean_error":
            raise TacticFailure(response.get("error", "native tactic failure"))
        state, goals = response.get("state"), response.get("goals")
        if response.get("status") != "ok" or type(state) is not int or not isinstance(goals, list):
            raise LiveStateError("invalid native snapshot")
        names = [goal.get("name") for goal in goals if isinstance(goal, dict)]
        if len(names) != len(goals) or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise LiveStateError("native goals need unique identities")
        return NativeSnapshot(state, tuple(goals))

    async def start_declaration(self, header):
        header, name = checked_header(header, prefix=self.prefix)
        if self.proc is not None:
            if header == self.header and self.root is not None:
                return self.root
            raise LiveStateError("one target per native session")
        helper = HELPER.read_text().removeprefix("import Pantograph\n")
        imports = "\n".join("import " + module for module in dict.fromkeys(("Pantograph", "Lean", *self.imports)))
        source = (imports + "\n" + helper + "\n" + self.prefix
            + f"\nset_option maxHeartbeats {self.heartbeats}\nset_option maxRecDepth {self.rec_depth}\n"
            + f"run_elab do\n  if (← _root_.Lean.MonadEnv.getEnv).contains `{name} then\n"
            + '    throwError "GNN_TARGET_ALREADY_AVAILABLE"\n'
            + header + " := by\n  gnn_native_search\n")
        self.temporary = tempfile.TemporaryDirectory(prefix="gnn_native_")
        path = Path(self.temporary.name) / "Session.lean"
        path.write_text(source, encoding="utf-8")
        lean, env = lean_environment(str(self.source_root), str(self.pantograph_root))
        self.proc = await asyncio.create_subprocess_exec(lean, str(path), cwd=self.source_root, env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=256 * 1024 * 1024, start_new_session=True)
        self.stderr_task = asyncio.create_task(self._stderr(self.proc))
        try:
            response = await asyncio.wait_for(self._response(), timeout=self.startup_timeout)
            if self.had_errors:
                raise LiveStateError("native input elaboration failed: " + "\n".join(self.logs))
            self.root, self.header = self._snapshot(response), header
            return self.root
        except RuntimeError as error:
            await self.close()
            raise LiveStateError(str(error)) from error
        except BaseException:
            await self.close()
            raise

    async def step(self, parent, tactic):
        if parent.closed or self.proc is None:
            raise LiveStateError("cannot step closed/uninitialized state")
        request = json.dumps({"command": "step", "state": parent.state_id, "tactic": tactic}, ensure_ascii=False)
        self.proc.stdin.write((request + "\n").encode())
        await self.proc.stdin.drain()
        try:
            return self._snapshot(await asyncio.wait_for(self._response(), timeout=self.timeout))
        except (asyncio.TimeoutError, asyncio.CancelledError):
            await self.close()
            raise

    async def close(self):
        process, self.proc = self.proc, None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        if self.stderr_task is not None:
            self.stderr_task.cancel()
            await asyncio.gather(self.stderr_task, return_exceptions=True)
            self.stderr_task = None
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None
