"""Evidence for failed Pantograph sessions, owned by the application.

The client reports every request it serializes (``pantograph_async.evidence``).
Each session created by :func:`create_model_sexpr_client` keeps a bounded
:class:`RequestHistory` of those events, annotated with the application phase
that issued them (search, replay, artifact check, certification). When a worker
is replaced after a session failure, the failed attempt is written to a
run-owned :class:`FailureArchive`: the theorem's seed, the attempt context, the
client's final failure snapshot (including the REPL's stderr record) and the
session's request history. ``scripts/replay_pantograph_failure.py`` replays it
without a model.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator

from .command_journal import default_journal

__all__ = [
    "FAILURE_SCHEMA_VERSION",
    "FailureArchive",
    "RequestHistory",
    "execution_phase",
    "failure_record",
    "failure_summary",
    "runtime_identity",
]

FAILURE_SCHEMA_VERSION = 1

_PHASE: ContextVar[str | None] = ContextVar("pantograph_execution_phase", default=None)


@contextmanager
def execution_phase(name: str) -> Iterator[None]:
    """Label the Pantograph requests issued inside this block (e.g. ``replay``)."""
    token = _PHASE.set(name)
    try:
        yield
    finally:
        _PHASE.reset(token)


class RequestHistory:
    """A session's recent request events, bounded in count and bytes.

    Used as the client observer. Each event also goes to the opt-in command
    journal. When old events are dropped, ``dropped`` counts them, and a replay
    of the remaining history is no longer exact.
    """

    def __init__(self, *, max_events: int = 4096, max_bytes: int = 4 * 1024 * 1024) -> None:
        if max_events <= 0 or max_bytes <= 0:
            raise ValueError("History bounds must be positive.")
        self.max_events = max_events
        self.max_bytes = max_bytes
        self._events: deque[tuple[dict[str, Any], int]] = deque()
        self._bytes = 0
        self.dropped = 0
        self._lock = threading.Lock()

    def __call__(self, event: Any) -> None:
        record = {**event.to_dict(), "phase": _PHASE.get()}
        size = len(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        with self._lock:
            self._events.append((record, size))
            self._bytes += size
            while len(self._events) > self.max_events or (self._bytes > self.max_bytes and len(self._events) > 1):
                _, removed = self._events.popleft()
                self._bytes -= removed
                self.dropped += 1
        journal = default_journal()
        if journal is not None:
            journal.event(record)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record, _ in self._events]


def failure_summary(snapshot: Any) -> str:
    """One readable line: the failed operation and what the REPL reported."""
    if snapshot is None:
        return "no failure snapshot"
    record = snapshot.repl_failure
    text = f"{snapshot.operation} failed with {snapshot.error_type}"
    if record is not None:
        detail = (record.detail or "").replace("\n", " ")[:300]
        text += f" (REPL {record.exception_category} in {record.phase}: {detail})"
    elif snapshot.association == "ambiguous":
        text += " (REPL record ambiguous)"
    return text


def _seed(theorem: Any) -> dict[str, Any] | None:
    seed = getattr(theorem, "seed", theorem)
    as_dict = getattr(seed, "as_dict", None)
    try:
        return as_dict() if callable(as_dict) else None
    except Exception:
        return None


_ENVIRONMENTS: dict[tuple[str, str, tuple[str, ...]], str] = {}


def runtime_identity(server: Any) -> dict[str, Any]:
    """The Lean runtime a session ran in: environment fingerprint (toolchain, Lake
    manifest, REPL executable, imports and every compiled file), effective
    session options, and the client and companion identities.

    The fingerprint uses the deployment's environment-component hashing and is
    computed once per environment per process; call it off the event loop."""
    from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import (
        environment_components, fingerprint_environment_components,
    )
    from pantograph_async.compatibility import companion_base
    from pantograph_async.diagnostics import distribution_version

    spec = server.launch_spec
    key = (str(spec.source_root), str(spec.repl), tuple(spec.imports))
    if key not in _ENVIRONMENTS:
        _ENVIRONMENTS[key] = fingerprint_environment_components(
            environment_components(spec.source_root, spec.repl, spec.imports))
    options = getattr(server, "effective_options", None)
    return {
        "environment_fingerprint": _ENVIRONMENTS[key],
        "imports": list(spec.imports),
        "effective_options": options.to_dict() if options is not None else None,
        "client_version": distribution_version(),
        "companion_commit": companion_base()["source_commit"],
    }


def failure_record(
    *,
    context: dict[str, Any],
    theorem: Any,
    error: BaseException,
    server: Any,
    timeout_owner: str | None = None,
    identity: dict[str, Any] | None = None,
    cleanup_error: BaseException | None = None,
) -> dict[str, Any]:
    """Plain data for one failed attempt; no live states, handles or tensors.

    ``identity`` is :func:`runtime_identity` of the failed session; without it
    the bundle is not replayable. A cleanup failure is kept apart from the
    primary error."""
    snapshot = getattr(server, "last_failure", None)
    history = getattr(server, "request_history", None)
    requests = history.events() if history is not None else []
    complete = (
        history is not None and not history.dropped and snapshot is not None
        and not snapshot.request_truncated
        and not any(event.get("request_truncated") for event in requests)
        and any(event.get("operation") == "goal.start" for event in requests)
        and identity is not None and "environment_fingerprint" in identity
    )
    seed = _seed(theorem)
    missing = [
        reason for reason, absent in (
            ("seed", seed is None), ("failure snapshot", snapshot is None),
            ("request history", history is None),
            ("dropped history", history is not None and history.dropped > 0),
            ("runtime identity", identity is None or "environment_fingerprint" not in identity),
        ) if absent
    ]
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "recorded_at": time.time(),
        "context": dict(context),
        "identity": identity or {},
        "seed": seed,
        "error": {"type": type(error).__name__, "message": str(error)[:4096]},
        "cleanup_error": (None if cleanup_error is None else
                          {"type": type(cleanup_error).__name__, "message": str(cleanup_error)[:4096]}),
        "final": bool(snapshot is not None and snapshot.final),
        "timeout_owner": timeout_owner,
        "snapshot": snapshot.to_dict() if snapshot is not None else None,
        "summary": failure_summary(snapshot),
        "requests": requests,
        "history_dropped": history.dropped if history is not None else None,
        "replayable": complete,
        "missing": missing,
    }


class FailureArchive:
    """Run-owned failure bundles: ``ledger.jsonl`` and ``attempts/<id>/failure.json``.

    Accounting is restored from what is already on disk, so reopening the same
    root (a resumed run, a restarted API runtime) neither reuses an ID nor
    resets the bounds. Every attempt directory is created exclusively, so one
    bundle never overwrites another. The byte bound covers bundles and ledger
    lines, with room reserved for the single ``capture_limit`` line written
    once a bound is reached; later failures are counted only.

    ``begin`` persists a preliminary bundle before the failed session is
    cleaned up; ``finish`` replaces that same bundle with the final evidence.
    """

    LIMIT_RESERVE = 1024

    def __init__(self, root: str | Path, *, max_bytes: int = 256 * 1024 * 1024, max_bundles: int = 2000) -> None:
        if max_bundles <= 0 or max_bytes <= self.LIMIT_RESERVE:
            raise ValueError("Archive bounds must be positive and leave room for the limit marker.")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.max_bundles = max_bundles
        self._lock = threading.Lock()
        self.uncaptured = 0
        attempts = self.root / "attempts"
        self._bundles = sum(1 for _ in attempts.glob("*/failure.json")) if attempts.is_dir() else 0
        self._bytes = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file()) if self.root.is_dir() else 0
        ledger = self.root / "ledger.jsonl"
        # The marker is written once per archive; the limit itself follows the current bounds.
        self._marked = ledger.is_file() and any(
            '"capture_limit"' in line for line in ledger.read_text(encoding="utf-8").splitlines())
        self._limited = False

    def _encode(self, failure_id: str, record: dict[str, Any], status: str) -> bytes:
        return json.dumps({**record, "failure_id": failure_id, "status": status},
                          ensure_ascii=False, indent=1).encode("utf-8")

    def _limit(self, failure_id: str) -> None:
        self.uncaptured += 1
        self._limited = True
        if not self._marked:
            self._marked = True
            self._ledger({"kind": "capture_limit", "max_bytes": self.max_bytes,
                          "max_bundles": self.max_bundles, "at_failure": failure_id}, reserved=True)

    def begin(self, record: dict[str, Any]) -> dict[str, Any]:
        """Persist a preliminary bundle; returns the summary kept with the attempt."""
        with self._lock:
            failure_id = f"f{self._bundles + 1:06d}-{os.getpid()}-{os.urandom(3).hex()}"
            summary = {"failure_id": failure_id, "summary": record.get("summary"),
                       "replayable": record.get("replayable"), "stored": False}
            data = self._encode(failure_id, record, "preliminary")
            line = self._ledger_line({"kind": "failure", **summary, "stored": True,
                                      "path": f"attempts/{failure_id}/failure.json", "context": record.get("context")})
            budget = self.max_bytes - self.LIMIT_RESERVE
            if self._limited or self._bundles >= self.max_bundles or self._bytes + len(data) + len(line) > budget:
                self._limit(failure_id)
                return summary
            directory = self.root / "attempts" / failure_id
            directory.mkdir(parents=True, exist_ok=False)
            self._publish(directory / "failure.json", data)
            self._bundles += 1
            self._bytes += len(data)
            self._append(line)
            summary.update(stored=True, path=f"attempts/{failure_id}/failure.json")
            return summary

    def finish(self, summary: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        """Replace this attempt's own preliminary bundle with the final evidence."""
        summary = {**summary, "summary": record.get("summary"), "replayable": record.get("replayable")}
        if not summary.get("stored"):
            return summary
        with self._lock:
            path = self.root / summary["path"]
            old = path.stat().st_size
            data = self._encode(summary["failure_id"], record, "final")
            line = self._ledger_line({"kind": "finalized", "failure_id": summary["failure_id"],
                                      "summary": summary["summary"], "replayable": summary["replayable"]})
            if self._bytes - old + len(data) + len(line) > self.max_bytes - self.LIMIT_RESERVE:
                # Keep the preliminary bundle rather than exceed the bound.
                self._limit(summary["failure_id"])
                return {**summary, "final": False}
            self._publish(path, data)
            self._bytes += len(data) - old
            self._append(line)
            return {**summary, "final": True}

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        """Persist a complete failure in one step."""
        return self.finish(self.begin(record), record)

    @staticmethod
    def _publish(path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)

    @staticmethod
    def _ledger_line(line: dict[str, Any]) -> bytes:
        return (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8")

    def _ledger(self, line: dict[str, Any], *, reserved: bool = False) -> None:
        data = self._ledger_line(line)
        if not reserved and self._bytes + len(data) > self.max_bytes - self.LIMIT_RESERVE:
            return
        self._append(data)

    def _append(self, data: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "ledger.jsonl").open("ab") as stream:
            stream.write(data)
            stream.flush()
        self._bytes += len(data)
