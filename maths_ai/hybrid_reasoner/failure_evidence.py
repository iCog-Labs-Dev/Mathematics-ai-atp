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


def failure_record(
    *,
    context: dict[str, Any],
    theorem: Any,
    error: BaseException,
    server: Any,
    timeout_owner: str | None = None,
) -> dict[str, Any]:
    """Plain data for one failed attempt; no live states, handles or tensors."""
    snapshot = getattr(server, "last_failure", None)
    history = getattr(server, "request_history", None)
    requests = history.events() if history is not None else []
    complete = (
        history is not None and not history.dropped and snapshot is not None
        and not snapshot.request_truncated
        and not any(event.get("request_truncated") for event in requests)
        and any(event.get("operation") == "goal.start" for event in requests)
    )
    seed = _seed(theorem)
    try:
        from pantograph_async.compatibility import companion_base
        from pantograph_async.diagnostics import distribution_version
        identity = {"client_version": distribution_version(),
                    "companion_commit": companion_base()["source_commit"]}
    except Exception as exc:
        identity = {"error": f"{type(exc).__name__}: {exc}"[:256]}
    missing = [
        reason for reason, absent in (
            ("seed", seed is None), ("failure snapshot", snapshot is None),
            ("request history", history is None),
            ("dropped history", history is not None and history.dropped > 0),
        ) if absent
    ]
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "recorded_at": time.time(),
        "context": dict(context),
        "identity": identity,
        "seed": seed,
        "error": {"type": type(error).__name__, "message": str(error)[:4096]},
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

    Bounded in bytes and in bundles. Once a bound is reached, one
    ``capture_limit`` ledger line says so and later failures are counted only.
    Bundles are written to a temporary name and renamed, so a bundle on disk is
    complete.
    """

    def __init__(self, root: str | Path, *, max_bytes: int = 256 * 1024 * 1024, max_bundles: int = 2000) -> None:
        if max_bytes <= 0 or max_bundles <= 0:
            raise ValueError("Archive bounds must be positive.")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.max_bundles = max_bundles
        self._lock = threading.Lock()
        self._bytes = 0
        self._bundles = 0
        self._limited = False
        self.uncaptured = 0

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        """Persist a failure; return the plain summary kept with the attempt."""
        with self._lock:
            self._bundles += 1
            failure_id = f"f{self._bundles:06d}-{os.getpid()}"
            summary = {
                "failure_id": failure_id, "summary": record.get("summary"),
                "replayable": record.get("replayable"), "stored": False,
            }
            data = json.dumps({**record, "failure_id": failure_id}, ensure_ascii=False, indent=1).encode("utf-8")
            if self._limited or self._bundles > self.max_bundles or self._bytes + len(data) > self.max_bytes:
                self.uncaptured += 1
                if not self._limited:
                    self._limited = True
                    self._ledger({"kind": "capture_limit", "max_bytes": self.max_bytes,
                                  "max_bundles": self.max_bundles, "at_failure": failure_id})
                return summary
            directory = self.root / "attempts" / failure_id
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "failure.json.tmp"
            temporary.write_bytes(data)
            os.replace(temporary, directory / "failure.json")
            self._bytes += len(data)
            summary.update(stored=True, path=str((directory / "failure.json").relative_to(self.root)))
            self._ledger({"kind": "failure", **summary, "context": record.get("context")})
            return summary

    def _ledger(self, line: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "ledger.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, ensure_ascii=False) + "\n")
            stream.flush()
