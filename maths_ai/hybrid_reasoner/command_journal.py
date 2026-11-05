"""Bounded diagnostic journal of the commands sent through retained execution.

A terminal Pantograph error ends the session, and the exception names only the
failing command. Reproducing it needs the commands that came before, in order,
with the phase (exploration, replay, artifact check) each ran in. The journal
writes each command before it is dispatched and its outcome after, one JSON line
at a time, flushed immediately, so a crash or a terminal error still leaves the
last request on disk.

It is off unless ``MATHS_AI_COMMAND_JOURNAL`` names a file. Records are bounded,
the file stops growing at ``max_bytes`` (a final ``truncated`` record says so),
and nothing about model tensors or the environment is written.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

__all__ = ["CommandJournal", "JOURNAL_ENV", "default_journal"]

JOURNAL_ENV = "MATHS_AI_COMMAND_JOURNAL"
_TEXT_LIMIT = 2048


def _bounded(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _TEXT_LIMIT:
        return value[:_TEXT_LIMIT] + f"…[{len(value) - _TEXT_LIMIT} more]"
    return value


class CommandJournal:
    """Append-only JSON lines: ``command`` before dispatch, ``outcome`` after."""

    def __init__(self, path: str | Path, *, max_bytes: int = 8 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._sequence = 0
        self._truncated = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def command(self, **fields: Any) -> int:
        """Record a command about to be sent; returns its sequence number."""
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        self._write({"kind": "command", "seq": sequence, **fields})
        return sequence

    def outcome(self, sequence: int, outcome: str, **fields: Any) -> None:
        self._write({"kind": "outcome", "seq": sequence, "outcome": outcome, **fields})

    def _write(self, record: dict[str, Any]) -> None:
        record = {"time": time.time(), "pid": os.getpid(), **{k: _bounded(v) for k, v in record.items()}}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            if self._truncated:
                return
            size = self.path.stat().st_size if self.path.exists() else 0
            if size + len(line.encode()) > self.max_bytes:
                self._truncated = True
                line = json.dumps({"kind": "truncated", "max_bytes": self.max_bytes}) + "\n"
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())


_DEFAULT: dict[str, CommandJournal] = {}
_DEFAULT_LOCK = threading.Lock()


def default_journal() -> Optional[CommandJournal]:
    """The process journal named by ``MATHS_AI_COMMAND_JOURNAL``, if any."""
    path = os.environ.get(JOURNAL_ENV)
    if not path:
        return None
    with _DEFAULT_LOCK:
        journal = _DEFAULT.get(path)
        if journal is None:
            journal = _DEFAULT[path] = CommandJournal(path)
        return journal
