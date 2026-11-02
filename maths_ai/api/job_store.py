"""Persistent evaluation jobs: versioned SQLite metadata and atomic result files.

The API coordinator is the only writer. Every call opens its own connection in
a worker thread with a bounded busy timeout, so no connection is shared across
threads. A result file is written to a temporary name and renamed into place
before the database records it; a terminal state and its result reference are
committed in one transaction, so a crash between the two never makes a job
look complete. Native handles never enter the store.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

QUEUED, ADMITTING, RUNNING, CANCELLING = "queued", "admitting", "running", "cancelling"
COMPLETED, CANCELLED, FAILED, INTERRUPTED = "completed", "cancelled", "failed", "interrupted"
ACTIVE_STATES = (QUEUED, ADMITTING, RUNNING, CANCELLING)
TERMINAL_STATES = (COMPLETED, CANCELLED, FAILED, INTERRUPTED)

_TRANSITIONS = {
    QUEUED: {ADMITTING, CANCELLING, CANCELLED, FAILED, INTERRUPTED},
    ADMITTING: {RUNNING, CANCELLING, FAILED, INTERRUPTED},
    RUNNING: {COMPLETED, CANCELLING, FAILED, INTERRUPTED},
    CANCELLING: {CANCELLED, FAILED, INTERRUPTED, COMPLETED},
}


class JobStateError(RuntimeError):
    """A transition the job lifecycle does not allow."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JobStore:
    def __init__(self, state_dir: Path, *, busy_timeout_s: float = 5.0) -> None:
        self.state_dir = Path(state_dir)
        self.results_dir = self.state_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "jobs.sqlite3"
        self.busy_timeout_s = busy_timeout_s
        self._lock = asyncio.Lock()

    # Connection handling --------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.busy_timeout_s, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    async def _call(self, function, *args):
        async with self._lock:
            return await asyncio.to_thread(self._in_connection, function, *args)

    def _in_connection(self, function, *args):
        connection = self._connect()
        try:
            return function(connection, *args)
        finally:
            connection.close()

    # Schema and startup ---------------------------------------------------

    async def open(self) -> list[str]:
        """Create or check the schema, then interrupt every unfinished job.

        Returns the identifiers of the jobs interrupted by this restart. They
        are never resumed: a client resubmits explicitly.
        """
        return await self._call(self._open)

    def _open(self, db: sqlite3.Connection) -> list[str]:
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                db.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif int(row["value"]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Job store schema {row['value']} is not the supported version {SCHEMA_VERSION}"
                )
            db.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    phase TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    request TEXT NOT NULL,
                    settings TEXT NOT NULL,
                    provenance TEXT,
                    progress TEXT NOT NULL,
                    error_category TEXT,
                    error_detail TEXT,
                    result_file TEXT,
                    result_partial INTEGER,
                    runtime_generation INTEGER
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS attempts (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    repeat INTEGER NOT NULL,
                    item INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    PRIMARY KEY (job_id, repeat, item)
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS repeat_stats (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    repeat INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    PRIMARY KEY (job_id, repeat)
                )"""
            )
            interrupted = [
                row["id"]
                for row in db.execute(
                    f"SELECT id FROM jobs WHERE state IN ({','.join('?' * len(ACTIVE_STATES))})",
                    ACTIVE_STATES,
                )
            ]
            now = utc_now()
            for job_id in interrupted:
                db.execute(
                    "UPDATE jobs SET state = ?, updated_at = ?, finished_at = ?, error_category = ?, "
                    "error_detail = ? WHERE id = ?",
                    (INTERRUPTED, now, now, "service-restart",
                     "The service restarted before the job finished; resubmit to run it again.", job_id),
                )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        self._reconcile_files(db)
        return interrupted

    def _reconcile_files(self, db: sqlite3.Connection) -> None:
        """Remove result files no committed job refers to (a crash before commit)."""
        referenced = {
            row["result_file"] for row in db.execute("SELECT result_file FROM jobs WHERE result_file IS NOT NULL")
        }
        for path in self.results_dir.iterdir():
            if path.name not in referenced:
                path.unlink(missing_ok=True)

    # Jobs ----------------------------------------------------------------

    async def create(self, request: dict[str, Any], settings: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        await self._call(self._create, job_id, request, settings)
        return job_id

    def _create(self, db, job_id, request, settings) -> None:
        now = utc_now()
        db.execute(
            "INSERT INTO jobs (id, state, phase, created_at, updated_at, request, settings, progress) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, QUEUED, "queued", now, now, json.dumps(request), json.dumps(settings), json.dumps({})),
        )

    async def get(self, job_id: str) -> dict[str, Any] | None:
        return await self._call(self._get, job_id)

    def _get(self, db, job_id) -> dict[str, Any] | None:
        row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        for key in ("request", "settings", "provenance", "progress"):
            job[key] = json.loads(job[key]) if job[key] is not None else None
        job["result_partial"] = None if job["result_partial"] is None else bool(job["result_partial"])
        return job

    async def count_active(self) -> int:
        return await self._call(
            lambda db: db.execute(
                f"SELECT COUNT(*) FROM jobs WHERE state IN ({','.join('?' * len(ACTIVE_STATES))})",
                ACTIVE_STATES,
            ).fetchone()[0]
        )

    async def transition(self, job_id: str, state: str, **fields: Any) -> dict[str, Any]:
        """Move ``job_id`` to ``state``, refusing a transition the lifecycle forbids."""
        return await self._call(self._transition, job_id, state, fields)

    def _transition(self, db, job_id, state, fields) -> dict[str, Any]:
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = row["state"]
            if state != current and state not in _TRANSITIONS.get(current, set()):
                raise JobStateError(f"Job {job_id} cannot move from {current} to {state}")
            self._write(db, job_id, state, fields)
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        return self._get(db, job_id)

    def _write(self, db, job_id, state, fields) -> None:
        now = utc_now()
        columns = {"state": state, "updated_at": now}
        if state in TERMINAL_STATES:
            columns["finished_at"] = now
        for key, value in fields.items():
            if key in ("provenance", "progress"):
                value = json.dumps(value)
            elif key == "result_partial":
                value = int(bool(value))
            columns[key] = value
        assignments = ", ".join(f"{column} = ?" for column in columns)
        db.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*columns.values(), job_id))

    async def record_chunk(
        self, job_id: str, repeat: int, attempts: list[dict[str, Any]], repeat_stats: dict[str, Any],
        progress: dict[str, Any],
    ) -> None:
        """Persist one chunk's attempts, its repeat's running stats and progress together."""
        await self._call(self._record_chunk, job_id, repeat, attempts, repeat_stats, progress)

    def _record_chunk(self, db, job_id, repeat, attempts, repeat_stats, progress) -> None:
        db.execute("BEGIN IMMEDIATE")
        try:
            for attempt in attempts:
                db.execute(
                    "INSERT OR REPLACE INTO attempts VALUES (?, ?, ?, ?)",
                    (job_id, repeat, int(attempt["index"]), json.dumps(attempt)),
                )
            db.execute(
                "INSERT OR REPLACE INTO repeat_stats VALUES (?, ?, ?)",
                (job_id, repeat, json.dumps(repeat_stats)),
            )
            db.execute(
                "UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?",
                (json.dumps(progress), utc_now(), job_id),
            )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    async def attempts(self, job_id: str) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        return await self._call(self._attempts, job_id)

    def _attempts(self, db, job_id):
        attempts = [
            json.loads(row["data"])
            for row in db.execute(
                "SELECT data FROM attempts WHERE job_id = ? ORDER BY repeat, item", (job_id,)
            )
        ]
        stats = {
            row["repeat"]: json.loads(row["data"])
            for row in db.execute("SELECT repeat, data FROM repeat_stats WHERE job_id = ?", (job_id,))
        }
        return attempts, stats

    async def finish(
        self, job_id: str, state: str, result: dict[str, Any], *, partial: bool, **fields: Any
    ) -> dict[str, Any]:
        """Publish ``result`` atomically, then commit it with the terminal ``state``."""
        if state not in TERMINAL_STATES:
            raise JobStateError(f"{state} is not a terminal state")
        name = f"{job_id}.json"
        path = self.results_dir / name
        temporary = self.results_dir / f".{name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
        body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=1)

        def publish() -> None:
            with open(temporary, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)

        await asyncio.to_thread(publish)
        try:
            return await self.transition(
                job_id, state, result_file=name, result_partial=partial, **fields
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    async def result(self, job_id: str) -> dict[str, Any] | None:
        job = await self.get(job_id)
        if job is None or job["result_file"] is None:
            return None
        path = self.results_dir / job["result_file"]
        try:
            return json.loads(await asyncio.to_thread(path.read_text, "utf-8"))
        except FileNotFoundError:
            return None
