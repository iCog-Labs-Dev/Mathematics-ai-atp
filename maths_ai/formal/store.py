"""The durable registry of formal runs, their attempts and their replays.

One SQLite file holds every coordinator-owned formal run: the original
request and its digest, the resolved deployment and pinned profile
configuration, the admitted tactic allowance and how much of it attempts have
consumed, the admitted input's content addresses, and the latest result.
Dataset evaluation jobs are not stored here.

Consumption is never replenished. An attempt reserves the run's whole
remaining allowance when it starts. When it finishes, the reported count is
settled; an attempt that never reports back (the service stopped or the
runtime was lost) is charged its full reservation, and the run records that
its consumption is an upper bound rather than an exact count.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    request_digest TEXT NOT NULL,
    request_json TEXT NOT NULL,
    deployment_id TEXT NOT NULL,
    backend TEXT NOT NULL,
    profile TEXT NOT NULL,
    profile_config_hash TEXT NOT NULL,
    environment_id TEXT NOT NULL,
    environment_hash TEXT NOT NULL,
    model_json TEXT NOT NULL,
    input_kind TEXT NOT NULL,
    input_artifact_hash TEXT,
    source_artifact_hash TEXT,
    source_origin TEXT,
    requested_max_steps INTEGER NOT NULL,
    admitted_max_steps INTEGER NOT NULL,
    consumed_steps INTEGER NOT NULL DEFAULT 0,
    consumed_exact INTEGER NOT NULL DEFAULT 1,
    requested_wall_clock_ms INTEGER,
    admitted_wall_clock_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    disposition TEXT,
    termination TEXT,
    result_json TEXT,
    certificate_hash TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    run_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    admitted_at REAL NOT NULL,
    started_at REAL,
    ended_at REAL,
    reserved_steps INTEGER NOT NULL,
    used_steps INTEGER,
    used_exact INTEGER,
    outcome TEXT,
    elapsed_ms INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    PRIMARY KEY (run_id, attempt)
);
CREATE TABLE IF NOT EXISTS replays (
    run_id TEXT NOT NULL,
    replay INTEGER NOT NULL,
    at REAL NOT NULL,
    status TEXT NOT NULL,
    attestation_json TEXT NOT NULL,
    PRIMARY KEY (run_id, replay)
);
"""

ACTIVE = ("admitting", "queued", "running", "cancelling")


@dataclass
class RunRecord:
    run_id: str
    request_digest: str
    request: dict[str, Any]
    deployment_id: str
    backend: str
    profile: str
    profile_config_hash: str
    environment_id: str
    environment_hash: str
    model: dict[str, Any]
    input_kind: str
    input_artifact_hash: Optional[str]
    source_artifact_hash: Optional[str]
    source_origin: Optional[str]
    requested_max_steps: int
    admitted_max_steps: int
    consumed_steps: int
    consumed_exact: bool
    requested_wall_clock_ms: Optional[int]
    admitted_wall_clock_ms: int
    status: str
    disposition: Optional[str]
    termination: Optional[str]
    result: Optional[dict[str, Any]]
    certificate_hash: Optional[str]
    attempts: int
    cancel_requested: bool
    created_at: float
    updated_at: float

    @property
    def remaining_steps(self) -> int:
        return max(0, self.admitted_max_steps - self.consumed_steps)


def _record(row: sqlite3.Row) -> RunRecord:
    values = dict(row)
    return RunRecord(
        run_id=values["run_id"],
        request_digest=values["request_digest"],
        request=json.loads(values["request_json"]),
        deployment_id=values["deployment_id"],
        backend=values["backend"],
        profile=values["profile"],
        profile_config_hash=values["profile_config_hash"],
        environment_id=values["environment_id"],
        environment_hash=values["environment_hash"],
        model=json.loads(values["model_json"]),
        input_kind=values["input_kind"],
        input_artifact_hash=values["input_artifact_hash"],
        source_artifact_hash=values["source_artifact_hash"],
        source_origin=values["source_origin"],
        requested_max_steps=values["requested_max_steps"],
        admitted_max_steps=values["admitted_max_steps"],
        consumed_steps=values["consumed_steps"],
        consumed_exact=bool(values["consumed_exact"]),
        requested_wall_clock_ms=values["requested_wall_clock_ms"],
        admitted_wall_clock_ms=values["admitted_wall_clock_ms"],
        status=values["status"],
        disposition=values["disposition"],
        termination=values["termination"],
        result=None if values["result_json"] is None else json.loads(values["result_json"]),
        certificate_hash=values["certificate_hash"],
        attempts=values["attempts"],
        cancel_requested=bool(values["cancel_requested"]),
        created_at=values["created_at"],
        updated_at=values["updated_at"],
    )


class FormalRunStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(SCHEMA)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(attempts)")}
        if "elapsed_ms" not in columns or "result_json" not in columns:
            raise RuntimeError("Formal store uses an older contract; use a fresh state directory.")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _write(self, statements: list[tuple[str, tuple]]) -> None:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                for sql, params in statements:
                    self._db.execute(sql, params)
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            row = self._db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else _record(row)

    def insert(self, record: RunRecord) -> bool:
        """Insert a new run; False if the id already exists (the caller reconciles)."""
        now = time.time()
        try:
            self._write([(
                "INSERT INTO runs (run_id, request_digest, request_json, deployment_id, backend, profile, "
                "profile_config_hash, environment_id, environment_hash, model_json, input_kind, "
                "input_artifact_hash, source_artifact_hash, source_origin, requested_max_steps, "
                "admitted_max_steps, consumed_steps, consumed_exact, requested_wall_clock_ms, "
                "admitted_wall_clock_ms, status, attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1, ?, ?, ?, 0, ?, ?)",
                (
                    record.run_id, record.request_digest, json.dumps(record.request, sort_keys=True),
                    record.deployment_id, record.backend, record.profile, record.profile_config_hash,
                    record.environment_id, record.environment_hash, json.dumps(record.model, sort_keys=True),
                    record.input_kind, record.input_artifact_hash, record.source_artifact_hash,
                    record.source_origin, record.requested_max_steps, record.admitted_max_steps,
                    record.requested_wall_clock_ms, record.admitted_wall_clock_ms, record.status, now, now,
                ),
            )])
        except sqlite3.IntegrityError:
            return False
        return True

    def update(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        columns = []
        values = []
        for key, value in fields.items():
            if key == "result":
                key, value = "result_json", None if value is None else json.dumps(value, sort_keys=True)
            if key in ("cancel_requested", "consumed_exact"):
                value = int(bool(value))
            columns.append(f"{key} = ?")
            values.append(value)
        columns.append("updated_at = ?")
        values.append(time.time())
        self._write([(f"UPDATE runs SET {', '.join(columns)} WHERE run_id = ?", (*values, run_id))])

    # Attempts ------------------------------------------------------------------

    def begin_attempt(self, run_id: str, admitted_at: float) -> tuple[int, int]:
        """Open the next attempt, reserving the run's whole remaining allowance."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT attempts, admitted_max_steps, consumed_steps FROM runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                attempt = row["attempts"] + 1
                reserved = max(0, row["admitted_max_steps"] - row["consumed_steps"])
                self._db.execute(
                    "INSERT INTO attempts (run_id, attempt, admitted_at, reserved_steps) VALUES (?, ?, ?, ?)",
                    (run_id, attempt, admitted_at, reserved),
                )
                self._db.execute(
                    "UPDATE runs SET attempts = ?, status = 'queued', disposition = NULL, termination = NULL, "
                    "result_json = NULL, certificate_hash = NULL, cancel_requested = 0, "
                    "updated_at = ? WHERE run_id = ?",
                    (attempt, time.time(), run_id),
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return attempt, reserved

    def mark_started(self, run_id: str, attempt: int) -> None:
        self._write([
            ("UPDATE attempts SET started_at = ? WHERE run_id = ? AND attempt = ?", (time.time(), run_id, attempt)),
            ("UPDATE runs SET status = 'running', updated_at = ? WHERE run_id = ? AND status = 'queued'",
             (time.time(), run_id)),
        ])

    def settle_attempt(
        self,
        run_id: str,
        attempt: int,
        *,
        used_steps: Optional[int],
        outcome: str,
        run_fields: dict[str, Any],
    ) -> None:
        """Charge an attempt's steps and record the run's new state in one transaction.

        ``used_steps`` of ``None`` means the attempt's cost is unknown: its whole
        reservation is charged and the run's consumption stops being exact.
        """
        run_fields = dict(run_fields)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT reserved_steps, used_steps FROM attempts WHERE run_id = ? AND attempt = ?",
                    (run_id, attempt),
                ).fetchone()
                if row is None or row["used_steps"] is not None:
                    raise RuntimeError(f"Attempt {attempt} of {run_id} is not open.")
                reserved = row["reserved_steps"]
                exact = used_steps is not None
                charged = reserved if used_steps is None else min(int(used_steps), reserved)
                self._db.execute(
                    "UPDATE attempts SET ended_at = ?, used_steps = ?, used_exact = ?, outcome = ?, elapsed_ms = ?, result_json = ? "
                    "WHERE run_id = ? AND attempt = ?",
                    (time.time(), charged, int(exact), outcome,
                     run_fields.pop("elapsed_ms", 0), json.dumps(run_fields.get("result")), run_id, attempt),
                )
                fields = dict(run_fields)
                assignments = ["consumed_steps = consumed_steps + ?", "updated_at = ?"]
                values: list[Any] = [charged, time.time()]
                if not exact:
                    assignments.append("consumed_exact = 0")
                for key, value in fields.items():
                    if key == "result":
                        key, value = "result_json", None if value is None else json.dumps(value, sort_keys=True)
                    assignments.append(f"{key} = ?")
                    values.append(value)
                self._db.execute(
                    f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?", (*values, run_id)
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def attempt_rows(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM attempts WHERE run_id = ? ORDER BY attempt", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def consumed_wall_ms(self, run_id: str) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COALESCE(SUM(elapsed_ms), 0) FROM attempts WHERE run_id = ?", (run_id,)
            ).fetchone()[0]

    def open_attempt(self, run_id: str) -> Optional[int]:
        with self._lock:
            row = self._db.execute(
                "SELECT attempt FROM attempts WHERE run_id = ? AND used_steps IS NULL "
                "ORDER BY attempt DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return None if row is None else row["attempt"]

    def interrupted_runs(self) -> list[RunRecord]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM runs WHERE status IN ({', '.join('?' * len(ACTIVE))})", ACTIVE
            ).fetchall()
        return [_record(row) for row in rows]

    # Replays -------------------------------------------------------------------

    def add_replay(self, run_id: str, status: str, attestation: dict[str, Any]) -> int:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT COALESCE(MAX(replay), 0) AS n FROM replays WHERE run_id = ?", (run_id,)
                ).fetchone()
                number = row["n"] + 1
                self._db.execute(
                    "INSERT INTO replays (run_id, replay, at, status, attestation_json) VALUES (?, ?, ?, ?, ?)",
                    (run_id, number, time.time(), status, json.dumps(attestation, sort_keys=True)),
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return number

    def replays(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT replay, at, status, attestation_json FROM replays WHERE run_id = ? ORDER BY replay",
                (run_id,),
            ).fetchall()
        return [
            {"replay": row["replay"], "at": row["at"], "status": row["status"],
             "attestation": json.loads(row["attestation_json"])}
            for row in rows
        ]
