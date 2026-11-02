"""Admission of RL theorem seeds: offline decoding, then a Lean preflight.

A candidate is a source row that decoded and rendered into a replay. It is
admitted only after one scoped client has replayed it and the goal Lean built
matched the seed. Everything else is quarantined with its typed reason and
source identity, so a reconstruction failure is never counted as a failed
proof and never reaches training. Client, protocol and transport failures are
infrastructure and are never used as row labels. A single replay that costs the
client its session (a request timeout, or the REPL printing an uncaught Lean
exception such as a heartbeat limit instead of a JSON reply) is skipped and the
preflight continues on a fresh client. A broken environment still stops it,
because every fresh client runs the startup probes.

Training and ``scripts/validate_rl_seeds.py`` use these same functions, so a
record gets the same classification in both.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Iterator

from pantograph_async import (
    PantographClient,
    PantographEnv,
    PantographProtocolError,
    PantographRequestTimeout,
)
from pantograph_async.diagnostics import distribution_source

from maths_ai.hybrid_reasoner.pantograph_model_sexpr import materialize_seed
from maths_ai.lean_replay import (
    RENDERER_VERSION,
    LeanGoalSeed,
    ReplayInputError,
    render_replay,
    seed_from_text,
)

from .dataset import iter_dataset_rows

EXAMPLES_PER_REASON = 20
# Printed-state size buckets (characters) for coverage reporting.
SIZE_BUCKETS = (200, 400, 800, 1600)


def _domain(source: dict[str, object]) -> str:
    """The Mathlib area of a row (``Mathlib/RingTheory``), or its input kind."""
    path = source.get("file_path")
    if isinstance(path, str) and path:
        return "/".join(path.split("/")[:2]).removesuffix(".lean")
    return "file" if "file" in source else "unknown"


def _size_bucket(size: int) -> str:
    for limit in SIZE_BUCKETS:
        if size <= limit:
            return f"<={limit}"
    return f">{SIZE_BUCKETS[-1]}"


@dataclass
class TheoremItem:
    seed: LeanGoalSeed
    tactic_label: str  # ground-truth tactic from the dataset row ("" in file mode)
    size: int


@dataclass
class AdmissionReport:
    """Counts and bounded examples for one admission pass."""

    source: dict[str, object] = field(default_factory=dict)
    scanned: int = 0
    candidates: int = 0
    admitted: int = 0
    cache_hits: int = 0
    skipped: Counter = field(default_factory=Counter)
    quarantined: Counter = field(default_factory=Counter)
    examples: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    coverage: dict[str, Counter] = field(default_factory=dict)
    name_drift: int = 0
    context_index_shift: int = 0
    stopped_by: str = "source-exhausted"
    preflight_s: float = 0.0
    environment: dict[str, object] = field(default_factory=dict)

    def cover(self, source: dict[str, object], size: int, outcome: str) -> None:
        """Count an outcome by domain and by size, to show what the pool covers."""
        for key in (f"domain:{_domain(source)}", f"size:{_size_bucket(size)}"):
            self.coverage.setdefault(key, Counter())[outcome] += 1

    def quarantine(
        self, seed_source: dict[str, object], error: ReplayInputError, size: int, **extra
    ) -> None:
        reason = error.reason.value
        self.quarantined[reason] += 1
        self.cover(seed_source, size, "quarantined")
        bucket = self.examples.setdefault(reason, [])
        if len(bucket) < EXAMPLES_PER_REASON:
            bucket.append({"source": seed_source, "error": error.as_dict(), **extra})

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "renderer": RENDERER_VERSION,
            "environment": self.environment,
            "scanned": self.scanned,
            "candidates": self.candidates,
            "admitted": self.admitted,
            "quarantined": dict(sorted(self.quarantined.items())),
            "skipped": dict(sorted(self.skipped.items())),
            "cache_hits": self.cache_hits,
            "coverage": {key: dict(value) for key, value in sorted(self.coverage.items())},
            "admitted_with_name_drift": self.name_drift,
            "admitted_with_context_index_shift": self.context_index_shift,
            "stopped_by": self.stopped_by,
            "preflight_s": round(self.preflight_s, 3),
            "examples": self.examples,
        }

    def summary(self) -> str:
        quarantined = ", ".join(f"{k} {v}" for k, v in sorted(self.quarantined.items()))
        skipped = ", ".join(f"{k} {v}" for k, v in sorted(self.skipped.items()))
        return (
            f"Seeds: {self.scanned} scanned, {self.candidates} candidates, "
            f"{self.admitted} Lean-validated; quarantined [{quarantined or 'none'}]; "
            f"skipped [{skipped or 'none'}]; stopped by {self.stopped_by}."
        )


def write_report(report: AdmissionReport, directory: Path) -> Path:
    """Write the report to a new file in ``directory``; never overwrite one."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"admission-{stamp}.json"
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


# Offline candidates ----------------------------------------------------------


def _admit_offline(
    seed_factory, tactic: str, size: int, source: dict[str, object],
    max_state_chars: int | None, report: AdmissionReport,
) -> TheoremItem | None:
    report.scanned += 1
    if max_state_chars is not None and size > max_state_chars:
        report.skipped["over-size"] += 1
        report.cover(source, size, "skipped")
        return None
    try:
        seed = seed_factory()
        render_replay(seed)
    except ReplayInputError as exc:
        report.quarantine(source, exc, size)
        return None
    report.candidates += 1
    return TheoremItem(seed=seed, tactic_label=tactic, size=size)


def dataset_candidates(
    *,
    dataset_name: str,
    split: str,
    revision: str | None,
    max_scan_rows: int,
    max_state_chars: int | None,
    report: AdmissionReport,
) -> Iterator[TheoremItem]:
    """Structured seeds from dataset rows; at most ``max_scan_rows`` are read."""
    report.source = {
        "kind": "dataset",
        "dataset": dataset_name,
        "revision": revision,
        "split": split,
        "max_scan_rows": max_scan_rows,
    }
    for row in iter_dataset_rows(
        dataset_name=dataset_name, split=split, sample_limit=max_scan_rows, revision=revision
    ):
        item = _admit_offline(
            row.structured_seed,
            row.tactic or "",
            len(row.state),
            row.source_identity(),
            max_state_chars,
            report,
        )
        if item is not None:
            yield item


def file_candidates(
    path: Path,
    *,
    max_scan_rows: int,
    max_state_chars: int | None,
    report: AdmissionReport,
) -> Iterator[TheoremItem]:
    """Text seeds from JSONL rows ``{"goal": str, "hypotheses": [str], "tactic"?: str}``.

    Each hypothesis is one declaration as Lean prints it (``n m : Nat``,
    ``x : Nat := 3``) and goes through the shared declaration parser.
    """
    report.source = {"kind": "file", "path": str(path), "max_scan_rows": max_scan_rows}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if report.scanned >= max_scan_rows:
                return
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            goal, hypotheses = row["goal"], row.get("hypotheses", [])
            item = _admit_offline(
                lambda: seed_from_text(
                    goal=goal, hypotheses=hypotheses, source={"file": str(path), "line": line_number}
                ),
                row.get("tactic", ""),
                len(goal) + sum(len(h) for h in hypotheses),
                {"file": str(path), "line": line_number},
                max_state_chars,
                report,
            )
            if item is not None:
                yield item


# Lean preflight --------------------------------------------------------------


def environment_identity(env: PantographEnv) -> dict[str, object]:
    """Identities a cached validation is only valid for."""
    def digest(path: Path) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    repl = Path(env.pantograph_repl)
    manifest = Path(env.source_root) / "lake-manifest.json"
    toolchain = Path(env.source_root) / "lean-toolchain"
    return {
        "client": distribution_source(),
        "repl_sha256": digest(repl),
        "imports": list(env.imports),
        "source_root": str(env.source_root),
        "lake_manifest_sha256": digest(manifest),
        "lean_toolchain_sha256": digest(toolchain),
    }


class ValidationCache:
    """Validated replay fingerprints for one exact environment identity.

    Only admitted seeds are cached, never live handles; a different renderer,
    client, REPL or Mathlib identity starts an empty cache.
    """

    def __init__(self, path: Path | None, identity: dict[str, object]) -> None:
        self.path = path
        self.key = hashlib.sha256(
            json.dumps({"renderer": RENDERER_VERSION, **identity}, sort_keys=True).encode()
        ).hexdigest()
        self.entries: set[str] = set()
        if path is not None and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("identity_key") == self.key:
                self.entries = set(data.get("validated", []))

    def __contains__(self, fingerprint: str) -> bool:
        return fingerprint in self.entries

    def add(self, fingerprint: str) -> None:
        self.entries.add(fingerprint)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"identity_key": self.key, "validated": sorted(self.entries)}),
            encoding="utf-8",
        )
        temporary.replace(self.path)


class AdmissionSession:
    """Lean admission of candidate seeds, advanced a bounded step at a time.

    The session owns one client at a time. A replay that closes the client
    (request timeout, or unsolicited REPL output) is skipped (it says nothing
    about the row's correctness) and a fresh, re-probed client continues.
    Replays longer than ``max_replay_chars`` are skipped before sending.
    Other failures end the session. The deadline is absolute: time spent
    paused between steps counts against it. ``step`` returns only after the
    current Lean request and its scope have settled, so a caller may run other
    work between steps.
    """

    def __init__(
        self,
        candidates: Iterable[TheoremItem],
        client_factory: Callable[[], Awaitable[PantographClient]],
        *,
        max_admitted: int,
        time_budget_s: float,
        report: AdmissionReport,
        cache: ValidationCache | None = None,
        max_replay_chars: int | None = None,
    ) -> None:
        self._candidates = iter(candidates)
        self._client_factory = client_factory
        self.max_admitted = max_admitted
        self.report = report
        self.cache = cache
        self.max_replay_chars = max_replay_chars
        self.admitted: list[TheoremItem] = []
        self.rows_seen = 0
        self.done = False
        self._client: PantographClient | None = None
        self._started = time.monotonic()
        self._deadline = self._started + time_budget_s

    async def step(self, max_rows: int) -> list[TheoremItem]:
        """Admit from at most ``max_rows`` more candidates; return the newly admitted."""
        if self.done:
            return []
        newly: list[TheoremItem] = []
        if self._client is None:
            self._client = await self._client_factory()
        try:
            for _ in range(max_rows):
                item = self._next_candidate()
                if item is None:
                    break
                admitted = await self._admit(item)
                if admitted:
                    newly.append(item)
        except BaseException:
            await self.close()
            raise
        if self.done:
            await self.close()
        return newly

    def _next_candidate(self) -> TheoremItem | None:
        if len(self.admitted) >= self.max_admitted:
            self.report.stopped_by = "pool-full"
            self.done = True
            return None
        if time.monotonic() >= self._deadline:
            self.report.stopped_by = "time-budget"
            self.done = True
            return None
        item = next(self._candidates, None)
        if item is None:
            self.done = True
        else:
            self.rows_seen += 1
        return item

    async def _admit(self, item: TheoremItem) -> bool:
        report, cache = self.report, self.cache
        replay = render_replay(item.seed)
        if self.max_replay_chars is not None and len(replay.expression) > self.max_replay_chars:
            report.skipped["over-size-replay"] += 1
            report.cover(item.seed.source_dict, item.size, "skipped")
            return False
        fingerprint = replay.fingerprint
        if cache is not None and fingerprint in cache:
            report.cache_hits += 1
        else:
            client = self._client
            try:
                async with client.state_scope() as scope:
                    _, _, validation = await materialize_seed(scope, item.seed)
            except ReplayInputError as exc:
                report.quarantine(item.seed.source_dict, exc, item.size, replay=replay.as_dict())
                return False
            except (PantographRequestTimeout, PantographProtocolError) as exc:
                if client.is_usable:
                    raise
                kind = (
                    "replay-timeout"
                    if isinstance(exc, PantographRequestTimeout)
                    else "replay-repl-failure"
                )
                report.skipped[kind] += 1
                report.cover(item.seed.source_dict, item.size, "skipped")
                bucket = report.examples.setdefault(kind, [])
                if len(bucket) < EXAMPLES_PER_REASON:
                    bucket.append(
                        {
                            "source": item.seed.source_dict,
                            "replay_chars": len(replay.expression),
                            "error": str(exc)[:300],
                        }
                    )
                await client.aclose()
                self._client = await self._client_factory()
                return False
            if validation.name_drift:
                report.name_drift += 1
            if validation.source_indices != validation.replay_indices:
                report.context_index_shift += 1
            if cache is not None:
                cache.add(fingerprint)
        report.admitted += 1
        report.cover(item.seed.source_dict, item.size, "admitted")
        self.admitted.append(item)
        return True

    async def close(self) -> None:
        """Close the client and record the time spent; safe to call twice."""
        client, self._client = self._client, None
        try:
            if client is not None:
                await client.aclose()
        finally:
            if self._started is not None:
                self.report.preflight_s += time.monotonic() - self._started
                self._started = None
                if self.cache is not None:
                    self.cache.save()


async def preflight(
    candidates: Iterable[TheoremItem],
    client_factory: Callable[[], Awaitable[PantographClient]],
    *,
    max_admitted: int,
    time_budget_s: float,
    report: AdmissionReport,
    cache: ValidationCache | None = None,
    max_replay_chars: int | None = None,
) -> list[TheoremItem]:
    """Replay candidates in their own scopes until the pool or budget is full.

    Drains an ``AdmissionSession`` in one go; see it for the client, skip and
    deadline rules.
    """
    session = AdmissionSession(
        candidates,
        client_factory,
        max_admitted=max_admitted,
        time_budget_s=time_budget_s,
        report=report,
        cache=cache,
        max_replay_chars=max_replay_chars,
    )
    try:
        while not session.done:
            await session.step(64)
    finally:
        await session.close()
    return session.admitted
