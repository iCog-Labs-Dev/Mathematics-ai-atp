"""Bounds on retained snapshots, replay work and resident memory.

One policy applies: retain the pinned anchors and a bounded frontier, and
prune speculative alternatives under pressure. An evicted snapshot is never
recovered by rebuilding its goal; its obligation simply stops being
searchable, with unknown evidence.

Resident memory is sampled, not accounted per snapshot. A REPL process and the
shared collection process each get a growth bound over a baseline taken after
imports and model initialization. The collection workers are tasks in one
Python process, so that process is monitored once for the whole pool. The
bounds are stop and prune thresholds, not operating-system limits, and memory
need not fall right after deletion because allocators keep freed pages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping, Optional

from .proof_replay import ReplayLimits

__all__ = ["ResidentMemoryMonitor", "RetentionLimits", "resident_mb"]


@dataclass(frozen=True)
class RetentionLimits:
    """Per-theorem retention and replay bounds plus resident-memory growth bounds.

    The defaults are conservative development values; deployment defaults are
    set from measurements.
    """

    max_retained_snapshots: int = 256
    max_replay_tactic_steps: int = 256
    max_replay_combinations: int = 32
    max_replay_continuations: int = 4
    max_repl_rss_growth_mb: float = 2048.0
    max_collection_rss_growth_mb: float = 4096.0

    def __post_init__(self) -> None:
        for name in (
            "max_retained_snapshots",
            "max_replay_tactic_steps",
            "max_replay_combinations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if type(self.max_replay_continuations) is not int or self.max_replay_continuations < 0:
            raise ValueError("max_replay_continuations must be a nonnegative integer.")
        for name in ("max_repl_rss_growth_mb", "max_collection_rss_growth_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < math.inf:
                raise ValueError(f"{name} must be a positive number, got {value!r}.")

    @classmethod
    def from_config(cls, payload: Optional[Mapping]) -> "RetentionLimits":
        payload = dict(payload or {})
        known = {item.name for item in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown snapshot retention settings: {sorted(unknown)}")
        return cls(**payload)

    def as_dict(self) -> dict:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def replay_limits(self, *, certify_heartbeats: Optional[int] = None) -> ReplayLimits:
        return ReplayLimits(
            max_tactic_steps=self.max_replay_tactic_steps,
            max_combinations=self.max_replay_combinations,
            max_continuations=self.max_replay_continuations,
            certify_heartbeats=certify_heartbeats,
        )


def resident_mb(pid: Optional[int] = None) -> Optional[float]:
    """Resident set size in MiB from ``/proc``, or ``None`` where unavailable."""
    path = Path("/proc") / ("self" if pid is None else str(pid)) / "status"
    try:
        for line in path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


class ResidentMemoryMonitor:
    """Sampled resident-memory growth of one process over its baseline."""

    def __init__(self, limit_mb: float, *, pid: Optional[int] = None) -> None:
        self.limit_mb = limit_mb
        self.pid = pid
        self.baseline = resident_mb(pid)
        self.peak_growth = 0.0

    def growth_mb(self) -> Optional[float]:
        current = resident_mb(self.pid)
        if current is None or self.baseline is None:
            return None
        growth = current - self.baseline
        self.peak_growth = max(self.peak_growth, growth)
        return growth

    def exceeded(self) -> bool:
        growth = self.growth_mb()
        return growth is not None and growth > self.limit_mb
