"""The theorem-local owner of retained Lean snapshots.

``PantographStateScope`` owns the wire allocations; this registry is the only
application component that decides when one is released. Graph nodes, replay
attempts and in-flight operations hold ``Lease`` objects and never call
``scope.release`` themselves. A snapshot is deleted when its last lease is
dropped, so several sibling obligations can share one snapshot and removing
one of them cannot invalidate the others.

A result is adopted synchronously, right after the await that produced it, so
no cancellation point separates an allocation from its owner. If deletion
fails or is cancelled the client retires the session, and the registry then
invalidates every lease: an old snapshot can never be used again, and a
restarted worker starts a new theorem attempt.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

from pantograph_async import PantographStateError, PantographStateScope, TacticState

__all__ = [
    "Lease",
    "LeaseKind",
    "RetentionExceeded",
    "Reservation",
    "SnapshotRegistry",
]


class LeaseKind(str, Enum):
    ROOT = "root"
    GRAPH = "graph"
    OPERATION = "operation"
    REPLAY = "replay"
    CERTIFICATE = "certificate"


class RetentionExceeded(RuntimeError):
    """A request would need more retained snapshots than the configured bound."""


@dataclass(eq=False)
class Lease:
    """One holder's claim on one retained snapshot."""

    snapshot: int
    kind: LeaseKind
    holder: str
    released: bool = False


@dataclass(eq=False)
class _Record:
    state: TacticState
    leases: set[Lease]


@dataclass(eq=False)
class Reservation:
    """Exclusive capacity for results not yet returned by Lean."""

    registry: "SnapshotRegistry"
    remaining: int

    def adopt(self, state: TacticState, kind: LeaseKind, holder: str) -> Lease:
        self.registry._check_valid()
        if self not in self.registry._reservations or self.remaining < 1:
            raise PantographStateError("The result reservation is no longer active.")
        self.remaining -= 1
        if not self.remaining:
            self.registry._reservations.remove(self)
        return self.registry._adopt(state, kind, holder)

    def cancel(self) -> None:
        """Refund unused slots, including on rejection or cancellation."""
        self.registry._reservations.discard(self)
        self.remaining = 0


class SnapshotRegistry:
    """Leases over the snapshots of one theorem search, with a retention bound.

    ``max_snapshots`` counts distinct allocations plus reserved result slots,
    not leases or graph nodes.
    Every state-producing request goes through ``before_request``: an
    installed ``make_room`` hook may prune speculative work first, and the
    bound itself is then checked, so it holds throughout execution.
    """

    def __init__(
        self,
        scope: PantographStateScope,
        *,
        max_snapshots: Optional[int] = None,
        make_room: Optional[Callable[[int], Awaitable[None]]] = None,
    ) -> None:
        if max_snapshots is not None and (
            type(max_snapshots) is not int or max_snapshots < 1
        ):
            raise ValueError("max_snapshots must be a positive integer.")
        self.scope = scope
        self.max_snapshots = max_snapshots
        # Frees capacity (for example by pruning open search nodes) or raises
        # RetentionExceeded; called before every state-producing request.
        self.make_room = make_room
        self._records: dict[int, _Record] = {}
        self._reservations: set[Reservation] = set()
        self._next = 0
        self._invalid = False
        self.peak_snapshots = 0
        self.adopted = 0
        self.deleted = 0

    # Ownership

    def _adopt(self, state: TacticState, kind: LeaseKind, holder: str) -> Lease:
        """Take ownership of a freshly returned state. Synchronous by design."""
        self._check_valid()
        snapshot = self._next
        self._next += 1
        lease = Lease(snapshot, kind, holder)
        self._records[snapshot] = _Record(state, {lease})
        self.adopted += 1
        self.peak_snapshots = max(self.peak_snapshots, len(self._records))
        return lease

    def adopt_root(self, state: TacticState, holder: str) -> Lease:
        """Transfer the admitted root from the scope into the registry."""
        reservation = self.reserve()
        try:
            return reservation.adopt(state, LeaseKind.ROOT, holder)
        finally:
            reservation.cancel()

    @asynccontextmanager
    async def request(self, source: Lease):
        """Pin the input and reserve its result until adoption or request exit."""
        pin = self.share(source, LeaseKind.OPERATION, "in-flight input")
        reservation = None
        try:
            reservation = await self.before_request()
            yield reservation
        finally:
            if reservation is not None:
                reservation.cancel()
            await self.release(pin)

    def is_pinned(self, lease: Lease) -> bool:
        """Whether an allocation has holders that frontier pruning must keep."""
        return any(item.kind is not LeaseKind.GRAPH for item in self._record(lease).leases)

    def share(self, lease: Lease, kind: LeaseKind, holder: str) -> Lease:
        """Another holder's lease on the snapshot ``lease`` names."""
        record = self._record(lease)
        shared = Lease(lease.snapshot, kind, holder)
        record.leases.add(shared)
        return shared

    def state(self, lease: Lease) -> TacticState:
        return self._record(lease).state

    async def release(self, lease: Lease) -> None:
        """Drop a lease; delete the snapshot once no lease remains. Idempotent."""
        if lease.released or self._invalid:
            lease.released = True
            return
        record = self._records.get(lease.snapshot)
        lease.released = True
        if record is None:
            return
        record.leases.discard(lease)
        if record.leases:
            return
        try:
            if self.scope.is_usable:
                await self.scope.release(record.state)
                self.deleted += 1
            self._records.pop(lease.snapshot, None)
        except BaseException:
            # Deletion is now uncertain; the client retired the session.
            self.invalidate()
            raise

    async def close(self) -> None:
        """Release every remaining lease. Scope exit deletes anything left."""
        for snapshot in list(self._records):
            record = self._records.get(snapshot)
            if record is None:
                continue
            for lease in list(record.leases):
                await self.release(lease)
        self.invalidate()

    def invalidate(self) -> None:
        """Forget every snapshot without deleting; used after session loss."""
        for record in self._records.values():
            for lease in record.leases:
                lease.released = True
        self._records.clear()
        for reservation in tuple(self._reservations):
            reservation.cancel()
        self._invalid = True

    # Accounting

    @property
    def live_snapshots(self) -> int:
        return len(self._records)

    @property
    def live_leases(self) -> int:
        return sum(len(record.leases) for record in self._records.values())

    @property
    def reserved_snapshots(self) -> int:
        return sum(item.remaining for item in self._reservations)

    def reserve(self, count: int = 1) -> Reservation:
        """Claim capacity synchronously before yielding to another request."""
        self._check_valid()
        if type(count) is not int or count < 1:
            raise ValueError("Reservation count must be a positive integer.")
        if not self.has_capacity(count):
            raise RetentionExceeded(
                f"{self.live_snapshots} retained snapshots, {self.reserved_snapshots} "
                f"reserved and {count} more exceed the bound of {self.max_snapshots}."
            )
        reservation = Reservation(self, count)
        self._reservations.add(reservation)
        return reservation

    async def before_request(self, count: int = 1) -> Reservation:
        """Prune if necessary, then atomically claim result capacity."""
        self._check_valid()
        if self.make_room is not None:
            await self.make_room(count)
        return self.reserve(count)

    def has_capacity(self, count: int = 1) -> bool:
        return self.max_snapshots is None or (
            self.live_snapshots + self.reserved_snapshots + count <= self.max_snapshots
        )

    # Internals

    def _check_valid(self) -> None:
        if self._invalid:
            raise PantographStateError("The snapshot registry was invalidated.")

    def _record(self, lease: Lease) -> _Record:
        self._check_valid()
        record = self._records.get(lease.snapshot)
        if lease.released or record is None or lease not in record.leases:
            raise PantographStateError("The lease does not hold a retained snapshot.")
        return record
