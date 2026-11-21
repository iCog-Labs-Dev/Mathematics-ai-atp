"""Execution identities for search over retained Lean snapshots.

A search keeps the states Lean returns and runs later tactics on them
directly, so a canonical goal no longer identifies something Lean can execute.
Three identities are kept apart:

- the canonical ``Goal`` is the model observation (GNN and PLN input, and what
  learning stores);
- an ``ExecutionOccurrence`` is one obligation, a Lean metavariable, inside one
  retained snapshot;
- ``ObligationKey`` and ``DeclarationKey`` are portable addresses. They name an
  obligation or a local by how the search produced it, so a replay in another
  Lean lineage, where every metavariable and local has a new ID, can find the
  corresponding one. They never contain server handles or Lean IDs.

Goal positions and printed names are not identities: Lean reorders goal lists,
repeats identical targets and shadows names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from maths_ai.data_models.proof_components import ArgumentRef, Goal

if TYPE_CHECKING:
    from .execution_registry import Lease

__all__ = [
    "ChildObligation",
    "DeclarationKey",
    "ExecutionOccurrence",
    "ObligationKey",
    "bind_arguments",
    "child_declarations",
    "root_declarations",
]

DIRECT = "direct"
EXTRA = "extra"


@dataclass(frozen=True, order=True)
class ObligationKey:
    """The root obligation, or child ``slot`` of the accepted tactic ``attempt``.

    Attempts are numbered within one search (they are the graph's edge IDs), so
    a key is valid inside that search and in the replay artifact it exports.
    """

    attempt: Optional[int] = None
    slot: int = 0

    @classmethod
    def root(cls) -> "ObligationKey":
        return cls()

    @property
    def is_root(self) -> bool:
        return self.attempt is None


@dataclass(frozen=True, order=True)
class DeclarationKey:
    """The ``position``-th local of the admitted root, or the ``position``-th
    local that ``attempt`` introduced into its child ``slot``."""

    attempt: Optional[int]
    slot: int
    position: int

    @classmethod
    def root(cls, position: int) -> "DeclarationKey":
        return cls(None, 0, position)


@dataclass(frozen=True)
class ChildObligation:
    """One obligation an accepted tactic produced, in its result snapshot.

    ``kind`` is ``direct`` for goals the tactic returned for the selected goal
    and ``extra`` for reachable holes Lean's validation restored. ``introduced``
    lists the live IDs of the locals this goal has that the selected goal
    lacked, in context order.
    """

    slot: int
    kind: str
    goal_name: str
    goal: Goal
    introduced: tuple[str, ...]


@dataclass(eq=False)
class ExecutionOccurrence:
    """One obligation in one retained snapshot.

    ``declarations`` maps the live ID of each local in ``goal`` to its
    portable key. The lease is dropped once the snapshot is no longer needed;
    the occurrence's identities and observation stay readable after that.
    """

    occurrence_id: int
    key: ObligationKey
    goal_name: str
    goal: Goal
    declarations: dict[str, DeclarationKey] = field(default_factory=dict)
    lease: Optional["Lease"] = None

    @property
    def is_retained(self) -> bool:
        return self.lease is not None and not self.lease.released


def root_declarations(goal: Goal) -> dict[str, DeclarationKey]:
    """Every local of the admitted root is a root declaration, in context order."""
    return {
        local.internal_name: DeclarationKey.root(position)
        for position, local in enumerate(goal.locals or [])
        if local.internal_name
    }


def child_declarations(
    parent: Mapping[str, DeclarationKey],
    attempt: int,
    child: ChildObligation,
) -> dict[str, DeclarationKey]:
    """A child's keys: the parent's for locals it kept, new ones for those introduced."""
    present = {local.internal_name for local in child.goal.locals or []}
    keys = {name: key for name, key in parent.items() if name in present}
    for position, name in enumerate(child.introduced):
        keys[name] = DeclarationKey(attempt, child.slot, position)
    return keys


def _is_identifier(text: str) -> bool:
    parts = text.split(".")
    return bool(parts) and all(
        bool(core := part.rstrip("'!?")) and core.isidentifier() for part in parts
    )


def bind_arguments(goal: Goal, arguments: Sequence[str]) -> list[ArgumentRef]:
    """What each argument refers to in ``goal``, the way Lean resolves it.

    A name is a local when it is the visible name of the latest local that
    carries it, since that is the declaration Lean finds; any other identifier
    is a global and anything else a literal. Proposers that already know the
    declaration they chose should attach their own references instead.
    """
    latest: dict[str, str] = {}
    for local in goal.locals or []:
        if local.internal_name:
            latest[local.execution_or_live_name] = local.internal_name
    refs = []
    for text in arguments:
        name = text.rstrip(":").strip()
        if name in latest and "✝" not in name:
            refs.append(ArgumentRef(kind="local", text=text, internal_name=latest[name]))
        elif _is_identifier(name):
            refs.append(ArgumentRef(kind="global", text=text))
        else:
            refs.append(ArgumentRef(kind="literal", text=text))
    return refs
