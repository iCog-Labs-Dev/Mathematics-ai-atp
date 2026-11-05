"""Apply tactics to obligations of retained snapshots.

The executor runs one rendered command on one occurrence: it finds the
occurrence's goal in its own snapshot by metavariable name (never by a stored
position), applies the command there, and hands the result back as a lease.
It decodes the transition provenance into the obligations the tactic produced
for the selected goal, in tactic order, keeping siblings Lean carried over out
of that list. Nothing is reconstructed from a canonical goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Optional

from pantograph_async import PantographResourceError, PantographTacticRejected, TacticState

from maths_ai.data_models.model_identity import carry_model_identity
from maths_ai.data_models.proof_components import Goal

from .command_journal import default_journal
from .execution_registry import Lease, LeaseKind, SnapshotRegistry
from .execution_types import DIRECT, EXTRA, ChildObligation, ExecutionOccurrence
from .pantograph_adapter import execution_goal_to_goal
from .resource_outcomes import ResourceDiagnostic

__all__ = [
    "ExecutionReferenceError",
    "NullTacticExecutor",
    "RetainedOutcome",
    "RetainedTacticExecutor",
]


class ExecutionReferenceError(RuntimeError):
    """An occurrence does not name an open goal of its own snapshot."""


@dataclass
class RetainedOutcome:
    """The result of one command on one occurrence.

    On success ``lease`` holds the result snapshot for the caller, who either
    shares it with the graph or releases it. ``children`` are the obligations
    the tactic produced for the selected goal (direct goals, then restored
    holes) with their observations. ``siblings`` are the refreshed observations
    of the input's other goals; ``closed`` and ``changed`` name the ones the
    transition assigned or altered. ``selected_complete`` and
    ``selected_holes`` are only computed when there are no children: whether
    Lean can instantiate the selected goal's assignment without holes, and the
    unassigned metavariables it still depends on.
    """

    success: bool
    command: str
    error: Optional[str] = None
    lease: Optional[Lease] = None
    children: list[ChildObligation] = field(default_factory=list)
    siblings: dict[str, Goal] = field(default_factory=dict)
    closed: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    selected_complete: Optional[bool] = None
    selected_holes: tuple[str, ...] = ()
    resource: ResourceDiagnostic | None = None


def _canonical(parent: Goal, state: TacticState, name: str) -> Goal:
    goal = next(goal for goal in state.goals if goal.goal_name == name)
    return carry_model_identity(parent, execution_goal_to_goal(goal))


def _introduced(parent: Goal, child: Goal) -> tuple[str, ...]:
    known = {local.internal_name for local in parent.locals or []}
    return tuple(
        local.internal_name
        for local in child.locals or []
        if local.internal_name and local.internal_name not in known
    )


class RetainedTacticExecutor:
    """Runs commands on retained snapshots owned by a ``SnapshotRegistry``."""

    async def apply(
        self,
        registry: SnapshotRegistry,
        occurrence: ExecutionOccurrence,
        command: str,
        *,
        holder: str = "operation",
    ) -> RetainedOutcome:
        if not occurrence.is_retained:
            raise ExecutionReferenceError("The occurrence's snapshot is no longer retained.")
        state = registry.state(occurrence.lease)
        names = [goal.goal_name for goal in state.goals]
        if occurrence.goal_name not in names:
            raise ExecutionReferenceError(
                f"Goal {occurrence.goal_name} is not open in its snapshot."
            )
        index = names.index(occurrence.goal_name)
        journal = default_journal()
        sequence = journal.command(
            phase=holder, session=state.handle.session_id, scope=state.handle.scope_id,
            state=state.handle.state_id,
            goal=occurrence.goal_name, target=occurrence.goal.expression, command=command,
        ) if journal else 0
        async with registry.request(occurrence.lease) as reservation:
            try:
                result = await registry.scope.apply_tactic(state, command, goal_index=index)
            except PantographTacticRejected as exc:
                if journal:
                    journal.outcome(sequence, "rejected", error=str(exc))
                return RetainedOutcome(success=False, command=command, error=str(exc))
            except PantographResourceError as exc:
                if journal:
                    journal.outcome(sequence, "resource", error=str(exc))
                return RetainedOutcome(success=False, command=command, resource=ResourceDiagnostic.from_error(exc))
            except BaseException as exc:
                # A terminal error ends the session; keep its structured payload.
                if journal:
                    journal.outcome(
                        sequence, "terminal", error_type=type(exc).__name__, error=str(exc),
                        payload=json.dumps(getattr(exc, "payload", None), default=str),
                    )
                raise
            if journal:
                journal.outcome(
                    sequence, "accepted", next_state=result.handle.state_id,
                    validation=f"{result.validation.version}:{result.validation.scope}",
                    goals=len(result.goals),
                )
            lease = reservation.adopt(result, LeaseKind.OPERATION, holder)
            try:
                return await self._decode(registry, occurrence, result, lease, command)
            except PantographResourceError as exc:
                await registry.release(lease)
                return RetainedOutcome(success=False, command=command, resource=ResourceDiagnostic.from_error(exc))
            except BaseException:
                await registry.release(lease)
                raise

    async def _decode(
        self,
        registry: SnapshotRegistry,
        occurrence: ExecutionOccurrence,
        result: TacticState,
        lease: Lease,
        command: str,
    ) -> RetainedOutcome:
        transition = result.transition
        parent = occurrence.goal
        children = []
        for slot, (name, introduced) in enumerate(
            zip(transition.direct, transition.introduced)
        ):
            children.append(
                ChildObligation(
                    slot, DIRECT, name, _canonical(parent, result, name), tuple(introduced)
                )
            )
        for name in transition.extra:
            goal = _canonical(parent, result, name)
            children.append(
                ChildObligation(len(children), EXTRA, name, goal, _introduced(parent, goal))
            )
        siblings = {
            name: execution_goal_to_goal(
                next(goal for goal in result.goals if goal.goal_name == name)
            )
            for name in transition.siblings
        }
        complete = None
        holes: tuple[str, ...] = ()
        if not children:
            (selected,) = await registry.scope.inspect(result, [transition.selected])
            complete = selected.is_complete
            holes = selected.holes
        return RetainedOutcome(
            success=True,
            command=command,
            lease=lease,
            children=children,
            siblings=siblings,
            closed=transition.closed,
            changed=transition.changed,
            selected_complete=complete,
            selected_holes=holes,
        )


class NullTacticExecutor(RetainedTacticExecutor):
    """Stand-in executor for tests and tooling without a Lean backend.

    Every application is reported as failed rather than as a vacuous success:
    a success without children would look like a closed goal and fabricate
    proofs. Surfacing the failure keeps the missing backend visible.
    """

    def __init__(self, server=None):
        self.server = server

    async def apply(self, registry, occurrence, command, *, holder="operation"):
        return RetainedOutcome(
            success=False,
            command=command,
            error=(
                f"no TacticExecutor configured — cannot apply {command!r} to "
                f"{occurrence.goal.expression!r}; requires a Lean/Pantograph backend"
            ),
        )
