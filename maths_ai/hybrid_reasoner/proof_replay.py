"""Combine candidate recipes by replaying them in one Lean execution.

Branches of the AND/OR search are explored independently, each in its own
retained snapshot, so their assignments are separate: two branches can each
succeed while disagreeing about a shared witness. Combining them therefore
means replaying the chosen attempts one after another against a single
evolving state that starts at the admitted root, then asking the kernel to
certify the result.

Replay is a bounded backtracking search over the candidate recipes:

- obligations are processed in the recorded AND order; each is found in the
  current state by its metavariable, never by position;
- every argument is re-pointed at the same declaration through its key and
  the command is rendered again with the shared renderer;
- the attempt's child obligations are matched to the recorded slots, and a
  shape mismatch is a conflict;
- an obligation a shared assignment already discharged needs no tactic;
- a conditional step, one that closes its goal through obligations it still
  refers to (``rfl`` on ``?w = ?w``), is accepted when those obligations are
  still pending in this replay; they are solved later in the same state and
  the kernel check of the root decides;
- a rejected step or a conflict tries the next alternative from the retained
  prefix state; failed prefixes are remembered so they are not retried. A
  prefix state is released as soon as no alternative is left to try from it.

Every step reaches Lean through the executor, so it is subject to the same
snapshot reservation and memory policy as exploration; when no room can be
made, the replay ends with ``RESOURCE_LIMIT``.

When an obligation whose observation was changed by a shared assignment has
no working recipe, the state is kept as a continuation: the search can expand
that actual obligation, and replay resumes from it. A conflict belongs to one
combination; it never marks a tactic or a goal as impossible.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

from pantograph_async import CertificationResult

from maths_ai.data_models.model_identity import carry_model_identity
from maths_ai.data_models.proof_components import Goal, TacticCandidate
from maths_ai.lean_replay import GoalReplaySpec

from .execution_registry import Lease, LeaseKind, RetentionExceeded, SnapshotRegistry
from .execution_types import (
    DeclarationKey,
    ExecutionOccurrence,
    ObligationKey,
    child_declarations,
    root_declarations,
)
from .pantograph_adapter import execution_goal_to_goal
from .pantograph_model_sexpr import materialize_replay
from .proof_recipes import (
    ArtifactArgument,
    ArtifactStep,
    BindingConflict,
    RecipeSource,
    RecipeStep,
    ReplayArtifact,
    candidate_for,
    rebind_arguments,
)
from .retained_executor import ExecutionReferenceError, RetainedTacticExecutor
from .tactic_rendering import render_tactic_command

__all__ = [
    "ArtifactMismatch",
    "ReplayConflict",
    "ReplayContinuation",
    "ReplayCoordinator",
    "ReplayLimits",
    "ReplayResult",
    "ReplayStatus",
    "WitnessStep",
    "build_artifact",
    "reverify",
]


class ReplayStatus(str, Enum):
    CERTIFIED = "certified"
    CONFLICT = "conflict"
    BUDGET = "budget"
    INCONCLUSIVE = "inconclusive"
    RESOURCE_LIMIT = "resource_limit"



@dataclass(frozen=True)
class ReplayLimits:
    max_tactic_steps: int = 256
    max_combinations: int = 32
    max_continuations: int = 4
    certify_heartbeats: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("max_tactic_steps", "max_combinations", "max_continuations"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "max_continuations" else 1):
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class ReplayConflict:
    """Why one attempt failed inside one combination.

    ``kind`` is ``rejected`` (Lean refused the step), ``shape`` (the step's
    children do not match the recorded slots), ``binding`` (an argument has no
    unambiguous declaration), ``missing`` (the obligation is neither open nor
    discharged), ``incomplete`` or ``kernel_rejected`` (the finished replay did
    not certify).
    """

    node: int
    attempt: Optional[int]
    kind: str
    detail: str


@dataclass(frozen=True)
class WitnessStep:
    """One replayed step of a certified combination.

    ``observation_matched`` says whether the obligation looked to the model
    exactly as when the attempt was sampled, with the same rendered command;
    only such steps can give the original sample credit.
    """

    node: int
    attempt: int
    obligation: ObligationKey
    command: str
    observation_matched: bool
    step: ArtifactStep


@dataclass(eq=False)
class _Pending:
    key: ObligationKey
    node: int
    goal_name: str
    declarations: dict[str, DeclarationKey]
    goal: Goal


@dataclass(eq=False)
class ReplayContinuation:
    """A retained replay state whose next obligation needs a fresh search.

    ``occurrence`` is that obligation in the retained state, with its current
    observation; ``node`` is the graph node it stood in for. Replay resumes
    with ``rest`` after the obligation is solved there. ``deferred`` retains
    assigned obligations whose dependencies must finish before certification.
    """

    occurrence: ExecutionOccurrence
    node: int
    rest: tuple[_Pending, ...]
    witness: tuple[WitnessStep, ...]
    prefix: tuple[tuple[int, int], ...]
    anchor: int
    deferred: tuple[_Pending, ...] = ()


@dataclass
class ReplayResult:
    status: ReplayStatus
    certificate: Optional[CertificationResult] = None
    witness: tuple[WitnessStep, ...] = ()
    conflicts: list[ReplayConflict] = field(default_factory=list)
    conflict_count: int = 0
    steps: int = 0
    combinations: int = 0
    continuations: list[ReplayContinuation] = field(default_factory=list)

    @property
    def is_certified(self) -> bool:
        return self.status is ReplayStatus.CERTIFIED


class _Stop(Exception):
    def __init__(self, status: ReplayStatus):
        self.status = status


class ReplayCoordinator:
    """Runs bounded replays for one theorem search over one registry.

    The step and combination budgets are shared by all of the search's replays.
    """

    MEMO_LIMIT = 4096
    CONFLICT_LIMIT = 256

    def __init__(
        self,
        registry: SnapshotRegistry,
        *,
        limits: ReplayLimits | None = None,
        executor: RetainedTacticExecutor | None = None,
    ) -> None:
        self.registry = registry
        self.limits = limits or ReplayLimits()
        self.executor = executor or RetainedTacticExecutor()
        self._failed: set[tuple] = set()
        self._anchors = 0
        self.total_steps = 0
        self.total_combinations = 0

    async def combine(
        self,
        source: RecipeSource,
        root: ExecutionOccurrence,
        root_node: int,
        *,
        deadline: float | None = None,
    ) -> ReplayResult:
        """Replay the root's candidate recipes from the admitted root snapshot."""
        head = _Pending(root.key, root_node, root.goal_name, dict(root.declarations), root.goal)
        return await self._run(source, root.lease, (head,), (), (), 0, deadline)

    async def resume(
        self,
        source: RecipeSource,
        continuation: ReplayContinuation,
        node: int,
        *,
        deadline: float | None = None,
    ) -> ReplayResult:
        """Continue a replay once ``node``, the continuation's fresh search, is ready."""
        occurrence = continuation.occurrence
        head = _Pending(
            occurrence.key, node, occurrence.goal_name,
            dict(occurrence.declarations), occurrence.goal,
        )
        return await self._run(
            source, occurrence.lease, (head, *continuation.rest),
            continuation.witness, continuation.prefix, continuation.anchor, deadline,
            deferred=continuation.deferred,
        )

    async def _run(self, source, lease, pending, witness, prefix, anchor, deadline, *, deferred=()):
        result = ReplayResult(ReplayStatus.CONFLICT)
        self._source = source
        self._deadline = deadline
        self._result = result
        try:
            certified = await self._search(
                lease, tuple(pending), tuple(witness), tuple(prefix), anchor,
                deferred=deferred,
            )
            if certified is not None:
                result.status = ReplayStatus.CERTIFIED
                result.certificate, result.witness = certified
        except (_Stop, RetentionExceeded) as stop:
            result.status = (
                stop.status if isinstance(stop, _Stop) else ReplayStatus.RESOURCE_LIMIT
            )
            for continuation in result.continuations:
                await self.registry.release(continuation.occurrence.lease)
            result.continuations.clear()
        if result.is_certified:
            for continuation in result.continuations:
                await self.registry.release(continuation.occurrence.lease)
            result.continuations.clear()
        self.total_steps += result.steps
        self.total_combinations += result.combinations
        return result

    # The backtracking search

    def _check_budget(self) -> None:
        # The budgets cover every replay of one theorem search, not one call.
        if self.total_steps + self._result.steps >= self.limits.max_tactic_steps:
            raise _Stop(ReplayStatus.BUDGET)
        if self.total_combinations + self._result.combinations >= self.limits.max_combinations:
            raise _Stop(ReplayStatus.BUDGET)
        if self._deadline is not None and asyncio.get_running_loop().time() >= self._deadline:
            raise _Stop(ReplayStatus.BUDGET)

    def _conflict(self, node, attempt, kind, detail) -> None:
        self._result.conflict_count += 1
        if len(self._result.conflicts) < self.CONFLICT_LIMIT:
            self._result.conflicts.append(ReplayConflict(node, attempt, kind, detail))

    def _remember(self, key: tuple) -> None:
        if len(self._failed) < self.MEMO_LIMIT:
            self._failed.add(key)

    async def _search(
        self, lease: Lease, pending, witness, prefix, anchor,
        *, own: bool = False, deferred=(),
    ):
        """Replay ``pending`` from the state ``lease`` holds.

        ``own`` says the caller will not need ``lease`` again, so it can be
        released as soon as no alternative here can backtrack to it; the
        admitted root and continuation anchors are never owned.
        """
        if not pending:
            # Assigned siblings may only become complete after later steps.
            # Inspect them once, after progress, rather than rotating pending work.
            state = self.registry.state(lease)
            if deferred:
                statuses = await self.registry.scope.inspect(
                    state, [item.goal_name for item in deferred]
                )
                for item, status in zip(deferred, statuses):
                    if not status.is_complete:
                        self._conflict(
                            item.node, None, "incomplete",
                            f"{item.goal_name} still depends on {status.holes}",
                        )
                        self._result.combinations += 1
                        return None
            return await self._certify(lease, witness)
        head, rest = pending[0], pending[1:]
        state = self.registry.state(lease)
        listed = {goal.goal_name for goal in state.goals}
        if head.goal_name not in listed:
            (status,) = await self.registry.scope.inspect(state, [head.goal_name])
            if status.is_complete:
                # A shared assignment already discharged it; nothing to replay.
                return await self._search(
                    lease, rest, witness, prefix, anchor, own=own, deferred=deferred,
                )
            known = {item.goal_name for item in (*rest, *deferred)}
            if (
                status.status in {"assigned", "delayed"}
                and status.holes
                and set(status.holes) <= known
            ):
                return await self._search(
                    lease, rest, witness, prefix, anchor,
                    own=own, deferred=(*deferred, head),
                )
            self._conflict(head.node, None, "missing", f"{head.goal_name} is {status.status}")
            self._result.combinations += 1
            return None
        explored = self._source.observation(head.node)
        changed = explored.state_fingerprint() != head.goal.state_fingerprint()
        applied = False
        steps = [
            step
            for step in self._source.alternatives(head.node)
            if (anchor, (*prefix, (head.node, step.attempt))) not in self._failed
        ]
        for position, step in enumerate(steps):
            choice = (*prefix, (head.node, step.attempt))
            self._check_budget()
            certified, ran = await self._attempt(
                lease, head, rest, step, witness, choice, anchor, explored,
                release_input=own and position == len(steps) - 1, deferred=deferred,
            )
            if certified is not None:
                return certified
            applied = applied or ran
            self._remember((anchor, choice))
        # Only an obligation none of whose recorded attempts applies here, in a
        # context a shared assignment changed, is worth a fresh search.
        if (
            changed
            and not applied
            and len(self._result.continuations) < self.limits.max_continuations
        ):
            occurrence = ExecutionOccurrence(
                -1, head.key, head.goal_name, head.goal, dict(head.declarations),
                self.registry.share(lease, LeaseKind.REPLAY, "continuation"),
            )
            self._anchors += 1
            self._result.continuations.append(
                ReplayContinuation(occurrence, head.node, rest, witness, prefix, self._anchors, deferred)
            )
        self._result.combinations += 1
        return None

    async def _attempt(
        self, lease, head, rest, step: RecipeStep, witness, choice, anchor, explored,
        *, release_input: bool = False, deferred=(),
    ):
        try:
            arguments = rebind_arguments(
                step.arguments,
                self._source.declarations(head.node),
                head.goal,
                head.declarations,
            )
        except BindingConflict as exc:
            self._conflict(head.node, step.attempt, "binding", str(exc))
            return None, False
        command = render_tactic_command(candidate_for(step, arguments))
        if command is None:
            self._conflict(head.node, step.attempt, "binding", "the step no longer renders")
            return None, False
        occurrence = ExecutionOccurrence(
            -1, head.key, head.goal_name, head.goal, head.declarations, lease
        )
        self._result.steps += 1
        outcome = await self.executor.apply(self.registry, occurrence, command, holder="replay")
        if not outcome.success:
            self._conflict(head.node, step.attempt, "rejected", outcome.error or "rejected")
            return None, False
        try:
            shape = [(child.slot, child.kind, len(child.introduced)) for child in outcome.children]
            recorded = [(child.slot, child.kind, child.introduced) for child in step.children]
            if shape != recorded:
                self._conflict(
                    head.node, step.attempt, "shape",
                    f"replayed children {shape} differ from the recorded {recorded}",
                )
                return None, False
            if not outcome.children and outcome.selected_complete is False:
                # A conditional step closes its goal only through obligations
                # this replay still has to solve; certification decides.
                pending = {item.goal_name for item in (*rest, *deferred)}
                if not step.dependencies or not set(outcome.selected_holes) <= pending:
                    self._conflict(
                        head.node, step.attempt, "dependency",
                        f"holes {list(outcome.selected_holes)} are not pending obligations",
                    )
                    return None, False
                deferred = (*deferred, head)
            children = tuple(
                _Pending(
                    ObligationKey(step.attempt, child.slot),
                    record.node,
                    child.goal_name,
                    child_declarations(head.declarations, step.attempt, child),
                    child.goal,
                )
                for child, record in zip(outcome.children, step.children)
            )
            refreshed = tuple(
                replace(
                    item,
                    goal=carry_model_identity(item.goal, outcome.siblings[item.goal_name]),
                )
                if item.goal_name in outcome.siblings
                else item
                for item in rest
            )
            matched = (
                explored.state_fingerprint() == head.goal.state_fingerprint()
                and command == step.command
            )
            record = WitnessStep(
                head.node,
                step.attempt,
                head.key,
                command,
                matched,
                ArtifactStep(
                    head.key,
                    step.attempt,
                    step.tactic,
                    tuple(
                        ArtifactArgument(
                            ref.kind,
                            ref.text,
                            head.declarations.get(ref.internal_name) if ref.kind == "local" else None,
                        )
                        for ref in arguments
                    ),
                    tuple(recorded),
                ),
            )
            if release_input:
                # No alternative here is left to backtrack to; free the prefix.
                await self.registry.release(lease)
            certified = await self._search(
                outcome.lease, (*children, *refreshed), (*witness, record), choice, anchor,
                own=True, deferred=deferred,
            )
            return certified, True
        finally:
            await self.registry.release(outcome.lease)

    async def _certify(self, lease: Lease, witness):
        self._result.combinations += 1
        state = self.registry.state(lease)
        result = await self.registry.scope.certify(
            state, max_heartbeats=self.limits.certify_heartbeats
        )
        if result.is_certified:
            return result, witness
        if result.status == "inconclusive":
            raise _Stop(ReplayStatus.INCONCLUSIVE)
        kind = "kernel_rejected" if result.status == "rejected" else "incomplete"
        last = witness[-1] if witness else None
        self._conflict(last.node if last else -1, last.attempt if last else None, kind, result.reason or "")
        return None


class ArtifactMismatch(RuntimeError):
    """A replay artifact does not reproduce in this session."""


def build_artifact(
    statement: dict,
    result: ReplayResult,
    environment: dict,
) -> ReplayArtifact:
    """The portable record of a certified replay."""
    if not result.is_certified:
        raise ValueError("Only a certified replay has an artifact.")
    receipt = result.certificate.receipt
    return ReplayArtifact(
        statement=dict(statement),
        steps=tuple(step.step for step in result.witness),
        environment=dict(environment),
        receipt={
            "root_type": receipt.root_type,
            "proof": receipt.proof,
            "environment": receipt.environment,
            "lean": receipt.lean,
            "levels": list(receipt.levels),
            "axioms": list(receipt.axioms),
            "auxiliary": list(receipt.auxiliary),
            "identity": receipt.identity,
        },
    )


async def reverify(scope, artifact, *, max_heartbeats: Optional[int] = None) -> CertificationResult:
    """Replay ``artifact`` in ``scope`` from its statement and certify it again.

    Steps are applied in order to the obligations their keys name, arguments
    are re-pointed through their declaration keys, and the new receipt must
    prove the same statement with the same proof term against the same
    trusted environment. A step that does not apply, a slot shape that
    differs or a receipt that differs raises ``ArtifactMismatch``.
    """
    statement = artifact.statement
    spec = GoalReplaySpec(
        expression=statement["expression"],
        local_names=tuple(statement["local_names"]),
        universe_levels=tuple(statement["universe_levels"]),
    )
    registry = SnapshotRegistry(scope)
    try:
        state = await materialize_replay(scope, spec)
        lease = registry.adopt_root(state, "artifact")
        (root,) = state.goals
        goal = execution_goal_to_goal(root)
        open_obligations = {
            ObligationKey.root(): (root.goal_name, root_declarations(goal), goal)
        }
        executor = RetainedTacticExecutor()
        for step in artifact.steps:
            if step.obligation not in open_obligations:
                raise ArtifactMismatch(f"Step {step.attempt} names an unknown obligation.")
            name, declarations, current = open_obligations.pop(step.obligation)
            by_key = {key: internal for internal, key in declarations.items()}
            texts = []
            for argument in step.arguments:
                if argument.kind != "local":
                    texts.append(argument.text)
                    continue
                internal = by_key.get(argument.declaration)
                local = next(
                    (item for item in current.locals or [] if item.internal_name == internal),
                    None,
                )
                if local is None:
                    raise ArtifactMismatch(f"Step {step.attempt} names a missing declaration.")
                texts.append(local.execution_or_live_name)
            command = render_tactic_command(
                TacticCandidate(tactic_name=step.tactic, arguments=texts, probability=1.0)
            )
            occurrence = ExecutionOccurrence(-1, step.obligation, name, current, declarations, lease)
            try:
                outcome = await executor.apply(
                    registry, occurrence, command or step.tactic, holder="artifact"
                )
            except ExecutionReferenceError as exc:
                raise ArtifactMismatch(f"Step {step.attempt}: {exc}") from exc
            if not outcome.success:
                raise ArtifactMismatch(f"Step {step.attempt} was rejected: {outcome.error}")
            shape = tuple((child.slot, child.kind, len(child.introduced)) for child in outcome.children)
            if shape != step.children:
                await registry.release(outcome.lease)
                raise ArtifactMismatch(f"Step {step.attempt} produced {shape}, not {step.children}.")
            for child in outcome.children:
                open_obligations[ObligationKey(step.attempt, child.slot)] = (
                    child.goal_name,
                    child_declarations(declarations, step.attempt, child),
                    child.goal,
                )
            for key, (other, keys, observed) in list(open_obligations.items()):
                if other in outcome.siblings:
                    open_obligations[key] = (other, keys, carry_model_identity(observed, outcome.siblings[other]))
            await registry.release(lease)
            lease = outcome.lease
        result = await scope.certify(registry.state(lease), max_heartbeats=max_heartbeats)
        if not result.is_certified:
            raise ArtifactMismatch(f"Replay did not certify: {result.reason}")
        recorded = artifact.receipt
        if (result.receipt.root_type, result.receipt.proof) != (recorded["root_type"], recorded["proof"]):
            raise ArtifactMismatch("The new certificate proves a different statement or term.")
        if result.receipt.environment != recorded["environment"]:
            raise ArtifactMismatch("The proof was checked against a different trusted environment.")
        return result
    finally:
        await registry.close()
