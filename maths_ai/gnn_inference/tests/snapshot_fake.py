"""A scripted stand-in for a retained-snapshot Lean session.

It returns the client's real types (``TacticState`` with transition provenance,
``ObligationInspection``, ``CertificationResult``), so the registry, executor,
replay and search code run unchanged against it. Tactic behavior comes from a
``rules`` function; the actual Lean semantics of shared assignments are left to
the live tests.

A rule receives the selected goal's spec and the command and returns ``None``
(reject), or a ``Step``: the child specs the tactic produces, siblings it
closes, and siblings whose target it changes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Callable, Optional

from pantograph_async import (
    CertificateReceipt,
    CertificationResult,
    ExecutionGoal,
    ExecutionLocal,
    ObligationInspection,
    PantographStateError,
    PantographTacticRejected,
    StartedGoal,
    StateHandle,
    TacticState,
    TransitionProvenance,
    TransitionValidation,
    WireExpression,
)


@dataclass(frozen=True)
class Spec:
    """A goal: its target text and ``(name, type)`` locals added on top of its parent's."""

    target: str
    locals: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Step:
    children: tuple[Spec, ...] = ()
    closes: tuple[int, ...] = ()  # sibling positions (in the input's goal order)
    changes: tuple[tuple[int, str], ...] = ()  # (sibling position, new target)
    leave_holes: bool = False  # the selected goal's assignment is not complete
    # Targets of sibling goals the selected goal's assignment still refers to
    # while they are open; it is complete once they are assigned.
    depends: tuple[str, ...] = ()
    # Assigned sibling position -> target names of its remaining dependencies.
    sibling_depends: tuple[tuple[int, tuple[str, ...]], ...] = ()


Rule = Callable[[Spec, str], Optional[Step]]


def _expr(text: str) -> WireExpression:
    model = "(:c " + "".join(ch if ch.isalnum() else "_" for ch in text) + ")"
    return WireExpression(pp=text, sexp=model, model_sexp=model, model_sexp_version=1)


@dataclass
class _Goal:
    name: str
    spec: Spec
    locals: tuple[tuple[str, str, str], ...]  # (internal name, user name, type)
    target: str

    def wire(self) -> ExecutionGoal:
        return ExecutionGoal(
            goal_name=self.name,
            case_tag=None,
            is_conversion=False,
            fragment="tactic",
            target=_expr(self.target),
            locals=tuple(
                ExecutionLocal(
                    internal_name=internal,
                    user_name=user,
                    context_index=index,
                    binder_role=":explicit",
                    is_instance=False,
                    is_let=False,
                    is_inaccessible=False,
                    type=_expr(type_),
                )
                for index, (internal, user, type_) in enumerate(self.locals)
            ),
        )


@dataclass
class _Snapshot:
    root: str
    goals: list[_Goal]
    assigned: frozenset[str]
    # Selected goal -> the obligations its assignment still refers to.
    incomplete: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def unresolved(self, name: str) -> tuple[str, ...]:
        for selected, holes in self.incomplete:
            if selected == name:
                return tuple(hole for hole in holes if hole not in self.assigned)
        return ()


class FakeSnapshotScope:
    """``PantographStateScope``'s public surface over scripted snapshots."""

    is_usable = True

    def __init__(self, rules: Rule, *, roots: Optional[dict[str, Spec]] = None):
        self.rules = rules
        self.roots = roots or {}
        self.snapshots: dict[int, _Snapshot] = {}
        self.issued: dict[int, object] = {}
        self.calls: list[tuple] = []
        self._ids = itertools.count()
        self._names = itertools.count(1)
        self.deleted: list[int] = []

    # Context manager surface used by reasoners
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        self.snapshots.clear()
        self.issued.clear()

    def _name(self, prefix: str) -> str:
        return f"_{prefix}.{next(self._names)}"

    def _issue(self, snapshot: _Snapshot, transition=None):
        state_id = next(self._ids)
        self.snapshots[state_id] = snapshot
        handle = StateHandle("fake", "scope", state_id, state_id + 1)
        if transition is None:
            state = StartedGoal(handle, snapshot.root)
        else:
            state = TacticState(
                handle,
                tuple(goal.wire() for goal in snapshot.goals),
                TransitionValidation(1, "transition-expressions", True, False, False),
                (),
                transition,
            )
        self.issued[state_id] = state
        return state

    def _snapshot(self, state) -> _Snapshot:
        snapshot = self.snapshots.get(state.handle.state_id)
        if snapshot is None or self.issued.get(state.handle.state_id) is not state:
            raise PantographStateError("Fake state was released or not issued.")
        return snapshot

    # Public scope operations

    async def start_goal(self, expression: str, *, levels=()):
        self.calls.append(("goal.start", expression))
        root = self._name("root")
        spec = self.roots.get(expression, Spec(expression))
        goal = _Goal(root, spec, (), spec.target)
        return self._issue(_Snapshot(root, [goal], frozenset()))

    async def apply_tactic(self, state, tactic: str, *, goal_index: int = 0):
        self.calls.append(("goal.tactic", tactic))
        snapshot = self._snapshot(state)
        if isinstance(state, StartedGoal):
            selected = snapshot.goals[0]
            if tactic.startswith("intro") or tactic == "skip":
                # Materialization: the root's own locals come from the spec.
                names = tactic.split()[1:] if tactic.startswith("intro") else []
                locals_ = tuple(
                    (self._name("fv"), user, type_) for user, type_ in selected.spec.locals
                )
                if names and [user for _, user, _ in locals_] != names:
                    raise PantographTacticRejected("intro names do not match the spec")
                child = _Goal(self._name("g"), selected.spec, locals_, selected.spec.target)
                transition = TransitionProvenance(
                    selected.name, (child.name,), (tuple(n for n, _, _ in locals_),),
                    (), (), (), (), (),
                )
                return self._issue(
                    _Snapshot(snapshot.root, [child], snapshot.assigned), transition
                )
        selected = snapshot.goals[goal_index]
        step = self.rules(selected.spec, tactic)
        if step is None:
            raise PantographTacticRejected(f"fake rejected {tactic!r}")
        siblings = [goal for goal in snapshot.goals if goal is not selected]
        closed = [siblings[i] for i in step.closes]
        changed = {siblings[i].name: target for i, target in step.changes}
        children = []
        introduced = []
        for spec in step.children:
            new_locals = tuple((self._name("fv"), user, type_) for user, type_ in spec.locals)
            children.append(_Goal(self._name("g"), spec, selected.locals + new_locals, spec.target))
            introduced.append(tuple(name for name, _, _ in new_locals))
        remaining = [
            replace(goal, target=changed.get(goal.name, goal.target))
            for goal in siblings
            if goal not in closed
        ]
        assigned = snapshot.assigned | {selected.name} | {goal.name for goal in closed}
        holes = (("_hole",) if step.leave_holes else ()) + tuple(
            goal.name for goal in siblings if goal.spec.target in step.depends
        )
        sibling_holes = tuple(
            (siblings[position].name, tuple(
                goal.name for goal in siblings if goal.spec.target in targets
            ))
            for position, targets in step.sibling_depends
        )
        incomplete = snapshot.incomplete + (((selected.name, holes),) if holes else ()) + sibling_holes
        transition = TransitionProvenance(
            selected.name,
            tuple(goal.name for goal in children),
            tuple(introduced),
            tuple(goal.name for goal in remaining),
            tuple(name for name in changed if name in {g.name for g in remaining}),
            tuple(goal.name for goal in closed),
            (),
            (),
        )
        return self._issue(
            _Snapshot(snapshot.root, children + remaining, assigned, incomplete), transition
        )

    async def release(self, state) -> None:
        self.calls.append(("goal.delete", state.handle.state_id))
        if self.snapshots.pop(state.handle.state_id, None) is not None:
            self.deleted.append(state.handle.state_id)
        self.issued.pop(state.handle.state_id, None)

    async def inspect(self, state, goals):
        self.calls.append(("goal.inspect", tuple(goals)))
        snapshot = self._snapshot(state)
        listed = {goal.name: goal for goal in snapshot.goals}
        result = []
        for name in goals:
            if name in listed:
                result.append(ObligationInspection(name, "active", listed[name].wire(), (), False))
            elif snapshot.unresolved(name):
                result.append(
                    ObligationInspection(name, "assigned", None, snapshot.unresolved(name), False)
                )
            else:
                result.append(ObligationInspection(name, "complete", None, (), False))
        return tuple(result)

    async def certify(self, state, *, max_heartbeats=None):
        self.calls.append(("goal.certify", state.handle.state_id))
        snapshot = self._snapshot(state)
        if snapshot.goals:
            return CertificationResult("incomplete", f"{len(snapshot.goals)} goals remain", None)
        if any(snapshot.unresolved(name) for name, _ in snapshot.incomplete):
            return CertificationResult("incomplete", "The proof has unresolved holes", None)
        receipt = CertificateReceipt(
            snapshot.root, (), "0" * 16, f"{len(snapshot.assigned):016x}", "1" * 16,
            "fake", (), (),
        )
        return CertificationResult("certified", None, receipt)

    @property
    def live(self) -> int:
        return len(self.snapshots)

    def count(self, command: str) -> int:
        return sum(1 for call in self.calls if call[0] == command)


def table(**commands: Step) -> Rule:
    """A rule that maps a command (with ``_`` for spaces) to its step on any goal."""
    steps = {name.replace("_", " "): step for name, step in commands.items()}
    return lambda _spec, command: steps.get(command)
