"""Snapshot leases, argument bindings and the retained executor (no Lean process)."""

from __future__ import annotations

import asyncio
import unittest

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.hybrid_reasoner.execution_registry import (
    LeaseKind,
    RetentionExceeded,
    SnapshotRegistry,
)
from maths_ai.hybrid_reasoner.execution_types import (
    ChildObligation,
    DeclarationKey,
    ExecutionOccurrence,
    ObligationKey,
    bind_arguments,
    child_declarations,
    root_declarations,
)
from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_goal
from maths_ai.hybrid_reasoner.retained_executor import (
    ExecutionReferenceError,
    RetainedTacticExecutor,
)
from maths_ai.gnn_inference.tests.snapshot_fake import FakeSnapshotScope, Spec, Step, table
from pantograph_async import PantographStateError

ROOT = Spec("p ∧ q", (("p", "Prop"), ("q", "Prop"), ("hp", "p"), ("hq", "q")))
RULES = table(
    constructor=Step(children=(Spec("p"), Spec("q"))),
    exact_hp=Step(),
    intro_h=Step(children=(Spec("r", (("h", "s"),)),)),
    trivial=Step(),
)


async def admitted(scope, registry):
    start = await scope.start_goal("root")
    state = await scope.apply_tactic(start, "intro p q hp hq")
    await scope.release(start)
    lease = registry.adopt_root(state, "root")
    goal = execution_goal_to_goal(state.goals[0])
    return ExecutionOccurrence(
        0, ObligationKey.root(), state.goals[0].goal_name, goal,
        root_declarations(goal), lease,
    )


def run(body):
    return asyncio.run(body())


class RegistryTests(unittest.TestCase):
    def test_sibling_leases_share_one_allocation(self):
        async def body():
            scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
            registry = SnapshotRegistry(scope)
            root = await admitted(scope, registry)
            outcome = await RetainedTacticExecutor().apply(registry, root, "constructor")
            left = registry.share(outcome.lease, LeaseKind.GRAPH, "left")
            right = registry.share(outcome.lease, LeaseKind.GRAPH, "right")
            await registry.release(outcome.lease)
            self.assertEqual(registry.live_snapshots, 2)
            await registry.release(left)
            self.assertEqual(scope.live, 2)
            # The remaining sibling still holds the snapshot.
            self.assertIn(registry.state(right).handle.state_id, scope.snapshots)
            await registry.release(right)
            await registry.release(right)
            self.assertEqual(registry.live_snapshots, 1)
            self.assertEqual(scope.live, 1)

        run(body)

    def test_released_lease_cannot_be_used(self):
        async def body():
            scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
            registry = SnapshotRegistry(scope)
            root = await admitted(scope, registry)
            await registry.release(root.lease)
            with self.assertRaises(PantographStateError):
                registry.state(root.lease)
            with self.assertRaises(ExecutionReferenceError):
                await RetainedTacticExecutor().apply(registry, root, "trivial")

        run(body)

    def test_bound_counts_allocations_and_reserve_checks_first(self):
        async def body():
            scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
            registry = SnapshotRegistry(scope, max_snapshots=2)
            root = await admitted(scope, registry)
            extra = registry.share(root.lease, LeaseKind.REPLAY, "anchor")
            reservation = registry.reserve()
            self.assertFalse(registry.has_capacity())
            self.assertEqual(registry.reserved_snapshots, 1)
            reservation.cancel()
            self.assertTrue(registry.has_capacity())
            outcome = await RetainedTacticExecutor().apply(registry, root, "constructor")
            self.assertEqual(registry.live_snapshots, 2)
            self.assertEqual(registry.live_leases, 3)
            with self.assertRaises(RetentionExceeded):
                registry.reserve()
            await registry.release(outcome.lease)
            reservation = registry.reserve()
            reservation.cancel()
            await registry.release(extra)

        run(body)

    def test_close_releases_everything_and_invalidation_forgets(self):
        async def body():
            scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
            registry = SnapshotRegistry(scope)
            root = await admitted(scope, registry)
            outcome = await RetainedTacticExecutor().apply(registry, root, "constructor")
            await registry.close()
            self.assertEqual(scope.live, 0)
            self.assertTrue(root.lease.released and outcome.lease.released)
            other = SnapshotRegistry(scope)
            root = await admitted(scope, other)
            other.invalidate()
            self.assertTrue(root.lease.released)
            with self.assertRaises(PantographStateError):
                other.adopt_root(scope.issued[max(scope.issued)], "late")

        run(body)


class ExecutorTests(unittest.TestCase):
    def test_alternatives_reuse_the_retained_input_without_reconstruction(self):
        async def body():
            scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
            registry = SnapshotRegistry(scope)
            root = await admitted(scope, registry)
            starts = scope.count("goal.start")
            executor = RetainedTacticExecutor()
            first = await executor.apply(registry, root, "constructor")
            rejected = await executor.apply(registry, root, "nonsense")
            second = await executor.apply(registry, root, "exact hp")
            self.assertTrue(first.success and second.success)
            self.assertFalse(rejected.success)
            self.assertIsNone(rejected.lease)
            self.assertEqual(scope.count("goal.start"), starts)
            self.assertTrue(root.is_retained)
            self.assertEqual([child.kind for child in first.children], ["direct", "direct"])
            self.assertEqual([child.goal.expression for child in first.children], ["p", "q"])
            self.assertEqual(second.children, [])
            self.assertTrue(second.selected_complete)
            for outcome in (first, second):
                await registry.release(outcome.lease)
            self.assertEqual(registry.live_snapshots, 1)

        run(body)

    def test_the_selected_goal_is_found_by_name_not_position(self):
        async def body():
            scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
            registry = SnapshotRegistry(scope)
            root = await admitted(scope, registry)
            split = await RetainedTacticExecutor().apply(registry, root, "constructor")
            right = split.children[1]
            occurrence = ExecutionOccurrence(
                1, ObligationKey(0, 1), right.goal_name, right.goal,
                child_declarations(root.declarations, 0, right),
                registry.share(split.lease, LeaseKind.GRAPH, "right"),
            )
            closed = await RetainedTacticExecutor().apply(registry, occurrence, "trivial")
            state = registry.state(closed.lease)
            self.assertEqual(state.transition.selected, right.goal_name)
            self.assertEqual(
                [goal.goal_name for goal in state.goals], [split.children[0].goal_name]
            )
            self.assertEqual(list(closed.siblings), [split.children[0].goal_name])

        run(body)

    def test_a_decoding_failure_releases_the_result(self):
        async def body():
            scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
            registry = SnapshotRegistry(scope)
            root = await admitted(scope, registry)
            executor = RetainedTacticExecutor()

            async def broken(*_args):
                raise ValueError("decoding failed")

            executor._decode = broken
            with self.assertRaises(ValueError):
                await executor.apply(registry, root, "constructor")
            self.assertEqual(registry.live_snapshots, 1)
            self.assertEqual(scope.live, 1)

        run(body)

    def test_incomplete_assignment_is_reported(self):
        async def body():
            rules = table(close_partly=Step(leave_holes=True))
            scope = FakeSnapshotScope(rules, roots={"root": ROOT})
            registry = SnapshotRegistry(scope)
            root = await admitted(scope, registry)
            outcome = await RetainedTacticExecutor().apply(registry, root, "close partly")
            self.assertTrue(outcome.success)
            self.assertFalse(outcome.selected_complete)

        run(body)


def _local(name: str, internal: str, index: int) -> GoalLocal:
    return GoalLocal(
        user_name=name, internal_name=internal, context_index=index,
        binder_role=":explicit", type_pp="T", type_model_sexp="(:c T)",
    )


class BindingTests(unittest.TestCase):
    def goal(self, *locals_):
        return Goal(
            expression="P",
            goal_model_sexp="(:c P)",
            locals=[_local(name, internal, i) for i, (name, internal) in enumerate(locals_)],
            model_sexp_version=1,
        )

    def test_a_name_binds_to_the_latest_local_that_carries_it(self):
        goal = self.goal(("h", "_a"), ("h", "_b"), ("x", "_c"))
        refs = bind_arguments(goal, ["h", "x", "Nat.add_comm", "(1 : Nat)", "h✝"])
        self.assertEqual([ref.kind for ref in refs], ["local", "local", "global", "literal", "literal"])
        self.assertEqual(refs[0].internal_name, "_b")
        self.assertEqual(refs[1].internal_name, "_c")

    def test_declaration_keys_follow_introductions(self):
        goal = self.goal(("p", "_p"), ("hp", "_hp"))
        roots = root_declarations(goal)
        self.assertEqual(roots, {"_p": DeclarationKey.root(0), "_hp": DeclarationKey.root(1)})
        child_goal = self.goal(("p", "_p"), ("h", "_h"))
        child = ChildObligation(1, "direct", "g", child_goal, ("_h",))
        keys = child_declarations(roots, 7, child)
        self.assertEqual(keys, {"_p": DeclarationKey.root(0), "_h": DeclarationKey(7, 1, 0)})


if __name__ == "__main__":
    unittest.main()


def test_overlapping_requests_cannot_claim_the_same_result_slot():
    async def body():
        scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
        registry = SnapshotRegistry(scope, max_snapshots=2)
        root = await admitted(scope, registry)
        entered, proceed = asyncio.Event(), asyncio.Event()
        original = scope.apply_tactic

        async def blocked(*args, **kwargs):
            entered.set()
            await proceed.wait()
            return await original(*args, **kwargs)

        scope.apply_tactic = blocked
        executor = RetainedTacticExecutor()
        first = asyncio.create_task(executor.apply(registry, root, "trivial"))
        await entered.wait()
        try:
            assert registry.reserved_snapshots == 1
            assert registry.is_pinned(root.lease)
            try:
                await executor.apply(registry, root, "trivial")
            except RetentionExceeded:
                pass
            else:
                raise AssertionError("The second request reused reserved capacity")
        finally:
            proceed.set()
            outcome = await first
        assert registry.reserved_snapshots == 0
        assert registry.live_snapshots == registry.peak_snapshots == 2
        await registry.release(outcome.lease)
        await registry.close()
        assert scope.live == 0

    run(body)


def test_request_reservations_and_pins_are_refunded_on_cancel_error_and_rejection():
    async def body():
        scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
        registry = SnapshotRegistry(scope, max_snapshots=2)
        root = await admitted(scope, registry)
        executor = RetainedTacticExecutor()
        rejected = await executor.apply(registry, root, "nonsense")
        assert not rejected.success
        assert registry.reserved_snapshots == 0
        assert registry.live_leases == 1
        original = scope.apply_tactic
        entered = asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        scope.apply_tactic = blocked
        task = asyncio.create_task(executor.apply(registry, root, "trivial"))
        await entered.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert registry.reserved_snapshots == 0
        assert registry.live_leases == 1

        async def broken(*args, **kwargs):
            raise RuntimeError("transport failed")

        scope.apply_tactic = broken
        try:
            await executor.apply(registry, root, "trivial")
        except RuntimeError:
            pass
        else:
            raise AssertionError("The transport error was swallowed")
        assert registry.reserved_snapshots == 0
        assert registry.live_leases == 1
        scope.apply_tactic = original
        outcome = await executor.apply(registry, root, "trivial")
        await registry.release(outcome.lease)
        await registry.close()
        assert scope.live == 0

    run(body)


def test_a_snapshot_being_deleted_still_occupies_capacity():
    async def body():
        scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
        registry = SnapshotRegistry(scope, max_snapshots=2)
        root = await admitted(scope, registry)
        outcome = await RetainedTacticExecutor().apply(registry, root, "trivial")
        original = scope.release
        entered, proceed = asyncio.Event(), asyncio.Event()

        async def blocked(state):
            entered.set()
            await proceed.wait()
            await original(state)

        scope.release = blocked
        deletion = asyncio.create_task(registry.release(outcome.lease))
        await entered.wait()
        try:
            assert not registry.has_capacity()
            assert registry.live_snapshots == 2
        finally:
            proceed.set()
            await deletion
        assert registry.has_capacity()
        await registry.close()

    run(body)


def test_request_pins_a_graph_only_input_before_the_pruning_hook():
    async def body():
        scope = FakeSnapshotScope(RULES, roots={"root": ROOT})
        registry = SnapshotRegistry(scope, max_snapshots=2)
        root = await admitted(scope, registry)
        graph = registry.share(root.lease, LeaseKind.GRAPH, "frontier")
        await registry.release(root.lease)
        root.lease = graph
        assert not registry.is_pinned(graph)

        async def make_room(count):
            assert registry.is_pinned(graph)
            assert registry.live_leases == 2

        registry.make_room = make_room
        outcome = await RetainedTacticExecutor().apply(registry, root, "trivial")
        assert not registry.is_pinned(graph)
        assert registry.live_leases == 2
        await registry.release(outcome.lease)
        await registry.close()
        assert scope.live == 0

    run(body)
