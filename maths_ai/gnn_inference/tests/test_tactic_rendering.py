from __future__ import annotations

import asyncio
import unittest

from pantograph_async import (
    PantographGoalRejected,
    PantographTacticRejected,
    PantographTransportError,
    PantographProtocolError,
    PantographStateError,
)
from maths_ai.gnn_inference.tests.model_helpers import (
    FakeStateScope,
    ScopedFakeClient,
    ScriptedExecutor,
    structured_goal,
)
from maths_ai.hybrid_reasoner.execution_registry import LeaseKind, SnapshotRegistry
from maths_ai.hybrid_reasoner.execution_types import ExecutionOccurrence, ObligationKey

from maths_ai.data_models.proof_components import (
    Goal,
    GoalLocal,
    TacticCandidate,
)
from maths_ai.hybrid_reasoner.joint_inference import (
    HybridReasoner,
    PantographExecutor,
    render_tactic_command,
)
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import materialize_replay, materialize_seed
from maths_ai.lean_replay import ReplayInputError, render_replay, seed_from_text


def _tactic(name: str, *arguments: str) -> TacticCandidate:
    return TacticCandidate(tactic_name=name, arguments=list(arguments), probability=0.5)


class RenderTacticCommandTests(unittest.TestCase):
    def test_bracket_required_tactic_gets_brackets(self):
        self.assertEqual(render_tactic_command(_tactic("rw", "h")), "rw [h]")

    def test_multiple_rules_are_comma_separated(self):
        self.assertEqual(
            render_tactic_command(_tactic("rw", "h", "Nat.add_comm")),
            "rw [h, Nat.add_comm]",
        )

    def test_bracket_required_tactic_without_usable_names_is_unplayable(self):
        """`↑13` and `?m.2235` are expressions, not rules Lean can look up."""
        self.assertIsNone(render_tactic_command(_tactic("rw", "↑13")))
        self.assertIsNone(render_tactic_command(_tactic("rw", "?m.2235")))
        self.assertIsNone(render_tactic_command(_tactic("rw")))

    def test_unusable_arguments_are_filtered_from_the_rule_list(self):
        self.assertEqual(render_tactic_command(_tactic("rw", "↑13", "h")), "rw [h]")

    def test_bracket_optional_tactic_runs_bare_without_rules(self):
        self.assertEqual(render_tactic_command(_tactic("simp")), "simp")
        self.assertEqual(render_tactic_command(_tactic("simp", "↑13")), "simp")

    def test_bracket_optional_tactic_takes_brackets_with_rules(self):
        self.assertEqual(render_tactic_command(_tactic("simp", "h")), "simp [h]")
        self.assertEqual(
            render_tactic_command(_tactic("linarith", "h1", "h2")), "linarith [h1, h2]"
        )

    def test_ordinary_tactic_keeps_space_separated_arguments(self):
        self.assertEqual(render_tactic_command(_tactic("apply", "h")), "apply h")
        self.assertEqual(render_tactic_command(_tactic("intro", "p", "q")), "intro p q")
        self.assertEqual(
            render_tactic_command(_tactic("exact", "Nat.le_refl")), "exact Nat.le_refl"
        )

    def test_bare_tactic_renders_as_its_name(self):
        self.assertEqual(render_tactic_command(_tactic("rfl")), "rfl")

    def test_trailing_colons_and_blanks_are_stripped(self):
        """Arguments arrive rendered from DAG nodes, which can carry a `name :` form."""
        self.assertEqual(
            render_tactic_command(_tactic("apply", "h :", "  ")), "apply h"
        )

    def test_names_with_lean_punctuation_are_usable_rules(self):
        self.assertEqual(render_tactic_command(_tactic("rw", "h'")), "rw [h']")
        self.assertEqual(render_tactic_command(_tactic("rw", "α")), "rw [α]")
        self.assertEqual(
            render_tactic_command(_tactic("rw", "Nat.factorial_succ")),
            "rw [Nat.factorial_succ]",
        )


class _RecordingScope:
    """Captures the command and goal index the executor sends to Lean."""

    is_usable = True

    def __init__(self):
        self.commands: list[tuple[str, int]] = []

    async def apply_tactic(self, state, tactic, *, goal_index=0):
        self.commands.append((tactic, goal_index))
        raise PantographTacticRejected("tactic failed")  # the return path is not checked here

    async def release(self, state):
        pass


def _occurrence(registry, names):
    state = type("State", (), {"goals": [type("G", (), {"goal_name": name})() for name in names]})()
    lease = registry.adopt_root(state, "root")
    goal = structured_goal("p")
    return ExecutionOccurrence(0, ObligationKey.root(), names[-1], goal, {}, lease)


class ExecutorRenderingTests(unittest.TestCase):
    def test_the_rendered_command_runs_on_the_occurrence_goal(self):
        scope = _RecordingScope()
        registry = SnapshotRegistry(scope)
        occurrence = _occurrence(registry, ["other", "selected"])
        command = render_tactic_command(_tactic("rw", "h"))
        outcome = asyncio.run(PantographExecutor(object()).apply(registry, occurrence, command))
        self.assertEqual(scope.commands, [("rw [h]", 1)])
        self.assertFalse(outcome.success)

    def test_search_drops_an_unplayable_tactic_without_calling_lean(self):
        client = _HealthyClient()
        client._start_goal = _start_structured
        client._apply_tactic = _intro_structured
        executor = ScriptedExecutor(lambda _occurrence, command: self.fail(command), server=client)

        class Reasoner(HybridReasoner):
            def _build_gnn_engine(self, **_kwargs):
                return None

            def predict_next_tactic(self, goal):
                return [_tactic("rw", "↑13")]

        reasoner = Reasoner(
            config_path=None, tactic_model_path=None, argument_model_path=None,
            executor=executor, use_pln=False, max_nodes=5,
        )
        graph = asyncio.run(reasoner.prove(seed_from_text(goal="p", hypotheses=["p : Prop"])))
        self.assertEqual(executor.calls, [])  # never reached Lean
        self.assertEqual(graph.root.note, "executor rejected every candidate tactic")


async def _start_structured(expression):
    return "start"


async def _intro_structured(state, tactic):
    from maths_ai.gnn_inference.tests.model_helpers import pantograph_goal

    goal = structured_goal("p", ["p : Prop"])
    return type("State", (), {"goals": [pantograph_goal(goal)]})()


class _HealthyClient(ScopedFakeClient):
    def __init__(self):
        self.started = []

    async def _start_goal(self, expression):
        self.started.append(expression)
        return "goal-state"

    async def _apply_tactic(self, state, tactic):
        return f"{state}+{tactic}"


class ScopedReplayTests(unittest.TestCase):
    def test_typed_failures_propagate_without_restart_or_retry(self):
        for error in (
            PantographGoalRejected("bad goal"),
            PantographTransportError("closed pipe"),
            PantographProtocolError("bad contract"),
            PantographStateError("stale state"),
        ):
            client = _HealthyClient()
            calls = []

            async def reject(expression):
                calls.append(expression)
                raise error

            client._start_goal = reject
            # Lean rejecting the input is a property of the seed; the rest are
            # infrastructure failures and keep their own types.
            expected = (
                ReplayInputError if isinstance(error, PantographGoalRejected) else type(error)
            )
            with self.assertRaises(expected):
                asyncio.run(
                    materialize_seed(FakeStateScope(client), seed_from_text(goal="p → p"))
                )
            self.assertEqual(len(calls), 1)

    def test_hypotheses_are_quantified_then_introduced(self):
        healthy = _HealthyClient()
        replay = render_replay(
            seed_from_text(goal="q ∨ p", hypotheses=["p : Prop", "h : p ∨ q"])
        )
        state = asyncio.run(materialize_replay(FakeStateScope(healthy), replay))
        self.assertEqual(healthy.started, ["∀ (p : Prop), ∀ (h : p ∨ q), q ∨ p"])
        self.assertEqual(healthy.started_levels, ())
        self.assertEqual(state, "goal-state+intro p h")

    def test_universe_levels_reach_goal_start(self):
        healthy = _HealthyClient()
        replay = render_replay(
            seed_from_text(goal="a = a", hypotheses=["α : Type u_1", "a : α"])
        )
        asyncio.run(materialize_replay(FakeStateScope(healthy), replay))
        self.assertEqual(healthy.started, ["∀ (α : Type u_1), ∀ (a : α), a = a"])
        self.assertEqual(healthy.started_levels, ("u_1",))

    def test_structured_replay_preserves_binder_roles_and_local_let(self):
        healthy = _HealthyClient()
        goal = Goal(
            expression="x = x",
            goal_model_sexp="(:app (:c Eq) (:fv FV2) (:fv FV2))",
            goal_sexp="((:c Eq) (:fv _a) (:fv _x) (:fv _x))",
            locals=[
                GoalLocal(
                    user_name="α",
                    internal_name="_a",
                    context_index=0,
                    binder_role=":implicit",
                    type_pp="Type",
                    type_model_sexp="(:sort 1)",
                    type_sexp="(:sort 1)",
                ),
                GoalLocal(
                    user_name="inst",
                    internal_name="_inst",
                    context_index=1,
                    binder_role=":instance-implicit",
                    is_instance=True,
                    type_pp="Inhabited α",
                    type_model_sexp="(:app (:c Inhabited) (:fv FV0))",
                    type_sexp="((:c Inhabited) (:fv _a))",
                ),
                GoalLocal(
                    user_name="x",
                    internal_name="_x",
                    context_index=2,
                    binder_role=":let",
                    is_let=True,
                    type_pp="α",
                    type_model_sexp="(:fv FV0)",
                    type_sexp="(:fv _a)",
                    value_pp="default",
                    value_model_sexp="(:c default)",
                    value_sexp="((:c default) (:fv _a) (:fv _inst))",
                ),
            ],
            model_sexp_version=1,
        )

        state = asyncio.run(materialize_replay(FakeStateScope(healthy), goal.replay_spec()))

        self.assertEqual(
            healthy.started,
            [
                "∀ {α : (Sort 1)}, ∀ [inst : (@Inhabited @α)], "
                "let x : @α := (@default @α @inst); (@Eq @α @x @x)"
            ],
        )
        self.assertEqual(state, "goal-state+intro α inst x")
