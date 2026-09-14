"""Tactic surface syntax and dead-server discrimination (no Lean, no subprocess).

Two seams from the RL search log diagnosis:

* ``render_tactic_command`` — argument shape is per tactic. ``rw h`` is a parse
  error at the column where ``[`` was expected; the playable form is ``rw [h]``.
* ``_server_is_dead`` / ``_start_state`` — a ``ServerError`` means either a
  crashed REPL or a live REPL rejecting one goal. Only the first justifies a
  restart; re-raising the second lets the search mark the node exhausted.
"""

from __future__ import annotations

import asyncio
import unittest

from pantograph.server import ServerError

from maths_ai.data_models.proof_components import Goal, TacticCandidate
from maths_ai.hybrid_reasoner.joint_inference import (
    HybridReasoner,
    PantographExecutor,
    _server_is_dead,
    render_tactic_command,
)


def _tactic(name: str, *arguments: str) -> TacticCandidate:
    return TacticCandidate(tactic_name=name, arguments=list(arguments), probability=0.5)


class RenderTacticCommandTests(unittest.TestCase):
    def test_bracket_required_tactic_gets_brackets(self):
        self.assertEqual(render_tactic_command(_tactic("rw", "h")), "rw [h]")

    def test_multiple_rules_are_comma_separated(self):
        self.assertEqual(
            render_tactic_command(_tactic("rw", "h", "Nat.add_comm")), "rw [h, Nat.add_comm]"
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
        self.assertEqual(render_tactic_command(_tactic("linarith", "h1", "h2")), "linarith [h1, h2]")

    def test_ordinary_tactic_keeps_space_separated_arguments(self):
        self.assertEqual(render_tactic_command(_tactic("apply", "h")), "apply h")
        self.assertEqual(render_tactic_command(_tactic("intro", "p", "q")), "intro p q")
        self.assertEqual(render_tactic_command(_tactic("exact", "Nat.le_refl")), "exact Nat.le_refl")

    def test_bare_tactic_renders_as_its_name(self):
        self.assertEqual(render_tactic_command(_tactic("rfl")), "rfl")

    def test_trailing_colons_and_blanks_are_stripped(self):
        """Arguments arrive rendered from DAG nodes, which can carry a `name :` form."""
        self.assertEqual(render_tactic_command(_tactic("apply", "h :", "  ")), "apply h")

    def test_names_with_lean_punctuation_are_usable_rules(self):
        self.assertEqual(render_tactic_command(_tactic("rw", "h'")), "rw [h']")
        self.assertEqual(render_tactic_command(_tactic("rw", "Nat.factorial_succ")),
                         "rw [Nat.factorial_succ]")


class _RecordingServer:
    """Captures the command string the executor sends to Lean."""

    def __init__(self):
        self.commands: list[str] = []
        self.proc = object()

    async def goal_tactic_async(self, state, tactic):
        self.commands.append(tactic)
        raise ServerError("tactic failed")  # the return path is not what these tests check


class ExecutorRenderingTests(unittest.TestCase):
    def test_executor_sends_bracketed_form(self):
        server = _RecordingServer()
        executor = PantographExecutor(server)
        outcome = asyncio.run(executor.apply(server, object(), _tactic("rw", "h")))
        self.assertEqual(server.commands, ["rw [h]"])
        self.assertFalse(outcome.success)

    def test_executor_drops_unplayable_tactic_without_calling_lean(self):
        server = _RecordingServer()
        executor = PantographExecutor(server)
        outcome = asyncio.run(executor.apply(server, object(), _tactic("rw", "↑13")))
        self.assertEqual(server.commands, [])  # never reached Lean
        self.assertFalse(outcome.success)
        self.assertIn("bracketed rule list", outcome.error)


class _FakeEnv:
    """Stands in for PantographEnv, counting restarts."""

    def __init__(self, server_factory):
        self._server_factory = server_factory
        self.restarts = 0

    async def create_server(self):
        self.restarts += 1
        return self._server_factory()

    def describe(self) -> str:
        return "fake-env"


class _CrashableServer:
    """A REPL that raises on ``goal_start_async``.

    ``dies=True`` reproduces ``run_async``'s behaviour on an empty read from a
    panicked process: ``_close()`` first (which clears ``proc``), then raise.
    ``dies=False`` is a live server rejecting one goal.
    """

    def __init__(self, *, dies: bool, error: Exception | None = None):
        self.proc = object()
        self._dies = dies
        self._error = error or ServerError("elaboration failed")
        self.closed = False

    def _close(self):
        self.proc = None
        self.closed = True

    async def goal_start_async(self, expression):
        if self._dies:
            self._close()
        raise self._error

    async def goal_tactic_async(self, state, tactic):
        raise AssertionError("not reached")


class _HealthyServer:
    def __init__(self):
        self.proc = object()
        self.started: list[str] = []

    def _close(self):
        self.proc = None

    async def goal_start_async(self, expression):
        self.started.append(expression)
        return "goal-state"

    async def goal_tactic_async(self, state, tactic):
        return f"{state}+{tactic}"


def _reasoner(server, env) -> HybridReasoner:
    """A reasoner with only the attributes ``_start_state`` reaches.

    The full ``__init__`` builds a GNN engine and a PLN subprocess; the restart
    seam depends on none of that.
    """
    reasoner = object.__new__(HybridReasoner)
    reasoner.server = server
    reasoner.executor = PantographExecutor(server)
    reasoner._env = env
    return reasoner


class ServerIsDeadTests(unittest.TestCase):
    def test_live_server_is_not_dead(self):
        self.assertFalse(_server_is_dead(_HealthyServer()))

    def test_closed_server_is_dead(self):
        server = _HealthyServer()
        server._close()
        self.assertTrue(_server_is_dead(server))

    def test_object_without_proc_attribute_reads_as_dead(self):
        self.assertTrue(_server_is_dead(object()))


class RestartDiscriminationTests(unittest.TestCase):
    def test_live_rejection_is_reraised_without_restarting(self):
        """An elaboration error is the search's business, not the supervisor's."""
        server = _CrashableServer(dies=False)
        env = _FakeEnv(_HealthyServer)
        reasoner = _reasoner(server, env)
        with self.assertRaises(ServerError):
            asyncio.run(reasoner._start_state(Goal(expression="p → p", hypotheses=[])))
        self.assertEqual(env.restarts, 0)

    def test_crashed_server_triggers_restart_and_retry(self):
        healthy = _HealthyServer()
        env = _FakeEnv(lambda: healthy)
        reasoner = _reasoner(_CrashableServer(dies=True), env)
        state = asyncio.run(reasoner._start_state(Goal(expression="p → p", hypotheses=[])))
        self.assertEqual(env.restarts, 1)
        self.assertEqual(state, "goal-state")
        self.assertIs(reasoner.server, healthy)

    def test_restart_reinstalls_server_on_the_executor(self):
        """A stale executor would keep writing to the reaped subprocess."""
        healthy = _HealthyServer()
        env = _FakeEnv(lambda: healthy)
        reasoner = _reasoner(_CrashableServer(dies=True), env)
        asyncio.run(reasoner._start_state(Goal(expression="p → p", hypotheses=[])))
        self.assertIs(reasoner.executor.server, healthy)

    def test_assertion_error_from_reaped_process_triggers_restart(self):
        """``run_async`` asserts ``self.proc`` on every call after the first crash."""
        server = _CrashableServer(dies=False, error=AssertionError("Server not running."))
        env = _FakeEnv(_HealthyServer)
        reasoner = _reasoner(server, env)
        asyncio.run(reasoner._start_state(Goal(expression="p → p", hypotheses=[])))
        self.assertEqual(env.restarts, 1)

    def test_broken_pipe_triggers_restart(self):
        server = _CrashableServer(dies=False, error=BrokenPipeError())
        env = _FakeEnv(_HealthyServer)
        reasoner = _reasoner(server, env)
        asyncio.run(reasoner._start_state(Goal(expression="p → p", hypotheses=[])))
        self.assertEqual(env.restarts, 1)

    def test_hypotheses_are_quantified_then_introduced(self):
        healthy = _HealthyServer()
        reasoner = _reasoner(healthy, _FakeEnv(lambda: healthy))
        state = asyncio.run(
            reasoner._start_state(Goal(expression="q ∨ p", hypotheses=["p : Prop", "h : p ∨ q"]))
        )
        self.assertEqual(healthy.started, ["∀ (p : Prop), ∀ (h : p ∨ q), q ∨ p"])
        self.assertEqual(state, "goal-state+intro p h")

    def test_local_let_is_preserved_when_reconstructing_a_goal(self):
        healthy = _HealthyServer()
        reasoner = _reasoner(healthy, _FakeEnv(lambda: healthy))
        state = asyncio.run(
            reasoner._start_state(
                Goal(expression="x = 1", hypotheses=["let x : Nat := 1", "h : x = 1"])
            )
        )
        self.assertEqual(healthy.started, ["let x : Nat := 1\n∀ (h : x = 1), x = 1"])
        self.assertEqual(state, "goal-state+intro h")


if __name__ == "__main__":
    unittest.main()
