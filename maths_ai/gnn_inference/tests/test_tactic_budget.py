"""One tactic-step allowance across exploration and combination replay."""

from __future__ import annotations

import asyncio
import unittest

import pytest

from maths_ai.gnn_inference.tests.snapshot_fake import Spec, Step
from maths_ai.gnn_inference.tests.test_retained_search import (
    PUCT,
    SEED,
    _admit,
    _Client,
    _commands,
    _Scripted,
)
from maths_ai.hybrid_reasoner import joint_inference
from maths_ai.hybrid_reasoner.hypergraph import SearchEndReason
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor
from maths_ai.hybrid_reasoner.tactic_budget import TacticAllowance, TacticBudgetExhausted

# A conjunction: three exploration commands, then three replay commands.
RULES = _commands({
    ("P ∧ Q", "constructor"): Step(children=(Spec("P"), Spec("Q"))),
    ("P", "trivial"): Step(),
    ("Q", "trivial"): Step(),
})
PROPOSALS = {"P ∧ Q": [("constructor", 0.9)], "P": [("trivial", 0.8)], "Q": [("trivial", 0.7)]}


def test_the_allowance_counts_and_refuses():
    allowance = TacticAllowance(2)
    allowance.reserve()
    allowance.reserve()
    assert allowance.exhausted and allowance.remaining == 0
    with pytest.raises(TacticBudgetExhausted):
        allowance.reserve()
    assert allowance.as_dict() == {"limit": 2, "issued": 2}
    for bad in (-1, 1.5, True, None):
        with pytest.raises(ValueError):
            TacticAllowance(bad)


class TacticBudgetTests(unittest.TestCase):
    def setUp(self):
        self._original = joint_inference.materialize_seed
        joint_inference.materialize_seed = _admit

    def tearDown(self):
        joint_inference.materialize_seed = self._original

    def search(self, max_tactic_steps, rules=RULES, root=Spec("P ∧ Q"), proposals=PROPOSALS, **kwargs):
        client = _Client(rules, root)
        executor = PantographExecutor(client)
        reasoner = _Scripted(proposals, executor=executor, **kwargs)
        graph = asyncio.run(reasoner.prove(SEED, max_tactic_steps=max_tactic_steps))
        self.assertIsNone(executor.allowance, "the allowance must not outlive its search")
        return graph, client.scopes[0]

    def test_an_exact_allowance_covers_exploration_and_replay(self):
        graph, scope = self.search(6)
        self.assertTrue(graph.is_solved())
        self.assertEqual(graph.execution["original_tactic_calls"], 3)
        self.assertEqual(graph.execution["replay_tactic_calls"], 3)
        self.assertEqual(graph.execution["tactic_steps_issued"], 6)
        self.assertEqual(graph.execution["tactic_step_limit"], 6)
        self.assertEqual(scope.live, 0)

    def test_running_out_during_replay_ends_the_search_without_a_proof(self):
        graph, scope = self.search(5)
        self.assertFalse(graph.is_solved())
        self.assertEqual(graph.end_reason, SearchEndReason.STEP_BUDGET)
        self.assertEqual(graph.execution["tactic_steps_issued"], 5)
        self.assertIsNone(graph.artifact)
        self.assertEqual(scope.live, 0)

    def test_running_out_during_exploration_ends_the_search_without_a_proof(self):
        graph, scope = self.search(2)
        self.assertFalse(graph.is_solved())
        self.assertEqual(graph.end_reason, SearchEndReason.STEP_BUDGET)
        self.assertEqual(graph.execution["tactic_steps_issued"], 2)
        self.assertEqual(scope.live, 0)

    def test_rejected_commands_count(self):
        rules = _commands({("P", "trivial"): Step()})
        proposals = {"P": [("bogus", 0.9), ("trivial", 0.5)]}
        graph, _ = self.search(1, rules=rules, root=Spec("P"), proposals=proposals)
        # The rejected command used the only step; nothing is left to try trivial.
        self.assertFalse(graph.is_solved())
        self.assertEqual(graph.end_reason, SearchEndReason.STEP_BUDGET)
        self.assertEqual(graph.execution["tactic_steps_issued"], 1)

    def test_a_zero_allowance_sends_nothing_to_lean(self):
        graph, scope = self.search(0)
        self.assertEqual(graph.end_reason, SearchEndReason.STEP_BUDGET)
        self.assertEqual(graph.execution["original_tactic_calls"], 0)
        self.assertEqual(graph.execution["tactic_steps_issued"], 0)
        # Only the admission's own setup reached the fake scope.
        self.assertEqual(scope.count("goal.tactic"), 1)

    def test_no_allowance_is_unbounded_and_unreported(self):
        graph, _ = self.search(None)
        self.assertTrue(graph.is_solved())
        self.assertIsNone(graph.execution["tactic_steps_issued"])

    def test_puct_shares_the_same_allowance(self):
        solved, _ = self.search(6, **PUCT)
        self.assertTrue(solved.is_solved())
        self.assertEqual(solved.execution["tactic_steps_issued"], 6)
        short, scope = self.search(4, **PUCT)
        self.assertFalse(short.is_solved())
        self.assertEqual(short.end_reason, SearchEndReason.STEP_BUDGET)
        self.assertLessEqual(short.execution["tactic_steps_issued"], 4)
        self.assertEqual(scope.live, 0)
