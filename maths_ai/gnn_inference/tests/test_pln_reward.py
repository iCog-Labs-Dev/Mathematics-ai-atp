from __future__ import annotations

import unittest

from maths_ai.data_models.proof_components import Goal, STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import BackupValidity, ProofHypergraph, NodeStatus

from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import (
    RewardConfig,
    potential,
    edge_terminal_reward,
    edge_shaping,
    edge_shaped_reward,
)
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    HarvestConfig,
    compute_backups,
    extract_transitions,
)


def _tac(name: str = "apply", p: float = 1.0) -> TacticCandidate:
    return TacticCandidate(tactic_name=name, arguments=[], probability=p)


class PLNRewardTests(unittest.TestCase):
    def test_potential_terminal_is_zero(self) -> None:
        cfg = RewardConfig(use_score=False)
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        # QED edge → root becomes SOLVED → Φ(terminal) = 0 regardless of any STV.
        g.add_edge(g.root_id, _tac(), ranked_subgoals=[])
        self.assertEqual(g.nodes[g.root_id].status, NodeStatus.SOLVED)
        self.assertEqual(potential(g.nodes[g.root_id], cfg), 0.0)

    def test_potential_strength_vs_score(self) -> None:
        g = ProofHypergraph(Goal(expression="P∧Q", hypotheses=[]))
        stv = STV(strength=0.5, confidence=0.4)
        edge = g.add_edge(g.root_id, _tac(), ranked_subgoals=[(Goal(expression="P", hypotheses=[]), stv)])
        child = g.nodes[edge.child_ids[0]]
        self.assertAlmostEqual(potential(child, RewardConfig(use_score=False)), 0.5)
        self.assertAlmostEqual(potential(child, RewardConfig(use_score=True)), 0.5 * 0.4)

    def test_qed_terminal_reward(self) -> None:
        cfg = RewardConfig(terminal_success=1.0, step_penalty=0.01)
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        edge = g.add_edge(g.root_id, _tac(), ranked_subgoals=[])
        self.assertAlmostEqual(edge_terminal_reward(edge, g, cfg), 1.0 - 0.01)

    def test_shaping_telescopes(self) -> None:
        cfg = RewardConfig(gamma=0.9, use_score=False)
        g = ProofHypergraph(Goal(expression="root", hypotheses=[]))  # root stv None ⇒ Φ=0
        stv = STV(strength=0.5, confidence=1.0)
        edge = g.add_edge(g.root_id, _tac(), ranked_subgoals=[(Goal(expression="C", hypotheses=[]), stv)])
        # Φ(parent)=0, Φ(child)=0.5 ⇒ shaping = γ·0.5 − 0 = 0.45
        self.assertAlmostEqual(edge_shaping(edge, g, cfg), 0.9 * 0.5 - 0.0)


class SearchHarvestTests(unittest.TestCase):
    def _and_graph(self):
        """root(P∧Q) --tac--> [A(P), B(Q)], neither solved yet."""
        g = ProofHypergraph(Goal(expression="P∧Q", hypotheses=[]))
        edge = g.add_edge(
            g.root_id,
            _tac(),
            ranked_subgoals=[
                (Goal(expression="P", hypotheses=[]), STV(strength=0.6, confidence=1.0)),
                (Goal(expression="Q", hypotheses=[]), STV(strength=0.4, confidence=1.0)),
            ],
        )
        return g, edge

    def test_backup_solved_and_dead(self) -> None:
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        g.add_edge(g.root_id, _tac(), ranked_subgoals=[])  # QED
        self.assertEqual(compute_backups(g).node_targets[g.root_id].value, 1.0)

    def test_backup_and_node_product(self) -> None:
        g, edge = self._and_graph()
        a_id, b_id = edge.child_ids

        # Open subgoals have no observed numeric outcome.
        self.assertEqual(
            compute_backups(g).node_targets[g.root_id].validity,
            BackupValidity.UNKNOWN,
        )

        # Solve A only ⇒ A=1, B=0 ⇒ product = 0 (AND: all must close).
        g.add_edge(a_id, _tac(), ranked_subgoals=[])
        vals = compute_backups(g).node_targets
        self.assertEqual(vals[a_id].value, 1.0)
        self.assertEqual(vals[g.root_id].validity, BackupValidity.UNKNOWN)

        # Solve B ⇒ product(1,1) = 1 and root propagates to SOLVED.
        g.add_edge(b_id, _tac(), ranked_subgoals=[])
        vals = compute_backups(g).node_targets
        self.assertEqual(vals[b_id].value, 1.0)
        self.assertEqual(vals[g.root_id].value, 1.0)
        self.assertEqual(g.nodes[g.root_id].status, NodeStatus.SOLVED)

    def test_backup_min_combine(self) -> None:
        g, edge = self._and_graph()
        a_id, _ = edge.child_ids
        g.add_edge(a_id, _tac(), ranked_subgoals=[])  # A solved (1), B unresolved (0)
        vals = compute_backups(g, HarvestConfig(and_combine="min")).node_targets
        self.assertEqual(vals[g.root_id].validity, BackupValidity.UNKNOWN)

    def test_extract_transitions_fields(self) -> None:
        g, edge = self._and_graph()
        a_id, b_id = edge.child_ids
        g.add_edge(a_id, _tac(), ranked_subgoals=[])
        g.add_edge(b_id, _tac(), ranked_subgoals=[])  # fully solved

        reward_cfg = RewardConfig(gamma=0.9, terminal_success=1.0, step_penalty=0.0, use_score=False)
        transitions = extract_transitions(g, reward_cfg)
        # One transition per edge: root-edge + two QED edges.
        self.assertEqual(len(transitions), 3)

        root_t = next(t for t in transitions if t.node_id == g.root_id)
        # Root subgoals both solved ⇒ successor_value = product(1,1) = 1.
        self.assertEqual(root_t.successor_value, 1.0)
        self.assertAlmostEqual(root_t.return_, root_t.reward + 0.9 * 1.0)

    def test_edge_ids_filter(self) -> None:
        g, edge = self._and_graph()
        # Only harvest the root edge (on-policy selection).
        transitions = extract_transitions(g, edge_ids=[edge.id])
        self.assertEqual(len(transitions), 0)


if __name__ == "__main__":
    unittest.main()
