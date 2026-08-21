from __future__ import annotations

import unittest

from maths_ai.data_models.proof_components import Goal, STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph, NodeStatus

from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import (
    RewardConfig,
    potential,
    edge_terminal_reward,
    edge_shaping,
    edge_shaped_reward,
)
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    HarvestConfig,
    backup_values,
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
        self.assertEqual(backup_values(g)[g.root_id], 1.0)

    def test_backup_and_node_product(self) -> None:
        g, edge = self._and_graph()
        a_id, b_id = edge.child_ids

        # Open subgoals are not failure evidence.
        self.assertFalse(backup_values(g)[g.root_id].known)

        # Solve A only: B remains unknown, so the AND-edge remains unknown.
        g.add_edge(a_id, _tac(), ranked_subgoals=[])
        vals = backup_values(g)
        self.assertEqual(vals[a_id], 1.0)
        self.assertFalse(vals[g.root_id].known)

        # Solve B ⇒ product(1,1) = 1 and root propagates to SOLVED.
        g.add_edge(b_id, _tac(), ranked_subgoals=[])
        vals = backup_values(g)
        self.assertEqual(vals[b_id], 1.0)
        self.assertEqual(vals[g.root_id], 1.0)
        self.assertEqual(g.nodes[g.root_id].status, NodeStatus.SOLVED)

    def test_backup_min_combine(self) -> None:
        g, edge = self._and_graph()
        a_id, _ = edge.child_ids
        g.add_edge(a_id, _tac(), ranked_subgoals=[])  # A solved, B unresolved
        vals = backup_values(g, HarvestConfig(and_combine="min"))
        self.assertFalse(vals[g.root_id].known)

    def test_known_dead_child_beats_unknown_sibling(self) -> None:
        g, edge = self._and_graph()
        a_id, _ = edge.child_ids
        g.mark_node_exhausted(a_id, note="local candidate policy exhausted")
        g.mark_node_exhausted(g.root_id, note="local candidate policy exhausted")
        vals = backup_values(g)
        self.assertEqual(vals[a_id], 0.0)
        self.assertEqual(vals[g.root_id], 0.0)

    def test_unknown_child_omits_transition(self) -> None:
        g, edge = self._and_graph()
        self.assertEqual(extract_transitions(g, edge_ids=[edge.id]), [])

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
        # Root subgoals both solved ⇒ children_value = product(1,1) = 1; value_target = 1.
        self.assertEqual(root_t.children_value, 1.0)
        self.assertEqual(root_t.value_target, 1.0)
        self.assertAlmostEqual(root_t.return_, root_t.reward + 0.9 * 1.0)

    def test_edge_ids_filter(self) -> None:
        g, edge = self._and_graph()
        # The edge filter selects the root edge, but unresolved children make
        # its return unknown, so issue #24 correctly omits it from harvest.
        transitions = extract_transitions(g, edge_ids=[edge.id])
        self.assertEqual(transitions, [])


if __name__ == "__main__":
    unittest.main()
