from __future__ import annotations

import unittest

from maths_ai.data_models.proof_components import Goal, STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import (
    NodeClosureReason,
    NodeStatus,
    ProofHypergraph,
    SearchEndReason,
)

from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import (
    RewardConfig,
    potential,
    edge_terminal_reward,
    edge_shaping,
    edge_shaped_reward,
)
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    BackupValidity,
    HarvestConfig,
    compute_backups,
    extract_actor_transitions,
    extract_critic_samples,
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

    def test_solved_node_and_qed_edge_are_known_success(self) -> None:
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        edge = g.add_edge(g.root_id, _tac(), ranked_subgoals=[])
        tables = compute_backups(g)
        self.assertEqual(tables.edge_outcomes[edge.id].value, 1.0)
        self.assertEqual(tables.node_targets[g.root_id].value, 1.0)

    def test_open_children_leave_and_edge_and_parent_unknown(self) -> None:
        g, edge = self._and_graph()
        tables = compute_backups(g)
        self.assertEqual(tables.edge_outcomes[edge.id].validity, BackupValidity.UNKNOWN)
        self.assertEqual(tables.node_targets[g.root_id].validity, BackupValidity.UNKNOWN)

    def test_solved_plus_solved_is_known_and_success(self) -> None:
        g, edge = self._and_graph()
        for child_id in edge.child_ids:
            g.add_edge(child_id, _tac(), ranked_subgoals=[])
        tables = compute_backups(g)
        self.assertEqual(tables.edge_outcomes[edge.id].value, 1.0)
        self.assertEqual(tables.node_targets[g.root_id].value, 1.0)

    def test_solved_plus_exhausted_is_known_and_failure(self) -> None:
        g, edge = self._and_graph()
        solved_id, exhausted_id = edge.child_ids
        g.add_edge(solved_id, _tac(), ranked_subgoals=[])
        g.mark_node_exhausted(
            exhausted_id,
            reason=NodeClosureReason.NO_CANDIDATES,
        )
        tables = compute_backups(g)
        self.assertEqual(tables.edge_outcomes[edge.id].value, 0.0)

    def test_solved_plus_unelaborated_is_unknown(self) -> None:
        g, edge = self._and_graph()
        solved_id, unelaborated_id = edge.child_ids
        g.add_edge(solved_id, _tac(), ranked_subgoals=[])
        g.mark_node_unelaborated(unelaborated_id)
        tables = compute_backups(g)
        self.assertEqual(tables.edge_outcomes[edge.id].validity, BackupValidity.UNKNOWN)
        self.assertEqual(
            tables.node_targets[unelaborated_id].validity,
            BackupValidity.UNKNOWN,
        )

    def test_exhausted_plus_unelaborated_is_known_and_failure(self) -> None:
        g, edge = self._and_graph()
        exhausted_id, unelaborated_id = edge.child_ids
        g.mark_node_exhausted(exhausted_id, reason=NodeClosureReason.NO_CANDIDATES)
        g.mark_node_unelaborated(unelaborated_id)
        self.assertEqual(compute_backups(g).edge_outcomes[edge.id].value, 0.0)

    def test_solved_alternative_makes_parent_known_success(self) -> None:
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        solved = g.add_edge(g.root_id, _tac("exact"), ranked_subgoals=[])
        unknown = g.add_edge(
            g.root_id,
            _tac("apply"),
            ranked_subgoals=[(Goal(expression="Q", hypotheses=[]), None)],
        )
        g.mark_node_unelaborated(unknown.child_ids[0])
        tables = compute_backups(g)
        self.assertEqual(tables.edge_outcomes[solved.id].value, 1.0)
        self.assertEqual(tables.edge_outcomes[unknown.id].validity, BackupValidity.UNKNOWN)
        self.assertEqual(tables.node_targets[g.root_id].value, 1.0)

    def test_failed_plus_unknown_alternatives_leave_parent_unknown(self) -> None:
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        failed = g.add_edge(
            g.root_id,
            _tac("first"),
            ranked_subgoals=[(Goal(expression="Q", hypotheses=[]), None)],
        )
        unknown = g.add_edge(
            g.root_id,
            _tac("second"),
            ranked_subgoals=[(Goal(expression="R", hypotheses=[]), None)],
        )
        g.mark_node_exhausted(failed.child_ids[0], reason=NodeClosureReason.NO_CANDIDATES)
        g.mark_node_unelaborated(unknown.child_ids[0])
        g.mark_node_exhausted(g.root_id, reason=NodeClosureReason.CANDIDATES_EXHAUSTED)
        tables = compute_backups(g)
        self.assertEqual(tables.edge_outcomes[failed.id].value, 0.0)
        self.assertEqual(tables.edge_outcomes[unknown.id].validity, BackupValidity.UNKNOWN)
        self.assertEqual(tables.node_targets[g.root_id].validity, BackupValidity.UNKNOWN)

    def test_locally_exhausted_root_with_only_known_failures_is_zero(self) -> None:
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        edge = g.add_edge(
            g.root_id,
            _tac(),
            ranked_subgoals=[(Goal(expression="Q", hypotheses=[]), None)],
        )
        g.mark_node_exhausted(edge.child_ids[0], reason=NodeClosureReason.NO_CANDIDATES)
        g.mark_node_exhausted(g.root_id, reason=NodeClosureReason.CANDIDATES_EXHAUSTED)
        self.assertEqual(compute_backups(g).node_targets[g.root_id].value, 0.0)

    def test_depth_limit_fails_incoming_edge_but_not_child_critic(self) -> None:
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        edge = g.add_edge(
            g.root_id,
            _tac(),
            ranked_subgoals=[(Goal(expression="Q", hypotheses=[]), None)],
        )
        child_id = edge.child_ids[0]
        g.mark_node_exhausted(child_id, reason=NodeClosureReason.DEPTH_LIMIT)
        tables = compute_backups(g)
        self.assertEqual(tables.edge_outcomes[edge.id].value, 0.0)
        self.assertEqual(tables.node_targets[child_id].validity, BackupValidity.UNKNOWN)

    def test_root_depth_limit_is_a_known_configured_search_failure(self) -> None:
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        g.mark_node_exhausted(g.root_id, reason=NodeClosureReason.DEPTH_LIMIT)
        self.assertEqual(compute_backups(g).node_targets[g.root_id].value, 0.0)

    def test_clean_max_nodes_adds_only_root_zero(self) -> None:
        g, _edge = self._and_graph()
        g.search_end_reason = SearchEndReason.MAX_NODES
        samples = extract_critic_samples(g)
        self.assertEqual([(sample.node_id, sample.target) for sample in samples], [(g.root_id, 0.0)])

    def test_max_nodes_with_unelaborated_evidence_has_no_root_target(self) -> None:
        g, edge = self._and_graph()
        g.mark_node_unelaborated(edge.child_ids[0])
        g.search_end_reason = SearchEndReason.MAX_NODES
        self.assertEqual(extract_critic_samples(g), [])

    def test_actor_and_critic_harvest_are_separate(self) -> None:
        g, edge = self._and_graph()
        a_id, b_id = edge.child_ids
        g.add_edge(a_id, _tac(), ranked_subgoals=[])
        g.add_edge(b_id, _tac(), ranked_subgoals=[])  # fully solved

        reward_cfg = RewardConfig(gamma=0.9, terminal_success=1.0, step_penalty=0.0, use_score=False)
        transitions = extract_actor_transitions(g, reward_cfg)
        critic_samples = extract_critic_samples(g)
        self.assertEqual(len(transitions), 3)
        self.assertEqual(len({sample.node_id for sample in critic_samples}), 3)

        root_t = next(t for t in transitions if t.node_id == g.root_id)
        self.assertEqual(root_t.successor_value, 1.0)
        self.assertAlmostEqual(root_t.return_, root_t.reward + 0.9 * 1.0)

    def test_edge_ids_filter(self) -> None:
        g, edge = self._and_graph()
        transitions = extract_actor_transitions(g, edge_ids=[edge.id])
        self.assertEqual(transitions, [])

        for child_id in edge.child_ids:
            g.add_edge(child_id, _tac(), ranked_subgoals=[])
        transitions = extract_actor_transitions(g, edge_ids=[edge.id])
        self.assertEqual([transition.edge_id for transition in transitions], [edge.id])

    def test_invalid_and_combine_mode_is_rejected(self) -> None:
        g = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        g.add_edge(g.root_id, _tac(), ranked_subgoals=[])
        with self.assertRaisesRegex(ValueError, "Unknown AND combine mode"):
            compute_backups(g, HarvestConfig(and_combine="mean"))


if __name__ == "__main__":
    unittest.main()
