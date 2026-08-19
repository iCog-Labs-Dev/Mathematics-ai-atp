from __future__ import annotations

import unittest

from maths_ai.data_models.proof_components import Goal, STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import (
    BackupSource,
    BackupValidity,
    EdgeStatus,
    NodeClosureReason,
    NodeStatus,
    ProofHypergraph,
    SearchEndReason,
)
from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner, _Simulation
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import EdgeAction
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    HarvestConfig,
    compute_backups,
    extract_critic_samples,
    extract_minimal_hypertree,
    extract_transitions,
)


def tac(name: str = "apply", p: float = 1.0) -> TacticCandidate:
    return TacticCandidate(tactic_name=name, arguments=[], probability=p)


def stv() -> STV:
    return STV(strength=0.5, confidence=1.0)


class ValidityHarvestTests(unittest.TestCase):
    def test_solved_and_unknown_and_edge_is_unknown(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        edge = graph.add_edge(
            graph.root_id,
            tac(),
            [(Goal(expression="A", hypotheses=[]), stv()), (Goal(expression="B", hypotheses=[]), stv())],
        )
        graph.add_edge(edge.child_ids[0], tac("qed"), [])
        graph.nodes[edge.child_ids[1]].closure_reason = NodeClosureReason.ELABORATION_ERROR
        graph.nodes[edge.child_ids[1]].status = NodeStatus.DEAD
        graph.nodes[edge.child_ids[1]].exhausted = True
        tables = compute_backups(graph)
        self.assertEqual(tables.edge_outcomes[edge.id].validity, BackupValidity.UNKNOWN)
        transitions = extract_transitions(graph)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].edge_id, 1)

    def test_known_zero_dominates_unknown_child_on_and_edge(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        edge = graph.add_edge(
            graph.root_id,
            tac(),
            [(Goal(expression="A", hypotheses=[]), stv()), (Goal(expression="B", hypotheses=[]), stv())],
        )
        failed_edge = graph.add_edge(edge.child_ids[0], tac("fail"), [])
        failed_edge.status = EdgeStatus.DEAD
        graph.nodes[edge.child_ids[0]].status = NodeStatus.DEAD
        graph.nodes[edge.child_ids[0]].closure_reason = NodeClosureReason.CANDIDATES_EXHAUSTED
        graph.nodes[edge.child_ids[1]].status = NodeStatus.DEAD
        graph.nodes[edge.child_ids[1]].closure_reason = NodeClosureReason.ELABORATION_ERROR
        graph.nodes[edge.child_ids[1]].exhausted = True
        outcome = compute_backups(graph).edge_outcomes[edge.id]
        self.assertEqual(outcome, outcome.valid(0.0, BackupSource.SEARCH_FAILURE))

    def test_solved_alternative_dominates_unknown_parent(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        solved = graph.add_edge(graph.root_id, tac("qed", 0.9), [])
        unknown = graph.add_edge(
            graph.root_id, tac("unknown", 0.1),
            [(Goal(expression="A", hypotheses=[]), stv())],
        )
        graph.nodes[unknown.child_ids[0]].status = NodeStatus.DEAD
        graph.nodes[unknown.child_ids[0]].closure_reason = NodeClosureReason.ELABORATION_ERROR
        target = compute_backups(graph).node_targets[graph.root_id]
        self.assertEqual(target, target.valid(1.0, BackupSource.LEAN_STATUS))
        self.assertEqual(len(extract_transitions(graph)), 1)
        self.assertEqual(extract_transitions(graph)[0].edge_id, solved.id)

    def test_failed_alternative_does_not_dominate_unknown_parent(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        failed = graph.add_edge(
            graph.root_id, tac("failed"), [(Goal(expression="A", hypotheses=[]), stv())]
        )
        unknown = graph.add_edge(
            graph.root_id, tac("unknown"), [(Goal(expression="B", hypotheses=[]), stv())]
        )
        graph.nodes[failed.child_ids[0]].status = NodeStatus.DEAD
        graph.nodes[failed.child_ids[0]].closure_reason = NodeClosureReason.CANDIDATES_EXHAUSTED
        graph.nodes[unknown.child_ids[0]].status = NodeStatus.DEAD
        graph.nodes[unknown.child_ids[0]].closure_reason = NodeClosureReason.ELABORATION_ERROR
        target = compute_backups(graph).node_targets[graph.root_id]
        self.assertEqual(target.validity, BackupValidity.UNKNOWN)

    def test_unknown_simulation_counts_work_without_numeric_backup(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        edge = graph.add_edge(
            graph.root_id, tac(), [(Goal(expression="A", hypotheses=[]), stv())]
        )
        child = graph.nodes[edge.child_ids[0]]
        child.status = NodeStatus.DEAD
        child.exhausted = True
        child.closure_reason = NodeClosureReason.ELABORATION_ERROR
        edge.visit_stats.virtual_loss = 1
        reasoner = object.__new__(HybridReasoner)
        reasoner._backup_simulation(
            graph,
            _Simulation(chosen_edges={graph.root_id: edge.id}, leaves=[]),
        )
        self.assertEqual(edge.visit_stats.N, 1)
        self.assertEqual(edge.visit_stats.N_v, 0)
        self.assertEqual(edge.visit_stats.W, 0.0)
        self.assertEqual(edge.visit_stats.virtual_loss, 0)

    def test_clean_root_budget_failure_is_root_only(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        graph.end_reason = SearchEndReason.NUM_SIMULATIONS
        graph.nodes[graph.root_id].status = NodeStatus.EXPANDED
        tables = compute_backups(graph)
        self.assertEqual(tables.node_targets[graph.root_id].source, BackupSource.ROOT_EPISODE)
        self.assertEqual(extract_critic_samples(graph)[0].node_id, graph.root_id)

    def test_critic_source_and_unique_nodes(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        graph.add_edge(graph.root_id, tac(), [])
        samples = extract_critic_samples(graph)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].source, BackupSource.LEAN_STATUS)

    def test_all_subgoals_are_retained(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        edge = graph.add_edge(
            graph.root_id,
            tac(),
            [(Goal(expression=str(i), hypotheses=[]), stv()) for i in range(5)],
        )
        self.assertEqual(len(edge.child_ids), 5)

    def test_minimal_tree_requires_structural_solution(self):
        graph = ProofHypergraph(Goal(expression="P", hypotheses=[]))
        edge = graph.add_edge(graph.root_id, tac(), [])
        samples = extract_minimal_hypertree(
            graph, {edge.id: EdgeAction(tactic_id=1)}, mine_all_solved_nodes=False
        )
        self.assertEqual(len(samples), 1)


if __name__ == "__main__":
    unittest.main()
