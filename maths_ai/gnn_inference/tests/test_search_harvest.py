from __future__ import annotations

import unittest

from maths_ai.data_models.proof_components import STV, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import (
    BackupSource,
    BackupValidity,
    BackupValue,
    EdgeStatus,
    NodeClosureReason,
    NodeStatus,
    ProofHypergraph,
    SearchEndReason,
)
from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner, _Simulation
from maths_ai.gnn_inference.atp_lean_gnn.graph import dag_fingerprint, model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    EdgeAction,
    ObservationOutcome,
    PolicyObservation,
)
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import CandidatePoolSnapshot
from maths_ai.gnn_inference.tests.model_helpers import certify_witness, scored, structured_goal
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


def goal(expression: str):
    return structured_goal(expression, ["h : Prop"])


def fingerprint(goal_obj) -> str:
    return str(dag_fingerprint(model_goal_to_dag(goal_obj)))


class ValidityHarvestTests(unittest.TestCase):
    def test_ready_and_unknown_and_edge_is_unknown(self):
        graph = ProofHypergraph(goal("P"))
        edge = graph.add_edge(
            graph.root_id,
            scored(tac()),
            [(goal("A"), stv()), (goal("B"), stv())],
        )
        closing = graph.add_edge(edge.child_ids[0], scored(tac("qed")), [])
        graph.nodes[edge.child_ids[1]].closure_reason = NodeClosureReason.ELABORATION_ERROR
        graph.nodes[edge.child_ids[1]].status = NodeStatus.DEAD
        tables = compute_backups(graph)
        self.assertEqual(tables.edge_outcomes[edge.id].validity, BackupValidity.UNKNOWN)
        # The closing attempt is a candidate no replay certified.
        self.assertEqual(tables.edge_outcomes[closing.id].validity, BackupValidity.UNKNOWN)
        self.assertEqual(extract_transitions(graph), [])

    def test_known_zero_dominates_unknown_child_on_and_edge(self):
        graph = ProofHypergraph(goal("P"))
        edge = graph.add_edge(
            graph.root_id,
            scored(tac()),
            [(goal("A"), stv()), (goal("B"), stv())],
        )
        failed_edge = graph.add_edge(edge.child_ids[0], scored(tac("fail")), [])
        failed_edge.status = EdgeStatus.DEAD
        graph.nodes[edge.child_ids[0]].status = NodeStatus.DEAD
        graph.nodes[edge.child_ids[0]].closure_reason = NodeClosureReason.CANDIDATES_EXHAUSTED
        graph.nodes[edge.child_ids[1]].status = NodeStatus.DEAD
        graph.nodes[edge.child_ids[1]].closure_reason = NodeClosureReason.ELABORATION_ERROR
        outcome = compute_backups(graph).edge_outcomes[edge.id]
        self.assertEqual(outcome, outcome.valid(0.0, BackupSource.SEARCH_FAILURE))

    def test_solved_alternative_dominates_unknown_parent(self):
        graph = ProofHypergraph(goal("P"))
        solved = graph.add_edge(graph.root_id, scored(tac("qed", 0.9)), [])
        unknown = graph.add_edge(
            graph.root_id, scored(tac("unknown", 0.1)),
            [(goal("A"), stv())],
        )
        graph.nodes[unknown.child_ids[0]].status = NodeStatus.DEAD
        graph.nodes[unknown.child_ids[0]].closure_reason = NodeClosureReason.ELABORATION_ERROR
        certify_witness(graph, solved.id)
        target = compute_backups(graph).node_targets[graph.root_id]
        self.assertEqual(target, target.valid(1.0, BackupSource.LEAN_STATUS))
        self.assertEqual(len(extract_transitions(graph)), 1)
        self.assertEqual(extract_transitions(graph)[0].edge_id, solved.id)

    def test_failed_alternative_does_not_dominate_unknown_parent(self):
        graph = ProofHypergraph(goal("P"))
        failed = graph.add_edge(
            graph.root_id, scored(tac("failed")), [(goal("A"), stv())]
        )
        unknown = graph.add_edge(
            graph.root_id, scored(tac("unknown")), [(goal("B"), stv())]
        )
        graph.nodes[failed.child_ids[0]].status = NodeStatus.DEAD
        graph.nodes[failed.child_ids[0]].closure_reason = NodeClosureReason.CANDIDATES_EXHAUSTED
        graph.nodes[unknown.child_ids[0]].status = NodeStatus.DEAD
        graph.nodes[unknown.child_ids[0]].closure_reason = NodeClosureReason.ELABORATION_ERROR
        target = compute_backups(graph).node_targets[graph.root_id]
        self.assertEqual(target.validity, BackupValidity.UNKNOWN)

    def test_unknown_simulation_counts_work_without_numeric_backup(self):
        graph = ProofHypergraph(goal("P"))
        edge = graph.add_edge(
            graph.root_id, scored(tac()), [(goal("A"), stv())]
        )
        child = graph.nodes[edge.child_ids[0]]
        child.status = NodeStatus.DEAD
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
        graph = ProofHypergraph(goal("P"))
        graph.end_reason = SearchEndReason.NUM_SIMULATIONS
        graph.nodes[graph.root_id].status = NodeStatus.EXPANDED
        tables = compute_backups(graph)
        self.assertEqual(tables.node_targets[graph.root_id].source, BackupSource.ROOT_EPISODE)
        self.assertEqual(extract_critic_samples(graph)[0].node_id, graph.root_id)

    def test_critic_source_and_unique_nodes(self):
        graph = ProofHypergraph(goal("P"))
        edge = graph.add_edge(graph.root_id, scored(tac()), [])
        self.assertEqual(extract_critic_samples(graph), [])
        certify_witness(graph, edge.id)
        samples = extract_critic_samples(graph)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].source, BackupSource.LEAN_STATUS)

    def test_all_subgoals_are_retained(self):
        graph = ProofHypergraph(goal("P"))
        edge = graph.add_edge(
            graph.root_id,
            scored(tac()),
            [(goal(str(i)), stv()) for i in range(5)],
        )
        self.assertEqual(len(edge.child_ids), 5)

    def test_an_open_source_with_only_failed_alternatives_is_unknown(self):
        graph = ProofHypergraph(goal("P"))
        edge = graph.add_edge(graph.root_id, scored(tac()), [(goal("A"), stv())])
        child = graph.nodes[edge.child_ids[0]]
        failed = graph.add_edge(child.id, scored(tac("fail")), [(goal("Z"), stv())])
        graph.nodes[failed.child_ids[0]].closure_reason = NodeClosureReason.CANDIDATES_EXHAUSTED
        graph.nodes[failed.child_ids[0]].status = NodeStatus.DEAD
        failed.status = EdgeStatus.DEAD
        # A can still widen: its one known alternative failing is no target.
        self.assertEqual(compute_backups(graph).node_targets[child.id], BackupValue.unknown())
        # Once its source is closed, the failure is genuine for this search.
        graph.mark_node_exhausted(child.id, reason=NodeClosureReason.WIDENING_BOUND)
        self.assertEqual(
            compute_backups(graph).node_targets[child.id],
            BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE),
        )

    def test_minimal_tree_requires_the_certified_witness(self):
        graph = ProofHypergraph(goal("P"))
        edge = graph.add_edge(graph.root_id, scored(tac()), [])
        observations = [
            PolicyObservation(
                node_id=graph.root_id,
                goal=graph.root.goal,
                action=EdgeAction(
                    tactic_id=1,
                    candidate_pool=CandidatePoolSnapshot(()),
                    graph_fingerprint=fingerprint(graph.root.goal),
                ),
                outcome=ObservationOutcome.ACCEPTED,
                first_round=0,
                edge_id=edge.id,
                log_prior=0.0,
                provider="test",
            )
        ]
        # A ready candidate is not a proof hypertree.
        self.assertEqual(
            extract_minimal_hypertree(graph, observations, mine_all_solved_nodes=False), []
        )
        certify_witness(graph, edge.id, matched=False)
        self.assertEqual(
            extract_minimal_hypertree(graph, observations, mine_all_solved_nodes=False), []
        )
        certify_witness(graph, edge.id)
        samples = extract_minimal_hypertree(graph, observations, mine_all_solved_nodes=False)
        self.assertEqual(len(samples), 1)

    def test_soft_targets_ignore_subtrees_with_uncertified_candidates(self):
        graph = ProofHypergraph(goal("P"))
        edge = graph.add_edge(graph.root_id, scored(tac()), [(goal("A"), stv())])
        graph.add_edge(edge.child_ids[0], scored(tac("qed")), [])
        edge.visit_stats.N = edge.visit_stats.N_v = 4
        edge.visit_stats.W = 4.0
        # Q = 1 rests on a READY candidate the kernel never checked.
        target = compute_backups(graph, visit_threshold=1).node_targets[graph.root_id]
        self.assertEqual(target.validity, BackupValidity.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
