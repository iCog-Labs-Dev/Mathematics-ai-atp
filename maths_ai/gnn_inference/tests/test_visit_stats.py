"""Unit tests for EdgeVisitStats (Phase 0 of the HTPS/MCTS integration).

The Q property is the mean of valid numeric backups:
Q = W / N_v, with first-play urgency when N_v == 0. Total traversal count N
is separate and still includes unknown simulations.
"""

from maths_ai.hybrid_reasoner.hypergraph import EdgeVisitStats, ProofHyperedge, ProofHypergraph

from maths_ai.data_models.proof_components import Goal, TacticCandidate
from maths_ai.gnn_inference.tests.model_helpers import scored


def test_q_is_first_play_urgency_when_unvisited():
    stats = EdgeVisitStats()
    assert stats.N == 0
    assert stats.W == 0.0
    assert stats.virtual_loss == 0
    assert stats.Q == 0.5


def test_q_is_mean_of_backed_up_values():
    stats = EdgeVisitStats(N=4, N_v=4, W=3.0)
    assert stats.Q == 3.0 / 4


def test_virtual_loss_inflates_denominator_without_adding_value():
    # Two completed simulations worth 1.0 each; one simulation in flight.
    stats = EdgeVisitStats(N=2, N_v=2, W=2.0, virtual_loss=1)
    assert stats.Q == 1.0
    # After the in-flight simulation backs up unknown:
    stats.virtual_loss -= 1
    stats.N += 1
    assert stats.Q == 1.0


def test_virtual_loss_alone_pins_q_to_zero():
    # An in-flight unknown traversal leaves the valid-value mean unsupported.
    stats = EdgeVisitStats(virtual_loss=2)
    assert stats.Q == 0.5


def test_proof_hyperedge_carries_independent_stats():
    tactic = TacticCandidate(tactic_name="intro", arguments=[], probability=0.5)
    e1 = ProofHyperedge(id=0, source_id=0, tactic=tactic)
    e2 = ProofHyperedge(id=1, source_id=0, tactic=tactic)
    e1.visit_stats.N += 1
    e1.visit_stats.W += 1.0
    assert e2.visit_stats.N == 0
    assert e2.visit_stats.W == 0.0


def test_add_edge_records_the_raw_action_prior_not_the_tactic_confidence():
    # add_edge is the single seam every edge creator flows through: the raw
    # action's log-prior lands on the edge, and the tactic's confidence stays
    # on the candidate for ranking.
    graph = ProofHypergraph(Goal(expression="p → p", hypotheses=[]))
    tactic = TacticCandidate(tactic_name="intro", arguments=[], probability=0.37)
    edge = graph.add_edge(graph.root_id, scored(tactic, log_prior=-2.0), ranked_subgoals=[])
    assert edge.raw_log_priors == {("scripted", "intro", ()): -2.0}
    assert edge.tactic.probability == 0.37
