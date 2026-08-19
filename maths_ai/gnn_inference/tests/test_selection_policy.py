"""Unit tests for PUCT edge selection (Phase 1 of the HTPS/MCTS integration)
and the search-mode validator.

puct_score(stats, total_node_visits, c)
  = Q + c * stats.prior_prob * sqrt(total_node_visits) / (1 + N + virtual_loss)

where Q = W / N_v with first-play-urgency 0.5 when no valid backup exists, and
prior_prob is the policy prior stamped onto the stats at edge creation.
"""

import pytest

from maths_ai.hybrid_reasoner.hypergraph import EdgeVisitStats
from maths_ai.hybrid_reasoner.selection_policy import puct_score, resolve_search_params


def test_prior_dominates_among_unvisited_edges():
    # Both edges unvisited: Q is the identical FPU 0.5, so the ordering is
    # exactly the prior ordering.
    high = EdgeVisitStats(prior_prob=0.9)
    low = EdgeVisitStats(prior_prob=0.1)
    total = 4
    assert puct_score(high, total, c=1.0) > puct_score(low, total, c=1.0)


def test_q_dominates_at_high_visit_counts():
    # A well-visited winning edge (Q=1) with a tiny prior beats a well-visited
    # losing edge (Q=0) with a large prior: the exploration bonus has decayed
    # by 1/(1+N) while Q has converged.
    winning = EdgeVisitStats(N=100, N_v=100, W=100.0, prior_prob=0.01)
    losing = EdgeVisitStats(N=100, N_v=100, W=0.0, prior_prob=0.99)
    total = 200
    assert puct_score(winning, total, c=1.0) > puct_score(losing, total, c=1.0)


def test_virtual_loss_suppresses_in_flight_edge():
    # Identical priors and no valid backups; one edge has a simulation in
    # flight. Exploration still sees the traversal through N + virtual_loss.
    in_flight = EdgeVisitStats(virtual_loss=1, prior_prob=0.5)
    untouched = EdgeVisitStats(prior_prob=0.5)
    total = 1
    assert puct_score(untouched, total, c=1.0) > puct_score(in_flight, total, c=1.0)


def test_exploration_bonus_grows_with_sibling_visits():
    # The same edge scores higher when its siblings have absorbed more
    # simulations: sqrt(total_node_visits) rises while its own denominator
    # stays fixed, pushing selection back toward under-visited branches.
    stats = EdgeVisitStats(N=1, W=0.5, prior_prob=0.5)
    assert puct_score(stats, 100, c=1.0) > puct_score(stats, 4, c=1.0)


def test_c_scales_the_exploration_term_only():
    # At c=0 the score is exactly Q; raising c adds the prior-weighted bonus.
    stats = EdgeVisitStats(N=2, W=1.0, prior_prob=0.8)
    assert puct_score(stats, 10, c=0.0) == stats.Q
    assert puct_score(stats, 10, c=2.0) > puct_score(stats, 10, c=1.0)


class TestResolveSearchParams:
    def test_legacy_all_none_returns_placeholders(self):
        assert resolve_search_params("legacy", None, None, None) == (1, 1, 0.0)

    def test_legacy_with_explicit_budget_rejected(self):
        with pytest.raises(ValueError, match="num_simulations"):
            resolve_search_params("legacy", 8, None, None)
        with pytest.raises(ValueError, match="sim_batch_size"):
            resolve_search_params("legacy", None, 4, None)
        with pytest.raises(ValueError, match="puct_c"):
            resolve_search_params("legacy", None, None, 1.0)

    def test_puct_requires_num_simulations(self):
        with pytest.raises(ValueError, match="num_simulations"):
            resolve_search_params("puct", None, None, None)

    def test_puct_resolves_none_defaults(self):
        assert resolve_search_params("puct", 16, None, None) == (16, 4, 1.0)

    def test_puct_explicit_values_pass_through(self):
        assert resolve_search_params("puct", 16, 2, 0.5) == (16, 2, 0.5)

    def test_rp_reserved(self):
        with pytest.raises(NotImplementedError):
            resolve_search_params("rp", None, None, None)

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError, match="selection_policy"):
            resolve_search_params("greedy", None, None, None)
