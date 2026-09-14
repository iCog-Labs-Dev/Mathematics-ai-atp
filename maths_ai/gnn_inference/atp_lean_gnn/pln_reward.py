"""PLN reward system for the actor-critic (Approach 1 shaping + Approach 5 terminals).

The reward is assembled over the ``ProofHypergraph`` the hybrid reasoner already produces.
PLN enters ONLY as a potential-based shaping term ``γ·Φ(s') − Φ(s)`` with ``Φ = σ`` (PLN
strength), which — by the potential-shaping invariance theorem — cannot change the optimal
policy for any PLN, however unreliable. The trustworthy signal that defines the objective is
the terminal reward (QED / closed subgoal = +1), read from the AND-OR solved status.

See ``docs/dev_plans/pln_reward_integration_approaches.md`` and
``docs/dev_plans/pln_actor_critic_integration.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from maths_ai.hybrid_reasoner.hypergraph import (
    EdgeStatus,
    NodeStatus,
    ProofHyperedge,
    ProofHypergraph,
    ProofNode,
)


@dataclass(frozen=True)
class RewardConfig:
    """Parameters of the shaped reward.

    ``gamma``            — discount used both in the shaping term and the return bootstrap.
    ``terminal_success`` — reward for closing a branch (QED / all subgoals discharged).
    ``terminal_failure`` — reward for a dead node (tactic rejected / provably stuck).
    ``step_penalty``     — small per-transition cost to prefer short proofs.
    ``use_score``        — Φ = σ·c (strength×confidence) if True, else Φ = σ (strength).
    """
    gamma: float = 0.99
    terminal_success: float = 1.0
    terminal_failure: float = 0.0
    step_penalty: float = 0.01
    use_score: bool = False


def potential(node: ProofNode, cfg: RewardConfig) -> float:
    """Shaping potential ``Φ(s)`` for a node.

    Convention ``Φ(terminal) = 0`` (required for the shaping's optimum-invariance): a solved
    or dead node contributes no potential. Otherwise ``Φ = σ`` (or ``σ·c`` if ``use_score``),
    read from the node's PLN STV; a node with no STV (e.g. the root before scoring) has Φ = 0.
    """
    if node.status in (NodeStatus.SOLVED, NodeStatus.DEAD):
        return 0.0
    if node.stv is None:
        return 0.0
    return float(node.stv.score) if cfg.use_score else float(node.stv.strength)


def edge_terminal_reward(edge: ProofHyperedge, graph: ProofHypergraph, cfg: RewardConfig) -> float:
    """Trustworthy per-edge terminal reward ``r_term`` (Approach 5), minus the step penalty.

    +``terminal_success`` when this tactic directly closes the branch — a solved edge with no
    remaining subgoals (QED). ``terminal_failure`` when the edge is dead (the tactic was
    rejected or led to a dead subgoal). Interior progress carries only the step penalty; its
    value flows through the bootstrap / AND-OR backup, not an immediate reward.
    """
    if edge.status == EdgeStatus.DEAD:
        return cfg.terminal_failure - cfg.step_penalty
    if edge.status == EdgeStatus.SOLVED and not edge.child_ids:
        return cfg.terminal_success - cfg.step_penalty
    return -cfg.step_penalty


def edge_shaping(edge: ProofHyperedge, graph: ProofHypergraph, cfg: RewardConfig) -> float:
    """Potential-based shaping summed over the subgoals the tactic produced.

    ``Σ_j ( γ·Φ(child_j) − Φ(parent) )``. Because Φ is a state function, this telescopes along
    every root→leaf path and cannot change the optimal policy for any PLN — the property that
    makes an unreliable PLN safe to use here.
    """
    parent = graph.nodes[edge.source_id]
    phi_parent = potential(parent, cfg)
    if not edge.child_ids:
        # QED edge: successor is terminal, Φ(terminal)=0.
        return cfg.gamma * 0.0 - phi_parent
    total = 0.0
    for child_id in edge.child_ids:
        child = graph.nodes[child_id]
        total += cfg.gamma * potential(child, cfg) - phi_parent
    return total


def edge_shaped_reward(edge: ProofHyperedge, graph: ProofHypergraph, cfg: RewardConfig) -> float:
    """Full Approach-1 per-edge reward: ``r_term + ( γΦ(s') − Φ(s) )``."""
    return edge_terminal_reward(edge, graph, cfg) + edge_shaping(edge, graph, cfg)
