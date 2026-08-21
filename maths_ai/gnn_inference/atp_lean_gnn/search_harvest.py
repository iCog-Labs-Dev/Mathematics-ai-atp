"""Harvest actor-critic training targets from a finished ``ProofHypergraph`` (B3, AND-OR).

The hybrid reasoner's search produces an AND-OR proof graph whose solved/dead status is
back-propagated to the root. This module turns that graph into per-transition training
targets: a value target for the critic (from the AND-OR value backup) and a return for the
actor (shaped reward + bootstrapped successor value). The actor advantage
``Â = return − V_pred(s)`` is finished in the training loop, where the critic's prediction at
collection time is known.

Value backup carries both a value and evidence validity:
  * Lean-confirmed SOLVED node → known 1.0
  * locally exhausted failure  → known 0.0
  * globally truncated, unexpanded, or unelaborated → UNKNOWN
  * AND edges fail on one known-zero child, succeed only when every child is known-one
  * OR nodes succeed on one known-one edge; zero needs local exhaustion of every edge
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from maths_ai.data_models.proof_components import Goal, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import (
    EdgeStatus,
    NodeStatus,
    ProofHypergraph,
)

from .pln_reward import RewardConfig, edge_shaped_reward


@dataclass(frozen=True)
class HarvestConfig:
    and_combine: str = "product"  # "product" or "min"


@dataclass(frozen=True)
class BackupValue:
    """A backed-up value plus whether the search observed valid evidence for it."""
    value: Optional[float]
    known: bool

    def __eq__(self, other):
        if isinstance(other, (float, int)):
            return self.known and self.value == float(other)
        return super().__eq__(other)


@dataclass
class HarvestedTransition:
    """One (state, action) training example extracted from the search graph."""
    node_id: int
    goal: Goal                 # state s (expression + hypotheses)
    tactic: TacticCandidate    # action a = (tactic, args)
    reward: float              # shaped per-edge reward r'  (r_term + γΦ(s')−Φ(s))
    children_value: float      # AND-combined backup value of the edge's subgoals (bootstrap)
    value_target: Optional[float]  # unique parent-state critic target, if known
    return_: float             # reward + γ·children_value  (actor target return)
    edge_id: int = -1          # source hyperedge — the on-policy join key to EdgeAction


def _and_combine(values: list[BackupValue], cfg: HarvestConfig) -> BackupValue:
    if not values:
        return BackupValue(1.0, True)  # Lean-confirmed childless edge
    # A single known failed obligation proves an AND-edge cannot succeed.
    if any(v.known and v.value == 0.0 for v in values):
        return BackupValue(0.0, True)
    if not all(v.known for v in values):
        return BackupValue(None, False)
    if cfg.and_combine == "min":
        return BackupValue(min(v.value for v in values), True)
    product = 1.0
    for v in values:
        product *= v.value
    return BackupValue(product, True)


def backup_values(
    graph: ProofHypergraph,
    cfg: HarvestConfig | None = None,
) -> dict[int, BackupValue]:
    """Compute the AND-OR value backup for every node (memoized, cycle-safe)."""
    cfg = cfg or HarvestConfig()
    memo: dict[int, BackupValue] = {}
    in_progress: set[int] = set()

    def value(node_id: int) -> BackupValue:
        if node_id in memo:
            return memo[node_id]
        if node_id in in_progress:
            # Cycle (a subgoal identical to an ancestor): treat as unresolved.
            return BackupValue(None, False)
        node = graph.nodes[node_id]
        if node.status == NodeStatus.SOLVED:
            memo[node_id] = BackupValue(1.0, True)
            return memo[node_id]
        if node.status == NodeStatus.DEAD:
            memo[node_id] = BackupValue(0.0, True)
            return memo[node_id]

        in_progress.add(node_id)
        edge_values: list[BackupValue] = []
        for edge_id in node.outgoing_edge_ids:
            edge = graph.edges[edge_id]
            if edge.status == EdgeStatus.DEAD:
                edge_values.append(BackupValue(0.0, True))
                continue
            child_values = [value(cid) for cid in edge.child_ids]
            edge_values.append(_and_combine(child_values, cfg))
        in_progress.discard(node_id)

        # OR: any Lean-confirmed solution proves success.  A zero is valid only
        # after the node's local candidate policy was fully exhausted.
        if any(v.known and v.value == 1.0 for v in edge_values):
            val = BackupValue(1.0, True)
        elif node.exhausted and edge_values and all(v.known and v.value == 0.0 for v in edge_values):
            val = BackupValue(0.0, True)
        elif node.exhausted and not edge_values:
            val = BackupValue(0.0, True)
        else:
            val = BackupValue(None, False)
        memo[node_id] = val
        return val

    return {node_id: value(node_id) for node_id in graph.nodes}


def extract_transitions(
    graph: ProofHypergraph,
    reward_cfg: RewardConfig | None = None,
    harvest_cfg: HarvestConfig | None = None,
    *,
    edge_ids: list[int] | None = None,
) -> list[HarvestedTransition]:
    """Turn the search graph into per-transition training targets.

    One ``HarvestedTransition`` per hyperedge (an applied tactic). ``edge_ids`` restricts to a
    subset — the on-policy training loop passes only the edges whose tactic was sampled from
    the current policy, so the collected targets are on-policy. Without it, every edge is
    harvested (useful for tests and off-policy analysis).
    """
    reward_cfg = reward_cfg or RewardConfig()
    harvest_cfg = harvest_cfg or HarvestConfig()
    values = backup_values(graph, harvest_cfg)

    chosen = edge_ids if edge_ids is not None else list(graph.edges.keys())
    transitions: list[HarvestedTransition] = []
    critic_harvested: set[int] = set()
    for edge_id in chosen:
        edge = graph.edges[edge_id]
        parent = graph.nodes[edge.source_id]
        reward = edge_shaped_reward(edge, graph, reward_cfg)
        children = _and_combine([values[cid] for cid in edge.child_ids], harvest_cfg)
        # A policy-gradient return depending on unknown evidence is omitted.
        if not children.known:
            continue
        children_value = children.value
        return_ = reward + reward_cfg.gamma * children_value
        parent_value = values[parent.id]
        value_target = None
        if parent_value.known and parent.id not in critic_harvested:
            value_target = parent_value.value
            critic_harvested.add(parent.id)
        transitions.append(
            HarvestedTransition(
                node_id=parent.id,
                goal=parent.goal,
                tactic=edge.tactic,
                reward=reward,
                children_value=children_value,
                value_target=value_target,
                return_=return_,
                edge_id=edge_id,
            )
        )
    return transitions
