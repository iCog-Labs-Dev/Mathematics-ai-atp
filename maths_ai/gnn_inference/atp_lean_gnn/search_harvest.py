"""Extract validity-aware actor and critic targets from an AND-OR search graph.

Positive evidence comes only from a kernel-certified replay: an attempt on the
certified witness whose replayed observation and command matched the ones it
was sampled with is worth 1, and so is its source state. Candidate readiness
is not evidence; a READY AND-edge is unknown until certified, and child labels
are never multiplied into an edge's success.

Failure evidence needs a state the evidence cannot depend on: a goal without
metavariables. An exhausted hole-free state whose attempts are all known
failures, or that had no playable attempt, is worth 0; an attempt whose
hole-free child is known to fail is worth 0. Anything involving holes, a
replay conflict, pruning, infrastructure or incomplete verification is
unknown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from maths_ai.data_models.proof_components import Goal, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import (
    EdgeStatus,
    NodeClosureReason,
    ProofHypergraph,
    SearchEndReason,
)
from .pln_reward import RewardConfig, edge_shaped_reward


class BackupValidity(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BackupValue:
    value: Optional[float]
    validity: BackupValidity

    def __post_init__(self) -> None:
        if self.validity is BackupValidity.KNOWN:
            if self.value is None or not math.isfinite(self.value) or not 0.0 <= self.value <= 1.0:
                raise ValueError("KNOWN backup values must be finite numbers in [0, 1]")
        elif self.value is not None:
            raise ValueError("UNKNOWN backup values must not carry a numeric value")

    @classmethod
    def known(cls, value: float) -> "BackupValue":
        return cls(float(value), BackupValidity.KNOWN)

    @classmethod
    def unknown(cls) -> "BackupValue":
        return cls(None, BackupValidity.UNKNOWN)

    @property
    def is_known(self) -> bool:
        return self.validity is BackupValidity.KNOWN


@dataclass(frozen=True)
class BackupTables:
    edge_outcomes: dict[int, BackupValue]
    node_targets: dict[int, BackupValue]


@dataclass(frozen=True)
class HarvestConfig:
    """Harvest options. Child labels are never combined into an edge's
    success, so there is no AND-combination mode."""


@dataclass(frozen=True)
class ActorTransition:
    node_id: int
    goal: Goal
    tactic: TacticCandidate
    reward: float
    successor_value: float
    return_: float
    edge_id: int = -1


@dataclass(frozen=True)
class CriticSample:
    node_id: int
    goal: Goal
    target: float


_INCOMING_EDGE_FAILURE_REASONS = frozenset({
    NodeClosureReason.DEPTH_LIMIT,
    NodeClosureReason.CYCLE,
})
_STATE_LABEL_REASONS = frozenset({
    NodeClosureReason.CANDIDATES_EXHAUSTED,
    NodeClosureReason.NO_CANDIDATES,
})


def compute_backups(graph: ProofHypergraph, cfg: HarvestConfig | None = None) -> BackupTables:
    """Compute separate edge outcomes and unique state-level critic targets."""
    node_memo: dict[int, BackupValue] = {}
    edge_memo: dict[int, BackupValue] = {}
    in_progress: set[int] = set()
    witnessed = graph.witnessed_edges()
    witnessed_nodes = {graph.edges[edge_id].source_id for edge_id in witnessed}

    def standalone_node(node_id: int) -> BackupValue:
        if node_id in node_memo:
            return node_memo[node_id]
        if node_id in in_progress:
            return BackupValue.unknown()
        node = graph.nodes[node_id]
        if node_id in witnessed_nodes:
            result = BackupValue.known(1.0)
            node_memo[node_id] = result
            return result
        if node.has_holes:
            result = BackupValue.unknown()
            node_memo[node_id] = result
            return result
        if node_id == graph.root_id and node.closure_reason == NodeClosureReason.DEPTH_LIMIT:
            result = BackupValue.known(0.0)
            node_memo[node_id] = result
            return result
        if node.closure_reason not in (NodeClosureReason.NONE, *_STATE_LABEL_REASONS):
            result = BackupValue.unknown()
            node_memo[node_id] = result
            return result
        in_progress.add(node_id)
        outcomes = [edge_outcome(edge_id) for edge_id in node.outgoing_edge_ids]
        in_progress.remove(node_id)
        if (
            node.closure_reason in _STATE_LABEL_REASONS
            and outcomes
            and all(value.is_known and value.value == 0.0 for value in outcomes)
        ) or (node.closure_reason in _STATE_LABEL_REASONS and not outcomes):
            result = BackupValue.known(0.0)
        else:
            result = BackupValue.unknown()
        node_memo[node_id] = result
        return result

    def child_failure(node_id: int) -> bool:
        node = graph.nodes[node_id]
        if node.has_holes:
            return False
        if node.closure_reason in _INCOMING_EDGE_FAILURE_REASONS:
            return True
        value = standalone_node(node_id)
        return value.is_known and value.value == 0.0

    def edge_outcome(edge_id: int) -> BackupValue:
        if edge_id in edge_memo:
            return edge_memo[edge_id]
        edge = graph.edges[edge_id]
        if edge_id in witnessed:
            result = BackupValue.known(1.0)
        elif edge.status == EdgeStatus.DEAD and not edge.child_ids:
            result = BackupValue.known(0.0)
        elif any(child_failure(child_id) for child_id in edge.child_ids):
            # A required obligation that cannot depend on other assignments fails.
            result = BackupValue.known(0.0)
        else:
            result = BackupValue.unknown()
        edge_memo[edge_id] = result
        return result

    for edge_id in graph.edges:
        edge_outcome(edge_id)
    for node_id in graph.nodes:
        standalone_node(node_id)
    return BackupTables(edge_outcomes=edge_memo, node_targets=node_memo)


def extract_actor_transitions(
    graph: ProofHypergraph,
    reward_cfg: RewardConfig | None = None,
    harvest_cfg: HarvestConfig | None = None,
    *,
    edge_ids: list[int] | None = None,
) -> list[ActorTransition]:
    """Harvest one actor row for each selected edge with valid successor evidence."""
    reward_cfg = reward_cfg or RewardConfig()
    tables = compute_backups(graph, harvest_cfg)
    chosen = edge_ids if edge_ids is not None else list(graph.edges)
    transitions: list[ActorTransition] = []
    for edge_id in chosen:
        outcome = tables.edge_outcomes[edge_id]
        if not outcome.is_known:
            continue
        edge = graph.edges[edge_id]
        reward = edge_shaped_reward(edge, graph, reward_cfg)
        successor_value = float(outcome.value)
        transitions.append(ActorTransition(
            node_id=edge.source_id,
            goal=graph.nodes[edge.source_id].goal,
            tactic=edge.tactic,
            reward=reward,
            successor_value=successor_value,
            return_=reward + reward_cfg.gamma * successor_value,
            edge_id=edge_id,
        ))
    return transitions


def extract_critic_samples(
    graph: ProofHypergraph,
    harvest_cfg: HarvestConfig | None = None,
) -> list[CriticSample]:
    """Harvest one valid critic row per unique state, including root budget failure."""
    tables = compute_backups(graph, harvest_cfg)
    samples = [
        CriticSample(node_id=node_id, goal=graph.nodes[node_id].goal, target=float(target.value))
        for node_id, target in tables.node_targets.items()
        if target.is_known
    ]
    execution = getattr(graph, "execution", {}) or {}
    if (
        graph.search_end_reason == SearchEndReason.MAX_NODES
        and not graph.is_solved()
        # A ready root whose recipes did not certify is unresolved, not a failure.
        and not graph.has_candidate()
        and not graph.root.has_holes
        and execution.get("last_replay") is None
        and not any(
            node.closure_reason in {
                NodeClosureReason.ELABORATION_ERROR,
                NodeClosureReason.INFRASTRUCTURE_FAILURE,
                NodeClosureReason.RESOURCE_PRUNED,
            }
            for node in graph.nodes.values()
        )
        and not any(sample.node_id == graph.root_id for sample in samples)
    ):
        samples.append(CriticSample(node_id=graph.root_id, goal=graph.root.goal, target=0.0))
    return samples
