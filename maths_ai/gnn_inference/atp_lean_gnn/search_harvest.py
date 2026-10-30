"""Validity-aware actor and critic targets from a finished proof hypergraph.

The actor consumes one outcome per executed tactic edge. The critic consumes one target
per proof-state node. Unknown evidence is represented by ``BackupValue.unknown()`` and is
never converted into a numeric training label.

Positive evidence comes only from a kernel-certified replay: an attempt on the
certified witness whose replayed observation and command matched the ones it was
sampled with is worth 1, and so is its source state. Candidate readiness is not
evidence; a READY or JOINT attempt the witness does not contain is unknown.

Failure evidence needs a state the evidence cannot depend on: a goal without
metavariables. A hole-free child known to fail makes its attempt fail; a state
whose own search closed with every attempt failing is worth 0. Anything involving
holes, pruning, an unobservable goal or infrastructure is unknown, and so is a
visit-mean estimate over a subtree that contains any of these or an uncertified
candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from maths_ai.data_models.proof_components import Goal, TacticCandidate
from maths_ai.hybrid_reasoner.hypergraph import (
    BackupSource,
    BackupValidity,
    BackupValue,
    EdgeStatus,
    UNOBSERVED_CLOSURES,
    NodeClosureReason,
    NodeStatus,
    ProofHypergraph,
    SearchEndReason,
)

from .pln_reward import RewardConfig, edge_shaped_reward
from .graph import dag_fingerprint, model_goal_to_dag

if TYPE_CHECKING:
    from .pln_rl_training import EdgeAction, PolicyObservation
    from .policy_actions import ArgumentReference, CandidatePoolSnapshot


@dataclass(frozen=True)
class HarvestConfig:
    and_combine: str = "product"

    def __post_init__(self) -> None:
        if self.and_combine not in ("product", "min"):
            raise ValueError("and_combine must be 'product' or 'min'.")


@dataclass(frozen=True)
class BackupTables:
    """Learning evidence separated by its architectural consumer."""

    edge_outcomes: dict[int, BackupValue]
    node_targets: dict[int, BackupValue]


@dataclass(frozen=True)
class CriticSample:
    """One unique proof-state target ``y(s)`` with evidence provenance."""

    node_id: int
    goal: Goal
    graph_fingerprint: str
    target: float
    source: BackupSource


@dataclass(frozen=True)
class TacticImitationSample:
    """One complete policy action on the kernel-certified proof hypertree, with
    the candidate pool its argument references point into."""

    goal: Goal
    graph_fingerprint: str
    tactic_id: int
    candidate_pool: "CandidatePoolSnapshot"
    arguments: tuple["ArgumentReference", ...] = ()


@dataclass
class ActorTransition:
    """One valid executed-edge outcome used by the score-function estimator."""

    node_id: int
    goal: Goal
    tactic: TacticCandidate
    reward: float
    successor_value: float
    return_: float
    edge_id: int = -1


def _and_numeric(values: list[float], cfg: HarvestConfig) -> float:
    if not values:
        return 1.0
    if cfg.and_combine == "min":
        return min(values)
    product = 1.0
    for value in values:
        product *= value
    return product


def _combined_source(backups: list[BackupValue]) -> BackupSource:
    sources = {backup.source for backup in backups}
    for source in (
        BackupSource.CRITIC_BOOTSTRAP,
        BackupSource.VISIT_MEAN,
        BackupSource.ROOT_EPISODE,
        BackupSource.SEARCH_FAILURE,
        BackupSource.LEAN_STATUS,
    ):
        if source in sources:
            return source
    raise AssertionError("A valid backup must have a source.")


def _and_backup(backups: list[BackupValue], cfg: HarvestConfig) -> BackupValue:
    if not backups:
        return BackupValue.valid(1.0, BackupSource.LEAN_STATUS)
    if any(
        backup.validity == BackupValidity.VALID and backup.value == 0.0
        for backup in backups
    ):
        return BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE)
    if any(backup.validity == BackupValidity.UNKNOWN for backup in backups):
        return BackupValue.unknown()
    values = [backup.value for backup in backups]
    assert all(value is not None for value in values)
    return BackupValue.valid(
        _and_numeric([float(value) for value in values], cfg),
        _combined_source(backups),
    )


def _dead_path_value(closure_reason: NodeClosureReason) -> BackupValue:
    if closure_reason in UNOBSERVED_CLOSURES:
        return BackupValue.unknown()
    return BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE)


def _soft_target(graph: ProofHypergraph, node_id: int, visit_threshold: int) -> BackupValue:
    node = graph.nodes[node_id]
    supported = [
        graph.edges[edge_id]
        for edge_id in node.outgoing_edge_ids
        if graph.edges[edge_id].status != EdgeStatus.DEAD
        and graph.edges[edge_id].visit_stats.N_v >= visit_threshold
    ]
    if not supported:
        return BackupValue.unknown()
    return BackupValue.valid(
        max(edge.visit_stats.Q for edge in supported),
        BackupSource.VISIT_MEAN,
    )


def compute_backups(
    graph: ProofHypergraph,
    cfg: HarvestConfig | None = None,
    *,
    visit_threshold: int | None = None,
) -> BackupTables:
    """Compute validity-aware edge outcomes and unique node targets.

    A depth-limit or cycle closure is a valid failure of its incoming tactic path, but it
    is not a standalone target for the same state launched with a fresh root budget.
    """

    cfg = cfg or HarvestConfig()
    path_memo: dict[int, BackupValue] = {}
    edge_outcomes: dict[int, BackupValue] = {}
    in_progress: set[int] = set()
    witnessed = graph.witnessed_edges()
    witnessed_nodes = {graph.edges[edge_id].source_id for edge_id in witnessed}

    def soft_target_blocked(node_id: int, seen: set[int] | None = None) -> bool:
        """Whether the subtree's visit means rest on anything but observed outcomes."""
        seen = set() if seen is None else seen
        if node_id in seen:
            return False
        seen.add(node_id)
        node = graph.nodes[node_id]
        if node.closure_reason in UNOBSERVED_CLOSURES or node.has_holes:
            return True
        for edge_id in node.outgoing_edge_ids:
            edge = graph.edges[edge_id]
            if edge.status in (EdgeStatus.READY, EdgeStatus.JOINT) and edge_id not in witnessed:
                return True
            if any(soft_target_blocked(child_id, seen) for child_id in edge.child_ids):
                return True
        return False

    def child_failed(child_id: int) -> bool:
        if graph.nodes[child_id].has_holes:
            return False
        value = path_value(child_id)
        return value.validity == BackupValidity.VALID and value.value == 0.0

    def edge_value(edge_id: int) -> BackupValue:
        if edge_id in edge_outcomes:
            return edge_outcomes[edge_id]
        edge = graph.edges[edge_id]
        if edge_id in witnessed:
            outcome = BackupValue.valid(1.0, BackupSource.LEAN_STATUS)
        elif not edge.child_ids:
            outcome = (
                BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE)
                if edge.status == EdgeStatus.DEAD
                else BackupValue.unknown()
            )
        elif any(child_failed(child_id) for child_id in edge.child_ids):
            # A required obligation that cannot depend on other assignments fails.
            outcome = BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE)
        else:
            outcome = _and_backup([path_value(child_id) for child_id in edge.child_ids], cfg)
        edge_outcomes[edge_id] = outcome
        return outcome

    def path_value(node_id: int) -> BackupValue:
        if node_id in path_memo:
            return path_memo[node_id]
        if node_id in in_progress:
            return BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE)

        node = graph.nodes[node_id]
        # The node's own search closed it; pruning or an unreadable goal did not.
        searched_out = node.exhausted and node.closure_reason not in UNOBSERVED_CLOSURES
        if node_id in witnessed_nodes:
            result = BackupValue.valid(1.0, BackupSource.LEAN_STATUS)
        elif node.has_holes:
            # Its outcome can depend on assignments made elsewhere.
            result = BackupValue.unknown()
        elif node.status == NodeStatus.DEAD and not node.outgoing_edge_ids:
            result = _dead_path_value(node.closure_reason)
        elif not node.outgoing_edge_ids:
            result = BackupValue.unknown()
        else:
            in_progress.add(node_id)
            outcomes = [edge_value(edge_id) for edge_id in node.outgoing_edge_ids]
            in_progress.discard(node_id)

            estimated_successes = [
                outcome
                for outcome in outcomes
                if outcome.validity == BackupValidity.VALID and outcome.value == 1.0
            ]
            if estimated_successes:
                result = estimated_successes[0]
            elif any(outcome.validity == BackupValidity.UNKNOWN for outcome in outcomes):
                result = BackupValue.unknown()
            elif searched_out and all(outcome.value == 0.0 for outcome in outcomes):
                result = BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE)
            else:
                valid = [
                    outcome for outcome in outcomes
                    if outcome.validity == BackupValidity.VALID
                ]
                best = max(valid, key=lambda outcome: float(outcome.value)) if valid else None
                # An open proposal source can still add an alternative, so
                # its known edges failing is not a complete failure.
                if best is None or (not searched_out and best.value == 0.0):
                    result = BackupValue.unknown()
                else:
                    result = best

            if result.validity == BackupValidity.UNKNOWN and visit_threshold is not None:
                soft = _soft_target(graph, node_id, visit_threshold)
                if soft.validity == BackupValidity.VALID and not soft_target_blocked(node_id):
                    result = soft

        path_memo[node_id] = result
        return result

    for node_id in graph.nodes:
        path_value(node_id)
    for edge_id in graph.edges:
        edge_value(edge_id)

    node_targets = dict(path_memo)
    for node in graph.nodes.values():
        if node.id in witnessed_nodes:
            continue
        if node.closure_reason in (
            NodeClosureReason.DEPTH_LIMIT,
            NodeClosureReason.CYCLE,
        ) or node.closure_reason in UNOBSERVED_CLOSURES:
            node_targets[node.id] = BackupValue.unknown()

    execution = getattr(graph, "execution", {}) or {}
    clean_root_budget_failure = (
        graph.end_reason
        in (
            SearchEndReason.MAX_NODES,
            SearchEndReason.NUM_SIMULATIONS,
            SearchEndReason.DEADLINE,
        )
        and not graph.is_solved()
        # A ready root whose recipes did not certify is unresolved, not a failure.
        and not graph.has_candidate()
        and not graph.root.has_holes
        and execution.get("last_replay") is None
        and not any(
            node.closure_reason in UNOBSERVED_CLOSURES
            for node in graph.nodes.values()
        )
    )
    if clean_root_budget_failure:
        node_targets[graph.root_id] = BackupValue.valid(0.0, BackupSource.ROOT_EPISODE)

    return BackupTables(edge_outcomes=edge_outcomes, node_targets=node_targets)


def extract_critic_samples(
    graph: ProofHypergraph,
    *,
    visit_threshold: int | None = None,
    harvest_cfg: HarvestConfig | None = None,
) -> list[CriticSample]:
    tables = compute_backups(graph, harvest_cfg, visit_threshold=visit_threshold)
    samples: list[CriticSample] = []
    for node_id, backup in tables.node_targets.items():
        if backup.validity == BackupValidity.UNKNOWN:
            continue
        assert backup.value is not None
        node = graph.nodes[node_id]
        samples.append(
            CriticSample(
                node_id=node_id,
                goal=node.goal.require_model_state(),
                graph_fingerprint=dag_fingerprint(model_goal_to_dag(node.goal)),
                target=backup.value,
                source=backup.source,
            )
        )
    return samples


def extract_minimal_hypertree(
    graph: ProofHypergraph,
    observations: Sequence["PolicyObservation"],
    *,
    mine_all_solved_nodes: bool = True,
) -> list[TacticImitationSample]:
    """Mine policy actions only from the kernel-certified proof hypertree.

    The certified witness is the one Lean-confirmed hypertree a search
    produces, so root-only and all-node mining (``mine_all_solved_nodes``)
    mine the same steps. Each witnessed edge gives one sample: its first
    accepted observation (the raw action that executed it), not one per alias
    or draw. A step whose replayed observation or command differed from the
    sampled one gives no sample.
    """
    del mine_all_solved_nodes
    edge_actions: dict[int, "EdgeAction"] = {}
    for observation in observations:
        if observation.edge_id is not None:
            edge_actions.setdefault(observation.edge_id, observation.action)

    witnessed = graph.witnessed_edges()
    samples: list[TacticImitationSample] = []
    emitted: set[int] = set()
    for step in graph.witness:
        edge_id = step.attempt
        if edge_id not in witnessed or edge_id in emitted:
            continue
        emitted.add(edge_id)
        action = edge_actions.get(edge_id)
        if action is None:
            continue
        node = graph.nodes[graph.edges[edge_id].source_id]
        samples.append(
            TacticImitationSample(
                goal=node.goal.require_model_state(),
                graph_fingerprint=action.graph_fingerprint,
                tactic_id=action.tactic_id,
                candidate_pool=action.candidate_pool,
                arguments=tuple(action.arguments),
            )
        )
    return samples


def extract_transitions(
    graph: ProofHypergraph,
    reward_cfg: RewardConfig | None = None,
    harvest_cfg: HarvestConfig | None = None,
    *,
    edge_ids: list[int] | None = None,
) -> list[ActorTransition]:
    """Return actor rows only for executed edges with valid outcomes."""

    reward_cfg = reward_cfg or RewardConfig()
    harvest_cfg = harvest_cfg or HarvestConfig()
    tables = compute_backups(graph, harvest_cfg)
    chosen = edge_ids if edge_ids is not None else list(graph.edges)
    transitions: list[ActorTransition] = []
    for edge_id in chosen:
        outcome = tables.edge_outcomes[edge_id]
        if outcome.validity == BackupValidity.UNKNOWN:
            continue
        assert outcome.value is not None
        edge = graph.edges[edge_id]
        parent = graph.nodes[edge.source_id]
        reward = edge_shaped_reward(edge, graph, reward_cfg)
        transitions.append(
            ActorTransition(
                node_id=parent.id,
                goal=parent.goal,
                tactic=edge.tactic,
                reward=reward,
                successor_value=outcome.value,
                return_=reward + reward_cfg.gamma * outcome.value,
                edge_id=edge_id,
            )
        )
    return transitions
