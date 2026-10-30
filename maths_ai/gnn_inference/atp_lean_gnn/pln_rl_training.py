"""End-to-end PLN actor-critic RL: collect search trees (async) → train (sync).

Two phases share the caller's event loop; model updates begin after collection:

  * **collect** — run the hybrid reasoner's AND-OR search (async, PLN de-blocked via
    ``evaluate_async``) to produce ``ProofHypergraph`` s for a batch of theorems.
  * **train** — harvest per-transition targets from those graphs (``search_harvest``) and take
    a synchronous actor-critic gradient step: advantage = ``return − V_pred(s)`` with the
    AND-OR-backed return, critic regressing the AND-OR backup value, plus the annealed BC
    anchor. PLN enters the reward only as the potential-based shaping term (Approach 1).

The trainer takes an injected ``featurize`` callable (``Goal -> PyG Data``) so it is testable
without a Lean/Pantograph backend. ``make_featurizer`` builds the strict normalized model
S-expression path used by live collection and both optimizer paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

import torch
from torch_geometric.data import Batch, Data

from .actor_critic import ActorCriticWithArgsClassifier
from .actor_critic_loss import compute_bc_anchor_loss, compute_critic_loss, compute_entropy_bonus
from .policy_actions import ArgumentReference, CandidatePoolSnapshot, fingerprint
from .retrieval_artifacts import goal_query
from .graph import dag_fingerprint, model_goal_to_dag
from .pyg import build_premise_mask, dag_to_pyg
from .pln_reward import RewardConfig
from .search_harvest import (
    ActorTransition,
    CriticSample,
    HarvestConfig,
    TacticImitationSample,
    extract_critic_samples,
    extract_transitions,
)

from maths_ai.data_models.proof_components import Goal
from maths_ai.hybrid_reasoner.hypergraph import UnsupportedObservation
from maths_ai.lean_replay import LeanGoalSeed


_STRUCTURAL_LABELS = {
    "State",
    "Goal",
    "Hyp",
    ":app",
    ":arg",
    ":forall",
    ":instance-of",
    ":lambda",
    ":let",
    ":lit",
    ":mdata",
    ":metavar",
    ":proj",
    ":proof-of",
}
_FV_RE = re.compile(r"^FV\d+$")


def _is_structural_label(label: str) -> bool:
    return (
        label in _STRUCTURAL_LABELS
        or label.startswith(":")
        or label.startswith("HypRole:")
        or label.startswith("BinderRole:")
        or label.startswith("ArgRole:")
        or label.startswith("ArgPosition:")
        or _FV_RE.fullmatch(label) is not None
    )


# Roles the model serializer emits for binders, application arguments, and
# context locals (``instance``/``let`` refine the local's binder role).
_BINDER_ROLES = (":explicit", ":implicit", ":strict-implicit", ":instance-implicit")
_ARG_ROLES = (":explicit", ":implicit", ":implicit-type", ":instance", ":proof")
_HYP_ROLES = (*(role.removeprefix(":") for role in _BINDER_ROLES), "instance", "let")


def supported_structural_labels(*, max_locals: int, max_arity: int) -> list[str]:
    """Every structural label a state within these bounds can contain.

    A vocabulary built from a seed goal alone lacks labels its successors
    introduce, such as ``FV1`` after ``intro``. Vocabularies for synthetic runs
    include this set; states beyond the bounds still fail the missing-label check.
    """
    if max_locals < 0 or max_arity < 0:
        raise ValueError("Structural label bounds must be nonnegative.")
    labels = [
        *sorted(_STRUCTURAL_LABELS),
        *(f"BinderRole:{role}" for role in _BINDER_ROLES),
        *(f"ArgRole:{role}" for role in _ARG_ROLES),
        *(f"ArgPosition:{position}" for position in range(max_arity)),
        *(f"HypRole:{role}" for role in _HYP_ROLES),
        *(f"FV{index}" for index in range(max_locals)),
    ]
    assert all(_is_structural_label(label) for label in labels)
    return labels


def _oov_metrics(datas: list[Data]) -> dict[str, float]:
    return {
        "unknown_label_count": float(
            sum(int(data.unknown_label_count) for data in datas)
        ),
        "structural_unknown_label_count": float(
            sum(int(data.structural_unknown_label_count) for data in datas)
        ),
        "semantic_unknown_label_count": float(
            sum(int(data.semantic_unknown_label_count) for data in datas)
        ),
        "total_label_count": float(sum(int(data.total_label_count) for data in datas)),
    }


def _require_graph_fingerprint(data: Data, expected: str, owner: str) -> None:
    actual = str(data.graph_fingerprint)
    if actual != expected:
        raise ValueError(
            f"{owner} graph changed between collection and training: "
            f"expected {expected}, got {actual}."
        )


def _validate_update_size(
    datas: list[Data],
    *,
    max_update_nodes: int,
    max_update_edges: int,
) -> tuple[int, int]:
    if max_update_nodes < 0 or max_update_edges < 0:
        raise ValueError("RL update node and edge budgets cannot be negative.")
    node_count = sum(int(data.num_nodes) for data in datas)
    edge_count = sum(int(data.edge_index.size(1)) for data in datas)
    if (max_update_nodes and node_count > max_update_nodes) or (
        max_update_edges and edge_count > max_update_edges
    ):
        raise ValueError(
            "Collected RL update exceeds its graph budget: "
            f"graphs={len(datas)}, nodes={node_count}, edges={edge_count}, "
            f"max_nodes={max_update_nodes}, max_edges={max_update_edges}."
        )
    return node_count, edge_count


@dataclass(frozen=True)
class EdgeAction:
    """The integer record of one sampled action — everything the train phase needs to
    recompute its log-prob under the current parameters (no autograd graph is stashed).

    ``arguments`` are the decoder's typed references (a local node or a library
    lemma), resolved against ``candidate_pool``, the immutable support the action
    was drawn from; the update re-evaluates them against that same pool.

    ``multiplicity`` is how many of the k i.i.d. draws produced this same action: the
    action was applied to Lean once (dedup for the executor) but its policy-gradient term
    must be weighted by ``m`` or high-probability actions are under-weighted.
    """
    tactic_id: int
    candidate_pool: CandidatePoolSnapshot
    graph_fingerprint: str
    arguments: tuple[ArgumentReference, ...] = ()
    multiplicity: int = 1

    def __post_init__(self) -> None:
        if not self.graph_fingerprint:
            raise ValueError("EdgeAction requires a graph fingerprint.")

    @property
    def raw_action_id(self) -> tuple:
        """The concrete choice: tactic id and each argument's source and identity."""
        return (self.tactic_id, tuple((ref.source, ref.identity) for ref in self.arguments))


class ObservationOutcome(str, Enum):
    ACCEPTED = "accepted"          # Lean accepted the command: it has an edge
    REJECTED = "rejected"          # Lean checked the command and rejected it
    UNRENDERABLE = "unrenderable"  # the draw could not become a command


@dataclass(frozen=True)
class PolicyObservation:
    """Every draw of one raw action at one search node, with its checked outcome.

    ``action`` holds the raw tactic id and argument references, and its
    ``multiplicity`` is how many settled draws produced them, across every
    proposal round at the node. Two raw actions that render the same command
    share its one execution outcome but stay separate observations, so each
    keeps its own argument references and gradient. An accepted observation names
    its ``edge_id``; its return comes from that edge. A rejected or
    unrenderable one enters the loss actor-only, with
    ``return = terminal_failure − step_penalty`` and no critic target: the
    state may still be provable; only this action failed. ``log_prior`` is the
    proposal-time prior (None when no complete action existed); it is a record,
    and the loss recomputes log-probabilities instead of using it.
    """
    node_id: int
    goal: Goal
    action: EdgeAction
    outcome: ObservationOutcome
    first_round: int
    edge_id: int | None = None
    log_prior: float | None = None
    provider: str = ""

    def __post_init__(self) -> None:
        if (self.outcome == ObservationOutcome.ACCEPTED) != (self.edge_id is not None):
            raise ValueError("Only an accepted observation has an edge.")
        if self.outcome == ObservationOutcome.UNRENDERABLE:
            if self.log_prior is not None:
                raise ValueError("An unrenderable draw has no complete-action prior.")
        elif self.log_prior is None:
            raise ValueError("An executable observation needs its action prior.")


def make_featurizer(node_vocab: dict[str, int]) -> Callable[[Goal], Data]:
    """Build the canonical ``Goal -> PyG Data`` model-S-expression featurizer."""
    dag_featurize = make_dag_featurizer(node_vocab)

    def featurize(goal: Goal) -> Data:
        _dag, data = dag_featurize(goal)
        return data

    return featurize


def make_dag_featurizer(node_vocab: dict[str, int]):
    """Like ``make_featurizer`` but return ``(dag, data)``.

    The RL reasoner needs the ``DAGBuilder`` alongside the tensors: the pointer's sampled
    argument indices are positions in this DAG's node list, and decoding them to Lean
    argument strings (``_resolve_local_node_name``) requires the nodes themselves. Collect
    and train MUST featurize through the same path so the stored indices stay aligned.
    """
    def featurize(goal: Goal):
        dag = model_goal_to_dag(goal.require_model_state())
        data = dag_to_pyg(dag, node_vocab, add_reverse_edges=True)
        data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
        if dag.state_root_id is None:
            raise ValueError("RL DAG has no State root.")
        data.state_node_index = torch.tensor([dag.state_root_id], dtype=torch.long)
        data.graph_fingerprint = dag_fingerprint(dag)
        unknown_labels = [node.label for node in dag.nodes if node.label not in node_vocab]
        structural_unknown = sorted(
            {label for label in unknown_labels if _is_structural_label(label)}
        )
        if structural_unknown:
            raise UnsupportedObservation(
                "Prepared node vocabulary is missing required structural labels: "
                + ", ".join(structural_unknown)
            )
        data.unknown_label_count = len(unknown_labels)
        data.structural_unknown_label_count = 0
        data.semantic_unknown_label_count = sum(
            not _is_structural_label(label) for label in unknown_labels
        )
        data.structural_label_count = sum(
            _is_structural_label(node.label) for node in dag.nodes
        )
        data.semantic_label_count = int(data.num_nodes) - int(data.structural_label_count)
        data.total_label_count = int(data.num_nodes)
        return dag, data

    return featurize


def compute_transition_loss(
    model: ActorCriticWithArgsClassifier,
    transitions: list[ActorTransition],
    critic_samples: list[CriticSample],
    featurize: Callable[[Goal], Data],
    tactic_to_id: dict[str, int],
    *,
    device: torch.device | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    bc_weight: float = 0.0,
    max_update_nodes: int = 0,
    max_update_edges: int = 0,
) -> tuple[torch.Tensor, dict[str, float]] | None:
    """One actor-critic loss over a batch of harvested transitions (tactic-level policy grad).

    Advantage ``Â(s,τ) = return − V_pred(s)`` with the AND-OR-backed ``return`` from the
    harvest; the critic regresses the separately harvested node targets. Transitions whose tactic
    is outside ``tactic_to_id`` are dropped. Returns
    ``None`` if nothing survives. The BC anchor uses each transition's own tactic id as the
    supervised label (the search-chosen action), annealed by ``bc_weight``.
    """
    device = device or torch.device("cpu")
    actor_datas: list[Data] = []
    action_ids: list[int] = []
    returns: list[float] = []
    for t in transitions:
        tid = tactic_to_id.get(t.tactic.tactic_name)
        if tid is None:
            continue
        actor_datas.append(featurize(t.goal))
        action_ids.append(tid)
        returns.append(t.return_)

    critic_datas = []
    for sample in critic_samples:
        data = featurize(sample.goal)
        _require_graph_fingerprint(data, sample.graph_fingerprint, "Critic sample")
        critic_datas.append(data)
    if not actor_datas and not critic_datas:
        return None

    node_count, edge_count = _validate_update_size(
        [*actor_datas, *critic_datas],
        max_update_nodes=max_update_nodes,
        max_update_edges=max_update_edges,
    )
    zero = next(model.parameters()).sum() * 0.0
    if actor_datas:
        actor_batch = Batch.from_data_list(actor_datas).to(device)
        _node_emb, _state_emb, tactic_logits, actor_values = model.encode(actor_batch)
        actions = torch.tensor(action_ids, dtype=torch.long, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        raw_adv = returns_t - actor_values.squeeze(-1).detach()
        advantages = (raw_adv - raw_adv.mean()) / (raw_adv.std(unbiased=False) + 1e-8)
        log_probs = torch.log_softmax(tactic_logits, dim=-1)
        selected_logp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        actor_loss = -(selected_logp * advantages).mean()
        entropy = compute_entropy_bonus(tactic_logits)
    else:
        actor_loss = zero
        entropy = zero
        returns_t = torch.empty(0, device=device)
        tactic_logits = None
        actions = None

    if critic_datas:
        critic_batch = Batch.from_data_list(critic_datas).to(device)
        _nodes, _state, _logits, critic_values = model.encode(critic_batch)
        value_targets_t = torch.tensor(
            [sample.target for sample in critic_samples],
            dtype=torch.float32,
            device=device,
        )
        critic_loss = compute_critic_loss(critic_values, value_targets_t)
    else:
        critic_loss = zero
        value_targets_t = torch.empty(0, device=device)

    total = actor_loss + critic_weight * critic_loss - entropy_weight * entropy

    if bc_weight != 0.0 and tactic_logits is not None and actions is not None:
        bc_loss = compute_bc_anchor_loss(tactic_logits, actions)
        total = total + bc_weight * bc_loss
        bc_val = float(bc_loss.item())
    else:
        bc_val = 0.0

    metrics = {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(entropy.item()),
        "bc_loss": bc_val,
        "total_loss": float(total.item()),
        "mean_return": float(returns_t.mean().item()) if returns_t.numel() else 0.0,
        "mean_value_target": (
            float(value_targets_t.mean().item()) if value_targets_t.numel() else 0.0
        ),
        "num_transitions": float(len(actor_datas)),
        "num_critic_samples": float(len(critic_samples)),
        "update_nodes": float(node_count),
        "update_edges": float(edge_count),
        **_oov_metrics([*actor_datas, *critic_datas]),
    }
    for source in sorted({sample.source for sample in critic_samples}, key=lambda item: item.value):
        metrics[f"critic_source_{source.value}"] = float(
            sum(sample.source == source for sample in critic_samples)
        )
    return total, metrics


def _check_action(goal: Goal, data: Data, action: EdgeAction, owner: str) -> None:
    _require_graph_fingerprint(data, action.graph_fingerprint, owner)
    if action.candidate_pool is None:
        raise ValueError(f"{owner} needs its immutable candidate pool.")
    if action.candidate_pool.graph_fingerprint != data.graph_fingerprint:
        raise ValueError(f"{owner}: the candidate pool's graph changed.")
    if action.candidate_pool.query_fingerprint != fingerprint(goal_query(goal)):
        raise ValueError(f"{owner}: the query text changed between collection and update.")


def _candidate_budget(model, pools, *, max_candidates: int, max_bytes: int) -> tuple[int, int]:
    count = sum(len(p.local_ids) + len(p.lemma_ids) for p in pools)
    size = sum(
        (len(p.local_ids) * model.hidden_dim + len(p.lemma_ids) * model.decoder.text_dim) * 4
        for p in pools
    )
    if count > max_candidates or size > max_bytes:
        raise ValueError("Collected update exceeds its candidate tensor budget.")
    return count, size


def compute_onpolicy_loss(
    model: ActorCriticWithArgsClassifier,
    transitions: list[ActorTransition],
    critic_samples: list[CriticSample],
    observations: list[PolicyObservation],
    featurize: Callable[[Goal], Data],
    *,
    device: torch.device | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    bc_weight: float = 0.0,
    reward_cfg: RewardConfig | None = None,
    max_update_nodes: int = 0,
    max_update_edges: int = 0,
    max_update_candidates: int = 100000,
    max_update_candidate_bytes: int = 268435456,
) -> tuple[torch.Tensor, dict[str, float]] | None:
    """On-policy actor-critic loss over complete source-aware actions.

    Recompute pattern (refinement 2): the collect phase stored only the typed
    actions and the candidate pools they were drawn from (``EdgeAction``); here
    ``model.evaluate_actions`` rebuilds the complete action's log-probability
    (tactic, argument count, each argument's source and identity) with gradient
    under the current parameters. Exactly on-policy only when no optimizer step
    ran between collect and this call.

    Per accepted observation whose edge has a harvested transition:
      ``L_actor = −m · log π(a) · Â.detach()``
    with ``Â = return − V_pred(s)`` normalized over every row with multiplicity
    weights, and ``m`` the observation's multiplicity (refinement 1). Each edge's
    transition is harvested once and shared by every raw action observed on it.
    The critic targets are supplied separately through ``critic_samples``.

    Per rejected or unrenderable observation (refinement 4): the same actor term with
    ``return = terminal_failure − step_penalty`` and NO critic target.
    """
    device = device or torch.device("cpu")
    reward_cfg = reward_cfg or RewardConfig()

    actor_datas: list[Data] = []
    tactic_ids: list[int] = []
    argument_rows: list[tuple[ArgumentReference, ...]] = []
    candidate_pools: list[CandidatePoolSnapshot] = []
    multiplicities: list[float] = []
    returns: list[float] = []
    is_success: list[bool] = []

    by_edge = {t.edge_id: t for t in transitions}
    failure_return = reward_cfg.terminal_failure - reward_cfg.step_penalty
    failures = 0
    for observation in observations:
        action = observation.action
        if observation.outcome == ObservationOutcome.ACCEPTED:
            t = by_edge.get(observation.edge_id)
            if t is None:
                continue  # the edge's return is unknown: no actor row
            goal, return_, success = t.goal, t.return_, True
        else:
            goal, return_, success = observation.goal, failure_return, False
            failures += 1
        data = featurize(goal)
        _check_action(goal, data, action, f"{observation.outcome.value.capitalize()} action")
        actor_datas.append(data)
        tactic_ids.append(action.tactic_id)
        argument_rows.append(action.arguments)
        candidate_pools.append(action.candidate_pool)
        multiplicities.append(float(action.multiplicity))
        returns.append(return_)
        is_success.append(success)

    critic_datas = []
    for sample in critic_samples:
        data = featurize(sample.goal)
        _require_graph_fingerprint(data, sample.graph_fingerprint, "Critic sample")
        critic_datas.append(data)
    if not actor_datas and not critic_datas:
        return None

    node_count, edge_count = _validate_update_size(
        [*actor_datas, *critic_datas],
        max_update_nodes=max_update_nodes,
        max_update_edges=max_update_edges,
    )
    zero = next(model.parameters()).sum() * 0.0
    policy_diagnostics: dict[str, float] = {}
    if actor_datas:
        candidate_count, candidate_bytes = _candidate_budget(
            model, candidate_pools,
            max_candidates=max_update_candidates, max_bytes=max_update_candidate_bytes,
        )
        batch = Batch.from_data_list(actor_datas).to(device)
        tactic_ids_t = torch.tensor(tactic_ids, dtype=torch.long, device=device)
        evaluated = model.evaluate_actions(
            batch, tactic_ids_t, argument_rows, candidate_pools=candidate_pools
        )
        arg_logp, values, tactic_logits = evaluated.arg_logp, evaluated.value, evaluated.tactic_logits
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        mult_t = torch.tensor(multiplicities, dtype=torch.float32, device=device)
        success_t = torch.tensor(is_success, dtype=torch.bool, device=device)
        raw_adv = returns_t - values.detach()
        mean_adv = (raw_adv * mult_t).sum() / mult_t.sum()
        variance = ((raw_adv - mean_adv).square() * mult_t).sum() / mult_t.sum()
        advantages = (raw_adv - mean_adv) / (variance.sqrt() + 1e-8)
        joint_logp = evaluated.logp
        actor_loss = -(mult_t * joint_logp * advantages).sum() / mult_t.sum()
        # Score-function surrogate of the complete action's entropy.
        entropy_bonus = -((joint_logp.detach() + 1) * joint_logp * mult_t).sum() / mult_t.sum()
        sampled_entropy = float((-(joint_logp.detach() * mult_t).sum() / mult_t.sum()).item())
        policy_diagnostics = {
            "mean_count_logp": float(evaluated.count_logp.detach().mean()),
            "mean_source_logp": float(evaluated.source_logp.detach().mean()),
            "mean_identity_logp": float(evaluated.identity_logp.detach().mean()),
            "update_candidate_count": float(candidate_count),
            "update_candidate_bytes": float(candidate_bytes),
        }
    else:
        actor_loss = zero
        entropy_bonus = zero
        sampled_entropy = 0.0
        returns_t = torch.empty(0, device=device)
        arg_logp = torch.empty(0, device=device)
        success_t = torch.empty(0, dtype=torch.bool, device=device)
        tactic_ids_t = None
        tactic_logits = None

    if critic_datas:
        critic_batch = Batch.from_data_list(critic_datas).to(device)
        _nodes, _state, _logits, critic_values = model.encode(critic_batch)
        critic_targets_t = torch.tensor(
            [sample.target for sample in critic_samples],
            dtype=torch.float32,
            device=device,
        )
        critic_loss = compute_critic_loss(critic_values, critic_targets_t)
    else:
        critic_loss = zero
    total = actor_loss + critic_weight * critic_loss - entropy_weight * entropy_bonus

    if (
        bc_weight != 0.0
        and tactic_logits is not None
        and tactic_ids_t is not None
        and success_t.any()
    ):
        # BC anchor toward the search-executed tactic on success rows only — anchoring
        # toward rejected tactics would clone the failures.
        labels = torch.where(success_t, tactic_ids_t, torch.full_like(tactic_ids_t, -1))
        bc_loss = compute_bc_anchor_loss(tactic_logits, labels)
        total = total + bc_weight * bc_loss
        bc_val = float(bc_loss.item())
    else:
        bc_val = 0.0

    metrics = {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": sampled_entropy,
        "entropy_gradient_surrogate": float(-entropy_bonus.detach()),
        **policy_diagnostics,
        "bc_loss": bc_val,
        "total_loss": float(total.item()),
        "mean_return": float(returns_t.mean().item()) if returns_t.numel() else 0.0,
        "mean_arg_logp": float(arg_logp.mean().item()) if arg_logp.numel() else 0.0,
        "num_transitions": float(int(success_t.sum().item())),
        "num_failures": float(failures),
        "num_critic_samples": float(len(critic_samples)),
        "update_nodes": float(node_count),
        "update_edges": float(edge_count),
        **_oov_metrics([*actor_datas, *critic_datas]),
    }
    for source in sorted({sample.source for sample in critic_samples}, key=lambda item: item.value):
        metrics[f"critic_source_{source.value}"] = float(
            sum(sample.source == source for sample in critic_samples)
        )
    return total, metrics


def train_step(
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    graphs: list,
    featurize: Callable[[Goal], Data],
    tactic_to_id: dict[str, int],
    *,
    reward_cfg: RewardConfig | None = None,
    harvest_cfg: HarvestConfig | None = None,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    bc_weight: float = 0.0,
    max_update_nodes: int = 0,
    max_update_edges: int = 0,
) -> dict[str, float]:
    """Harvest transitions from a batch of search graphs and take one gradient step."""
    reward_cfg = reward_cfg or RewardConfig()
    harvest_cfg = harvest_cfg or HarvestConfig()
    transitions: list[ActorTransition] = []
    critic_samples: list[CriticSample] = []
    for graph in graphs:
        transitions.extend(extract_transitions(graph, reward_cfg, harvest_cfg))
        critic_samples.extend(extract_critic_samples(graph, harvest_cfg=harvest_cfg))

    if not transitions and not critic_samples:
        return {"num_transitions": 0.0}

    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = compute_transition_loss(
        model, transitions, critic_samples, featurize, tactic_to_id,
        device=device, critic_weight=critic_weight,
        entropy_weight=entropy_weight, bc_weight=bc_weight,
        max_update_nodes=max_update_nodes, max_update_edges=max_update_edges,
    )
    if result is None:
        return {"num_transitions": 0.0}
    loss, metrics = result
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return metrics


async def collect_batch(reasoner, theorems: list[LeanGoalSeed]) -> list:
    """Run one shared reasoner sequentially over a batch of theorem seeds.

    Requires a live Pantograph client + ``petta`` binary (env-gated). A reasoner runs one search at a time;
    independent pool workers provide cross-theorem concurrency. Returns the finished
    ``ProofHypergraph`` s.
    """
    return [await reasoner.prove(seed) for seed in theorems]


async def collect_and_train(
    reasoner,
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    theorems: list[LeanGoalSeed],
    featurize: Callable[[Goal], Data],
    tactic_to_id: dict[str, int],
    **train_kwargs,
) -> dict[str, float]:
    """Await collection on the owning loop, then perform one synchronous update."""
    graphs = await collect_batch(reasoner, theorems)
    return train_step(model, optimizer, graphs, featurize, tactic_to_id, **train_kwargs)


# ---------------------------------------------------------------------------
# On-policy path (training-mode expand): RLHybridReasoner results → gradient step
# ---------------------------------------------------------------------------


def train_step_onpolicy(
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    results: list,
    featurize: Callable[[Goal], Data],
    *,
    reward_cfg: RewardConfig | None = None,
    harvest_cfg: HarvestConfig | None = None,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    bc_weight: float = 0.0,
    max_update_nodes: int = 0,
    max_update_edges: int = 0,
    max_update_candidates: int = 100000,
    max_update_candidate_bytes: int = 268435456,
) -> dict[str, float]:
    """One gradient step over a batch of ``RLSearchResult`` s (rl_reasoner.py).

    Per result: harvest only the edges the policy produced
    (``extract_transitions(edge_ids=…)``), once each, and join every accepted
    observation to its edge's transition; rejected and unrenderable observations
    enter as failure rows. Exactly ONE ``optimizer.step()`` per
    collect round — the recomputed log-probs are on-policy only while the parameters match
    those that sampled the actions.
    """
    reward_cfg = reward_cfg or RewardConfig()
    harvest_cfg = harvest_cfg or HarvestConfig()

    transitions: list[ActorTransition] = []
    critic_samples: list[CriticSample] = []
    observations: list[PolicyObservation] = []
    accepted_edges = 0
    # edge_id is unique only within one graph (refinement 6): re-key each result's
    # edges by a per-batch offset before merging.
    offset = 0
    for result in results:
        edge_ids = sorted(result.accepted_edge_ids())
        accepted_edges += len(edge_ids)
        ts = extract_transitions(result.graph, reward_cfg, harvest_cfg, edge_ids=edge_ids)
        for t in ts:
            t.edge_id += offset
        transitions.extend(ts)
        critic_samples.extend(
            extract_critic_samples(result.graph, harvest_cfg=harvest_cfg)
        )
        observations.extend(
            replace(o, edge_id=o.edge_id + offset) if o.edge_id is not None else o
            for o in result.observations
        )
        if edge_ids:
            offset += edge_ids[-1] + 1

    unknown_edges_skipped = accepted_edges - len(transitions)
    unknown_nodes_skipped = sum(len(result.graph.nodes) for result in results) - len(critic_samples)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_result = compute_onpolicy_loss(
        model, transitions, critic_samples, observations, featurize,
        device=device, critic_weight=critic_weight, entropy_weight=entropy_weight,
        bc_weight=bc_weight, reward_cfg=reward_cfg,
        max_update_nodes=max_update_nodes, max_update_edges=max_update_edges,
        max_update_candidates=max_update_candidates,
        max_update_candidate_bytes=max_update_candidate_bytes,
    )
    if loss_result is None:
        return {
            "num_transitions": 0.0,
            "num_failures": 0.0,
            "unknown_label_count": 0.0,
            "structural_unknown_label_count": 0.0,
            "semantic_unknown_label_count": 0.0,
            "total_label_count": 0.0,
            "onpolicy_optimizer_step": 0.0,
        }
    loss, metrics = loss_result
    metrics["unknown_edges_skipped"] = float(unknown_edges_skipped)
    metrics["unknown_nodes_skipped"] = float(unknown_nodes_skipped)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    metrics["onpolicy_optimizer_step"] = 1.0
    return metrics


async def collect_batch_onpolicy(reasoner, theorems: list[LeanGoalSeed]) -> list:
    """Run the RL reasoner SEQUENTIALLY over the batch and return its ``RLSearchResult`` s.

    Sequential because one reasoner instance holds one action stash at a time
    (refinement 6); PLN concurrency *within* each search still applies via
    ``evaluate_async``. Cross-theorem concurrency needs per-search reasoner instances
    and is deferred.
    """
    results = []
    for seed in theorems:
        results.append(await reasoner.prove(seed))
    return results


async def collect_and_train_onpolicy(
    reasoner,
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    theorems: list[LeanGoalSeed],
    featurize: Callable[[Goal], Data],
    **train_kwargs,
) -> dict[str, float]:
    """One on-policy round: sequential collect (async) → one gradient step (sync)."""
    results = await collect_batch_onpolicy(reasoner, theorems)
    return train_step_onpolicy(model, optimizer, results, featurize, **train_kwargs)


# ---------------------------------------------------------------------------
# Decoupled HTPS-style step (Phases 2–3): supervised regression on stored pairs
# ---------------------------------------------------------------------------


def train_step_htps_style(
    model: ActorCriticWithArgsClassifier,
    optimizer_htps: torch.optim.Optimizer,
    tactic_batch: list[TacticImitationSample],
    critic_batch: list[CriticSample],
    featurize: Callable[[Goal], Data],
    *,
    w_critic_soft: float = 0.5,
    arg_loss_weight: float = 0.5,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, float]:
    """One decoupled supervised step: tactic imitation + soft critic regression.

    ``tactic_batch`` rows are complete actions (tactic id, argument references and
    their candidate pool) mined from the certified proof hypertree
    (``extract_minimal_hypertree``); ``critic_batch`` rows are (goal, Q-or-status)
    targets (``extract_critic_samples``). Losses:

      * ``L_tactic_imitation`` — the negative log-probability of the tactic plus,
        weighted by ``arg_loss_weight``, that of its arguments (count, source and
        identity), teacher-forced through ``evaluate_actions`` against the pool the
        action was drawn from.
      * ``L_critic_soft`` — MSE of the value head against the stored targets on
        critic rows, raw scale (the advantage-normalization convention applies to the
        actor's score-function estimator only).

      ``loss = L_tactic_imitation + w_critic_soft · L_critic_soft``

    **One-step-per-collect exemption:** this function may run many times per round.
    That invariant belongs to ``compute_onpolicy_loss``'s score-function estimator,
    whose recomputed log-probs are on-policy only under unchanged parameters. Here
    every input is a stored ``(input, label)`` pair whose validity does not depend on
    which parameters generated it — supervised regression, no importance ratio to
    invalidate. ``optimizer_htps`` must be a separate optimizer instance from the
    on-policy one (separate Adam moments; the driver asserts identity inequality).
    """
    device = device or torch.device("cpu")
    if not tactic_batch and not critic_batch:
        return {"tactic_imitation_loss": 0.0, "critic_soft_loss": 0.0, "htps_total_loss": 0.0,
                "num_imitation_rows": 0.0, "num_critic_rows": 0.0,
                "htps_optimizer_step": 0.0}

    imitation_datas: list[Data] = []
    tactic_labels: list[int] = []
    argument_rows: list[tuple[ArgumentReference, ...]] = []
    pools: list[CandidatePoolSnapshot] = []
    for s in tactic_batch:
        data = featurize(s.goal)
        _require_graph_fingerprint(data, s.graph_fingerprint, "Tactic imitation sample")
        imitation_datas.append(data)
        tactic_labels.append(s.tactic_id)
        argument_rows.append(s.arguments)
        pools.append(s.candidate_pool)
    critic_datas = []
    for sample in critic_batch:
        data = featurize(sample.goal)
        _require_graph_fingerprint(data, sample.graph_fingerprint, "Critic replay sample")
        critic_datas.append(data)

    n_imitation = len(tactic_batch)
    model.train()
    optimizer_htps.zero_grad(set_to_none=True)
    zero = next(model.parameters()).sum() * 0.0
    if imitation_datas:
        imitation_batch = Batch.from_data_list(imitation_datas).to(device)
        labels_t = torch.tensor(tactic_labels, dtype=torch.long, device=device)
        evaluated = model.evaluate_actions(
            imitation_batch, labels_t, argument_rows, candidate_pools=pools
        )
        tactic_loss = -evaluated.tactic_logp.mean()
        arg_loss = -evaluated.arg_logp.mean()
    else:
        tactic_loss = zero
        arg_loss = zero

    imitation_loss = tactic_loss + arg_loss_weight * arg_loss

    if critic_datas:
        critic_batch_data = Batch.from_data_list(critic_datas).to(device)
        _critic_nodes, _critic_state, _critic_logits, critic_values = model.encode(
            critic_batch_data
        )
        targets_t = torch.tensor(
            [sample.target for sample in critic_batch],
            dtype=torch.float32,
            device=device,
        )
        critic_soft_loss = compute_critic_loss(critic_values, targets_t)
    else:
        critic_soft_loss = zero

    total = imitation_loss + w_critic_soft * critic_soft_loss
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer_htps.step()

    return {
        "tactic_imitation_loss": float(imitation_loss.item()),
        "critic_soft_loss": float(critic_soft_loss.item()),
        "htps_total_loss": float(total.item()),
        "num_imitation_rows": float(n_imitation),
        "num_critic_rows": float(len(critic_batch)),
        "htps_optimizer_step": 1.0,
        **_oov_metrics([*imitation_datas, *critic_datas]),
    }
