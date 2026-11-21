"""Pre-accumulation full-batch objective references used only by equivalence tests."""

from __future__ import annotations
import torch
from torch_geometric.data import Batch
from maths_ai.gnn_inference.atp_lean_gnn.actor_critic_loss import (
    compute_bc_anchor_loss,
    compute_critic_loss,
    compute_entropy_bonus,
)
from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    _check_action,
    ObservationOutcome,
)


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
    for source in sorted(
        {sample.source for sample in critic_samples}, key=lambda item: item.value
    ):
        metrics[f"critic_source_{source.value}"] = float(
            sum(sample.source == source for sample in critic_samples)
        )
    return total, metrics


def _candidate_budget(
    model, pools, *, max_candidates: int, max_bytes: int
) -> tuple[int, int]:
    if any(
        type(limit) is not int or limit < 1 for limit in (max_candidates, max_bytes)
    ):
        raise ValueError("Candidate count and byte limits must be positive integers.")
    count = sum(len(p.local_ids) + len(p.lemma_ids) for p in pools)
    size = sum(
        (
            len(p.local_ids) * model.hidden_dim
            + len(p.lemma_ids) * model.decoder.text_dim
        )
        * 4
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
        if observation.outcome is ObservationOutcome.RESOURCE_LIMIT:
            continue
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
        _check_action(
            model,
            goal,
            data,
            action,
            f"{observation.outcome.value.capitalize()} action",
        )
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
            model,
            candidate_pools,
            max_candidates=max_update_candidates,
            max_bytes=max_update_candidate_bytes,
        )
        batch = Batch.from_data_list(actor_datas).to(device)
        tactic_ids_t = torch.tensor(tactic_ids, dtype=torch.long, device=device)
        evaluated = model.evaluate_actions(
            batch, tactic_ids_t, argument_rows, candidate_pools=candidate_pools
        )
        arg_logp, values, tactic_logits = (
            evaluated.arg_logp,
            evaluated.value,
            evaluated.tactic_logits,
        )
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
        entropy_bonus = (
            -((joint_logp.detach() + 1) * joint_logp * mult_t).sum() / mult_t.sum()
        )
        sampled_entropy = float(
            (-(joint_logp.detach() * mult_t).sum() / mult_t.sum()).item()
        )
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
    for source in sorted(
        {sample.source for sample in critic_samples}, key=lambda item: item.value
    ):
        metrics[f"critic_source_{source.value}"] = float(
            sum(sample.source == source for sample in critic_samples)
        )
    return total, metrics
