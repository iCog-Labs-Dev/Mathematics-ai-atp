from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_actor_loss(
    tactic_logits: Tensor,  # [B, num_tactics]
    actions: Tensor,        # [B] sampled tactic ids
    advantages: Tensor,     # [B] detached advantages
) -> Tensor:
    log_probs = F.log_softmax(tactic_logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    return -(selected_log_probs * advantages.detach()).mean()


def compute_critic_loss(
    predicted_values: Tensor,  # [B, 1]
    returns: Tensor,           # [B]
) -> Tensor:
    return F.mse_loss(predicted_values.squeeze(-1), returns)


def compute_entropy_bonus(
    tactic_logits: Tensor,           # [B, num_tactics]
    mask: Tensor | None = None,      # [B, num_tactics] bool
) -> Tensor:
    if mask is not None:
        tactic_logits = tactic_logits.masked_fill(~mask, float("-inf"))
    probs = F.softmax(tactic_logits, dim=-1)
    log_probs = F.log_softmax(tactic_logits, dim=-1)
    # Compute entropy = -sum(p * log(p)) over valid classes
    safe_log_probs = torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))
    entropy = -(probs * safe_log_probs).sum(dim=-1)
    return entropy.mean()


def compute_argument_rl_loss(
    arg_logits_list: list[Tensor],       # per-step arg logits, each [B, N_max]
    selected_arg_indices: list[Tensor],  # per-step selected arg positions, each [B]
    success_mask: Tensor,                # [B] bool — did the (tactic, args) succeed in Lean?
) -> Tensor:
    if not arg_logits_list or not selected_arg_indices:
        return torch.tensor(0.0, device=success_mask.device)

    if not success_mask.any():
        # No successful samples → no argument gradient.
        # Return a zero tensor attached to the computation graph (for autograd),
        # but avoid the `-inf * 0.0 = nan` trap from masked logits.
        device = arg_logits_list[0].device if arg_logits_list else success_mask.device
        return torch.tensor(0.0, device=device, requires_grad=True)

    losses = []
    success_indices = success_mask.nonzero(as_tuple=False).view(-1)

    for step_k, arg_logits_k in enumerate(arg_logits_list):
        if step_k >= len(selected_arg_indices):
            break

        target_indices_k = selected_arg_indices[step_k]
        logits_successful = arg_logits_k[success_indices]
        targets_successful = target_indices_k[success_indices]

        valid = targets_successful >= 0
        for idx in range(logits_successful.size(0)):
            if valid[idx] and torch.isneginf(logits_successful[idx, targets_successful[idx]]):
                valid[idx] = False

        if not valid.any():
            continue

        step_loss = F.cross_entropy(
            logits_successful[valid].clamp(min=-1e4),
            targets_successful[valid]
        )
        losses.append(step_loss)

    if losses:
        return torch.stack(losses).mean()
    else:
        # All success samples had invalid/masked indices → no valid argument gradient.
        device = arg_logits_list[0].device if arg_logits_list else success_mask.device
        return torch.tensor(0.0, device=device, requires_grad=True)


def compute_bc_anchor_loss(
    tactic_logits: Tensor,          # [B, num_tactics]
    labels: Tensor,                 # [B] ground-truth tactic ids; -1 marks "no label"
) -> Tensor:
    """Behavioral-cloning anchor: supervised cross-entropy toward the corpus label.

    A dense, low-variance gradient that keeps the policy near supervised competence while
    the PLN reward is near-constant and terminals are rare. Injected as ``−log π(label|s)``
    (exact target, no sampling), NOT as a reward — so the value target stays uncontaminated.
    Examples labeled ``-1`` (no ground-truth tactic, e.g. a rollout-only state) are ignored.
    """
    valid = labels >= 0
    if not valid.any():
        return tactic_logits.sum() * 0.0
    return F.cross_entropy(tactic_logits[valid], labels[valid])


def compute_actor_critic_combined_loss(
    tactic_logits: Tensor,
    value_estimates: Tensor,
    arg_logits_list: list[Tensor],
    actions: Tensor,
    returns: Tensor,
    selected_arg_indices: list[Tensor],
    success_mask: Tensor,
    *,
    tactic_mask: Tensor | None = None,
    labels: Tensor | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    arg_loss_weight: float = 0.5,
    bc_weight: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    # Raw advantage (detached — a fixed credit weight, not a differentiable function).
    raw_advantages = returns - value_estimates.squeeze(-1).detach()
    # Normalize per batch for the actor only, to bound policy-gradient variance. The
    # critic still regresses the unnormalized returns below. The +1e-8 guards the
    # zero-variance / single-example case.
    advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)

    actor_loss = compute_actor_loss(tactic_logits, actions, advantages)
    critic_loss = compute_critic_loss(value_estimates, returns)
    entropy = compute_entropy_bonus(tactic_logits, mask=tactic_mask)
    arg_loss = compute_argument_rl_loss(arg_logits_list, selected_arg_indices, success_mask)

    # Behavioral-cloning anchor (annealed via bc_weight); zero-weight ⇒ pure RL loss.
    if bc_weight != 0.0 and labels is not None:
        bc_loss = compute_bc_anchor_loss(tactic_logits, labels)
    else:
        bc_loss = torch.zeros((), device=tactic_logits.device, dtype=tactic_logits.dtype)

    total = (
        actor_loss
        + critic_weight * critic_loss
        - entropy_weight * entropy
        + arg_loss_weight * arg_loss
        + bc_weight * bc_loss
    )

    metrics = {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(entropy.item()),
        "arg_loss": float(arg_loss.item()),
        "bc_loss": float(bc_loss.item()),
        "total_loss": float(total.item()),
        "mean_advantage": float(raw_advantages.mean().item()),
        "mean_value": float(value_estimates.mean().item()),
        "mean_return": float(returns.mean().item()),
    }
    return total, metrics
