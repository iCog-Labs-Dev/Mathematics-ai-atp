from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_actor_loss(
    tactic_logits: Tensor,  # [B, num_tactics]
    actions: Tensor,  # [B] sampled tactic ids
    advantages: Tensor,  # [B] detached advantages
) -> Tensor:
    log_probs = F.log_softmax(tactic_logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    return -(selected_log_probs * advantages.detach()).mean()


def compute_critic_loss(
    predicted_values: Tensor,  # [B, 1]
    returns: Tensor,  # [B]
) -> Tensor:
    return F.mse_loss(predicted_values.squeeze(-1), returns)


def compute_entropy_bonus(
    tactic_logits: Tensor,  # [B, num_tactics]
    mask: Tensor | None = None,  # [B, num_tactics] bool
) -> Tensor:
    if mask is not None:
        tactic_logits = tactic_logits.masked_fill(~mask, float("-inf"))
    probs = F.softmax(tactic_logits, dim=-1)
    log_probs = F.log_softmax(tactic_logits, dim=-1)
    # Compute entropy = -sum(p * log(p)) over valid classes
    safe_log_probs = torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))
    entropy = -(probs * safe_log_probs).sum(dim=-1)
    return entropy.mean()


def compute_bc_anchor_loss(
    tactic_logits: Tensor,  # [B, num_tactics]
    labels: Tensor,  # [B] ground-truth tactic ids; -1 marks "no label"
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


def compute_complete_policy_loss(
    evaluated,
    returns,
    *,
    critic_weight=0.5,
    entropy_weight=0.01,
    labels=None,
    bc_weight=0.0,
):
    returns = returns.detach()
    advantage = returns - evaluated.value.detach()
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    actor = -(advantage * evaluated.logp).mean()
    critic = F.mse_loss(evaluated.value, returns)
    entropy_loss = ((evaluated.logp.detach() + 1) * evaluated.logp).mean()
    anchor = (
        compute_bc_anchor_loss(evaluated.tactic_logits, labels)
        if labels is not None and bc_weight
        else evaluated.logp.sum() * 0
    )
    total = (
        actor
        + critic_weight * critic
        + entropy_weight * entropy_loss
        + bc_weight * anchor
    )
    return total, dict(
        bc_loss=float(anchor.detach()),
        actor_loss=float(actor.detach()),
        critic_loss=float(critic.detach()),
        entropy=float(-evaluated.logp.detach().mean()),
        arg_loss=float(-evaluated.arg_logp.detach().mean()),
        total_loss=float(total.detach()),
    )
