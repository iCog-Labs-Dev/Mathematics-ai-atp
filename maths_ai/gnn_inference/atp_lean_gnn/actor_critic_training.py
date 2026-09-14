from __future__ import annotations

import time
import torch
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from .actor_critic import ActorCriticWithArgsClassifier
from .actor_critic_loss import compute_actor_critic_combined_loss
from .reward import RewardSource
from .reporting import console_print
from .training_safety import require_finite_loss


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def _should_log_batch(batch_index: int, total_batches: int, *, log_every_batches: int) -> bool:
    return (
        batch_index == 1
        or batch_index == total_batches
        or batch_index % log_every_batches == 0
    )


def _extract_tactic_names(batch) -> list[str]:
    if hasattr(batch, "tactic_name"):
        names = batch.tactic_name
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
        return [str(names)]
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    return [""] * batch_size


def _extract_tactic_mask(batch, batch_size, num_tactics, device):
    """Return a [B, T] bool legal-tactic mask from ``batch.legal_action_mask``, or None.

    Returns None when the field is absent (the current pre-env default), which leaves
    the actor and entropy unmasked — behavior identical to an all-ones mask. When
    present, guards against a fully-masked row (which would make the tactic softmax
    NaN) by falling back to all-legal for that row.
    """
    if not hasattr(batch, "legal_action_mask"):
        return None
    mask = batch.legal_action_mask
    if not torch.is_tensor(mask):
        return None
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.numel() != batch_size * num_tactics:
        return None
    mask = mask.view(batch_size, num_tactics)
    all_masked = ~mask.any(dim=1)
    if all_masked.any():
        mask = mask.clone()
        mask[all_masked] = True
    return mask


def build_param_groups(
    model: ActorCriticWithArgsClassifier,
    base_lr: float,
    arg_lr_multiplier: float = 0.1,
) -> list[dict]:
    """Create optimizer parameter groups with differential learning rates."""
    arg_param_ids = set(
        id(p) for p in list(model.argument_selector.parameters())
                      + list(model.tactic_embedding.parameters())
    )
    return [
        {
            "params": [p for p in model.parameters() if id(p) not in arg_param_ids],
            "lr": base_lr,
        },
        {
            "params": [p for p in model.parameters() if id(p) in arg_param_ids],
            "lr": base_lr * arg_lr_multiplier,
        },
    ]


def train_one_epoch_actor_critic(
    model: ActorCriticWithArgsClassifier,
    loader: DataLoader,
    *,
    reward_source: RewardSource,
    optimizer: Optimizer,
    grad_scaler,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    total_epochs: int,
    log_every_batches: int,
    use_amp: bool,
    pin_memory: bool,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    arg_loss_weight: float = 0.5,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Train one epoch of Actor-Critic RL."""
    model.train()
    total_metrics = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "arg_loss": 0.0,
        "total_loss": 0.0,
    }
    total_examples = 0
    total_batches = len(loader)
    start_time = time.perf_counter()

    console_print(
        f"  Starting epoch {epoch:02d}/{total_epochs:02d} "
        f"with {total_batches} train batches (Actor-Critic RL)..."
    )

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=(device.type == "cuda" and pin_memory))
        targets = batch.y.view(-1)
        tactic_names = _extract_tactic_names(batch)

        num_tactics = model.actor.base.out_features
        tactic_mask = _extract_tactic_mask(batch, int(targets.numel()), num_tactics, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            # Single GNN forward: encode once, sample the tactic, then run only the
            # pointer on the already-computed encoder outputs.
            node_embeddings, state_emb, tactic_logits, values = model.encode(
                batch, tactic_mask=tactic_mask
            )

            # Sample tactic actions
            tactic_dist = torch.distributions.Categorical(logits=tactic_logits)
            actions = tactic_dist.sample()

            # Argument logits conditioned on the sampled actions (no second GNN pass)
            arg_logits_list = model.select_arguments(
                batch, node_embeddings, state_emb, actions, tactic_names=tactic_names
            )

            # Select argument indices via argmax
            selected_arg_indices = []
            for arg_logits_k in arg_logits_list:
                selected_arg_indices.append(arg_logits_k.argmax(dim=1))

            # Simulate Lean outcomes: success if actions match ground truth targets
            success_mask = (actions == targets)
            returns = reward_source.get_rewards_batch(actions, targets, success_mask)

            loss, batch_metrics = compute_actor_critic_combined_loss(
                tactic_logits=tactic_logits,
                value_estimates=values,
                arg_logits_list=arg_logits_list,
                actions=actions,
                returns=returns,
                selected_arg_indices=selected_arg_indices,
                success_mask=success_mask,
                tactic_mask=tactic_mask,
                critic_weight=critic_weight,
                entropy_weight=entropy_weight,
                arg_loss_weight=arg_loss_weight,
            )

        require_finite_loss(
            loss,
            architecture=model.model_spec.architecture,
            amp_dtype=amp_dtype,
            epoch=epoch,
            batch_index=batch_index,
            components=batch_metrics,
        )
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        batch_size = int(targets.numel())
        for k in total_metrics:
            total_metrics[k] += batch_metrics[k] * batch_size
        total_examples += batch_size

        if _should_log_batch(batch_index, total_batches, log_every_batches=log_every_batches):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            console_print(
                f"    train batch {batch_index:>5}/{total_batches} | "
                f"seen={total_examples} | "
                f"tot_loss={total_metrics['total_loss'] / max(total_examples, 1):.4f} | "
                f"act_loss={total_metrics['actor_loss'] / max(total_examples, 1):.4f} | "
                f"crt_loss={total_metrics['critic_loss'] / max(total_examples, 1):.4f} | "
                f"elapsed={elapsed}"
            )

    n = max(total_examples, 1)
    return {k: v / n for k, v in total_metrics.items()}


@torch.no_grad()
def evaluate_model_actor_critic(
    model: ActorCriticWithArgsClassifier,
    loader: DataLoader,
    *,
    reward_source: RewardSource,
    device: torch.device,
    split_name: str | None = None,
    log_every_batches: int | None = None,
    use_amp: bool = False,
    pin_memory: bool = False,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    arg_loss_weight: float = 0.5,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Evaluate Actor-Critic model."""
    model.eval()
    total_metrics = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "arg_loss": 0.0,
        "total_loss": 0.0,
    }
    top1_correct = 0
    total_examples = 0
    total_batches = len(loader)
    start_time = time.perf_counter()

    if split_name is not None:
        console_print(f"  Evaluating {split_name} split ({total_batches} batches, Actor-Critic)...")

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=(device.type == "cuda" and pin_memory))
        targets = batch.y.view(-1)
        tactic_names = _extract_tactic_names(batch)

        num_tactics = model.actor.base.out_features
        tactic_mask = _extract_tactic_mask(batch, int(targets.numel()), num_tactics, device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            # During evaluation, actions are chosen greedily via argmax.
            node_embeddings, state_emb, tactic_logits, values = model.encode(
                batch, tactic_mask=tactic_mask
            )
            actions = tactic_logits.argmax(dim=1)

            arg_logits_list = model.select_arguments(
                batch, node_embeddings, state_emb, actions, tactic_names=tactic_names
            )

            selected_arg_indices = []
            for arg_logits_k in arg_logits_list:
                selected_arg_indices.append(arg_logits_k.argmax(dim=1))

            success_mask = (actions == targets)
            returns = reward_source.get_rewards_batch(actions, targets, success_mask)

            _, batch_metrics = compute_actor_critic_combined_loss(
                tactic_logits=tactic_logits,
                value_estimates=values,
                arg_logits_list=arg_logits_list,
                actions=actions,
                returns=returns,
                selected_arg_indices=selected_arg_indices,
                success_mask=success_mask,
                tactic_mask=tactic_mask,
                critic_weight=critic_weight,
                entropy_weight=entropy_weight,
                arg_loss_weight=arg_loss_weight,
            )

        top1_correct += int((actions == targets).sum().item())
        batch_size = int(targets.numel())
        for k in total_metrics:
            total_metrics[k] += batch_metrics[k] * batch_size
        total_examples += batch_size

        if (
            split_name is not None
            and log_every_batches is not None
            and _should_log_batch(batch_index, total_batches, log_every_batches=log_every_batches)
        ):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            console_print(
                f"    {split_name} batch {batch_index:>5}/{total_batches} | "
                f"seen={total_examples} | "
                f"elapsed={elapsed}"
            )

    n = max(total_examples, 1)
    metrics = {k: v / n for k, v in total_metrics.items()}
    metrics["tactic_accuracy"] = top1_correct / n
    return metrics
