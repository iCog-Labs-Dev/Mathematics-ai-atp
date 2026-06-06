"""Argument-aware training and evaluation loops.

These mirror the functions in ``training.py`` but use the
``TacticWithArgsClassifier`` and the combined tactic + argument loss.
The baseline training pipeline remains completely untouched.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from .argument_selector import TacticWithArgsClassifier, compute_combined_loss
from .labels import get_tactic_arity
from .reporting import console_print


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
    """Extract per-sample tactic family names from a PyG Batch."""
    if hasattr(batch, "tactic_name"):
        names = batch.tactic_name
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
        return [str(names)]
    # Fallback: return empty strings (will use DEFAULT_ARITY)
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    return [""] * batch_size


def _extract_arg_targets(batch, max_args: int, device: torch.device) -> torch.Tensor:
    """Extract ground-truth argument node indices [B, max_args], padded with -1."""
    if not (hasattr(batch, "arg_node_indices") and hasattr(batch, "arg_count")):
        batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
        return torch.full((batch_size, max_args), -1, dtype=torch.long, device=device)

    batch_size = int(batch.y.size(0))
    all_indices = batch.arg_node_indices.to(device=device, dtype=torch.long)
    counts = batch.arg_count.tolist()

    if len(counts) != batch_size:
        return torch.full((batch_size, max_args), -1, dtype=torch.long, device=device)

    targets = torch.full((batch_size, max_args), -1, dtype=torch.long, device=device)
    split_indices = torch.split(all_indices, counts)
    ptr = batch.ptr.to(device=device)

    for i, sample_indices in enumerate(split_indices):
        n = min(len(sample_indices), max_args)
        if n > 0:
            shifted = sample_indices[:n].clone()
            valid = shifted >= 0
            shifted[valid] = shifted[valid] + ptr[i]
            targets[i, :n] = shifted

    return targets


def train_one_epoch_with_args(
    model: TacticWithArgsClassifier,
    loader: DataLoader,
    *,
    optimizer: AdamW,
    grad_scaler,
    device: torch.device,
    grad_clip: float,
    unknown_tactic_id: int,
    arg_loss_weight: float,
    epoch: int,
    total_epochs: int,
    log_every_batches: int,
    use_amp: bool,
    pin_memory: bool,
) -> dict[str, float | int]:
    """Train one epoch with combined tactic + argument loss."""
    model.train()
    total_tactic_loss = 0.0
    total_arg_loss = 0.0
    total_combined_loss = 0.0
    total_examples = 0
    total_batches = len(loader)
    start_time = time.perf_counter()

    console_print(
        f"  Starting epoch {epoch:02d}/{total_epochs:02d} "
        f"with {total_batches} train batches (arg-aware)..."
    )

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=(device.type == "cuda" and pin_memory))
        targets = batch.y.view(-1)
        tactic_names = _extract_tactic_names(batch)
        arg_targets = _extract_arg_targets(batch, model.max_args, device)
        tactic_arities = [get_tactic_arity(n) for n in tactic_names]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            tactic_logits, arg_logits_list = model(
                batch,
                teacher_tactic_ids=targets,
                tactic_names=tactic_names,
            )
            loss, metrics = compute_combined_loss(
                tactic_logits,
                arg_logits_list,
                targets,
                arg_targets,
                batch.batch,
                tactic_arity_per_sample=tactic_arities,
                arg_loss_weight=arg_loss_weight,
                unknown_tactic_id=unknown_tactic_id,
                # Diagnosis fields
                node_labels=batch.x,
                node_types=batch.node_type
            )

        grad_scaler.scale(loss).backward()
        if torch.isfinite(loss):
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            print(f"  WARNING: Skipping batch {batch_index} due to non-finite loss: {loss.item()}")
            # Enhanced debug info
            with torch.no_grad():
                print(f"    tactic_loss={metrics['tactic_loss']:.4f}, arg_loss={metrics['arg_loss']:.4f}")
                # Check for NaNs in gradients/weights if needed, but here we just skip the step
            optimizer.zero_grad(set_to_none=True)

        batch_size = int(targets.numel())
        total_tactic_loss += metrics["tactic_loss"] * batch_size
        total_arg_loss += metrics["arg_loss"] * batch_size
        total_combined_loss += metrics["total_loss"] * batch_size
        total_examples += batch_size

        if _should_log_batch(batch_index, total_batches, log_every_batches=log_every_batches):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            console_print(
                f"    train batch {batch_index:>5}/{total_batches} | "
                f"seen={total_examples} | "
                f"tac_loss={total_tactic_loss / max(total_examples, 1):.4f} | "
                f"arg_loss={total_arg_loss / max(total_examples, 1):.4f} | "
                f"elapsed={elapsed}"
            )

    n = max(total_examples, 1)
    return {
        "tactic_loss": total_tactic_loss / n,
        "arg_loss": total_arg_loss / n,
        "combined_loss": total_combined_loss / n,
        "example_count": total_examples,
    }


@torch.no_grad()
def evaluate_model_with_args(
    model: TacticWithArgsClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    unknown_tactic_id: int,
    arg_loss_weight: float,
    split_name: str | None = None,
    log_every_batches: int | None = None,
    use_amp: bool = False,
    pin_memory: bool = False,
) -> dict[str, float | int]:
    """Evaluate model with combined metrics."""
    model.eval()
    total_tactic_loss = 0.0
    total_arg_loss = 0.0
    total_combined_loss = 0.0
    top1_correct = 0
    known_count = 0
    total_count = 0
    total_batches = len(loader)
    start_time = time.perf_counter()

    if split_name is not None:
        console_print(f"  Evaluating {split_name} split ({total_batches} batches, arg-aware)...")

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=(device.type == "cuda" and pin_memory))
        targets = batch.y.view(-1)
        tactic_names = _extract_tactic_names(batch)
        arg_targets = _extract_arg_targets(batch, model.max_args, device)
        tactic_arities = [get_tactic_arity(n) for n in tactic_names]

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            tactic_logits, arg_logits_list = model(
                batch,
                tactic_names=tactic_names,
            )
            _, metrics = compute_combined_loss(
                tactic_logits,
                arg_logits_list,
                targets,
                arg_targets,
                batch.batch,
                tactic_arity_per_sample=tactic_arities,
                arg_loss_weight=arg_loss_weight,
                unknown_tactic_id=unknown_tactic_id,
            )

        bs = int(targets.numel())
        total_tactic_loss += metrics["tactic_loss"] * bs
        total_arg_loss += metrics["arg_loss"] * bs
        total_combined_loss += metrics["total_loss"] * bs

        # Tactic top-1 accuracy (excluding UNK)
        known_mask = targets != unknown_tactic_id
        kc = int(known_mask.sum().item())
        if kc > 0:
            preds = tactic_logits[known_mask].argmax(dim=1)
            top1_correct += int((preds == targets[known_mask]).sum().item())
        known_count += kc
        total_count += bs

        if (
            split_name is not None
            and log_every_batches is not None
            and _should_log_batch(batch_index, total_batches, log_every_batches=log_every_batches)
        ):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            console_print(
                f"    {split_name} batch {batch_index:>5}/{total_batches} | "
                f"known={known_count} | elapsed={elapsed}"
            )

    n = max(total_count, 1)
    return {
        "tactic_loss": total_tactic_loss / n,
        "arg_loss": total_arg_loss / n,
        "combined_loss": total_combined_loss / n,
        "tactic_top1_accuracy": top1_correct / max(known_count, 1),
        "known_label_count": known_count,
        "evaluated_count": total_count,
    }
