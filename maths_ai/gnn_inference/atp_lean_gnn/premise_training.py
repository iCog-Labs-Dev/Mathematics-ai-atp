"""Premise-aware training and evaluation loops.

These extend the argument-aware loops in ``argument_training.py`` by adding
premise ranking loss from the unified candidate pool.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from .argument_selector import TacticWithArgsClassifier, compute_combined_loss
from .labels import get_tactic_arity
from .lemma_index import LemmaIndex
from .premise_pool import build_unified_pools
from .premise_scoring import PremiseScorer, compute_premise_ranking_loss
from .reporting import console_print


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def _should_log_batch(
    batch_index: int, total_batches: int, *, log_every_batches: int
) -> bool:
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
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    return [""] * batch_size


def _extract_arg_targets(
    batch, max_args: int, device: torch.device
) -> torch.Tensor:
    """Extract ground-truth argument node indices [B, max_args], padded with -1."""
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    targets = torch.full((batch_size, max_args), -1, dtype=torch.long, device=device)
    
    if hasattr(batch, "arg_node_indices") and hasattr(batch, "arg_count"):
        flat_targets = batch.arg_node_indices.to(device=device, dtype=torch.long)
        counts = batch.arg_count.tolist()
        offset = 0
        for i, count in enumerate(counts):
            n_copy = min(count, max_args)
            if n_copy > 0:
                targets[i, :n_copy] = flat_targets[offset : offset + n_copy]
            offset += count
            
    return targets


def _extract_arg_lemma_ids(
    batch, max_args: int, device: torch.device
) -> torch.Tensor:
    """Extract ground-truth lemma IDs [B, max_args], padded with -1."""
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    targets = torch.full((batch_size, max_args), -1, dtype=torch.long, device=device)
    
    if hasattr(batch, "arg_lemma_ids") and hasattr(batch, "arg_count"):
        flat_targets = batch.arg_lemma_ids.to(device=device, dtype=torch.long)
        counts = batch.arg_count.tolist()
        offset = 0
        for i, count in enumerate(counts):
            n_copy = min(count, max_args)
            if n_copy > 0:
                targets[i, :n_copy] = flat_targets[offset : offset + n_copy]
            offset += count
            
    return targets


def train_one_epoch_with_premises(
    model: TacticWithArgsClassifier,
    scorer: PremiseScorer,
    loader: DataLoader,
    lemma_index: LemmaIndex,
    *,
    optimizer: AdamW,
    grad_scaler,
    device: torch.device,
    grad_clip: float,
    unknown_tactic_id: int,
    arg_loss_weight: float,
    premise_loss_weight: float,
    k: int = 500,
    epoch: int,
    total_epochs: int,
    log_every_batches: int,
    use_amp: bool,
    pin_memory: bool,
) -> dict[str, float | int]:
    """Train one epoch with combined tactic + argument + premise ranking loss."""
    model.train()
    model.backbone.eval()  # frozen backbone stays in eval mode
    scorer.train()

    total_tactic_loss = 0.0
    total_arg_loss = 0.0
    total_premise_loss = 0.0
    total_combined_loss = 0.0
    total_examples = 0
    total_batches = len(loader)
    start_time = time.perf_counter()

    console_print(
        f"  Starting epoch {epoch:02d}/{total_epochs:02d} "
        f"with {total_batches} train batches (premise-aware)..."
    )

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(
            device, non_blocking=(device.type == "cuda" and pin_memory)
        )
        targets = batch.y.view(-1)
        tactic_names = _extract_tactic_names(batch)
        arg_targets = _extract_arg_targets(batch, model.max_args, device)
        arg_lemma_targets = _extract_arg_lemma_ids(batch, model.max_args, device)
        tactic_arities = [get_tactic_arity(n) for n in tactic_names]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Forward pass through model
            tactic_logits, arg_logits_list = model(
                batch,
                teacher_tactic_ids=targets,
                tactic_names=tactic_names,
            )

            # Combined tactic + argument loss
            ta_loss, ta_metrics = compute_combined_loss(
                tactic_logits,
                arg_logits_list,
                targets,
                arg_targets,
                batch.batch,
                tactic_arity_per_sample=tactic_arities,
                arg_loss_weight=arg_loss_weight,
                unknown_tactic_id=unknown_tactic_id,
            )

            # Recompute embeddings (detached) for the premise scoring branch
            with torch.no_grad():
                node_embeddings = model.backbone.encode_nodes(batch)
                state_emb = model.backbone.readout(node_embeddings, batch)
            node_embeddings = node_embeddings.detach()
            state_emb = state_emb.detach()

            # Get tactic embeddings
            tactic_emb = model.tactic_embedding(targets)

            # Build premise mask
            premise_mask = batch.premise_mask.to(
                dtype=torch.bool, device=device
            )

            # Build unified candidate pools
            pools = build_unified_pools(
                state_emb,
                node_embeddings,
                premise_mask,
                batch.batch,
                lemma_index=lemma_index,
                k=k,
            )

            # Score candidates
            score_list = scorer(state_emb, tactic_emb, pools)

            # Premise ranking loss
            p_loss, p_metrics = compute_premise_ranking_loss(
                score_list,
                pools,
                arg_targets,
                arg_lemma_targets,
            )

            # Total loss
            total_loss = ta_loss + premise_loss_weight * p_loss

        grad_scaler.scale(total_loss).backward()
        grad_scaler.unscale_(optimizer)
        trainable_params = [p for p in list(model.parameters()) + list(scorer.parameters()) if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        batch_size = int(targets.numel())
        total_tactic_loss += ta_metrics["tactic_loss"] * batch_size
        total_arg_loss += ta_metrics["arg_loss"] * batch_size
        total_premise_loss += p_metrics["premise_loss"] * batch_size
        total_combined_loss += float(total_loss.item()) * batch_size
        total_examples += batch_size

        if _should_log_batch(
            batch_index, total_batches, log_every_batches=log_every_batches
        ):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            n = max(total_examples, 1)
            console_print(
                f"    train batch {batch_index:>5}/{total_batches} | "
                f"seen={total_examples} | "
                f"tac={total_tactic_loss / n:.4f} "
                f"arg={total_arg_loss / n:.4f} "
                f"prem={total_premise_loss / n:.4f} | "
                f"elapsed={elapsed}"
            )

    n = max(total_examples, 1)
    return {
        "tactic_loss": total_tactic_loss / n,
        "arg_loss": total_arg_loss / n,
        "premise_loss": total_premise_loss / n,
        "combined_loss": total_combined_loss / n,
        "example_count": total_examples,
    }


@torch.no_grad()
def evaluate_model_with_premises(
    model: TacticWithArgsClassifier,
    scorer: PremiseScorer,
    loader: DataLoader,
    lemma_index: LemmaIndex,
    *,
    device: torch.device,
    unknown_tactic_id: int,
    arg_loss_weight: float,
    premise_loss_weight: float,
    k: int = 500,
    split_name: str | None = None,
    log_every_batches: int | None = None,
    use_amp: bool = False,
    pin_memory: bool = False,
) -> dict[str, float | int]:
    """Evaluate model with combined tactic + argument + premise metrics."""
    model.eval()
    scorer.eval()

    total_tactic_loss = 0.0
    total_arg_loss = 0.0
    total_premise_loss = 0.0
    total_combined_loss = 0.0
    top1_correct = 0
    known_count = 0

    premise_valid = 0
    premise_target_present = 0
    premise_top1_correct = 0
    premise_top5_correct = 0
    premise_mrr_sum = 0.0

    total_count = 0
    total_batches = len(loader)
    start_time = time.perf_counter()

    if split_name is not None:
        console_print(
            f"  Evaluating {split_name} split "
            f"({total_batches} batches, premise-aware)..."
        )

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(
            device, non_blocking=(device.type == "cuda" and pin_memory)
        )
        targets = batch.y.view(-1)
        tactic_names = _extract_tactic_names(batch)
        arg_targets = _extract_arg_targets(batch, model.max_args, device)
        arg_lemma_targets = _extract_arg_lemma_ids(batch, model.max_args, device)
        tactic_arities = [get_tactic_arity(n) for n in tactic_names]

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            tactic_logits, arg_logits_list = model(
                batch, tactic_names=tactic_names
            )
            ta_loss, ta_metrics = compute_combined_loss(
                tactic_logits,
                arg_logits_list,
                targets,
                arg_targets,
                batch.batch,
                tactic_arity_per_sample=tactic_arities,
                arg_loss_weight=arg_loss_weight,
                unknown_tactic_id=unknown_tactic_id,
            )

            with torch.no_grad():
                node_embeddings = model.backbone.encode_nodes(batch)
                state_emb = model.backbone.readout(node_embeddings, batch)
            tactic_ids = tactic_logits.argmax(dim=1)
            tactic_emb = model.tactic_embedding(tactic_ids)
            premise_mask = batch.premise_mask.to(
                dtype=torch.bool, device=device
            )

            pools = build_unified_pools(
                state_emb,
                node_embeddings,
                premise_mask,
                batch.batch,
                lemma_index=lemma_index,
                k=k,
            )
            score_list = scorer(state_emb, tactic_emb, pools)
            p_loss, p_metrics = compute_premise_ranking_loss(
                score_list, pools, arg_targets, arg_lemma_targets
            )

        bs = int(targets.numel())
        total_tactic_loss += ta_metrics["tactic_loss"] * bs
        total_arg_loss += ta_metrics["arg_loss"] * bs
        total_premise_loss += p_metrics["premise_loss"] * bs
        total_combined_loss += (
            ta_metrics["total_loss"]
            + premise_loss_weight * p_metrics["premise_loss"]
        ) * bs
        premise_valid += p_metrics["valid_samples"]
        premise_target_present += p_metrics["target_present_count"]
        premise_top1_correct += p_metrics["top1_correct"]
        premise_top5_correct += p_metrics["top5_correct"]
        premise_mrr_sum += p_metrics["mrr_sum"]

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
            and _should_log_batch(
                batch_index,
                total_batches,
                log_every_batches=log_every_batches,
            )
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
        "premise_loss": total_premise_loss / n,
        "combined_loss": total_combined_loss / n,
        "tactic_top1_accuracy": top1_correct / max(known_count, 1),
        "premise_recall": premise_valid / max(premise_target_present, 1),
        "premise_mrr": premise_mrr_sum / max(premise_valid, 1),
        "premise_top1_accuracy": premise_top1_correct / max(premise_valid, 1),
        "premise_top5_accuracy": premise_top5_correct / max(premise_valid, 1),
        "known_label_count": known_count,
        "premise_target_present_count": premise_target_present,
        "premise_valid_count": premise_valid,
        "evaluated_count": total_count,
    }
