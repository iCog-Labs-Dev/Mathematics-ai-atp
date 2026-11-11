"""Resumable structured-head training over immutable frozen feature shards."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import torch

from .bundle import file_sha256
from .structured_decoder import FORMAT
from .structured_features import atomic_json, atomic_torch


@dataclass(frozen=True)
class StructuredTrainingConfig:
    epochs: int = 10
    batch_size: int = 16
    max_batch_tokens: int = 8192
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    checkpoint_every: int = 25
    max_updates: int = 0

    def __post_init__(self):
        for name in ("epochs", "batch_size", "max_batch_tokens", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "max_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("learning_rate", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be finite and nonnegative")


def metrics(totals, candidate_rows, compiled_rows, feature_rows):
    counts = dict(totals)
    rate = lambda key, denominator: counts.get(key, 0) / max(1, denominator)
    return {**counts, "candidate_rows": candidate_rows, "compiled_rows": compiled_rows,
        "feature_rows": feature_rows, "uncompiled_misses": candidate_rows - compiled_rows,
        "feature_exclusions": compiled_rows - feature_rows,
        "unvisited_feature_rows": feature_rows - counts.get("rows", 0),
        "mean_loss": rate("loss_sum", counts.get("rows", 0)),
        "mean_token_loss": rate("token_loss_sum", counts.get("rows", 0)),
        "mean_pointer_loss": rate("pointer_loss_sum", counts.get("rows", 0)),
        "teacher_forced_exact_rate": rate("teacher_forced_exact_hits", candidate_rows),
        "token_accuracy": rate("token_hits", counts.get("tokens", 0)),
        "structural_token_accuracy": rate("structural_hits", counts.get("structural_tokens", 0)),
        "name_literal_byte_accuracy": rate("byte_hits", counts.get("byte_tokens", 0)),
        "local_accuracy_given_gold_source": rate("local_hits", counts.get("local_positions", 0)),
        "library_accuracy_given_retrieved_gold": rate("library_hits", counts.get("library_reachable_positions", 0)),
        "lexical_accuracy_given_gold_source": rate("bound_hits", counts.get("bound_positions", 0)),
        "frozen_tactic_top1": rate("frozen_tactic_hits", counts.get("known_tactics", 0)),
        "tactic_conditioning": "gold", "previous_arguments": "gold_teacher_forcing",
        "generated_action_execution_measured": False, "proof_success_measured": False}


@torch.no_grad()
def evaluate(model, features, config):
    model.eval()
    total = Counter()
    for rows in features.batches("val", batch_size=config.batch_size, max_tokens=config.max_batch_tokens,
                                 seed=config.seed, epoch=0):
        _, counts = model(rows)
        total.update(counts)
    info = features.manifest["splits"]["val"]
    return metrics(total, info["candidate_rows"], info["compiled_rows"], info["feature_rows"])


def train_decoder(model, features, output, config, *, resume=False):
    output = Path(output).resolve()
    if (model.hidden_dim != features.manifest["binding"]["hidden_dim"]
            or model.text_dim != features.manifest["binding"]["text_dim"]):
        raise ValueError("decoder and frozen feature dimensions differ")
    if any(features.manifest["splits"][split]["feature_rows"] < 1 for split in ("train", "val")):
        raise ValueError("nonempty encoded train and val splits required")
    semantic = {key: value for key, value in asdict(config).items() if key not in {"epochs", "max_updates", "checkpoint_every"}}
    binding = {"model": model.configuration(), "feature_manifest_sha256": file_sha256(features.root / "manifest.json"),
        "feature_binding_sha256": features.manifest["binding_sha256"], "training": semantic,
        "trainer_sha256": file_sha256(Path(__file__)),
        "decoder_sha256": file_sha256(Path(__file__).with_name("structured_decoder.py"))}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    state = {"epoch": 0, "next_batch": 0, "global_updates": 0, "train_totals": {}, "history": [], "best_score": None}
    last_validation = None
    if output.exists():
        if not resume:
            raise FileExistsError("training output exists; use --resume with identical bindings")
        if json.loads((output / "binding.json").read_text()) != binding:
            raise ValueError("training resume changed data/model/optimization bindings")
        # Let optimizer.load_state_dict place moments on parameter devices while
        # retaining CPU step/RNG tensors for non-capturable AdamW.
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if saved["binding"] != binding or saved["model_type"] != FORMAT:
            raise ValueError("checkpoint binding differs from training run")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        state = saved["state"]
        last_validation = saved.get("val_metrics")
        torch.set_rng_state(saved["rng_state"].cpu())
        random.setstate(saved["python_rng_state"])
        if saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng_state"]])
    else:
        output.mkdir(parents=True)
        atomic_json(output / "binding.json", binding)

    def checkpoint(status, validation=None, *, best=False):
        payload = {"model_type": FORMAT, "binding": binding, "config": asdict(config), "state": state,
            "status": status, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == "cuda" else None,
            "val_metrics": validation}
        atomic_torch(output / "last.pt", payload)
        if best:
            atomic_torch(output / "best.pt", payload)
        summary = {"status": status, "epoch": state["epoch"], "next_batch": state["next_batch"],
            "global_updates": state["global_updates"], "binding": binding,
            "val_metrics": validation, "history": state["history"]}
        atomic_json(output / "report.json", summary)
        return summary

    def validate_and_save(status):
        validation = evaluate(model, features, config)
        score = [validation["teacher_forced_exact_rate"], -validation["mean_loss"]]
        best = state["best_score"] is None or tuple(score) > tuple(state["best_score"])
        if best:
            state["best_score"] = score
        print(f"update={state['global_updates']} val_loss={validation['mean_loss']:.4f} "
              f"teacher_forced_exact={validation.get('teacher_forced_exact_hits', 0)}/{validation['candidate_rows']} "
              f"library_missing={validation.get('library_missing_positions', 0)}", flush=True)
        return checkpoint(status, validation, best=best)

    if state["epoch"] >= config.epochs:
        return checkpoint("complete", last_validation)
    checkpoint("initialized" if state["global_updates"] == 0 else "resumed")
    while state["epoch"] < config.epochs:
        if config.max_updates and state["global_updates"] >= config.max_updates:
            return validate_and_save("update_budget_reached")
        totals = Counter(state["train_totals"])
        for batch_id, rows in enumerate(features.batches("train", batch_size=config.batch_size,
                max_tokens=config.max_batch_tokens, seed=config.seed, epoch=state["epoch"])):
            if batch_id < state["next_batch"]:
                continue
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, counts = model(rows)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite structured training loss; last atomic checkpoint retained")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            totals.update(counts)
            state.update(next_batch=batch_id + 1, global_updates=state["global_updates"] + 1, train_totals=dict(totals))
            if state["global_updates"] % config.checkpoint_every == 0:
                checkpoint("training")
                print(f"epoch={state['epoch'] + 1}/{config.epochs} update={state['global_updates']} "
                      f"train_loss={totals['loss_sum'] / totals['rows']:.4f}", flush=True)
            if config.max_updates and state["global_updates"] >= config.max_updates:
                return validate_and_save("update_budget_reached")
        validation = evaluate(model, features, config)
        info = features.manifest["splits"]["train"]
        training = metrics(totals, info["candidate_rows"], info["compiled_rows"], info["feature_rows"])
        state["history"].append({"epoch": state["epoch"] + 1, "train": training, "val": validation})
        state.update(epoch=state["epoch"] + 1, next_batch=0, train_totals={})
        score = [validation["teacher_forced_exact_rate"], -validation["mean_loss"]]
        best = state["best_score"] is None or tuple(score) > tuple(state["best_score"])
        if best:
            state["best_score"] = score
        checkpoint("complete" if state["epoch"] >= config.epochs else "epoch_complete", validation, best=best)
        print(f"epoch={state['epoch']}/{config.epochs} train_loss={training['mean_loss']:.4f} "
              f"val_loss={validation['mean_loss']:.4f} "
              f"teacher_forced_exact={validation.get('teacher_forced_exact_hits', 0)}/{validation['candidate_rows']}", flush=True)
    return json.loads((output / "report.json").read_text())
