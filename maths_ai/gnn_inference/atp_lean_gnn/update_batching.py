"""CPU update plans and finite computational-batch resource limits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch_geometric.data import Data

from maths_ai.data_models.proof_components import Goal


class UpdateBudgetError(ValueError):
    """A single validated row exceeds a computational-batch budget."""


@dataclass(frozen=True)
class UpdateLimits:
    nodes: int = 50000
    edges: int = 250000
    candidates: int = 100000
    candidate_bytes: int = 268435456

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"Update {name} budget must be a positive integer.")

    def contract(self):
        return {
            "version": 1,
            "reduction": "global-objective-denominators-and-population-advantages",
            "forward_mode": "actor-critic-eval-with-gradients",
            "limits": vars(self).copy(),
        }


@dataclass(frozen=True)
class RowCost:
    nodes: int
    edges: int
    candidates: int = 0
    candidate_bytes: int = 0

    def __add__(self, other):
        return RowCost(
            *(a + b for a, b in zip(vars(self).values(), vars(other).values()))
        )

    def fits(self, limits):
        return all(value <= getattr(limits, key) for key, value in vars(self).items())


@dataclass(frozen=True)
class UpdateRow:
    goal: Goal
    fingerprint: str
    target: float
    tactic_id: int | None = None
    action: object | None = None
    weight: float = 1.0
    accepted: bool = False
    source: str = ""


@dataclass(frozen=True)
class PreparedRow:
    record: UpdateRow
    cost: RowCost


@dataclass(frozen=True)
class UpdatePlan:
    phase: str
    actors: tuple[PreparedRow, ...]
    critics: tuple[PreparedRow, ...]
    actor_batches: tuple[tuple[int, ...], ...]
    critic_batches: tuple[tuple[int, ...], ...]
    limits: UpdateLimits
    metrics: dict[str, float]

    @property
    def empty(self):
        return not self.actors and not self.critics


def candidate_cost(model, action):
    if action is None:
        return 0, 0
    pool = action.candidate_pool
    count = len(pool.local_ids) + len(pool.lemma_ids)
    # The decoder gathers local vectors and gathers/adapts library vectors.
    # Count each row conservatively even when the decoder shares adapted pools.
    elements = len(pool.local_ids) * model.hidden_dim + len(pool.lemma_ids) * (
        model.decoder.text_dim + model.hidden_dim
    )
    return count, elements * next(model.parameters()).element_size()


def graph_cost(data: Data, candidates=0, candidate_bytes=0):
    if data.x.device.type != "cpu" or data.edge_index.device.type != "cpu":
        raise ValueError("Update featurization must produce CPU graph tensors.")
    return RowCost(
        int(data.num_nodes), int(data.edge_index.size(1)), candidates, candidate_bytes
    )


def pack(rows, limits):
    batches, current, cost = [], [], RowCost(0, 0)
    for index, row in enumerate(rows):
        proposed = cost + row.cost
        if current and not proposed.fits(limits):
            batches.append(tuple(current))
            current, cost = [], RowCost(0, 0)
        current.append(index)
        cost = cost + row.cost
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def prepare_plan(
    model, actors, critics, featurize, validate_action: Callable, *, phase, limits
):
    metrics = {
        key: 0.0
        for key in (
            "unknown_label_count",
            "structural_unknown_label_count",
            "semantic_unknown_label_count",
            "total_label_count",
        )
    }

    def prepare(records, role):
        rows = []
        for index, record in enumerate(records):
            if (
                not math.isfinite(record.target)
                or not math.isfinite(record.weight)
                or record.weight <= 0
            ):
                raise ValueError(
                    f"{phase} {role} row {index}: nonfinite target or invalid multiplicity."
                )
            data = featurize(record.goal)
            if str(data.graph_fingerprint) != record.fingerprint:
                raise ValueError(
                    f"{phase} {role} row {index}: graph changed between collection and training."
                )
            if record.action is not None:
                validate_action(
                    model,
                    record.goal,
                    data,
                    record.action,
                    f"{phase} {role} row {index}",
                )
            if (
                record.tactic_id is not None
                and not 0 <= record.tactic_id < model.actor.base.out_features
            ):
                raise ValueError(f"{phase} {role} row {index}: invalid tactic ID.")
            cost = graph_cost(data, *candidate_cost(model, record.action))
            if not cost.fits(limits):
                raise UpdateBudgetError(
                    f"{phase} {role} row {index} graph {record.fingerprint} exceeds graph budget "
                    f"or candidate budget: cost={vars(cost)}, limits={vars(limits)}"
                )
            for key in metrics:
                metrics[key] += float(getattr(data, key))
            rows.append(PreparedRow(record, cost))
        return tuple(rows)

    actor_rows, critic_rows = prepare(actors, "actor"), prepare(critics, "critic")
    actor_batches, critic_batches = pack(actor_rows, limits), pack(critic_rows, limits)
    total = RowCost(0, 0)
    peaks = RowCost(0, 0)
    for rows, batches in ((actor_rows, actor_batches), (critic_rows, critic_batches)):
        for indexes in batches:
            cost = RowCost(0, 0)
            for index in indexes:
                cost = cost + rows[index].cost
            total = total + cost
            peaks = RowCost(
                *(max(a, b) for a, b in zip(vars(peaks).values(), vars(cost).values()))
            )
    for key in vars(total):
        metric = {"candidates": "candidate_count"}.get(key, key)
        metrics[f"update_{metric}"] = float(getattr(total, key))
        metrics[f"update_peak_{metric}"] = float(getattr(peaks, key))
    metrics.update(
        update_actor_batches=float(len(actor_batches)),
        update_critic_batches=float(len(critic_batches)),
        update_backward_batches=float(len(actor_batches) + len(critic_batches)),
        update_value_batches=0.0,
    )
    return UpdatePlan(
        phase, actor_rows, critic_rows, actor_batches, critic_batches, limits, metrics
    )


def materialize(model, rows, indexes, featurize):
    """Regenerate only the current batch and reject changed inputs before transfer."""
    datas = []
    for index in indexes:
        row = rows[index]
        data = featurize(row.record.goal)
        if str(data.graph_fingerprint) != row.record.fingerprint:
            raise ValueError("Update graph changed after preflight.")
        if graph_cost(data, *candidate_cost(model, row.record.action)) != row.cost:
            raise ValueError("Update row cost changed after preflight.")
        datas.append(data)
    return datas


def normalized_advantages(returns, values, weights):
    # Float64 scalar reductions avoid cancellation without retaining device graphs.
    raw = torch.tensor(returns, dtype=torch.float64) - torch.tensor(
        values, dtype=torch.float64
    )
    mult = torch.tensor(weights, dtype=torch.float64)
    mean = (raw * mult).sum() / mult.sum()
    variance = ((raw - mean).square() * mult).sum() / mult.sum()
    return ((raw - mean) / (variance.sqrt() + 1e-8)).tolist()
