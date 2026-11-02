"""Evaluation and single-search operations shared by every entry point.

The standalone ``--eval-only`` run, periodic evaluation during training and the
HTTP service all search through ``collect_round`` and report through
``summarize_evaluation``, so a proof rate means the same thing everywhere: the
fraction of searched seeds whose root was kernel-certified. Nothing here builds
an optimizer or changes model weights.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from maths_ai.hybrid_reasoner.hypergraph import (
    PRIOR_ALIAS_POLICY,
    PRIOR_NORMALIZATION,
    PRIOR_SEMANTICS,
    SearchEndReason,
)
from maths_ai.hybrid_reasoner.proposal import (
    PROPOSAL_MODES,
    SEED_DERIVATION_VERSION,
    ProposalSpec,
    input_identity,
    proposal_specs,
)
from maths_ai.lean_replay import ReplayInputError
from pantograph_async import PantographRequestTimeout, PantographTransportError

from .reporting import console_print
from .rl_live_collection import FATAL_SESSION_ERRORS, LiveReasonerPool
from .rl_reasoner import ACTOR_PRIOR, RLHybridReasoner, RLSearchResult
from .seed_admission import TheoremItem

if TYPE_CHECKING:
    from .rl_training_driver import RLTrainingConfig, TheoremPool


def search_settings(cfg: RLTrainingConfig) -> dict[str, object]:
    """Search semantics that must remain fixed across an exact RL resume."""

    return {
        "top_k_tactics": cfg.top_k_tactics,
        "max_depth": cfg.max_depth,
        "max_nodes": cfg.max_nodes,
        "selection_policy": cfg.selection_policy,
        "num_simulations": cfg.num_simulations,
        "sim_batch_size": cfg.sim_batch_size,
        "puct_c": cfg.puct_c,
        "use_pln": cfg.use_pln,
        "visit_threshold": cfg.visit_threshold,
        "widening": None if cfg.widening() is None else cfg.widening().as_dict(),
        "proposal": {
            "training_mode": "sample",
            "seed": cfg.seed,
            "seed_derivation": SEED_DERIVATION_VERSION,
            "greedy": "stepwise-argmax-v1",
            # One encoding per round, then each state's requested draws in
            # row-major order. A seed from another schedule does not reproduce
            # the same actions.
            "sampling_schedule": "encode-once-row-major-v1",
        },
        "prior": {
            "semantics": PRIOR_SEMANTICS,
            "normalization": PRIOR_NORMALIZATION,
            "alias_policy": PRIOR_ALIAS_POLICY,
            "provider": ACTOR_PRIOR,
        },
        "observations": "per-node-raw-action-ledger-v1",
        "command_identity": "rendered-lean-command-v1",
    }



# The proof rate is solved / searched seeds over every repeat. Version 2 is the
# first with selectable sampled or step-wise greedy proposals and per-search
# seeds; version-1 scores (tactic-only greedy) are not comparable with it.
EVAL_METRIC_VERSION = "searched-proof-rate-v2"


@dataclass(frozen=True)
class EvaluationSettings:
    """How evaluation searches the held-out pool; shared by every entry point."""

    mode: str = "sample"
    seed: int = 42
    repeats: int = 1

    def __post_init__(self) -> None:
        if self.mode not in PROPOSAL_MODES:
            raise ValueError(
                f"eval_proposal_mode must be one of {list(PROPOSAL_MODES)}, got {self.mode!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"eval_seed must be a non-negative integer, got {self.seed!r}")
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, int) or self.repeats < 1:
            raise ValueError(f"eval_repeats must be a positive integer, got {self.repeats!r}")

    def proposals(self, items: list[TheoremItem], repeat: int) -> list[ProposalSpec]:
        return proposal_specs(
            [item.seed for item in items],
            mode=self.mode,
            namespace="eval",
            base_seed=self.seed,
            pass_index=repeat,
        )

    def contract(self) -> dict[str, object]:
        """What a stored best proof rate was measured under."""
        return {
            "metric": EVAL_METRIC_VERSION,
            "mode": self.mode,
            "seed": self.seed,
            "repeats": self.repeats,
            "seed_derivation": SEED_DERIVATION_VERSION,
        }



def subset_identity(items: list[TheoremItem]) -> str:
    """Digest of an ordered list of inputs, by their stable identities."""
    body = json.dumps([input_identity(item.seed) for item in items])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()



def _describe(item: TheoremItem) -> str:
    source = item.seed.source_dict
    label = source.get("theorem") or source.get("file") or "seed"
    where = source.get("row_index", source.get("line"))
    return f"{label}#{where}" if where is not None else str(label)


def _searched(stats: dict[str, float]) -> float:
    """Inputs that were actually searched: a seed that failed to replay was not."""
    return stats["attempted"] - stats.get("seeds_inadmissible", 0.0)



# ---------------------------------------------------------------------------
# Fault-isolated collect
# ---------------------------------------------------------------------------


async def collect_round(
    reasoner: RLHybridReasoner,
    batch: list[TheoremItem],
    *,
    timeout_s: float,
    proposals: list[ProposalSpec],
    reasoner_pool: LiveReasonerPool | None = None,
    model_generation: int = 0,
) -> tuple[list[RLSearchResult], dict[str, float]]:
    """Collect with per-theorem fault isolation.

    ``proposals[i]`` is the proposal mode and seed of ``batch[i]``'s search.

    A pool uses isolated reasoners and Pantograph clients concurrently and
    replaces a worker whose client failed. The single-reasoner path remains
    available for tests and small runs; it cannot replace its only client, so a
    transport failure or a timeout that retired the client ends the run.

    Timeout handling: ``prove`` receives a monotonic ``deadline`` and stops
    cleanly between expansions / simulation batches, returning the partial
    graph — a multi-simulation search that runs out of time keeps its
    completed simulations as experience. ``asyncio.timeout`` stays as a
    backstop at ``timeout_s * 1.25`` for a hang inside a single Lean call,
    where cancellation (and the loss of that search) is the only option left.
    """
    if reasoner_pool is not None:
        collected = await reasoner_pool.collect_round(
            batch,
            timeout_s=timeout_s,
            proposals=proposals,
            model_generation=model_generation,
        )
        return collected.results, {**collected.stats, **search_counters(collected.results)}
    results: list[RLSearchResult] = []
    solved = 0
    failed = 0
    cooperative_deadlines = 0
    hard_timeouts = 0
    inadmissible = 0
    if len(proposals) != len(batch):
        raise ValueError(
            f"Expected one proposal per item: {len(proposals)} for {len(batch)} items."
        )
    for item, proposal in zip(batch, proposals):
        try:
            deadline = time.monotonic() + timeout_s
            async with asyncio.timeout(timeout_s * 1.25):
                result = await reasoner.prove(item.seed, proposal=proposal, deadline=deadline)
            results.append(result)
            if result.graph.is_solved():
                solved += 1
            if result.graph.end_reason == SearchEndReason.DEADLINE:
                cooperative_deadlines += 1
        except ReplayInputError as exc:
            # The seed itself failed to replay: no search ran, so nothing is learned.
            inadmissible += 1
            console_print(f"  [collect] inadmissible seed {_describe(item)}: {exc}")
        except (asyncio.TimeoutError, PantographRequestTimeout) as exc:
            failed += 1
            hard_timeouts += 1
            console_print(f"  [collect] timeout on: {_describe(item)}")
            server = getattr(reasoner, "server", None)
            if server is not None and not server.is_usable:
                raise PantographTransportError(
                    "A hard timeout retired the only Pantograph client; use a "
                    "reasoner pool to replace interrupted workers."
                ) from exc
        except (*FATAL_SESSION_ERRORS, PantographTransportError):
            raise
        except Exception as exc:  # noqa: BLE001 — a single search may not kill the run
            failed += 1
            console_print(f"  [collect] error on: {_describe(item)} — {exc}")
    stats = {
        "attempted": float(len(batch)),
        "collected": float(len(results)),
        "solved": float(solved),
        "searches_failed": float(failed),
        "searches_cooperative_deadline": float(cooperative_deadlines),
        "searches_hard_timed_out": float(hard_timeouts),
        "seeds_inadmissible": float(inadmissible),
        **search_counters(results),
    }
    return results, stats


def search_counters(results: list[RLSearchResult]) -> dict[str, float]:
    """Proposal, widening and observation work summed over completed searches.

    ``raw_draws`` counts every draw the searches made and ``settled_draws``
    those with a checked outcome (the observations' multiplicities); draws of
    an interrupted round are the difference. ``accepted_edges <=
    distinct_commands <= raw_draws`` holds for each search.
    """
    totals: dict[str, float] = {
        "proposal_rounds": 0.0,
        "widening_rounds": 0.0,
        "raw_draws": 0.0,
        "distinct_commands": 0.0,
        "accepted_edges": 0.0,
        "rejected_commands": 0.0,
        "cache_hits": 0.0,
        "completed_simulations": 0.0,
        "deferred_selections": 0.0,
        "settled_draws": 0.0,
        "observations": 0.0,
    }
    for result in results:
        if result.graph is None:
            continue
        for key, value in result.graph.search_counters().items():
            totals[key] = totals.get(key, 0.0) + value
        totals["settled_draws"] += float(result.draws())
        totals["observations"] += float(len(result.observations))
    return totals



# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


# Counters of one round that add across the chunks of a repeat.
ADDITIVE_ROUND_STATS = (
    "attempted",
    "collected",
    "solved",
    "searches_failed",
    "searches_cooperative_deadline",
    "searches_hard_timed_out",
    "seeds_inadmissible",
)


def chunk_stats(stats: dict[str, float]) -> dict[str, float]:
    """The additive part of a round's stats: outcome counts and search counters."""
    keys = (*ADDITIVE_ROUND_STATS, *search_counters([]))
    return {key: float(stats.get(key, 0.0)) for key in keys}


def merge_stats(total: dict[str, float], stats: dict[str, float]) -> dict[str, float]:
    """Add one chunk's additive stats into a repeat's running total."""
    for key, value in chunk_stats(stats).items():
        total[key] = total.get(key, 0.0) + value
    return total


def repeat_row(
    repeat: int, proposals: list[ProposalSpec], stats: dict[str, float]
) -> dict[str, Any]:
    """One repeat's report row: its proof rate over searched seeds and its seeds."""
    seeds = json.dumps([proposal.seed for proposal in proposals])
    return {
        "repeat": repeat,
        "proof_rate": stats["solved"] / (_searched(stats) or 1.0),
        "searched": _searched(stats),
        "proposal_seeds_sha256": hashlib.sha256(seeds.encode("utf-8")).hexdigest(),
        **stats,
    }


def summarize_evaluation(
    settings: EvaluationSettings,
    eval_items: list[TheoremItem],
    repeats: list[dict[str, Any]],
) -> dict[str, Any]:
    """The evaluation result over completed repeat rows.

    The proof rate counts searched seeds only; inadmissible seeds are reported
    separately. Both the whole-round evaluation and the service's chunked jobs
    report through this one function, so their results mean the same thing.
    """

    def total(key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in repeats)

    searched = total("searched")
    return {
        "proof_rate": total("solved") / (searched or 1.0),
        "solved": total("solved"),
        "attempted": total("attempted"),
        "searched": searched,
        "searches_failed": total("searches_failed"),
        "searches_hard_timed_out": total("searches_hard_timed_out"),
        "searches_cooperative_deadline": total("searches_cooperative_deadline"),
        "seeds_inadmissible": total("seeds_inadmissible"),
        **{key: total(key) for key in search_counters([])},
        "evaluation": settings.contract(),
        "eval_subset_size": len(eval_items),
        "eval_subset_sha256": subset_identity(eval_items),
        "repeats": repeats,
    }


async def evaluate_proof_rate(
    reasoner: RLHybridReasoner,
    eval_items: list[TheoremItem],
    *,
    settings: EvaluationSettings,
    timeout_s: float = 60.0,
    reasoner_pool: LiveReasonerPool | None = None,
    model_generation: int = 0,
) -> dict[str, Any]:
    """Proof rate on the fixed held-out pool — the model-selection metric.

    Runs ``settings.repeats`` passes over ``eval_items``, each with seeds
    derived from ``settings.seed``, the repeat and the input, so a pass is
    reproducible and independent of the worker that ran each search. The
    proof rate counts searched seeds only; inadmissible seeds are reported
    separately. The model is only read: no optimizer runs and the global RNG
    is not consumed. The fixed pool makes the number comparable across rounds,
    but only between runs with the same ``settings.contract()``.
    """
    if not eval_items:
        raise ValueError("The evaluation subset is empty; set eval_pool_size > 0.")
    was_training = reasoner.model.training
    reasoner.model.eval()
    repeats: list[dict[str, Any]] = []
    try:
        for repeat in range(settings.repeats):
            proposals = settings.proposals(eval_items, repeat)
            _results, stats = await collect_round(
                reasoner,
                eval_items,
                timeout_s=timeout_s,
                proposals=proposals,
                reasoner_pool=reasoner_pool,
                model_generation=model_generation,
            )
            repeats.append(repeat_row(repeat, proposals, stats))
    finally:
        if was_training:
            reasoner.model.train()
    return summarize_evaluation(settings, eval_items, repeats)


def evaluation_report(
    cfg: RLTrainingConfig,
    stats: dict[str, Any],
    *,
    pool: "TheoremPool",
    checkpoint: str | None,
    model_generation: int,
) -> dict[str, Any]:
    """The evaluation result with the settings and inputs it was measured on."""
    return {
        "kind": "rl-evaluation-report",
        "result": stats,
        "checkpoint": checkpoint,
        "model_generation": model_generation,
        "search_settings": search_settings(cfg),
        "budgets": {
            "theorem_timeout_s": cfg.theorem_timeout_s,
            "top_k_tactics": cfg.top_k_tactics,
            "max_depth": cfg.max_depth,
            "max_nodes": cfg.max_nodes,
        },
        "dataset": {
            "data_source": cfg.data_source,
            "dataset_name": cfg.dataset_name,
            "dataset_revision": cfg.dataset_revision,
            "split": cfg.leandojo_split,
            "theorem_file": str(cfg.theorem_file) if cfg.theorem_file else None,
            "eval_pool_size": cfg.eval_pool_size,
        },
        "admission": pool.admission,
        "workers": cfg.collection_workers,
        "reproducibility": (
            "Proposal draws repeat for the same seeds, model, inputs and device; "
            "Lean and wall-clock timing (deadlines, timeouts) are not reproducible."
        ),
    }


def write_evaluation_report(report: dict[str, Any], path: Path) -> Path:
    """Write ``report`` to a new file; an existing report is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path
