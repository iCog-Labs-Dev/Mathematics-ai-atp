"""RL training driver: warm-start from supervised checkpoints, curriculum over
LeanDojo proof states, BC annealing, fault tolerance, eval-by-proof-rate.

Encodes the three invariants:
  - one optimizer step per collect round (on-policy A2C) — the decoupled
    HTPS-style step (``train_step_htps_style``) is exempt: supervised regression
    on stored pairs through its own optimizer,
  - one featurizer instance shared between collect and train (index alignment),
  - vocabs always from prepared_root (embedding alignment across all phases).

Design decisions and alternatives are recorded in
``docs/dev_plans/rl_training_driver.md``.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections import deque
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW

from maths_ai.data_models.proof_components import Goal, LeanGoalSeed
from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_server
from maths_ai.hybrid_reasoner.hypergraph import BackupSource, SearchEndReason
from maths_ai.hybrid_reasoner.selection_policy import resolve_search_params

from .actor_critic import ActorCriticWithArgsClassifier
from .checkpointing import build_model_from_checkpoint, checkpoint_payload
from .graph_contract import GraphRepresentationSpec, require_graph_representation
from .graph import dag_fingerprint, model_goal_to_dag
from .dataset import DATASET_NAME, iter_dataset_rows
from .pln_reward import RewardConfig
from .pln_rl_training import make_dag_featurizer, train_step_htps_style, train_step_onpolicy
from .reporting import console_print
from .rl_live_collection import LiveReasonerPool
from .rl_resources import ResourcePlan, plan_resources
from .rl_reasoner import RLHybridReasoner, RLSearchResult
from .search_harvest import (
    CriticSample,
    TacticImitationSample,
    extract_critic_samples,
    extract_minimal_hypertree,
)
from .state import parse_state
from .training import load_prepared_metadata


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLTrainingConfig:
    """Configuration for the RL training driver (flat JSON, ``from_json`` below).

    ``warmstart_checkpoint`` is a version-3, self-describing supervised
    actor-critic checkpoint. Its manifest owns the encoder architecture.
    """

    warmstart_checkpoint: Path
    prepared_root: Path

    # Theorem sourcing: "dataset" streams LeanDojo proof states; "file" reads a
    # JSONL of {"goal": str, "hypotheses": [str, ...]} rows.
    data_source: str = "dataset"
    theorem_file: Path | None = None
    dataset_name: str = DATASET_NAME
    leandojo_split: str = "train"
    max_pool_size: int = 5000
    max_state_chars: int | None = 400
    eval_pool_size: int = 200
    seed: int = 42

    # Curriculum: sliding window over the size-sorted pool.
    curriculum_start_size: int = 200
    curriculum_growth_factor: float = 1.5
    curriculum_solve_threshold: float = 0.3
    curriculum_window_rounds: int = 10  # solve rate measured over this many recent rounds

    # BC anchor anneal (linear).
    bc_anneal_start: float = 0.5
    bc_anneal_end: float = 0.05
    bc_anneal_rounds: int = 200

    # Round loop.
    num_rounds: int = 500
    theorems_per_round: int = 8
    theorem_timeout_s: float = 120.0
    checkpoint_every: int = 20
    eval_every: int = 25
    max_dead_rounds: int = 3

    # Search budgets (RLHybridReasoner).
    top_k_tactics: int = 4
    max_depth: int = 8
    max_nodes: int = 64

    # PLN kill switch: False skips PLNInference construction, DTS sampler, rank_subgoals
    # calls, and the _expand fallback block. Subgoal nodes receive stv=None, which
    # ProofNode.local_score and potential() already handle — frontier ranking degrades to
    # GNN probability and reward shaping vanishes, both with zero changes to those modules.
    use_pln: bool = True

    # Search mode (see hybrid_reasoner/selection_policy.py, the single
    # authority for this contract). "legacy" runs the best-first loop and
    # forbids an explicit simulation budget; "puct" enables PUCT-guided
    # repeated simulation with virtual loss and per-edge visit statistics
    # and requires num_simulations. None = "left alone" (resolved by
    # resolve_search_params); "rp" is reserved for the deferred variant.
    selection_policy: str = "legacy"
    num_simulations: int | None = None
    sim_batch_size: int | None = None
    puct_c: float | None = None

    # Decoupled HTPS-style step (Phases 2–3). htps_steps_per_round=0 disables it
    # entirely — no queues fill semantics change, no second optimizer steps.
    visit_threshold: int = 4          # min valid edge backups before Q becomes a critic target
    critic_queue_size: int = 10000
    htps_steps_per_round: int = 0     # 0 ⇒ decoupled step disabled ⇒ current behavior
    htps_batch_size: int = 64
    htps_learning_rate: float = 1e-4
    w_critic_soft: float = 0.5
    tactic_queue_size: int = 10000
    mine_all_solved_nodes: bool = True  # False = root-only mining (ablation)

    # Optimizer / loss.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    critic_weight: float = 0.5
    entropy_weight: float = 0.01
    arg_loss_weight: float = 0.5
    max_update_nodes: int = 0
    max_update_edges: int = 0

    # Reward (Approach 1 potential shaping lives inside pln_reward).
    reward_gamma: float = 0.99
    reward_step_penalty: float = 0.01
    reward_terminal_success: float = 1.0
    reward_terminal_failure: float = 0.0

    run_root: Path = Path("runs/rl_actor_critic")
    device: str = "auto"

    # Live collection resources. Replicas are read-only until the round update
    # completes, preserving the on-policy boundary.
    collection_workers: int = 1
    collection_devices: list[str] | None = None
    update_device: str | None = None
    cpu_reserve: int = 2
    resource_check: str = "strict"
    worker_queue_size: int = 0
    server_start_timeout_s: float = 120.0
    worker_shutdown_timeout_s: float = 10.0

    source_root: Path | None = None
    pantograph_repl: Path | None = None
    pantograph_imports: list[str] | None = None
    server_timeout_s: int = 120

    _PATH_FIELDS = (
        "warmstart_checkpoint", "prepared_root", "theorem_file", "run_root",
        "source_root", "pantograph_repl",
    )

    def __post_init__(self) -> None:
        # Fail at config construction — before any Lean server spins up. Only
        # validates; the reasoner constructor runs the same resolver again and
        # stores the resolved values.
        resolve_search_params(
            self.selection_policy, self.num_simulations, self.sim_batch_size, self.puct_c
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "RLTrainingConfig":
        with open(path) as f:
            payload = json.load(f)
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "graph_representation":
                continue
            if key in cls._PATH_FIELDS and value is not None:
                kwargs[key] = Path(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            out[key] = str(value) if isinstance(value, Path) else value
        return out

    def resource_plan(self) -> ResourcePlan:
        """Resolve and validate the effective live-collection topology."""
        requested_update_device = self.update_device or self.device
        update_device = requested_update_device
        if requested_update_device == "auto":
            update_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        collection_devices = self.collection_devices
        if collection_devices is None and requested_update_device != "auto":
            collection_devices = [update_device]
        return plan_resources(
            collection_workers=self.collection_workers,
            collection_devices=collection_devices,
            update_device=update_device,
            cpu_reserve=self.cpu_reserve,
            resource_check=self.resource_check,
        )


def pantograph_env(cfg: RLTrainingConfig) -> PantographEnv:
    if cfg.pantograph_imports is not None:
        imports = tuple(cfg.pantograph_imports)
    elif cfg.source_root is not None:
        imports = ("Init", "Mathlib")
    else:
        imports = ("Init",)
    return PantographEnv(
        source_root=cfg.source_root,
        pantograph_repl=cfg.pantograph_repl,
        imports=imports,
        timeout=cfg.server_timeout_s,
    )


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
    }


# ---------------------------------------------------------------------------
# Theorem pool + curriculum
# ---------------------------------------------------------------------------


@dataclass
class TheoremItem:
    goal: LeanGoalSeed
    tactic_label: str  # ground-truth tactic from the dataset row ("" in file mode)
    size: int


# Universe placeholders such as ``?u.319125`` are emitted in extracted type
# annotations and are not unresolved term metavariables.
_UNIVERSE_METAVARIABLE_RE = re.compile(r"\?u(?:\.|_)?[0-9]+")
_METAVARIABLE_RE = re.compile(r"\?m\.\d+|\?[a-zA-Z_][a-zA-Z0-9_]*\b")


def _has_metavariable(goal: LeanGoalSeed) -> bool:
    """Return whether a goal or hypothesis contains an unresolved term hole."""
    def has_term_metavariable(text: str) -> bool:
        without_universes = _UNIVERSE_METAVARIABLE_RE.sub("", text)
        return _METAVARIABLE_RE.search(without_universes) is not None

    if has_term_metavariable(goal.expression):
        return True
    return any(has_term_metavariable(hypothesis) for hypothesis in goal.hypotheses)


def _row_state_to_goal(state_str: str) -> LeanGoalSeed:
    """Convert a dataset proof-state string into an unelaborated theorem seed."""
    parsed = parse_state(state_str)
    return LeanGoalSeed(
        expression=parsed.goal,
        hypotheses=[f"{h.name} : {h.type_expr}" for h in parsed.hypotheses],
    )


class TheoremPool:
    """Size-sorted pool of rollout roots with a sliding curriculum window.

    The eval pool is carved from the pool by hash BEFORE sorting windows are
    served, is fixed for the whole run, and never enters a training batch.
    """

    def __init__(self, items: list[TheoremItem], *, eval_pool_size: int, curriculum_size: int, seed: int) -> None:
        self._rng = random.Random(seed)
        items = sorted(items, key=lambda t: t.size)
        # Deterministic held-out split: every k-th item of the size-sorted pool, so the
        # eval set spans all difficulty levels and is identical across runs/resumes
        # (the pool itself is deterministic: same source, same filters, same sort).
        if eval_pool_size > 0 and items:
            stride = max(len(items) // eval_pool_size, 1)
            eval_indices = set(list(range(0, len(items), stride))[:eval_pool_size])
            self.eval_items = [t for i, t in enumerate(items) if i in eval_indices]
            self.train_items = [t for i, t in enumerate(items) if i not in eval_indices]
        else:
            self.eval_items = []
            self.train_items = list(items)
        self.curriculum_size = min(max(curriculum_size, 1), len(self.train_items)) if self.train_items else 0

    def sample_batch(self, batch_size: int) -> list[TheoremItem]:
        window = self.train_items[: self.curriculum_size]
        if not window:
            return []
        return self._rng.sample(window, min(batch_size, len(window)))

    def grow(self, factor: float) -> None:
        self.curriculum_size = min(int(self.curriculum_size * factor) or 1, len(self.train_items))


def build_theorem_pool(cfg: RLTrainingConfig) -> TheoremPool:
    """Build the pool from the configured source (dataset stream or JSONL file)."""
    items: list[TheoremItem] = []
    dropped = 0

    if cfg.data_source == "file":
        if cfg.theorem_file is None:
            raise ValueError("data_source='file' requires theorem_file")
        with open(cfg.theorem_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                goal = LeanGoalSeed(
                    expression=row["goal"], hypotheses=row.get("hypotheses", [])
                )
                size = len(goal.expression) + sum(len(h) for h in goal.hypotheses)
                if cfg.max_state_chars is not None and size > cfg.max_state_chars:
                    dropped += 1
                    continue
                if _has_metavariable(goal):
                    dropped += 1
                    continue
                items.append(TheoremItem(goal=goal, tactic_label=row.get("tactic", ""), size=size))
                if len(items) >= cfg.max_pool_size:
                    break
    elif cfg.data_source == "dataset":
        for row in iter_dataset_rows(
            dataset_name=cfg.dataset_name,
            split=cfg.leandojo_split,
            sample_limit=cfg.max_pool_size * 2,
        ):
            state_str = (row.state or "").strip()
            if not state_str or (
                cfg.max_state_chars is not None and len(state_str) > cfg.max_state_chars
            ):
                dropped += 1
                continue
            try:
                goal = _row_state_to_goal(state_str)
            except Exception:
                dropped += 1
                continue
            if not goal.expression:
                dropped += 1
                continue
            if _has_metavariable(goal):
                dropped += 1
                continue
            items.append(TheoremItem(goal=goal, tactic_label=row.tactic or "", size=len(state_str)))
            if len(items) >= cfg.max_pool_size:
                break
    else:
        raise ValueError(f"Unknown data_source '{cfg.data_source}' (use 'dataset' or 'file')")

    console_print(f"Theorem pool: {len(items)} usable states, {dropped} dropped.")
    return TheoremPool(
        items,
        eval_pool_size=cfg.eval_pool_size,
        curriculum_size=cfg.curriculum_start_size,
        seed=cfg.seed,
    )


# ---------------------------------------------------------------------------
# BC anneal
# ---------------------------------------------------------------------------


def bc_weight_at_round(round_idx: int, cfg: RLTrainingConfig) -> float:
    """Linear anneal from ``bc_anneal_start`` to ``bc_anneal_end`` over
    ``bc_anneal_rounds``; constant at the end value afterwards."""
    if cfg.bc_anneal_rounds <= 0 or round_idx >= cfg.bc_anneal_rounds:
        return cfg.bc_anneal_end
    t = round_idx / cfg.bc_anneal_rounds
    return cfg.bc_anneal_start + t * (cfg.bc_anneal_end - cfg.bc_anneal_start)


# ---------------------------------------------------------------------------
# Fault-isolated collect
# ---------------------------------------------------------------------------


async def collect_round(
    reasoner: RLHybridReasoner,
    batch: list[TheoremItem],
    *,
    timeout_s: float,
    greedy: bool = False,
    reasoner_pool: LiveReasonerPool | None = None,
    model_generation: int = 0,
) -> tuple[list[RLSearchResult], dict[str, float]]:
    """Collect with per-theorem fault isolation.

    A pool uses isolated reasoners and Pantograph servers concurrently. The
    single-reasoner path remains available for tests and small runs.

    Timeout handling: ``prove`` receives a monotonic ``deadline`` and stops
    cleanly between expansions / simulation batches, returning the partial
    graph — a multi-simulation search that runs out of time keeps its
    completed simulations as experience. ``asyncio.wait_for`` stays as a
    backstop at ``timeout_s * 1.25`` for a hang inside a single Lean call,
    where cancellation (and the loss of that search) is the only option left.
    """
    if reasoner_pool is not None:
        collected = await reasoner_pool.collect_round(
            batch,
            timeout_s=timeout_s,
            greedy=greedy,
            model_generation=model_generation,
        )
        return collected.results, collected.stats
    results: list[RLSearchResult] = []
    solved = 0
    failed = 0
    cooperative_deadlines = 0
    hard_timeouts = 0
    for item in batch:
        try:
            deadline = time.monotonic() + timeout_s
            result = await asyncio.wait_for(
                reasoner.prove(
                    item.goal.expression,
                    hypotheses=item.goal.hypotheses,
                    greedy=greedy,
                    deadline=deadline,
                ),
                timeout=timeout_s * 1.25,
            )
            results.append(result)
            if result.graph.is_solved():
                solved += 1
            if result.graph.end_reason == SearchEndReason.DEADLINE:
                cooperative_deadlines += 1
        except asyncio.TimeoutError:
            failed += 1
            hard_timeouts += 1
            console_print(f"  [collect] timeout on: {item.goal.expression[:60]}")
        except Exception as exc:  # noqa: BLE001 — a single search may not kill the run
            failed += 1
            console_print(f"  [collect] error on: {item.goal.expression[:60]} — {exc}")
    stats = {
        "attempted": float(len(batch)),
        "collected": float(len(results)),
        "solved": float(solved),
        "searches_failed": float(failed),
        "searches_cooperative_deadline": float(cooperative_deadlines),
        "searches_hard_timed_out": float(hard_timeouts),
    }
    return results, stats


# ---------------------------------------------------------------------------
# Checkpointing / resume
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    round_idx: int,
    curriculum_size: int,
    best_proof_rate: float,
    onpolicy_steps: int,
    path: Path,
    *,
    optimizer_htps: torch.optim.Optimizer | None = None,
    tactic_queue: "deque[TacticImitationSample] | None" = None,
    critic_queue: "deque[CriticSample] | None" = None,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    graph_representation: GraphRepresentationSpec,
    saved_search_settings: dict[str, object],
) -> None:
    """Write the resume state (Decision 1.4: optimizer-htps + queues included).

    Queue samples are serialized as structured dictionaries. Their canonical
    goals and graph fingerprints are required for exact replay alignment.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    training_state = {
        "rl_target_schema_version": 3,
        "optimizer_state_dict": optimizer.state_dict(),
        "round": round_idx,
        "curriculum_size": curriculum_size,
        "best_proof_rate": best_proof_rate,
        "onpolicy_steps": onpolicy_steps,
        "search_settings": saved_search_settings,
        "torch_rng_state": torch.get_rng_state(),
    }
    payload = checkpoint_payload(
        model_kind="actor_critic_with_args",
        model_spec=model.model_spec,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        model=model,
        graph_representation=graph_representation,
        **training_state,
    )
    if optimizer_htps is not None:
        payload["optimizer_htps_state_dict"] = optimizer_htps.state_dict()
    if tactic_queue is not None:
        payload["tactic_queue"] = [
            {
                "goal": s.goal.model_dump(mode="json"),
                "graph_fingerprint": s.graph_fingerprint,
                "tactic_id": s.tactic_id,
                "arg_indices": list(s.arg_indices),
            }
            for s in tactic_queue
        ]
    if critic_queue is not None:
        payload["critic_queue"] = [
            {
                "node_id": s.node_id,
                "goal": s.goal.model_dump(mode="json"),
                "graph_fingerprint": s.graph_fingerprint,
                "target": s.target,
                "source": s.source.value,
            }
            for s in critic_queue
        ]
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Greedy evaluation
# ---------------------------------------------------------------------------


async def evaluate_proof_rate(
    reasoner: RLHybridReasoner,
    eval_items: list[TheoremItem],
    *,
    timeout_s: float = 60.0,
    reasoner_pool: LiveReasonerPool | None = None,
    model_generation: int = 0,
) -> dict[str, float]:
    """Greedy proof rate on the fixed held-out pool — the model-selection metric.

    Greedy (argmax) actions measure the policy itself, not the sampler; the fixed
    pool makes the number comparable across rounds. Training return is neither
    (sampled policy, shifting curriculum window), which is why it does not pick
    ``best.pt``.
    """
    was_training = reasoner.model.training
    reasoner.model.eval()
    try:
        _results, stats = await collect_round(
            reasoner,
            eval_items,
            timeout_s=timeout_s,
            greedy=True,
            reasoner_pool=reasoner_pool,
            model_generation=model_generation,
        )
    finally:
        if was_training:
            reasoner.model.train()
    attempted = stats["attempted"] or 1.0
    return {
        "proof_rate": stats["solved"] / attempted,
        "solved": stats["solved"],
        "attempted": stats["attempted"],
        "searches_failed": stats["searches_failed"],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _create_run_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


async def _create_live_reasoner_pool(
    *,
    model: ActorCriticWithArgsClassifier,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    cfg: RLTrainingConfig,
    env: PantographEnv,
    plan: ResourcePlan,
    model_generation: int = 0,
) -> LiveReasonerPool:
    """Create workers whose reasoners preserve the complete MCTS configuration."""
    from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor

    async def server_factory(_worker_id: int):
        return await create_model_sexpr_server(env)

    def reasoner_factory(worker_id, device, replica, server):
        del worker_id
        return RLHybridReasoner(
            model=replica,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            executor=PantographExecutor(server),
            device=device,
            top_k_tactics=cfg.top_k_tactics,
            max_depth=cfg.max_depth,
            max_nodes=cfg.max_nodes,
            selection_policy=cfg.selection_policy,
            num_simulations=cfg.num_simulations,
            sim_batch_size=cfg.sim_batch_size,
            puct_c=cfg.puct_c,
            use_pln=cfg.use_pln,
            env=env,
        )

    async def search_factory(reasoner, theorem, greedy, deadline):
        return await reasoner.prove(
            theorem.goal.expression,
            hypotheses=theorem.goal.hypotheses,
            greedy=greedy,
            deadline=deadline,
        )

    return await LiveReasonerPool.create(
        model=model,
        plan=plan,
        server_factory=server_factory,
        reasoner_factory=reasoner_factory,
        search_factory=search_factory,
        is_solved=lambda result: result.graph.is_solved(),
        is_cooperative_deadline=lambda result: (
            result.graph.end_reason == SearchEndReason.DEADLINE
        ),
        startup_timeout_s=cfg.server_start_timeout_s,
        shutdown_timeout_s=cfg.worker_shutdown_timeout_s,
        queue_size=cfg.worker_queue_size,
        environment_description=env.describe(),
        model_generation=model_generation,
        log=console_print,
    )


async def run_rl_training(
    cfg: RLTrainingConfig,
    *,
    resume_run_dir: Path | None = None,
    reasoner_factory=None,
    pool: TheoremPool | None = None,
) -> dict[str, float]:
    """Run training while guaranteeing cleanup of every live Pantograph worker."""

    async with AsyncExitStack() as live_stack:
        return await _run_rl_training_impl(
            cfg,
            resume_run_dir=resume_run_dir,
            reasoner_factory=reasoner_factory,
            pool=pool,
            live_stack=live_stack,
        )


async def _run_rl_training_impl(
    cfg: RLTrainingConfig,
    *,
    resume_run_dir: Path | None,
    reasoner_factory,
    pool: TheoremPool | None,
    live_stack: AsyncExitStack,
) -> dict[str, float]:
    """The round loop: collect → one gradient step → anneal/checkpoint/eval.

    ``reasoner_factory(model, node_vocab, tactic_vocab, cfg) -> RLHybridReasoner``
    and ``pool`` are injectable for tests (a mock executor and a synthetic pool);
    both default to the live Pantograph path and the configured data source.
    """
    resource_plan = cfg.resource_plan()
    device = resource_plan.update_device
    console_print(f"RL resources: {resource_plan.describe()}")
    metadata = load_prepared_metadata(cfg.prepared_root)
    require_graph_representation(metadata.graph_representation)
    node_vocab, tactic_vocab = metadata.node_vocab, metadata.tactic_vocab

    checkpoint = torch.load(cfg.warmstart_checkpoint, map_location=device, weights_only=False)
    model, _manifest, model_spec = build_model_from_checkpoint(
        checkpoint,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        graph_representation=metadata.graph_representation,
        expected_model_kind="actor_critic_with_args",
    )
    model = model.to(device)
    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    console_print(
        f"Warm start (strict, {model_spec.architecture}): {cfg.warmstart_checkpoint}"
    )
    console_print(f"Canonical model parameters: {model_parameter_bytes} bytes")

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # Decoupled HTPS-style step: separate optimizer instance over the SAME
    # parameters, so the supervised imitation/soft-critic gradient keeps its own
    # Adam moment estimates instead of contaminating the on-policy ones.
    optimizer_htps = AdamW(
        model.parameters(), lr=cfg.htps_learning_rate, weight_decay=cfg.weight_decay
    )
    assert optimizer_htps is not optimizer  # the decoupled step must never touch the on-policy optimizer

    # Replay queues for the decoupled step (Decision 1.4: checkpointed).
    tactic_queue: deque[TacticImitationSample] = deque(maxlen=cfg.tactic_queue_size)
    critic_queue: deque[CriticSample] = deque(maxlen=cfg.critic_queue_size)
    rng = random.Random(cfg.seed)

    # Run dir / resume.
    start_round = 0
    onpolicy_steps = 0
    best_proof_rate = -1.0
    curriculum_size_override: Optional[int] = None
    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir)
        last_path = run_dir / "last.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"Resume requested but {last_path} does not exist")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("rl_target_schema_version") != 3:
            raise ValueError(
                "This checkpoint lacks canonical model-S-expression replay queues and "
                "cannot be resumed exactly. Use its model checkpoint as a warm start "
                "for a fresh RL run."
            )
        if state.get("search_settings") != search_settings(cfg):
            raise ValueError(
                "Resume configuration changes the saved search semantics. Start a fresh "
                "RL run when changing PUCT, depth, node, tactic, PLN, or visit settings."
            )
        resumed_model, _resume_manifest, resume_spec = build_model_from_checkpoint(
            state,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            graph_representation=metadata.graph_representation,
            expected_model_kind="actor_critic_with_args",
        )
        if resume_spec != model_spec:
            raise ValueError(
                "Resume checkpoint model specification does not match the configured "
                "warm-start checkpoint."
            )
        model.load_state_dict(resumed_model.state_dict(), strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        optimizer_htps.load_state_dict(state["optimizer_htps_state_dict"])
        tactic_queue.extend(
            TacticImitationSample(
                goal=Goal.model_validate(row["goal"]),
                graph_fingerprint=str(row["graph_fingerprint"]),
                tactic_id=int(row["tactic_id"]),
                arg_indices=tuple(int(value) for value in row["arg_indices"]),
            )
            for row in state.get("tactic_queue", [])
        )
        critic_queue.extend(
            CriticSample(
                node_id=int(row["node_id"]),
                goal=Goal.model_validate(row["goal"]),
                graph_fingerprint=str(row["graph_fingerprint"]),
                target=float(row["target"]),
                source=BackupSource(row["source"]),
            )
            for row in state.get("critic_queue", [])
        )
        for sample in [*tactic_queue, *critic_queue]:
            sample.goal.require_model_state()
            if (
                dag_fingerprint(model_goal_to_dag(sample.goal))
                != sample.graph_fingerprint
            ):
                raise ValueError(
                    "Resume checkpoint contains a replay sample whose canonical goal "
                    "does not match its graph fingerprint."
                )
        torch.set_rng_state(state["torch_rng_state"].cpu())
        start_round = int(state["round"]) + 1
        onpolicy_steps = int(state["onpolicy_steps"])
        best_proof_rate = float(state.get("best_proof_rate", -1.0))
        curriculum_size_override = int(state.get("curriculum_size", 0)) or None
        console_print(f"Resumed {run_dir} at round {start_round} (best proof rate {best_proof_rate:.3f})")
    else:
        run_dir = _create_run_dir(cfg.run_root)
        with open(run_dir / "config.json", "w") as f:
            run_config = cfg.to_dict()
            run_config["graph_representation"] = metadata.graph_representation.to_dict()
            json.dump(run_config, f, indent=2)
    metrics_path = run_dir / "metrics.jsonl"
    console_print(f"Run dir: {run_dir}")

    # Featurizer: ONE instance, shared by reasoner (via its own) and train step.
    torch.manual_seed(cfg.seed)

    # Reasoner (live Pantograph unless a factory is injected).
    model_generation = 0
    live_reasoner_pool: LiveReasonerPool | None = None
    if reasoner_factory is None:
        env = pantograph_env(cfg)
        env.verify()
        console_print(f"Pantograph environment: {env.describe()}")
        live_reasoner_pool = await _create_live_reasoner_pool(
            model=model,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            cfg=cfg,
            env=env,
            plan=resource_plan,
            model_generation=model_generation,
        )
        await live_stack.enter_async_context(live_reasoner_pool)
        reasoner = live_reasoner_pool.primary
    else:
        reasoner = reasoner_factory(model, node_vocab, tactic_vocab, cfg)

    # Theorem pool (injectable for tests).
    if pool is None:
        pool = build_theorem_pool(cfg)
    if curriculum_size_override:
        pool.curriculum_size = min(curriculum_size_override, len(pool.train_items))
    console_print(
        f"Pool: {len(pool.train_items)} train / {len(pool.eval_items)} eval; "
        f"curriculum window {pool.curriculum_size}"
    )

    reward_cfg = RewardConfig(
        gamma=cfg.reward_gamma,
        step_penalty=cfg.reward_step_penalty,
        terminal_success=cfg.reward_terminal_success,
        terminal_failure=cfg.reward_terminal_failure,
    )

    recent_solve_rates: list[float] = []
    last_metrics: dict[str, float] = {}
    dead_rounds = 0

    for round_idx in range(start_round, cfg.num_rounds):
        round_start = time.time()
        batch = pool.sample_batch(cfg.theorems_per_round)
        if not batch:
            console_print(f"Round {round_idx}: empty curriculum window — stopping.")
            break

        results, collect_stats = await collect_round(
            reasoner,
            batch,
            timeout_s=cfg.theorem_timeout_s,
            reasoner_pool=live_reasoner_pool,
            model_generation=model_generation,
        )

        bc_weight = bc_weight_at_round(onpolicy_steps, cfg)
        onpolicy_started = time.perf_counter()
        if results:
            # The score-function estimator performs at most one on-policy update
            # before replay mining. HTPS updates below use their separate optimizer.
            metrics = train_step_onpolicy(
                model,
                optimizer,
                results,
                reasoner.dag_featurize_data,
                reward_cfg=reward_cfg,
                grad_clip=cfg.grad_clip,
                device=device,
                critic_weight=cfg.critic_weight,
                entropy_weight=cfg.entropy_weight,
                arg_loss_weight=cfg.arg_loss_weight,
                bc_weight=bc_weight,
                max_update_nodes=cfg.max_update_nodes,
                max_update_edges=cfg.max_update_edges,
            )
        else:
            metrics = {
                "num_transitions": 0.0,
                "num_failures": 0.0,
                "unknown_label_count": 0.0,
                "structural_unknown_label_count": 0.0,
                "semantic_unknown_label_count": 0.0,
                "total_label_count": 0.0,
                "onpolicy_optimizer_step": 0.0,
            }
        metrics["onpolicy_update_s"] = time.perf_counter() - onpolicy_started
        onpolicy_optimizer_steps = int(metrics.get("onpolicy_optimizer_step", 0.0))
        onpolicy_steps += onpolicy_optimizer_steps

        # Mine every completed graph into replay before applying the configured
        # supervised HTPS updates. Replicas are not read anywhere in this phase.
        imitation_mined = 0
        for result in results:
            critic_queue.extend(
                extract_critic_samples(
                    result.graph,
                    visit_threshold=cfg.visit_threshold,
                )
            )
            mined = extract_minimal_hypertree(
                result.graph,
                result.edge_actions,
                mine_all_solved_nodes=cfg.mine_all_solved_nodes,
            )
            imitation_mined += len(mined)
            tactic_queue.extend(mined)
        metrics["imitation_samples_mined"] = float(imitation_mined)
        metrics["tactic_queue_len"] = float(len(tactic_queue))
        metrics["critic_queue_len"] = float(len(critic_queue))

        htps_optimizer_steps = 0
        htps_started = time.perf_counter()
        for _ in range(cfg.htps_steps_per_round):
            if not tactic_queue and not critic_queue:
                break
            tactic_batch = rng.sample(
                list(tactic_queue), min(cfg.htps_batch_size, len(tactic_queue))
            )
            critic_batch = rng.sample(
                list(critic_queue), min(cfg.htps_batch_size, len(critic_queue))
            )
            htps_metrics = train_step_htps_style(
                model,
                optimizer_htps,
                tactic_batch,
                critic_batch,
                reasoner.dag_featurize_data,
                w_critic_soft=cfg.w_critic_soft,
                arg_loss_weight=cfg.arg_loss_weight,
                grad_clip=cfg.grad_clip,
                device=device,
            )
            htps_optimizer_steps += int(
                htps_metrics.get("htps_optimizer_step", 0.0)
            )
            for key in (
                "tactic_imitation_loss",
                "critic_soft_loss",
                "htps_total_loss",
                "num_imitation_rows",
                "num_critic_rows",
            ):
                metrics[key] = htps_metrics[key]
            for key in (
                "unknown_label_count",
                "structural_unknown_label_count",
                "semantic_unknown_label_count",
                "total_label_count",
            ):
                metrics[f"htps_{key}"] = htps_metrics[key]
        metrics["htps_update_s"] = time.perf_counter() - htps_started
        metrics["htps_optimizer_steps"] = float(htps_optimizer_steps)

        parameter_updates_this_round = (
            onpolicy_optimizer_steps + htps_optimizer_steps
        )
        metrics["parameter_updates_this_round"] = float(
            parameter_updates_this_round
        )
        if parameter_updates_this_round:
            model_generation += parameter_updates_this_round
            if live_reasoner_pool is not None:
                collect_stats["replica_sync_s"] = live_reasoner_pool.synchronize(
                    model,
                    model_generation,
                )
                live_reasoner_pool.assert_generation(model_generation)
            else:
                collect_stats["replica_sync_s"] = 0.0
            dead_rounds = 0
        else:
            collect_stats["replica_sync_s"] = 0.0
            dead_rounds += 1

        collect_stats["model_generation"] = float(model_generation)
        collect_stats["replica_generation"] = float(
            live_reasoner_pool.model_generation
            if live_reasoner_pool is not None
            else model_generation
        )
        if dead_rounds >= cfg.max_dead_rounds:
            raise RuntimeError(
                "RL training produced no valid on-policy or HTPS optimizer update for "
                f"{cfg.max_dead_rounds} consecutive rounds. Inspect unknown-label and "
                "whole-search error metrics, and verify the Pantograph environment at "
                f"{cfg.source_root}."
            )

        # Curriculum: grow when the recent training-window solve rate crosses threshold.
        solve_rate = collect_stats["solved"] / (collect_stats["attempted"] or 1.0)
        recent_solve_rates.append(solve_rate)
        if len(recent_solve_rates) > cfg.curriculum_window_rounds:
            recent_solve_rates.pop(0)
        window_rate = sum(recent_solve_rates) / len(recent_solve_rates)
        if (
            len(recent_solve_rates) >= cfg.curriculum_window_rounds
            and window_rate >= cfg.curriculum_solve_threshold
            and pool.curriculum_size < len(pool.train_items)
        ):
            pool.grow(cfg.curriculum_growth_factor)
            recent_solve_rates.clear()
            console_print(f"  Curriculum widened to {pool.curriculum_size} (solve rate {window_rate:.2f})")

        row = {
            "round": round_idx,
            "bc_weight": bc_weight,
            "curriculum_size": pool.curriculum_size,
            "wall_clock_s": time.time() - round_start,
            **collect_stats,
            **metrics,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        console_print(
            f"Round {round_idx}: solved {collect_stats['solved']:.0f}/{collect_stats['attempted']:.0f}, "
            f"trans {metrics.get('num_transitions', 0):.0f}, fail {metrics.get('num_failures', 0):.0f}, "
            f"return {metrics.get('mean_return', 0.0):.3f}, loss {metrics.get('total_loss', 0.0):.3f}, "
            f"bc {bc_weight:.3f}, {row['wall_clock_s']:.1f}s"
        )
        last_metrics = row

        if (round_idx + 1) % cfg.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, round_idx, pool.curriculum_size, best_proof_rate,
                onpolicy_steps,
                run_dir / "last.pt",
                optimizer_htps=optimizer_htps,
                tactic_queue=tactic_queue,
                critic_queue=critic_queue,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                graph_representation=metadata.graph_representation,
                saved_search_settings=search_settings(cfg),
            )

        if cfg.eval_every > 0 and (round_idx + 1) % cfg.eval_every == 0 and pool.eval_items:
            if live_reasoner_pool is not None:
                live_reasoner_pool.assert_generation(model_generation)
            eval_stats = await evaluate_proof_rate(
                reasoner,
                pool.eval_items,
                timeout_s=cfg.theorem_timeout_s,
                reasoner_pool=live_reasoner_pool,
                model_generation=model_generation,
            )
            console_print(f"  Eval: proof rate {eval_stats['proof_rate']:.3f}")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"round": round_idx, "eval": eval_stats}) + "\n")
            if eval_stats["proof_rate"] > best_proof_rate:
                best_proof_rate = eval_stats["proof_rate"]
                save_checkpoint(
                    model, optimizer, round_idx, pool.curriculum_size, best_proof_rate,
                    onpolicy_steps,
                    run_dir / "best.pt",
                    optimizer_htps=optimizer_htps,
                    tactic_queue=tactic_queue,
                    critic_queue=critic_queue,
                    node_vocab=node_vocab,
                    tactic_vocab=tactic_vocab,
                    graph_representation=metadata.graph_representation,
                    saved_search_settings=search_settings(cfg),
                )
                console_print(f"  New best proof rate {best_proof_rate:.3f} → best.pt")

    # Final checkpoint so the run is always resumable from its end state.
    save_checkpoint(
        model, optimizer, cfg.num_rounds - 1, pool.curriculum_size, best_proof_rate,
        onpolicy_steps,
        run_dir / "last.pt",
        optimizer_htps=optimizer_htps,
        tactic_queue=tactic_queue,
        critic_queue=critic_queue,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        graph_representation=metadata.graph_representation,
        saved_search_settings=search_settings(cfg),
    )
    return last_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def driver_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="On-policy RL training over live Lean search")
    parser.add_argument("--config", type=str, required=True, help="Path to the RL training JSON config")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Hugging Face dataset identifier used for dataset-mode theorem streaming",
    )
    parser.add_argument("--resume", type=str, default=None, help="Run directory to resume (contains last.pt)")
    parser.add_argument("--eval-only", action="store_true", help="Only run the greedy proof-rate evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint override for --eval-only (defaults to warmstart_checkpoint)")
    args = parser.parse_args(argv)

    cfg = RLTrainingConfig.from_json(args.config)
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.eval_only:
        if args.checkpoint:
            cfg.warmstart_checkpoint = Path(args.checkpoint)
        cfg.num_rounds = 0
        cfg.eval_every = 0

        async def _eval() -> None:
            environment = pantograph_env(cfg)
            environment.verify()
            console_print(f"Pantograph environment: {environment.describe()}")

            resource_plan = cfg.resource_plan()
            device = resource_plan.update_device
            console_print(f"RL resources: {resource_plan.describe()}")
            metadata = load_prepared_metadata(cfg.prepared_root)
            require_graph_representation(metadata.graph_representation)
            node_vocab, tactic_vocab = metadata.node_vocab, metadata.tactic_vocab
            checkpoint = torch.load(
                cfg.warmstart_checkpoint,
                map_location=device,
                weights_only=False,
            )
            model, _manifest, _model_spec = build_model_from_checkpoint(
                checkpoint,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                graph_representation=metadata.graph_representation,
                expected_model_kind="actor_critic_with_args",
            )
            model = model.to(device)

            live_pool = await _create_live_reasoner_pool(
                model=model,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                cfg=cfg,
                env=environment,
                plan=resource_plan,
                model_generation=0,
            )
            async with live_pool:
                theorem_pool = build_theorem_pool(cfg)
                live_pool.assert_generation(0)
                stats = await evaluate_proof_rate(
                    live_pool.primary,
                    theorem_pool.eval_items,
                    timeout_s=cfg.theorem_timeout_s,
                    reasoner_pool=live_pool,
                    model_generation=0,
                )
                console_print(
                    f"Proof rate: {stats['proof_rate']:.3f} "
                    f"({stats['solved']:.0f}/{stats['attempted']:.0f})"
                )

        asyncio.run(_eval())
        return 0

    resume_dir = Path(args.resume) if args.resume else None
    asyncio.run(run_rl_training(cfg, resume_run_dir=resume_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(driver_main())
