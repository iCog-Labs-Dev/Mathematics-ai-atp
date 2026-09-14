"""RL training driver: warm-start from supervised checkpoints, curriculum over
LeanDojo proof states, BC annealing, fault tolerance, eval-by-proof-rate.

Encodes the three invariants:
  - one optimizer step per collect round (on-policy A2C),
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW

from maths_ai.data_models.proof_components import Goal
from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv

from .actor_critic import ActorCriticWithArgsClassifier
from .checkpointing import build_model_from_checkpoint, checkpoint_payload
from .dataset import iter_dataset_rows
from .pln_reward import RewardConfig
from .pln_rl_training import make_dag_featurizer, train_step_onpolicy
from .reporting import console_print
from .rl_reasoner import RLHybridReasoner, RLSearchResult
from .state import parse_state


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLTrainingConfig:
    """Configuration for the RL training driver (flat JSON, ``from_json`` below).

    ``warmstart_checkpoint`` is a version-2, self-describing supervised
    actor-critic checkpoint. Its manifest owns the encoder architecture.
    """

    warmstart_checkpoint: Path
    prepared_root: Path

    # Theorem sourcing: "dataset" streams LeanDojo proof states; "file" reads a
    # JSONL of {"goal": str, "hypotheses": [str, ...]} rows.
    data_source: str = "dataset"
    theorem_file: Path | None = None
    leandojo_split: str = "train"
    max_pool_size: int = 5000
    max_state_chars: int = 400
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
    # Stop the run when this many consecutive rounds harvest no transitions. A loop
    # that collects nothing is misconfigured — most often a Lean environment that
    # cannot elaborate the pool — and annealing quietly for hours hides it.
    max_dead_rounds: int = 3

    # Search budgets (RLHybridReasoner).
    top_k_tactics: int = 4
    max_depth: int = 8
    max_nodes: int = 64
    # Set to False to disable all PLN involvement (no petta subprocess, no reward
    # shaping, no PLN fallback QED). Terminal rewards and the step penalty remain.
    use_pln: bool = True

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

    # Lean environment the Pantograph server runs in. `source_root` is the Lake
    # project whose compiled artifacts the REPL should see; leaving it None starts a
    # core-Lean server that cannot elaborate Mathlib notation such as `ℕ` or `⌊…⌋₊`.
    source_root: Path | None = None
    pantograph_repl: Path | None = None
    pantograph_imports: list[str] | None = None  # None → resolved from source_root
    server_timeout_s: int = 120

    _PATH_FIELDS = (
        "warmstart_checkpoint", "prepared_root", "theorem_file", "run_root",
        "source_root", "pantograph_repl",
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "RLTrainingConfig":
        with open(path) as f:
            payload = json.load(f)
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
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


# ---------------------------------------------------------------------------
# Lean environment
# ---------------------------------------------------------------------------


def pantograph_env(cfg: RLTrainingConfig) -> PantographEnv:
    """Resolve the config's Lean-environment fields into one ``PantographEnv``.

    Both the training loop and the standalone evaluation path call this, so the
    server they start and the server a post-crash restart re-starts are described
    by the same value.

    ``imports`` defaults by ``source_root``: a run pointed at a Mathlib project
    wants ``Mathlib`` on the import line, and a run with no project cannot have it
    — importing a module outside ``LEAN_PATH`` makes the REPL fail at startup
    rather than degrade. ``pantograph_imports`` overrides the default when a
    project exports a different top-level module.
    """
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


# ---------------------------------------------------------------------------
# Theorem pool + curriculum
# ---------------------------------------------------------------------------


@dataclass
class TheoremItem:
    goal: Goal
    tactic_label: str  # ground-truth tactic from the dataset row ("" in file mode)
    size: int


# `?m.4519` is Lean's pretty-printed form of an unassigned metavariable — a hole the
# elaborator had not yet solved when the dataset row was captured mid-proof. Such a
# state is not a theorem statement: `goal_start` has no assignment to give the hole,
# so it fails to elaborate and the rollout is wasted before its first tactic.
_METAVARIABLE_RE = re.compile(r"\?m\.\d+|\?[a-zA-Z_][a-zA-Z0-9_]*\b")


def _has_metavariable(goal: Goal) -> bool:
    """True when the goal or any hypothesis mentions an unassigned metavariable."""
    if _METAVARIABLE_RE.search(goal.expression):
        return True
    return any(_METAVARIABLE_RE.search(hypothesis.render()) for hypothesis in goal.hypotheses)


def _row_state_to_goal(state_str: str) -> Goal:
    """Dataset row's pretty-printed state → ``Goal`` via the SAME parser the
    featurizer uses (``parse_state``), so pool goals and rollout goals agree on
    hypothesis splitting."""
    parsed = parse_state(state_str)
    return Goal(
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
                goal = Goal(expression=row["goal"], hypotheses=row.get("hypotheses", []))
                size = len(goal.expression) + sum(len(h.render()) for h in goal.hypotheses)
                if size > cfg.max_state_chars:
                    dropped += 1
                    continue
                if _has_metavariable(goal):
                    dropped += 1
                    continue
                items.append(TheoremItem(goal=goal, tactic_label=row.get("tactic", ""), size=size))
                if len(items) >= cfg.max_pool_size:
                    break
    elif cfg.data_source == "dataset":
        for row in iter_dataset_rows(split=cfg.leandojo_split, sample_limit=cfg.max_pool_size * 2):
            state_str = (row.state or "").strip()
            if not state_str or len(state_str) > cfg.max_state_chars:
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
    ``bc_anneal_rounds``; constant at the end value afterwards.

    ``round_idx`` counts rounds that produced an optimizer step, not loop
    iterations. The anneal exists to hand control from the supervised anchor to
    the policy gradient as the policy improves, and a round that collected no
    transitions took no step, so the policy did not change and the anchor must
    not be weakened for it.
    """
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
) -> tuple[list[RLSearchResult], dict[str, float]]:
    """Sequential collect with per-theorem fault isolation.

    A theorem whose search raises or times out contributes no result; the round
    trains on the survivors. Sequential because one reasoner holds one action
    stash at a time (refinement 6).
    """
    results: list[RLSearchResult] = []
    solved = 0
    failed = 0
    for item in batch:
        try:
            result = await asyncio.wait_for(
                reasoner.prove(item.goal.expression, hypotheses=[h.render() for h in item.goal.hypotheses], greedy=greedy),
                timeout=timeout_s,
            )
            results.append(result)
            if result.graph.is_solved():
                solved += 1
        except asyncio.TimeoutError:
            failed += 1
            console_print(f"  [collect] timeout on: {item.goal.expression[:60]}")
        except Exception as exc:  # noqa: BLE001 — a single search may not kill the run
            failed += 1
            console_print(f"  [collect] error on: {item.goal.expression[:60]} — {exc}")
    stats = {
        "attempted": float(len(batch)),
        "collected": float(len(results)),
        "solved": float(solved),
        "searches_failed": float(failed),
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
    path: Path,
    *,
    anneal_rounds_done: int = 0,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model_kind="actor_critic_with_args",
            model_spec=model.model_spec,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            model=model,
            optimizer_state_dict=optimizer.state_dict(),
            round=round_idx,
            curriculum_size=curriculum_size,
            best_proof_rate=best_proof_rate,
            # The BC-anneal clock counts optimizer steps, not loop iterations, so it
            # cannot be recomputed from `round` on resume.
            anneal_rounds_done=anneal_rounds_done,
            torch_rng_state=torch.get_rng_state(),
        ),
        path,
    )


# ---------------------------------------------------------------------------
# Greedy evaluation
# ---------------------------------------------------------------------------


async def evaluate_proof_rate(
    reasoner: RLHybridReasoner,
    eval_items: list[TheoremItem],
    *,
    timeout_s: float = 60.0,
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
            reasoner, eval_items, timeout_s=timeout_s, greedy=True
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


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_vocabs(prepared_root: Path) -> tuple[dict[str, int], dict[str, int]]:
    node_vocab_path = prepared_root / "vocab" / "node_vocab.json"
    tactic_vocab_path = prepared_root / "vocab" / "tactic_vocab.json"
    for p in (node_vocab_path, tactic_vocab_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing vocab file: {p}")
    with open(node_vocab_path) as f:
        node_vocab = {str(k): int(v) for k, v in json.load(f).items()}
    with open(tactic_vocab_path) as f:
        tactic_vocab = {str(k): int(v) for k, v in json.load(f).items()}
    return node_vocab, tactic_vocab


def _create_run_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


async def _create_live_reasoner(
    *,
    model: ActorCriticWithArgsClassifier,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    cfg: RLTrainingConfig,
    device: torch.device,
    env: PantographEnv,
) -> RLHybridReasoner:
    """Create the live search reasoner from the complete RL search config."""
    from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor

    server = await env.create_server()
    return RLHybridReasoner(
        model=model,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        executor=PantographExecutor(server),
        device=device,
        top_k_tactics=cfg.top_k_tactics,
        max_depth=cfg.max_depth,
        max_nodes=cfg.max_nodes,
        use_pln=cfg.use_pln,
        env=env,
    )


async def run_rl_training(
    cfg: RLTrainingConfig,
    *,
    resume_run_dir: Path | None = None,
    reasoner_factory=None,
    pool: TheoremPool | None = None,
) -> dict[str, float]:
    """The round loop: collect → one gradient step → anneal/checkpoint/eval.

    ``reasoner_factory(model, node_vocab, tactic_vocab, cfg) -> RLHybridReasoner``
    and ``pool`` are injectable for tests (a mock executor and a synthetic pool);
    both default to the live Pantograph path and the configured data source.
    """
    device = _resolve_device(cfg.device)
    node_vocab, tactic_vocab = _load_vocabs(cfg.prepared_root)

    # The checkpoint manifest owns the complete actor-critic architecture.
    checkpoint = torch.load(cfg.warmstart_checkpoint, map_location=device, weights_only=False)
    model, warmstart_manifest, warmstart_spec = build_model_from_checkpoint(
        checkpoint,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        expected_model_kind="actor_critic_with_args",
    )
    model = model.to(device)
    console_print(f"Warm start (strict): {cfg.warmstart_checkpoint}")

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Run dir / resume.
    start_round = 0
    best_proof_rate = -1.0
    curriculum_size_override: Optional[int] = None
    anneal_rounds_done = 0  # rounds with an optimizer step; the BC-anneal clock
    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir)
        last_path = run_dir / "last.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"Resume requested but {last_path} does not exist")
        state = torch.load(last_path, map_location=device, weights_only=False)
        resumed_model, _, resumed_spec = build_model_from_checkpoint(
            state,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            expected_model_kind="actor_critic_with_args",
        )
        if resumed_spec != warmstart_spec:
            raise ValueError("Resume checkpoint model specification differs from the warm start.")
        model = resumed_model.to(device)
        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        start_round = int(state["round"]) + 1
        best_proof_rate = float(state.get("best_proof_rate", -1.0))
        curriculum_size_override = int(state.get("curriculum_size", 0)) or None
        anneal_rounds_done = int(state.get("anneal_rounds_done", 0)) or 0
        console_print(f"Resumed {run_dir} at round {start_round} (best proof rate {best_proof_rate:.3f})")
    else:
        run_dir = _create_run_dir(cfg.run_root)
        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
    metrics_path = run_dir / "metrics.jsonl"
    console_print(f"Run dir: {run_dir}")

    # Featurizer: ONE instance, shared by reasoner (via its own) and train step.
    torch.manual_seed(cfg.seed)

    # Reasoner (live Pantograph unless a factory is injected).
    if reasoner_factory is None:
        env = pantograph_env(cfg)
        env.verify()
        console_print(f"Pantograph environment: {env.describe()}")
        reasoner = await _create_live_reasoner(
            model=model,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            cfg=cfg,
            device=device,
            env=env,
        )
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
    dead_rounds = 0  # consecutive rounds that harvested no transitions

    for round_idx in range(start_round, cfg.num_rounds):
        round_start = time.time()
        batch = pool.sample_batch(cfg.theorems_per_round)
        if not batch:
            console_print(f"Round {round_idx}: empty curriculum window — stopping.")
            break

        results, collect_stats = await collect_round(
            reasoner, batch, timeout_s=cfg.theorem_timeout_s
        )

        # Indexed by optimizer steps taken, not loop iterations: a dead round leaves
        # the policy unchanged, so the anchor it needs is unchanged too.
        bc_weight = bc_weight_at_round(anneal_rounds_done, cfg)
        if results:
            # Exactly ONE optimizer step per collect round (on-policy invariant).
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
                "num_critic_samples": 0.0,
                "num_failures": 0.0,
                "optimizer_step": 0.0,
            }

        took_optimizer_step = bool(metrics.get("optimizer_step", 0.0))
        if took_optimizer_step:
            anneal_rounds_done += 1
            dead_rounds = 0
        else:
            dead_rounds += 1
        if dead_rounds >= cfg.max_dead_rounds:
            raise RuntimeError(
                f"Round {round_idx}: {dead_rounds} consecutive rounds produced no valid "
                f"actor or critic training rows ({collect_stats['searches_failed']:.0f} "
                f"searches failed). Check that --source-root points at the compiled "
                f"mathlib_lean project and that the toolchains match."
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
            "anneal_rounds_done": anneal_rounds_done,
            "curriculum_size": pool.curriculum_size,
            "wall_clock_s": time.time() - round_start,
            **collect_stats,
            **metrics,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        # Two distinct failure counts, both on the line (Issue 3): `rej` counts
        # tactics the executor refused inside searches that ran, `err` counts whole
        # searches that raised or timed out and contributed no transitions. A run
        # whose environment is broken shows rej 0 with err equal to the batch size,
        # which the old line rendered as `fail 0`.
        console_print(
            f"Round {round_idx}: solved {collect_stats['solved']:.0f}/{collect_stats['attempted']:.0f}, "
            f"trans {metrics.get('num_transitions', 0):.0f}, "
            f"rej {metrics.get('num_failures', 0):.0f}, "
            f"err {collect_stats['searches_failed']:.0f}, "
            f"return {metrics.get('mean_return', 0.0):.3f}, loss {metrics.get('total_loss', 0.0):.3f}, "
            f"bc {bc_weight:.3f}, {row['wall_clock_s']:.1f}s"
        )
        last_metrics = row

        if (round_idx + 1) % cfg.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, round_idx, pool.curriculum_size, best_proof_rate,
                run_dir / "last.pt", anneal_rounds_done=anneal_rounds_done,
                node_vocab=node_vocab, tactic_vocab=tactic_vocab,
            )

        if cfg.eval_every > 0 and (round_idx + 1) % cfg.eval_every == 0 and pool.eval_items:
            eval_stats = await evaluate_proof_rate(
                reasoner, pool.eval_items, timeout_s=cfg.theorem_timeout_s
            )
            console_print(f"  Eval: proof rate {eval_stats['proof_rate']:.3f}")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"round": round_idx, "eval": eval_stats}) + "\n")
            if eval_stats["proof_rate"] > best_proof_rate:
                best_proof_rate = eval_stats["proof_rate"]
                save_checkpoint(
                    model, optimizer, round_idx, pool.curriculum_size, best_proof_rate,
                    run_dir / "best.pt", anneal_rounds_done=anneal_rounds_done,
                    node_vocab=node_vocab, tactic_vocab=tactic_vocab,
                )
                console_print(f"  New best proof rate {best_proof_rate:.3f} → best.pt")

    # Final checkpoint so the run is always resumable from its end state.
    save_checkpoint(
        model, optimizer, cfg.num_rounds - 1, pool.curriculum_size, best_proof_rate,
        run_dir / "last.pt", anneal_rounds_done=anneal_rounds_done,
        node_vocab=node_vocab, tactic_vocab=tactic_vocab,
    )
    return last_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def driver_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="On-policy RL training over live Lean search")
    parser.add_argument("--config", type=str, required=True, help="Path to the RL training JSON config")
    parser.add_argument("--resume", type=str, default=None, help="Run directory to resume (contains last.pt)")
    parser.add_argument("--eval-only", action="store_true", help="Only run the greedy proof-rate evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint override for --eval-only (defaults to warmstart_checkpoint)")
    parser.add_argument("--source-root", type=str, default=None,
                        help="Lake project root whose compiled .olean artifacts the Pantograph "
                             "REPL should see. Without it the REPL runs on core Lean only and "
                             "cannot elaborate Mathlib notation such as ℕ.")
    parser.add_argument("--pantograph-repl", type=str, default=None,
                        help="Pantograph REPL binary to run instead of the bundled one. Its Lean "
                             "toolchain must match --source-root's.")
    parser.add_argument("--pantograph-imports", type=str, default=None,
                        help="Comma-separated modules the server imports at startup "
                             "(default: Init,Mathlib when --source-root is set, else Init)")
    parser.add_argument("--server-timeout", type=int, default=None,
                        help="Per-request Pantograph timeout in seconds")
    args = parser.parse_args(argv)

    cfg = RLTrainingConfig.from_json(args.config)
    # Applied only when given, so a flag never overwrites a configured value with a default.
    if args.source_root:
        cfg.source_root = Path(args.source_root)
    if args.pantograph_repl:
        cfg.pantograph_repl = Path(args.pantograph_repl)
    if args.pantograph_imports:
        cfg.pantograph_imports = [m.strip() for m in args.pantograph_imports.split(",") if m.strip()]
    if args.server_timeout is not None:
        cfg.server_timeout_s = args.server_timeout
    if args.eval_only:
        if args.checkpoint:
            cfg.warmstart_checkpoint = Path(args.checkpoint)
        cfg.num_rounds = 0
        cfg.eval_every = 0

        async def _eval() -> None:
            # Verify the Lean environment before the checkpoint load: a wrong
            # --source-root then fails in under a second rather than after several
            # minutes of model construction.
            env = pantograph_env(cfg)
            env.verify()
            console_print(f"Pantograph environment: {env.describe()}")

            device = _resolve_device(cfg.device)
            node_vocab, tactic_vocab = _load_vocabs(cfg.prepared_root)
            checkpoint = torch.load(cfg.warmstart_checkpoint, map_location=device, weights_only=False)
            model, _, _ = build_model_from_checkpoint(
                checkpoint,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                expected_model_kind="actor_critic_with_args",
            )
            model = model.to(device)

            reasoner = await _create_live_reasoner(
                model=model,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                cfg=cfg,
                device=device,
                env=env,
            )
            pool = build_theorem_pool(cfg)
            stats = await evaluate_proof_rate(reasoner, pool.eval_items, timeout_s=cfg.theorem_timeout_s)
            console_print(f"Proof rate: {stats['proof_rate']:.3f} ({stats['solved']:.0f}/{stats['attempted']:.0f})")

        asyncio.run(_eval())
        return 0

    resume_dir = Path(args.resume) if args.resume else None
    asyncio.run(run_rl_training(cfg, resume_run_dir=resume_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(driver_main())
