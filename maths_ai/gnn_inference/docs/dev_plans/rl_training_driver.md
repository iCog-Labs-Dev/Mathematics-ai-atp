# RL training driver: rounds of collect → train over real theorems

## Purpose and the exact gap

Every mechanism of the on-policy loop exists and is tested: `RLHybridReasoner` samples and
stashes actions during search, `search_harvest` + `pln_reward` turn the finished
`ProofHypergraph` into returns and value targets, and
`pln_rl_training.collect_and_train_onpolicy` runs one round (sequential collect → exactly one
gradient step). What is missing is the **driver** that repeats this round over real data with
the operational scaffolding a multi-hour run needs: theorem sourcing, warm-start loading,
BC-anchor annealing, fault tolerance around Lean/petta, checkpointing, metrics, and periodic
proof-rate evaluation. `rl_smoke.py` proves the plumbing on one hard-coded goal and exits; it
is a smoke test, not a trainer.

## Inputs and their producers

| Input | Producer | Format |
|---|---|---|
| node/tactic vocabs | `prepare_dataset.py` → `artifacts/prepared/v1/vocab/*.json` | loaded via `load_prepared_metadata` |
| warm-start weights | phase-3 supervised actor-critic run → `runs/actor_critic_gnn/<run>/best.pt` | `{"model_state_dict": …}` for `ActorCriticWithArgsClassifier`, loaded strict |
| theorems | LeanDojo benchmark rows (`dataset.iter_dataset_rows`, train split) or a user JSONL | each row's `state` string → `parse_state` → `Goal(expression, hypotheses)` |
| Lean environment | live Pantograph `Server.create()` | one long-lived server per driver process |
| PLN | `PLNInference` (petta subprocess, de-blocked via `evaluate_async`) | constructed inside the reasoner |

The driver does NOT accept a pointer checkpoint. Warm-starting from the pointer directly would
skip the supervised actor-critic phase and start RL with a random critic *and* no
actor-residual training history; the staged path (pointer → supervised actor-critic →
RL) is the design the warm-start plan fixed. If phase 3 is ever skipped deliberately, the
fallback is one flag (`--from-pointer`) that routes through `load_from_pointer_checkpoint`;
default is the strict full load.

## Design

### Entry point and configuration

New script `atp_lean_gnn/rl_training_driver.py` with a thin launcher
`scripts/rl_train.py` (same pattern as `train_baseline.py` → `training.train_main`).
Configuration is one new JSON, `configs/rl_actor_critic.json`, parsed into a frozen
`RLDriverConfig` dataclass mirroring how `ActorCriticConfig` is handled in `training.py`:

```json
{
  "prepared_root": "artifacts/prepared/v1",
  "run_root": "runs/rl_actor_critic",
  "warmstart_checkpoint": "runs/actor_critic_gnn/<run>/best.pt",
  "seed": 42,
  "device": "auto",
  "model": { "hidden_dim": 512, "num_layers": 4, "dropout": 0.2, "max_args": 3 },
  "search": { "top_k_tactics": 4, "top_k_subgoals": 3, "max_depth": 8, "max_nodes": 64,
              "theorem_timeout_s": 120 },
  "reward": { "gamma": 0.99, "terminal_success": 1.0, "terminal_failure": 0.0,
              "step_penalty": 0.01 },
  "loss":   { "critic_weight": 0.5, "entropy_weight": 0.01, "arg_loss_weight": 0.5 },
  "bc":     { "start": 0.5, "end": 0.05, "anneal_rounds": 200 },
  "training": { "rounds": 500, "theorems_per_round": 8, "learning_rate": 1e-4,
                "weight_decay": 1e-4, "grad_clip": 1.0,
                "checkpoint_every": 20, "eval_every": 25 },
  "data": { "source": "dataset", "theorem_file": null,
            "max_state_chars": 400, "pool_size": 5000, "eval_pool_size": 200 }
}
```

`model` must match the warm-start checkpoint's architecture; the strict `load_state_dict`
makes a mismatch a hard error at startup, not a silent random init.

### Theorem sourcing and curriculum

`build_theorem_pool(config, metadata)` yields `Goal` objects:

- **dataset mode** — stream `iter_dataset_rows("train")`, parse each row's `state` with
  `parse_state`, build `Goal(expression=parsed.goal, hypotheses=["name : type", …])`. Rows
  are *intermediate proof states*, which is exactly what we want: they are one-to-a-few
  tactics from closure far more often than whole theorem statements, so QED terminals —
  the only strong reward — actually fire early in training (the Approach-5 easy-mixing
  rationale).
- **file mode** — JSONL of `{"goal": str, "hypotheses": [str, …]}` for hand-picked or
  synthetic curricula.

Curriculum = filter + sort, not a scheduler: drop states longer than `max_state_chars`
(unparseable/degenerate tails), take the first `pool_size`, sort ascending by
`len(hypotheses) + len(expression)` as the difficulty proxy, and sample each round's batch
from a window that slides from the easy end toward the full pool as the measured
solve-rate crosses a threshold (e.g. widen when >30% of the last window's searches solved).
Windowing is a few lines over a sorted list; anything fancier is untestable speculation
before the first real run.

### The round loop

```
load metadata (vocabs) ── load model ── strict load warmstart ── build reasoner ── server
for round in 1..rounds:
    theorems = sample(window, theorems_per_round)
    results  = asyncio.run(collect_batch_onpolicy(reasoner, theorems))   # sequential per theorem
    bc_w     = anneal(round)
    metrics  = train_step_onpolicy(model, optimizer, results, featurize, bc_weight=bc_w, …)
    append metrics + solve stats to metrics.jsonl
    every checkpoint_every: save last.pt
    every eval_every: greedy proof-rate on eval pool → save best.pt if improved
```

Invariants the loop encodes (not merely documents):

1. **One `optimizer.step()` per collect** — already the shape of
   `collect_and_train_onpolicy`; the driver never calls `train_step_onpolicy` twice on one
   round's results. This is what licenses the recompute-at-train-time log-probs.
2. **Featurizer identity** — `reasoner.dag_featurize_data` is passed to
   `train_step_onpolicy`, never a second featurizer instance, so collect-time argument
   indices align with train-time DAGs. Both are built from `metadata.node_vocab`.
3. **Vocab identity** — `metadata.node_vocab` / `metadata.tactic_vocab` from
   `prepared_root`, never `build_vocab` over rollout states (which would scramble the
   warm-started embeddings).

### Fault tolerance

A 500-round run will hit Lean timeouts, petta failures, and unparseable states; none may
kill the run, and none may corrupt a gradient step.

- **Per-theorem isolation**: wrap each `reasoner.prove` in `try/except` +
  `asyncio.wait_for(…, theorem_timeout_s)`. A failed/timed-out search contributes no
  `RLSearchResult`; the round trains on the survivors. Log the exception per theorem.
- **Featurize guard**: `parse_state`/`proof_state_to_dag` failures during pool building
  drop the row (counted, logged); during training they cannot occur because collect
  already featurized the same goals.
- **Server recovery**: if a `prove` raises a Pantograph transport error, dispose the
  server, `await Server.create()` again, rebuild the executor, continue. One retry per
  round; a second consecutive transport failure aborts the run with the checkpoint intact.
- **Empty round**: `train_step_onpolicy` already returns `{"num_transitions": 0.0, …}`
  without stepping when nothing was collected; the driver logs and moves on.

### Checkpointing and metrics

Reuse the existing run-dir pattern (`_create_run_dir(run_root)`, `config.json` snapshot,
`metrics.jsonl`, `last.pt`/`best.pt` via the same payload shape as `_save_checkpoint`:
`model_state_dict`, `optimizer_state_dict`, `round`, `bc_weight`, RNG states). `--resume
<run_dir>` restores all of it and continues the round counter — a multi-day run on a shared
server will be preempted.

Per-round metrics row: everything `train_step_onpolicy` returns, plus
`solved/attempted`, `searches_failed`, `wall_clock_s`, `bc_weight`, curriculum window
bounds. The existing `analyze_run.py` conventions apply to this `metrics.jsonl` unchanged.

### Evaluation: greedy proof rate

`evaluate_proof_rate(reasoner_factory, model, eval_goals, …) -> {"proof_rate", "solved",
"attempted", "mean_nodes"}`. Mechanics: a second `RLHybridReasoner` whose
`predict_next_tactic` runs `model.act(batch, greedy=True)` (one deterministic action per
node — evaluation measures the policy, not the sampler), `model.eval()`, no stashing
consumed, judged by `result.graph.is_solved()`. The eval pool is `eval_pool_size` states
held out from the pool by hash, fixed across rounds so the curve is comparable. `best.pt`
tracks this number — proof rate, not training loss, is the model-selection criterion.

Also exposed as `scripts/rl_train.py --eval-only --checkpoint <path>` so the same code
produces the final three-way comparison (supervised warm start vs. RL-tuned vs. GNN top-k
search) on the test split.

## File-by-file changes

**New**
- `atp_lean_gnn/rl_training_driver.py` — `RLDriverConfig` (+ JSON loader),
  `build_theorem_pool`, `anneal_bc`, `run_rl_training(config, resume=None)`,
  `evaluate_proof_rate`, `driver_main(argv)`.
- `scripts/rl_train.py` — thin launcher (`argparse`: `--config`, `--resume`,
  `--eval-only`, `--checkpoint`, `--from-pointer`).
- `configs/rl_actor_critic.json` — the config above.
- `tests/test_rl_training_driver.py` — see test plan.

**Modified**
- `atp_lean_gnn/rl_reasoner.py` — add `greedy: bool = False` to `RLHybridReasoner.__init__`,
  forwarded to `model.act`. One parameter, no mode branches elsewhere.

Nothing in `joint_inference.py`, the loss code, or the harvest changes.

## Alternatives considered

1. **Fold the driver into `training.py` as `train_rl` next to `train_actor_critic`.**
   Pros: one CLI, shared run-dir helpers. Cons: `training.py` is 1700 lines of *offline
   dataloader* training; the RL driver's shape (async collect against live Lean, no
   DataLoader, per-round rather than per-epoch) shares almost none of it beyond
   `_create_run_dir`/`_save_checkpoint`, which are importable. Chosen: separate module,
   import the run-dir helpers.
2. **Whole-theorem statements instead of dataset proof states as rollout roots.**
   Pros: matches the final evaluation task exactly. Cons: with a warm-started but
   RL-untrained policy, multi-step proofs from scratch rarely reach QED, so early rounds
   would be failure-only and the terminal-reward channel silent. Chosen: proof states
   (near-terminal by construction) with the curriculum window growing toward harder
   states; whole statements enter via file mode when proof rate supports it.
3. **Concurrent collect across theorems (per-search reasoner instances).**
   Pros: wall-clock. Cons: refinement 6 deferred this deliberately — the stash is
   per-reasoner, and Lean server sharing across concurrent searches is untested; the
   driver is the wrong place to introduce it. Chosen: sequential collect now; the
   `collect_batch_onpolicy` seam is where concurrency lands later without driver changes.
4. **Select `best.pt` by training return instead of greedy proof rate.**
   Pros: free (no eval searches). Cons: return is measured on the *sampled* policy over a
   *shifting* curriculum window — not comparable across rounds and gameable by entropy
   collapse. Chosen: periodic greedy proof rate on a fixed held-out pool.
5. **Anneal entropy weight alongside BC weight.** Deferred: two coupled schedules before
   any real run is over-tuning; entropy stays constant until metrics show premature
   determinism.

## Test plan

Unit tests run under `uv run python -m pytest`; no Lean/petta needed (fakes from
`test_rl_reasoner.py` are reused).

| Test | How |
|---|---|
| config round-trip | JSON → `RLDriverConfig` → `to_dict` equality; missing field errors |
| pool from file mode | tiny JSONL → `Goal`s in order, length filter applied |
| pool from dataset rows | monkeypatched `iter_dataset_rows` yielding fixed rows → parse + sort correct |
| BC anneal endpoints | `anneal_bc(0) == start`, `anneal_bc(anneal_rounds) == end`, monotone between |
| round loop happy path | fake reasoner (QED executor) + 2 rounds → params change, metrics.jsonl has 2 rows, last.pt written |
| per-theorem fault isolation | fake reasoner raising on theorem 2 of 3 → round completes with survivors, failure counted |
| empty round no-op | all-reject executor with empty vocab → no optimizer step (params unchanged), round logged |
| resume | run 2 rounds, resume, run 1 more → round counter is 3, optimizer state restored |
| greedy eval determinism | same model + pool twice → identical proof-rate dict |
| checkpoint selection | eval improves on round k → best.pt mtime changes; regresses → unchanged |

Live validation (manual, needs toolchain + warm-start weights): 5 rounds on the real
dataset with `theorems_per_round: 4`, assert nonzero transitions, finite losses, and a
written checkpoint — the driver-level analogue of `rl_smoke.py`.

## Staging

```
1. RLDriverConfig + theorem pool + BC anneal (pure functions) + their tests
2. run_rl_training round loop with fault isolation + checkpoint/resume + tests
3. Greedy eval path (reasoner greedy flag + evaluate_proof_rate + --eval-only) + tests
4. configs/rl_actor_critic.json + scripts/rl_train.py launcher
5. Manual live validation once phase-2/3 checkpoints exist
```

## Out of scope

- Concurrent collect (alternative 3) and batched GNN inference across frontiers.
- PPO-style multi-epoch updates per collect (would break the one-step invariant by design,
  needs ratio clipping — a separate plan if single-step proves sample-starved).
- Lemma-corpus arguments in the pointer action space.
- Hyperparameter search; the config defaults are starting points, not tuned values.
