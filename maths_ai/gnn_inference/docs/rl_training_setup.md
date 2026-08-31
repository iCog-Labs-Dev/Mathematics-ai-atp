# Starting RL training: weights, environment, and launch (remote server / JupyterLab)

Step-by-step operational guide: where the weight files go, what format the loaders
expect, how to prepare a fresh remote machine, and how to launch, monitor, resume, and
evaluate the RL run. The *mechanics* of what the loaders do internally — including the
HTPS simulation loop, the decoupled imitation/critic step, and the PLN kill switch — are
in `docs/rl_process_walkthrough.md`; this file is the checklist.

## 0. The two hand-offs at a glance

```
pointer best.pt ──(hand-off A: filtered load + zero-residual copy)──▶ supervised actor-critic
                                                                        │ trains actor+critic
supervised actor-critic best.pt ──(hand-off B: strict full load)──▶ RL driver
                                                                        │ collect→train rounds
                                                              runs/rl_actor_critic/<stamp>/best.pt
```

The RL driver does **not** accept a pointer checkpoint directly — it strict-loads a full
`ActorCriticWithArgsClassifier` state dict. If you only have pointer weights, run
hand-off A first (even briefly) to produce one.

The driver supports two search modes, controlled by `selection_policy` in the config:

| `selection_policy` | Search loop | When to use |
|---|---|---|
| `"legacy"` (default) | Best-first AND-OR: pop highest-`combined_rank` node, expand, propagate | Baseline; simpler to debug; no HTPS budget to tune |
| `"puct"` | HTPS repeated simulation: PUCT selection + virtual loss, batched leaf expansion, validity-aware `N/N_v/W/Q` backup | When you want MCTS-style exploration and the critic to serve as `v_T(g)` |

PLN involvement is controlled by `use_pln` in the config (default `true`). Set to
`false` for a pure terminal-reward run with no petta subprocess.

## 1. Checkpoint file format

The RL driver requires a version-3 checkpoint manifest. It reconstructs the encoder and
heads from the manifest, validates the model kind, node/tactic vocabulary fingerprints,
and the normalized model-S-expression graph contract. Use
`scripts/migrate_model_checkpoint.py` to create a version-3 model warm start; migration
drops old optimizer and replay state because validity provenance and normalized graph
identity cannot be reconstructed from text-only queues.

The live RL representation is `lean-model-sexp-v2`, declared in
`configs/model_sexpr_graph_contract.json`. The prepared dataset must contain
`metadata/graph_representation.json` with the same contract. Runtime loading rejects a
missing or mismatched contract instead of inferring one from tensor shapes or labels.

The expected file is:

- a **`.pt` file** written by `torch.save`, containing `manifest`, `model_state_dict`,
  `model_spec`, and vocabulary fingerprints.

This is exactly what this repo's training scripts write; checkpoints they produced need
no conversion. What the state dict must contain per hand-off:

| Hand-off | Expected tensors | Loaded by |
|---|---|---|
| A (pointer → supervised AC) | `TacticWithArgsClassifier`: `backbone.*`, `tactic_embedding.*`, `argument_selector.*` | `load_from_pointer_checkpoint` (filtered, shape-guarded, `strict=False`) |
| B (supervised AC → RL) | full `ActorCriticWithArgsClassifier` incl. `actor.*`, `critic.*` | RL driver (`strict=True`) |

Do not rename tensors from a foreign checkpoint to force a fit: matching shapes with a
different node/tactic vocab ordering loads without error and silently maps every
embedding row to the wrong token. The checkpoint is only valid together with the
`prepared_root` it was trained against.

RL run checkpoints written by the HTPS-enabled driver contain additional keys:

| Key | Contents | Missing = |
|---|---|---|
| `optimizer_htps_state_dict` | Adam moment state for the decoupled imitation/critic step | resume rejected |
| `tactic_queue` | structured canonical goals, graph fingerprints, tactic and argument indices | resume rejected |
| `critic_queue` | structured canonical goals, graph fingerprints, targets, and provenance | resume rejected |
| `graph_representation` | normalized model-S-expression graph contract | resume rejected |
| `rl_target_schema_version` | validity-aware replay schema version (`3`) | resume rejected |

Pre-version-3 checkpoints may be used as model warm starts after an audited prepared-data
migration, but text-only HTPS queues cannot be exact-resumed. Start fresh optimizers and
empty validity-aware queues for those runs.

## 2. Where to copy the files on the server

All paths are configuration, not convention — but use the repo layout so configs stay
readable. From the repo root:

```
runs/pointer_gnn/imported/best.pt            # ← your pointer weights (hand-off A input)
runs/actor_critic_gnn/<run>/best.pt          # ← produced by hand-off A (or copied in, if
                                             #    you already trained the AC phase elsewhere)
artifacts/prepared/v1/                       # ← the prepared dataset BOTH phases used:
    vocab/node_vocab.json                    #    these two files size & order every
    vocab/tactic_vocab.json                  #    embedding table — same files, all phases
    ...(split manifests / pyg tensors)...
```

Copy from your machine with `scp -r runs artifacts user@server:/path/to/new-maths/`, or
upload through the JupyterLab file browser (drag-and-drop; for multi-GB artifacts, zip
first and unzip in a terminal — the browser upload is per-file).

Note: the repo contains a symlink `maths_ai/gnn_inference/artifacts/prepared →
../../_support_files/artifacts/prepared` which is broken unless that target exists.
Either create the target directory or (simpler) point `prepared_root` in the configs at
the real absolute path of your prepared dataset.

## 3. Environment preparation (remote server, JupyterLab)

The RL phase runs live Lean and (when `use_pln=true`) petta subprocesses, so the Python
packages alone are not enough. Work in a **JupyterLab terminal** (File → New →
Terminal), not a notebook — the driver is a long-running CLI process.

### 3.1 Python side

```bash
cd /path/to/new-maths
uv sync                          # installs torch, torch-geometric, pydantic, faiss-cpu,
                                 # and pantograph (from the PyPantograph git source)
uv add datasets                  # HuggingFace streaming — required by the driver's
                                 # dataset mode (iter_dataset_rows); not yet in pyproject
```

CUDA check (the prepared configs use `"device": "auto"`, which falls back to CPU —
fine for the RL phase, whose bottleneck is Lean, but verify what you got):

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 3.2 Lean / Pantograph

Live RL uses the repository's strict model-S-expression server factory. It requires a
Lake project, a custom `pantograph-repl` built against that project, and the `Mathlib`
import. Use the setup helper to verify the pair and print the exact config flags:

```bash
uv run python maths_ai/gnn_inference/scripts/setup_sexpr_environment.py \
  --source-root /abs/path/to/lean_mathlib \
  --pantograph-repl /abs/path/to/lean_mathlib/.lake/build/bin/pantograph-repl
```

The helper checks the custom REPL, `Mathlib` environment, and model-S-expression capability
before a training run. The first build may compile Lean core and Mathlib, which can take
several minutes.

### 3.3 petta (PLN) — required only when `use_pln=true`

If you are running with `"use_pln": false` (terminal-reward-only mode), skip this
section entirely — `PLNInference` is never constructed and no petta subprocess ever
spawns.

When `use_pln=true` (the default), the PLN reward and ranking paths shell out to the
`petta` binary. `PLNInference` finds it via, in order: the `petta_bin` constructor arg,
the `PETTA_BIN` environment variable, or `shutil.which("petta")`. So either install it
on PATH (e.g. `/usr/local/bin/petta`) or:

```bash
export PETTA_BIN=/path/to/petta          # put this in ~/.bashrc for JupyterLab terminals
```

Check:

```bash
uv run python -c "
from maths_ai.pln_inference.model import PLNInference
p = PLNInference()
r = p.evaluate('p -> p', hypotheses=['p : Prop'])
print('petta OK:', r.status, r.stv, 'fallback =', r.is_fallback)
"
```

`is_fallback=True` with status `petta_unavailable` means the binary wasn't found. With
`use_pln=true`, every Φ would be exploration noise from the DTS bandit instead of PLN
scores — the run proceeds but reward shaping is uninformed.

### 3.4 Sanity: unit suite + live smoke

```bash
uv run python -m pytest maths_ai/gnn_inference/tests/ -q
# expected: all green except the pre-existing test_premise_pool failure
```

The smoke test validates the live chain end-to-end. Run the variant matching your
intended config:

```bash
# with PLN (default — needs petta):
uv run python -m maths_ai.gnn_inference.scripts.rl_smoke
# expected: "[rl_smoke] OK — collect → harvest → one on-policy gradient step completed."

# without PLN (use_pln=false — no petta needed):
uv run python -m maths_ai.gnn_inference.scripts.rl_smoke --use-pln false
```

## 4. Hand-off A: pointer weights → supervised actor-critic checkpoint

Skip this section if you already have a trained `ActorCriticWithArgsClassifier`
checkpoint; go to section 5.

1. Edit `maths_ai/gnn_inference/configs/actor_critic_graphsage_state.json`:

```json
"prepared_root": "/abs/path/artifacts/prepared/v1",
"pretrained_pointer_checkpoint": "runs/pointer_gnn/imported/best.pt",
```

   The `model` block (`hidden_dim: 512, num_layers: 4, max_args: 3`) must equal the
   pointer run's architecture — the loader shape-checks every transferred tensor and
   raises a `ValueError` naming the mismatched keys rather than leaving the model at
   random init.

2. Run the supervised actor-critic phase:

```bash
uv run python maths_ai/gnn_inference/scripts/train_baseline.py \
  --model-type actor_critic \
  --config maths_ai/gnn_inference/configs/actor_critic_graphsage_state.json
```

3. Verify the warm start took, on the console: the
   `Warm-start: loaded N tensors ...` line, and **first-epoch tactic accuracy ≈ the
   pointer run's final accuracy** (the zero-residual property: the actor's step-0
   distribution is the supervised classifier's; near-random first-epoch accuracy means
   the warm start failed and the shape guard should have said why).

4. Output: `runs/actor_critic_gnn/<timestamp>/best.pt` — the input to hand-off B.

## 5. Hand-off B: launch the RL driver

### 5.1 Config walkthrough

Edit `maths_ai/gnn_inference/configs/rl_actor_critic.json`. The fields you must
touch before a first run are the paths and the two mode knobs; everything else ships
with working defaults.

**Paths (required):**

```json
"warmstart_checkpoint": "runs/actor_critic_gnn/<timestamp>/best.pt",
"prepared_root": "/abs/path/artifacts/prepared/v1",
"run_root": "runs/rl_actor_critic",
"device": "auto",
"source_root": "/abs/path/to/lean_mathlib",
"pantograph_repl": "/abs/path/to/lean_mathlib/.lake/build/bin/pantograph-repl"
```

Do not add an architecture block to the RL config. GraphSAGE or GATv2, including its
readout, comes from the version-3 warm-start manifest.

`source_root` is the Lake project whose toolchain and Mathlib environment the live server
loads. `pantograph_repl` is the custom REPL binary built for that same environment. Both
are required for live RL; the strict server factory validates them, imports `Mathlib`, and
runs a capability probe on startup and after every restart. The probe exercises explicit
and instance locals and checks normalized `FV{i}` identities. A Python client or REPL from
another Lean/Mathlib pair is a protocol mismatch even when the executable starts.

**Search mode:**

```json
"selection_policy": "legacy",
"num_simulations": null,
"sim_batch_size": null,
"puct_c": null
```

`"legacy"` runs the original best-first loop and requires all three budget fields to
be `null`. To switch to HTPS/PUCT simulation, set:

```json
"selection_policy": "puct",
"num_simulations": 50,
"sim_batch_size": 8,
"puct_c": 1.0
```

`num_simulations` is the total number of simulations per `prove()` call (the HTPS
budget). `sim_batch_size` is how many partial hypertrees are selected and expanded in
one batch before backup. `puct_c` is the exploration constant in the PUCT score
`Q + c·P·sqrt(total)/(1 + N + VL)`. Validation runs at config construction: an
explicit budget under `"legacy"` and a missing `num_simulations` under `"puct"` both
raise `ValueError` before any Lean server starts.

Each PUCT leaf is serialized through Pantograph before the batched policy forward. If a
server restart occurs while a leaf batch is being executed, every materialized state from
the old server epoch is discarded. Those leaves are replayed under the new epoch and their
actions are resampled; stale actions are not recorded as executor failures.

The visit counters have separate meanings. `N` counts every completed traversal, including
an `UNKNOWN` result caused by an unelaborated node or infrastructure failure. `N_v` counts
only traversals with a valid numeric result, and `W` accumulates those numeric values. The
selection value is `Q = W/N_v`; when `N_v = 0`, first-play urgency is used. Therefore an
unknown simulation contributes exploration evidence through `N`, but it cannot create a
critic target of `0.0` or `1.0`.

**PLN kill switch:**

```json
"use_pln": true
```

`true` (default): PLN ranks every Lean-returned subgoal after a successful tactic
application, and the DTS bandit replaces fallback STVs. Reward shaping adds the PLN
potential Φ = `stv.strength` to every edge's terminal reward. PLN cannot close a node;
only a Lean transition with no remaining goals creates hard success.

`false`: no petta subprocess ever spawns. Every Lean-returned subgoal is linked in the
executor's order, with `stv=None` on every child node.
`ProofNode.local_score` degrades to GNN probability alone; `potential()` returns 0.0
so `edge_shaped_reward = edge_terminal_reward` everywhere. The PLN fallback block is
skipped — a node whose every tactic is rejected goes straight to exhausted.

**HTPS decoupled step (Phases 2–3):**

```json
"htps_steps_per_round": 0,
"htps_batch_size": 64,
"htps_learning_rate": 0.0001,
"w_critic_soft": 0.5,
"visit_threshold": 4,
"tactic_queue_size": 10000,
"critic_queue_size": 10000,
"mine_all_solved_nodes": true
```

`htps_steps_per_round=0` disables the decoupled step entirely — only the on-policy step
runs. Set to a positive integer (e.g. 4) to also run
supervised imitation + soft-critic regression steps per round through a separate
optimizer (does not affect the on-policy optimizer's moments).

After each collect round the driver mines every graph into two replay queues. The tactic
queue receives Lean-confirmed step-minimal proof edges. The critic queue receives only
valid `CriticSample`s with provenance: hard Lean/search labels, root budget labels, or
soft `Q` targets supported by at least `visit_threshold` valid backups. Unknown and
unelaborated nodes are omitted. Imitation and critic batches use separate public encoder
forwards before one supervised backward pass.

```
L = L_tactic_imitation + w_critic_soft · L_critic_soft
```

`mine_all_solved_nodes=true` mines the full minimal hypertree from every SOLVED node in
the graph; `false` mines only from the root (ablation).

**Per-round budgets (tune first):**

```json
"theorems_per_round": 8,
"theorem_timeout_s": 120.0
```

These two have the most direct effect on wall-clock time per round and GPU utilization.
Start with the defaults and tighten or loosen based on the round timing shown in the
console.

### 5.2 Launch

Launch inside a persistent terminal — a JupyterLab terminal dies with your browser
session unless wrapped:

```bash
tmux new -s rl
cd /path/to/new-maths && source ~/.elan/env
uv run python maths_ai/gnn_inference/scripts/rl_train.py \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  2>&1 | tee rl_train.log
# detach: Ctrl-b d      reattach later: tmux attach -t rl
```

What startup prints, in order — each line is a checkpoint you can verify:
`Warm start (strict): ...` → `Run dir: runs/rl_actor_critic/<stamp>` →
`Theorem pool: N usable states, M dropped` → `Pool: X train / Y eval; curriculum
window Z` → per-round lines:

```
Round 0: solved 2/8, trans 11, rej 19, err 0, return 0.213, loss 0.847, bc 0.500, 94.2s
```

With `htps_steps_per_round > 0`, the round line is followed by additional loss fields
once the queues are non-empty: `tactic_imitation_loss` and `critic_soft_loss`.

## 6. Monitoring, resuming, evaluating

**Monitor** — from a JupyterLab notebook, plot the metrics file the driver appends per
round:

```python
import json, pandas as pd
rows = [json.loads(l) for l in open("runs/rl_actor_critic/<stamp>/metrics.jsonl")]
train = pd.DataFrame([r for r in rows if "num_transitions" in r])
train.plot(
    x="round",
    y=["solved", "num_failures", "searches_failed", "mean_return", "total_loss"],
    subplots=True,
)
```

Healthy early signs: rejected tactics (`num_failures`) and whole-search errors
(`searches_failed`) trending down, `solved` and `mean_return` up, `entropy` positive
(not collapsing to 0), and curriculum-widened lines appearing.

When the decoupled HTPS step is enabled, additional columns appear once the queues fill:

| Metric | Meaning |
|---|---|
| `imitation_samples_mined` | Step-minimal proof edges extracted from this round's graphs |
| `tactic_queue_len` | Current size of the tactic imitation replay queue |
| `critic_queue_len` | Current size of the soft-critic replay queue |
| `tactic_imitation_loss` | Cross-entropy on proof-edge tactics + arguments (decoupled step) |
| `critic_soft_loss` | MSE of critic head vs. valid status/root/soft-Q targets (decoupled step) |
| `unknown_label_count` | Total graph labels mapped to `<UNK>` during the update |
| `structural_unknown_label_count` | Unknown graph-schema labels; nonzero values fail the update |
| `semantic_unknown_label_count` | Unknown Lean content labels mapped to `<UNK>` |

With `selection_policy="puct"`, also watch that `visit_stats.N > 0` on edges in the
saved graphs — a consistently zero visit count means the simulation loop is not
running, which would indicate the config validation was bypassed.

**Resume after a disconnect/preemption** — the driver checkpoints `last.pt` (model +
both optimizers + both queues + RNG + round counter) every `checkpoint_every` rounds:

```bash
uv run python maths_ai/gnn_inference/scripts/rl_train.py \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --resume runs/rl_actor_critic/<stamp>
```

The resume path restores both optimizers and both queues. Pre-HTPS checkpoints
(missing the HTPS keys) still resume cleanly with fresh optimizer moments and empty
queues.

**Evaluate** — greedy proof rate on the same held-out pool, for the comparison that
matters (did RL beat its own warm start):

```bash
# baseline: the supervised warm start, before any RL
uv run python maths_ai/gnn_inference/scripts/rl_train.py \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --eval-only --checkpoint runs/actor_critic_gnn/<timestamp>/best.pt

# RL-tuned
uv run python maths_ai/gnn_inference/scripts/rl_train.py \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --eval-only --checkpoint runs/rl_actor_critic/<stamp>/best.pt
```

`best.pt` in the run dir is already model-selected by this metric (evaluated every
`eval_every` rounds), so the final artifact of the whole pipeline is
`runs/rl_actor_critic/<stamp>/best.pt`.

## 7. Failure modes at startup, and what each means

| Error | Cause | Fix |
|---|---|---|
| `Missing vocab file: .../vocab/node_vocab.json` | `prepared_root` wrong or symlink broken | point at the real prepared dataset directory |
| `ValueError: checkpoint manifest ...` | model or vocabulary manifest does not match | migrate the intended checkpoint and use its prepared vocabulary |
| `Missing key(s) in state_dict: "actor.base.weight" ...` | a pointer checkpoint was given to hand-off B | run hand-off A first |
| `Warm-start shape mismatch` | hand-off A config ≠ pointer architecture | fix the supervised AC configuration before migration |
| `ValueError: selection_policy='legacy' cannot have an explicit simulation budget` | `num_simulations`/`sim_batch_size`/`puct_c` set non-null under `"legacy"` | set them all to `null` for legacy mode |
| `ValueError: selection_policy='puct' requires num_simulations` | `"puct"` set without a simulation budget | add `"num_simulations": <int>` |
| `RuntimeError: rank_subgoals requires PLN; ... use_pln=False` | `rank_subgoals` called directly on a `use_pln=False` reasoner | this is a guard, not a config error; indicates a code path that expects PLN was reached — check that the calling code respects the flag |
| every PLN result `is_fallback=True` | petta not found (only relevant when `use_pln=true`) | install petta / set `PETTA_BIN`; or switch to `"use_pln": false` |
| strict Pantograph server creation fails | missing or mismatched `source_root`, `pantograph_repl`, Lean toolchain, or `Mathlib` import | run `setup_sexpr_environment.py` with the same paths and rebuild the custom REPL |
| `ModuleNotFoundError: datasets` | streaming dep not installed | `uv add datasets` |
