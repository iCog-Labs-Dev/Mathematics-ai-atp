# RL training setup: pluggable GNNs, checkpoints, and live Lean search

This is the operational guide for preparing data, selecting a GraphSAGE or GATv2
actor, training the supervised warm start, and launching, monitoring, resuming, and
evaluating reinforcement learning (RL). The mechanism inside the search and update
loop is described in `docs/rl_process_walkthrough.md`.

Run commands from the repository root unless a command says otherwise.

## 1. End-to-end artifact flow

Architecture selection happens before RL training. Select an architecture in the
pointer model config, use an actor-critic config with the same complete `model` block,
and give the resulting actor-critic checkpoint to the RL driver.

```text
prepared dataset and vocabularies
            |
            v
matching pointer config: model.architecture = graphsage or gatv2
            |
            v
pointer best.pt
            |
            | strict transfer of encoder, tactic classifier,
            | tactic embedding, and argument selector
            v
matching supervised actor-critic config
            |
            v
actor-critic best.pt with a version-2 manifest
            |
            | RL reconstructs the model described by the manifest
            v
RL best.pt
```

The RL config does not contain `architecture`, `hidden_dim`, `num_layers`, `dropout`,
`use_node_type`, `max_args`, `heads`, or `readout`. It also does not accept a source
directory containing model code. The version-2 warm-start checkpoint is the single
source of truth for the actor encoder and its associated policy components.

This design keeps an RL run reproducible: the checkpoint records the normalized model
specification, architecture version, vocabulary fingerprints, and a fingerprint of the
trained encoder weights.

## 2. Select an architecture and readout

The stable architecture names are exactly `graphsage` and `gatv2`.

GraphSAGE has a state-root readout. Its `model` block is:

```json
{
  "architecture": "graphsage",
  "hidden_dim": 512,
  "dropout": 0.2,
  "use_node_type": true,
  "max_args": 3,
  "encoder": {
    "num_layers": 4
  }
}
```

GATv2 has a configurable multi-head attention encoder and graph readout. For example:

```json
{
  "architecture": "gatv2",
  "hidden_dim": 512,
  "dropout": 0.2,
  "use_node_type": true,
  "max_args": 3,
  "encoder": {
    "num_layers": 4,
    "heads": 8,
    "readout": "state_mean_attention"
  }
}
```

`hidden_dim` is the width of every node and proof-state embedding. For GATv2 it must be
divisible by `heads`, the number of attention heads. The supported GATv2 readouts are:

| Readout | Proof-state representation supplied to the policy and critic |
|---|---|
| `state` | embedding of the distinguished proof-state root node |
| `state_mean_attention` | learned fusion of the root, the mean node embedding, and a state-conditioned attention summary |
| `state_max_attention` | learned fusion of the root, the maximum node embedding, and a state-conditioned attention summary |
| `state_mean_max_attention` | learned fusion of the root, mean and maximum node embeddings, and a state-conditioned attention summary |

Use a pointer preset and actor-critic preset with the same suffix:

| Architecture and readout | Pointer preset | Actor-critic preset |
|---|---|---|
| GraphSAGE, state root | `pointer_graphsage_state.json` | `actor_critic_graphsage_state.json` |
| GATv2, state root | `pointer_gatv2_state.json` | `actor_critic_gatv2_state.json` |
| GATv2, mean plus attention | `pointer_gatv2_state_mean_attention.json` | `actor_critic_gatv2_state_mean_attention.json` |
| GATv2, maximum plus attention | `pointer_gatv2_state_max_attention.json` | `actor_critic_gatv2_state_max_attention.json` |
| GATv2, mean, maximum, plus attention | `pointer_gatv2_state_mean_max_attention.json` | `actor_critic_gatv2_state_mean_max_attention.json` |

Baseline presets with the same architecture and readout names also exist for supervised
tactic-classification experiments. The RL warm start must be an actor-critic checkpoint,
not a baseline or pointer checkpoint.

## 3. Regenerate the prepared dataset

Regenerate data prepared before the pluggable-architecture change. Each PyTorch
Geometric artifact now has a `.size.json` sidecar containing:

- `nodes`: number of graph nodes;
- `edges_forward`: number of stored directed edges;
- `edges_bidirectional`: number of edges after adding reverse edges and removing
  duplicates.

The supervised batch sampler reads the field matching `edge_mode`. Old sidecars do not
contain these edge counts and cannot enforce the current graph budgets correctly.

```bash
uv run python -m maths_ai.gnn_inference.scripts.prepare_dataset \
  --output-root maths_ai/gnn_inference/artifacts/prepared/v1 \
  --splits train,val,test \
  --force
```

`--force` replaces the selected output root. Copy any prepared artifacts you need to
retain before running it.

Use the same prepared root for pointer training, supervised actor-critic training, and
RL. In particular, use the same files at:

```text
<prepared-root>/vocab/node_vocab.json
<prepared-root>/vocab/tactic_vocab.json
```

The integer assignments in these vocabularies determine the rows of the model's
embedding and classifier tables. Version-2 loaders compare fingerprints of the complete
mappings, so equal vocabulary sizes with different assignments are rejected.

The repository may contain the symlink
`maths_ai/gnn_inference/artifacts/prepared -> ../../_support_files/artifacts/prepared`.
If its target does not exist on a new machine, set `prepared_root` in each config to the
real absolute dataset directory.

## 4. Copy artifacts to a remote machine

The paths are configuration values, but keeping the following layout makes the hand-offs
easy to audit:

```text
maths_ai/gnn_inference/runs/pointer_gatv2_state_mean_attention/<run>/best.pt
maths_ai/gnn_inference/runs/actor_critic_gatv2_state_mean_attention/<run>/best.pt
maths_ai/gnn_inference/artifacts/prepared/v1/
    vocab/node_vocab.json
    vocab/tactic_vocab.json
    train/pyg/...
    val/pyg/...
    test/pyg/...
```

Copy the run and prepared-data directories together, or set each config to their actual
absolute locations:

```bash
scp -r /local/path/to/maths_ai/gnn_inference/runs \
  user@server:/path/to/new-maths/maths_ai/gnn_inference/
scp -r /local/path/to/prepared/v1 \
  user@server:/path/to/new-maths/maths_ai/gnn_inference/artifacts/prepared/
```

Do not copy a checkpoint without the prepared vocabulary files that produced it.

## 5. Prepare the runtime environment

Use a JupyterLab terminal or another persistent shell for setup and training. The RL
driver is a long-running command-line process.

### 5.1 Python and CUDA

```bash
cd /path/to/new-maths
uv sync
uv add datasets
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

The `datasets` package supplies the Hugging Face streaming interface used when the RL
config has `"data_source": "dataset"`. It is not currently declared in
`pyproject.toml`.

The supervised trainers choose mixed precision from the selected encoder:

- GraphSAGE uses FP16 on CUDA when `training.use_amp` is true.
- GATv2 uses BF16 on CUDA hardware that supports BF16 and otherwise uses FP32. GATv2
  never uses FP16 because FP16 attention scores can become non-finite.
- CPU training uses FP32 for both architectures.

The supervised baseline, pointer, and actor-critic trainers stop with a diagnostic
`FloatingPointError` when a loss is non-finite. Do not continue from a run that reports
this error; inspect the named architecture, precision, loss component, and batch first.

### 5.2 Lean and Pantograph

PyPantograph requires a Lean 4 toolchain through `elan` and `lake`:

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source ~/.elan/env
```

For goals that use Mathlib notation or declarations, set `source_root` to a compiled
Lake project containing Mathlib. A server started without `source_root` imports core
Lean only and cannot elaborate notation such as `ℕ`.

You may set the environment in `configs/rl_actor_critic.json`:

```json
"source_root": "/abs/path/to/mathlib-lake-project",
"pantograph_repl": null,
"pantograph_imports": null,
"server_timeout_s": 120
```

With `source_root` set and `pantograph_imports` left as `null`, the driver imports
`Init,Mathlib`. An explicit `pantograph_repl` must use the same Lean toolchain as the
Lake project. Command-line values override the corresponding config values:

```bash
uv run python -m maths_ai.gnn_inference.scripts.rl_smoke \
  --source-root /abs/path/to/mathlib-lake-project
```

If a custom REPL is required:

```bash
uv run python -m maths_ai.gnn_inference.scripts.rl_smoke \
  --source-root /abs/path/to/mathlib-lake-project \
  --pantograph-repl /abs/path/to/pantograph-repl
```

The expected final line is:

```text
[rl_smoke] OK — collect → harvest → one on-policy gradient step completed.
```

### 5.3 Optional petta/PLN process

The probabilistic logic network (PLN) path invokes the `petta` executable. It resolves
the executable from an explicit constructor value, the `PETTA_BIN` environment
variable, or `PATH`, in that order.

```bash
export PETTA_BIN=/abs/path/to/petta
```

Check the process before enabling PLN:

```bash
uv run python -c "from maths_ai.pln_inference.model import PLNInference; p = PLNInference(); r = p.evaluate('p -> p', hypotheses=['p : Prop']); print(r.status, r.stv, r.is_fallback)"
```

`r.stv` is the strength-and-confidence value returned by PLN. An
`is_fallback=True` result with status `petta_unavailable` means the executable was not
found.

## 6. Train the pointer model

Choose one pointer preset from the table in the architecture section. Update its
`prepared_root` and `run_root`, then train it. This example selects the GATv2
`state_mean_attention` readout:

```bash
uv run python -m maths_ai.gnn_inference.scripts.train_baseline \
  --model-type pointer \
  --config maths_ai/gnn_inference/configs/pointer_gatv2_state_mean_attention.json
```

For GraphSAGE, use:

```bash
uv run python -m maths_ai.gnn_inference.scripts.train_baseline \
  --model-type pointer \
  --config maths_ai/gnn_inference/configs/pointer_graphsage_state.json
```

Each supervised preset combines an example-count limit with two graph-size limits:

- `training.batch_size` limits the number of examples in a batch;
- `training.max_batch_nodes` and `training.max_batch_edges` limit the combined graph
  size of the batch.

The graph-budget sampler forms a batch only while all enabled limits remain satisfied.
`edge_mode` selects whether `max_batch_edges` is checked against `edges_forward` or
`edges_bidirectional` in each prepared sidecar. The checked-in presets use
`"edge_mode": "bidirectional"`.

The output used by the next phase is the pointer run's `best.pt`.
The preset's `run_root` determines its directory; do not infer the path from the preset
filename.

## 7. Train the matching supervised actor-critic

Select the actor-critic preset with the same architecture and readout as the pointer
checkpoint. Set:

```json
"prepared_root": "/abs/path/to/prepared/v1",
"pretrained_pointer_checkpoint": "maths_ai/gnn_inference/runs/pointer_gatv2_state_mean_attention/<run>/best.pt"
```

The complete normalized `model` specification must match the pointer checkpoint. This
includes `architecture`, `hidden_dim`, `dropout`, `use_node_type`, `max_args`, and all
fields inside `encoder`. Pointer transfer also verifies both vocabulary fingerprints.
There is no partial or architecture-only transfer mode.

For the example GATv2 chain:

```bash
uv run python -m maths_ai.gnn_inference.scripts.train_baseline \
  --model-type actor_critic \
  --config maths_ai/gnn_inference/configs/actor_critic_gatv2_state_mean_attention.json
```

For GraphSAGE:

```bash
uv run python -m maths_ai.gnn_inference.scripts.train_baseline \
  --model-type actor_critic \
  --config maths_ai/gnn_inference/configs/actor_critic_graphsage_state.json
```

At startup, verify the `Warm-start: loaded N tensors ...` message. The output needed by
RL is the actor-critic run's `best.pt`.

## 8. Inspect or migrate the checkpoint

All runtime loaders require checkpoint format version 2. Inspect an actor-critic
checkpoint before starting a long RL run:

```bash
uv run python -c "import pprint, torch; checkpoint = torch.load('maths_ai/gnn_inference/runs/actor_critic_gatv2_state_mean_attention/<run>/best.pt', map_location='cpu', weights_only=False); pprint.pp(checkpoint.get('manifest'))"
```

Confirm that the manifest contains:

- `checkpoint_format_version: 2`;
- `model_kind: actor_critic_with_args`;
- the expected `model_spec`, including the GATv2 readout when applicable;
- `node_vocab_fingerprint` and `tactic_vocab_fingerprint`;
- `encoder_fingerprint`.

Runtime loaders reject version-1 checkpoints and bare state dictionaries. Migrate an
audited GraphSAGE actor-critic checkpoint offline with:

```bash
uv run python -m maths_ai.gnn_inference.scripts.migrate_model_checkpoint \
  --layout ac_graphsage_actor_critic \
  --checkpoint runs/legacy_actor_critic/best.pt \
  --config runs/legacy_actor_critic/config.json \
  --prepared-root maths_ai/gnn_inference/artifacts/prepared/v1 \
  --output runs/migrated_actor_critic/best.pt
```

The accepted migration layouts are:

- `graphsage_baseline`;
- `graphsage_pointer`;
- `ac_graphsage_actor_critic`;
- `gatv2_baseline`;
- `gatv2_pointer`.

For example, migrate a legacy GATv2 pointer checkpoint with:

```bash
uv run python -m maths_ai.gnn_inference.scripts.migrate_model_checkpoint \
  --layout gatv2_pointer \
  --checkpoint runs/legacy_gatv2_pointer/best.pt \
  --config runs/legacy_gatv2_pointer/config.json \
  --prepared-root maths_ai/gnn_inference/artifacts/prepared/v1 \
  --output runs/migrated_gatv2_pointer/best.pt
```

Migration remaps only the listed, audited layouts. It builds the current model, loads
all remapped parameters strictly, verifies public model outputs on a fixed batch, and
writes a version-2 manifest. There is no legacy GATv2 actor-critic migration layout; use
a matching pointer checkpoint to train a current actor-critic checkpoint first.

## 9. Configure and launch RL

Edit `maths_ai/gnn_inference/configs/rl_actor_critic.json`:

```json
"warmstart_checkpoint": "maths_ai/gnn_inference/runs/actor_critic_gatv2_state_mean_attention/<run>/best.pt",
"prepared_root": "/abs/path/to/prepared/v1",
"run_root": "runs/rl_actor_critic",
"device": "auto",
"source_root": "/abs/path/to/mathlib-lake-project",
"use_pln": false
```

The checked-in preset has `use_pln: false`. The `RLTrainingConfig` class default is
`true`, but the JSON value is explicit and therefore controls a normal launch.

With `use_pln: true`, petta supplies each search node's PLN value and the reward code
uses the change in that value as potential-based shaping. With `use_pln: false`:

- no `PLNInference` object or petta subprocess is created;
- every Lean-returned subgoal remains attached to its tactic edge in executor order;
- search frontier scoring uses the GNN probability with `stv=None`;
- the PLN fallback that can fabricate a QED edge after executor rejection is disabled;
- every shaping potential is zero, so the reward contains the configured terminal terms
  and per-transition step penalty, without PLN shaping.

The critic value `V(s)` means the probability that this configured search proves state `s`
within its tactic, depth, node, and time budgets. Search harvesting keeps validity separate
from the numeric value: Lean-confirmed closure supplies `1.0`, a locally exhausted state
supplies `0.0`, and an unelaborated, infrastructure-failed, globally truncated, open, or
otherwise unresolved state supplies no critic row. For a tactic that returns multiple
subgoals, one known failed child makes the tactic edge a known failure, all known solved
children make it a known success, and a solved child plus an unknown child remains unknown.
The actor trains from valid edge returns and executor-rejected actions. The critic trains
once per unique state from `CriticSample` rows, so a state with several accepted tactics
does not duplicate its target.

The RL search budgets and update budgets control different objects:

| Config field | Limit |
|---|---|
| `max_depth` | maximum tactic depth within one theorem search |
| `max_nodes` | maximum proof-search nodes within one theorem search |
| `theorems_per_round` | theorem searches collected before one optimizer update |
| `max_update_nodes` | total graph nodes across all transitions and rejected actions in one optimizer update |
| `max_update_edges` | total graph edges across all transitions and rejected actions in one optimizer update |

The metrics distinguish `num_transitions` (accepted edges with known outcomes),
`num_critic_samples` (unique valid node targets), `num_failures` (executor-rejected
actions), `unknown_edges_skipped`, and `unknown_nodes_skipped`. A round advances the
optimizer and BC-anneal clock only when at least one actor or critic row produces a loss.

The update is rejected if the complete collected round exceeds either enabled update
budget. Reduce `theorems_per_round` to collect a smaller update, or raise the budget only
after measuring available memory.

Launch inside `tmux` or another session that survives a browser disconnect:

```bash
tmux new -s rl
cd /path/to/new-maths
source ~/.elan/env
uv run python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  2>&1 | tee rl_train.log
```

You can also override the Lean environment without editing the JSON:

```bash
uv run python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --source-root /abs/path/to/mathlib-lake-project \
  --pantograph-repl /abs/path/to/pantograph-repl
```

The driver verifies the Lean environment before constructing the model. It then loads
the prepared vocabularies, validates the checkpoint manifest, reconstructs the exact
actor-critic architecture, and loads the full state dictionary strictly.

## 10. Monitor, resume, and evaluate

A round line distinguishes executor-rejected actions from complete search errors:

```text
Round 0: solved 2/8, trans 11, rej 19, err 0, return 0.213, loss 0.847, bc 0.500, 94.2s
```

- `rej` counts sampled actions rejected by Lean inside searches that ran;
- `err` counts complete theorem searches that raised an exception or timed out.

An environment that cannot elaborate the theorem pool commonly reports `rej 0` and an
`err` count near `theorems_per_round`. Check `source_root`, imports, the Lean toolchain,
and the Pantograph REPL before changing model settings.

The driver appends one JSON object per round to `metrics.jsonl`. A notebook can plot the
principal metrics:

```python
import json
import pandas as pd

path = "runs/rl_actor_critic/<run>/metrics.jsonl"
rows = [json.loads(line) for line in open(path)]
train = pd.DataFrame(row for row in rows if "num_transitions" in row)
train.plot(
    x="round",
    y=[
        "solved",
        "num_transitions",
        "num_critic_samples",
        "num_failures",
        "unknown_edges_skipped",
        "unknown_nodes_skipped",
        "total_loss",
    ],
    subplots=True,
)
```

Resume from the run directory containing `last.pt`:

```bash
uv run python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --resume runs/rl_actor_critic/<run>
```

Evaluate the supervised warm start and the RL-selected checkpoint on the same held-out
pool:

```bash
uv run python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --eval-only \
  --checkpoint maths_ai/gnn_inference/runs/actor_critic_gatv2_state_mean_attention/<run>/best.pt

uv run python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --eval-only \
  --checkpoint runs/rl_actor_critic/<run>/best.pt
```

The RL run's `best.pt` is selected by greedy proof rate at each evaluation interval.
`last.pt` contains the model, optimizer, random-number-generator state, and round counter
needed to resume training.

## 11. Lemma-index and premise-scorer compatibility

This section applies when an inference or training path uses a lemma index or a learned
premise scorer. Their manifests store the `encoder_fingerprint`, a hash derived from the
normalized model specification, both vocabulary mappings, and the trained encoder
weights.

A lemma index or premise-scorer checkpoint is valid only for the exact encoder that
created it. Switching from GraphSAGE to GATv2, changing a GATv2 readout, changing a
vocabulary assignment, or changing encoder weights produces a different fingerprint.
Rebuild the lemma index and retrain or replace the scorer for the active model. Loaders
reject mismatched fingerprints rather than using incompatible embeddings.

## 12. Common failures

| Error or symptom | Mechanism | Required action |
|---|---|---|
| missing `vocab/node_vocab.json` | `prepared_root` does not identify a complete prepared dataset | point all phases at the real prepared root |
| sidecar is missing `edges_forward` or `edges_bidirectional` | prepared data predates edge-mode-aware graph budgets | regenerate the prepared dataset |
| checkpoint has no version-2 manifest | a version-1 checkpoint or bare state dictionary was supplied | run the audited offline migration, or retrain when no migration layout exists |
| checkpoint model kind is `tactic_with_args` | a pointer checkpoint was supplied directly to RL | train the matching supervised actor-critic first |
| pointer and actor-critic model specifications differ | architecture, readout, dimensions, or another normalized model field does not match | use matching presets and identical complete `model` blocks |
| vocabulary fingerprint does not match | the prepared vocabulary mapping differs from the checkpoint's mapping | use the prepared dataset that created the checkpoint |
| encoder fingerprint does not match | encoder weights, architecture, readout, or vocabularies differ | rebuild the dependent index or scorer for the active checkpoint |
| GATv2 reports a non-finite loss | the current batch or precision produced invalid values | use the registered GATv2 precision policy and inspect the named loss and batch |
| collected RL update exceeds its graph budget | all graphs in the round exceed `max_update_nodes` or `max_update_edges` together | reduce `theorems_per_round` or raise a measured update limit |
| round shows many `err` values and no transitions | complete Lean searches are failing or timing out | verify `source_root`, imports, toolchain compatibility, and the REPL |
| round collects searches but reports `optimizer_step: 0` | every edge and node outcome was unknown, commonly because reconstructed goals did not elaborate | inspect closure reasons and the unknown-edge/node metrics before changing the policy |
| every PLN result is a fallback | petta is unavailable or failing | set `PETTA_BIN`, fix petta, or run explicitly with `use_pln: false` |
| `ModuleNotFoundError: datasets` | Hugging Face dataset streaming dependency is absent | install it with `uv add datasets` |

For an environment and integration diagnostic after these checks, run:

```bash
uv run python -m pytest maths_ai/gnn_inference/tests/ -q
uv run python -m maths_ai.gnn_inference.scripts.rl_smoke \
  --source-root /abs/path/to/mathlib-lake-project
```
