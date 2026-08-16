# Porting current AC infrastructure onto MCTS-AC-branch

## Goal

Bring `MCTS-AC-branch` onto the current AC shared infrastructure without removing its
PUCT/HTPS search, per-edge visit statistics, batched leaf expansion, soft critic replay,
or minimal-hypertree imitation.

The port has three dependent parts:

1. the Pantograph environment, restart, parsing, tactic-rendering, pool-filtering, and
   training-loop fixes from AC commit `1476ca6`;
2. the pluggable GraphSAGE/GATv2 model, checkpoint, batching, and encoder-bound artifact
   work from AC commits `013087a`, `2bacd79`, `5905e0f`, and `ea80fcc`;
3. a validity-aware learning backup that neither branch has implemented yet, so an
   unelaborated or otherwise unobserved state cannot become a fabricated critic target,
   actor return, or MCTS visit value.

The critic value `V(s)` is the probability that the configured search procedure proves
state `s` within its tactic, depth, node, simulation, and time budgets. A Lean-confirmed
proof supplies a hard target of `1.0`. A valid search failure can supply `0.0`. An
elaboration failure, infrastructure failure, or internal state left unresolved by a
global cutoff supplies no numeric label.

## Source and target baselines

At plan revision time:

| Branch | Commit | Relevant state |
|---|---|---|
| `MCTS-AC-branch` | `84ef4f6` | PUCT/HTPS phases, PLN kill switch, original Pantograph integration |
| `AC-branch` | `ea80fcc` | Pantograph fixes plus pluggable GraphSAGE/GATv2 infrastructure |

Verify both references before implementation. If either branch moved, re-run the file
ownership audit and update this plan before copying code. Do not reset either branch to a
remote-tracking reference.

The old version of this plan targeted only `1476ca6`. That is no longer a sufficient
source baseline. In particular, copying its `rl_smoke.py`, constructing
`ActorCriticWithArgsClassifier` directly, or keeping flat model-dimension fields in the RL
configuration would revert the pluggable-model contract now implemented on AC.

## Symbol table

| Symbol | Meaning |
|---|---|
| `s` | One proof-state node: a Lean goal expression plus its local hypotheses. |
| `e` | One tactic hyperedge from parent state `s` to every subgoal produced by that tactic. |
| `V_theta(s)` | The critic network's current prediction for state `s`, where `theta` denotes model parameters. |
| `y(s)` | A search-derived critic target for state `s`, present only when its evidence is valid. |
| `G(e)` | The actor return for tactic edge `e`, present only when the edge outcome is valid. |
| `N(e)` | Completed selections that traversed edge `e`, including selections whose value became unknown. |
| `N_v(e)` | Traversals of edge `e` that produced a valid numeric backup. |
| `W(e)` | Sum of the `N_v(e)` valid numeric backups for edge `e`. |
| `Q(e)` | Mean valid action value `W(e) / N_v(e)`, with first-play urgency when `N_v(e) = 0`. |
| `VL(e)` | In-flight simulations currently holding virtual loss on edge `e`. |

## Audit findings

### The Pantograph defects still apply to both search modes

Both legacy best-first and PUCT reach Lean through
`HybridReasoner._start_state`. Without a configured Lake project, Pantograph sees only
core `Init`; Mathlib notation and types fail before tactic execution. A crashed REPL must
be reaped and restarted in the same environment. A live elaboration rejection must not be
mistaken for a crashed server.

The AC search-log fixes remain required:

- one verified `PantographEnv` for initial server creation and every restart;
- correct dead-process detection and `_close()` usage;
- separate whole-search errors from executor-rejected tactics;
- behavior-cloning annealing indexed by optimizer steps rather than loop rounds;
- a consecutive dead-round halt;
- case-label removal during proof-state parsing;
- unresolved-metavariable rejection in both theorem-pool sources;
- tactic-specific Lean rendering for bracket-taking tactics.

### The current AC model contract is now shared infrastructure

AC no longer reconstructs the RL model from `hidden_dim`, `num_layers`, `dropout`,
`max_args`, and `use_node_type` fields in `RLTrainingConfig`. A version-2 checkpoint
manifest owns the complete `ModelSpec`, model kind, architecture version, vocabulary
fingerprints, and encoder fingerprint. The model is reconstructed through
`build_model_from_checkpoint` with strict state loading.

The port must include the complete architecture boundary, not only the RL loader:

- the `graphsage` and `gatv2` registry entries;
- the common state-encoder contract and model builders;
- version-2 checkpoint manifests and explicit checkpoint migration;
- GraphSAGE/GATv2-safe supervised batching and numerical checks;
- encoder fingerprints on lemma indices and premise scorers;
- public model operations in inference, analysis, supervised training, and RL;
- removal of direct GraphSAGE construction from callers.

Porting only `5905e0f` would leave the MCTS RL driver depending on modules and checkpoint
formats that the rest of the branch does not provide.

### The old elaboration fix still fabricates learning evidence

The old plan proposed `ProofNode.unelaborated: bool` while retaining float-only backups.
That does not satisfy the intended guarantee:

- `backup_values` would still use `unresolved_leaf_value`, whose default is `0.0`;
- `extract_transitions` would still copy the parent target into every edge transition;
- `_backup_simulation` would replace the missing outcome with the critic's own estimate;
- pending actions would still be flushed as failures although no tactic was executed.

An unelaborated state is a failure to observe the search outcome. It is not a third numeric
target, and the critic's prediction is not evidence that can replace the missing outcome.

### MCTS has four value consumers that must migrate together

The MCTS branch consumes graph values through:

1. on-policy actor returns and parent critic targets in `extract_transitions`;
2. node-level hard and soft samples in `extract_critic_samples`;
3. simulation backups into `EdgeVisitStats.N` and `EdgeVisitStats.W`;
4. later soft targets derived from the visit mean.

Changing only one consumer leaves another path that converts unknown evidence into zero or
self-reinforcing critic output.

### Two shared graph defects can fabricate hard success

`top_k_subgoals` currently drops Lean-returned obligations before constructing an AND-edge.
A tactic that creates four subgoals can therefore be declared solved after proving only
three. Ranking may order subgoal expansion, but it may not remove obligations.

The PLN fallback can also create a childless solved edge from a high heuristic score. A
childless edge may become a hard success only when Lean reports no remaining goals. An edge
with children may become a hard success only after every required child has a
Lean-confirmed proof. Those are the only routes to a hard target of `1.0`, a proof-trace
step, or an imitation sample.

## Chosen architecture

### 1. Preserve MCTS ownership while importing AC shared infrastructure

Do not merge `AC-branch` and do not accept deletions merely because AC lacks MCTS files.
The MCTS branch owns:

- `selection_policy.py`;
- PUCT selection and batched simulation in `joint_inference.py`;
- edge visit statistics in `hypergraph.py`;
- batched policy proposals and critic leaf evaluation in `rl_reasoner.py`;
- HTPS queues and the decoupled optimizer in `rl_training_driver.py`;
- soft critic and minimal-hypertree harvesting in `search_harvest.py`;
- HTPS-specific tests and documentation.

These features must be hand-ported around AC infrastructure. They must not be replaced by
the AC legacy-only versions.

### 2. Use the AC pluggable-model contract unchanged

Import the current AC architecture system as one unit:

```text
ModelSpec
  -> architecture registry
  -> StateGraphEncoder
  -> build_actor_critic_model
  -> ActorCriticWithArgsClassifier public operations
```

Both MCTS model paths remain architecture-independent:

- `RLHybridReasoner.predict_next_tactics_batch` calls `model.act`;
- `RLHybridReasoner._leaf_value` calls the public actor-critic encode/value operation;
- `compute_onpolicy_loss` calls `model.evaluate_actions`;
- `train_step_htps_style` calls public encoder, actor, critic, and pointer operations.

No MCTS module may import GraphSAGE or GATv2 directly. The checkpoint manifest chooses the
encoder. The RL configuration retains search and optimizer fields but removes duplicated
model architecture fields.

### 3. Keep structural search status separate from learning validity

Retain `NodeStatus.OPEN`, `EXPANDED`, `SOLVED`, and `DEAD` for search control. Add a typed
closure reason to `ProofNode`:

```text
NodeClosureReason.NONE
NodeClosureReason.CANDIDATES_EXHAUSTED
NodeClosureReason.NO_CANDIDATES
NodeClosureReason.DEPTH_LIMIT
NodeClosureReason.CYCLE
NodeClosureReason.ELABORATION_ERROR
```

`ELABORATION_ERROR` is the explicit unelaborated state. It may remain structurally `DEAD`
so neither legacy frontier selection nor PUCT leaf selection retries it. Learning code must
inspect the closure reason before interpreting `DEAD` as numeric failure.

Replace free-form inference from `note` with the enum at every
`mark_node_exhausted` caller. Keep `note` only for diagnostic text.

Add a typed search-end reason to the graph or `RLSearchResult`:

```text
ROOT_SOLVED
ROOT_DEAD
MAX_NODES
NUM_SIMULATIONS
DEADLINE
FRONTIER_EMPTY
EXTERNAL_ABORT
```

The root is the only node that receives the full global budget of one search. If
`NUM_SIMULATIONS`, `MAX_NODES`, or `DEADLINE` ends an otherwise healthy root episode at a
completed-search safe point and the root is not solved, that observed episode supplies one
root target of `0.0`. This label means that the configured search did not solve the root
within this run's budget; it does not claim that every proof path is impossible and it does
not set `NodeStatus.DEAD`. It must not assign zero to untouched internal nodes. A deadline
that cancels an in-flight Pantograph operation, an external abort, or a root whose evidence
is contaminated by elaboration or infrastructure failure supplies no root target.
`ROOT_DEAD` or `FRONTIER_EMPTY` supplies `0.0` only when recursive validity shows that every
required root alternative is a valid failure. Structural status by itself is insufficient.

### 4. Represent validity and provenance explicitly

Introduce an immutable backup value:

```text
BackupValue
    value: float | None
    validity: VALID | UNKNOWN
    source: LEAN_STATUS | SEARCH_FAILURE | ROOT_EPISODE
            | VISIT_MEAN | CRITIC_BOOTSTRAP | NONE
```

Invariants:

- `VALID` requires a finite value in `[0.0, 1.0]` and a non-`NONE` source;
- `UNKNOWN` requires `value is None` and source `NONE`;
- hard labels use `LEAN_STATUS`, `SEARCH_FAILURE`, or `ROOT_EPISODE`;
- MCTS estimates use `VISIT_MEAN` or `CRITIC_BOOTSTRAP`;
- no caller may coerce `UNKNOWN` to `0.0`, `0.5`, `NaN`, or the critic prediction.

Remove `HarvestConfig.unresolved_leaf_value`. It encodes the old ambiguity as a
configuration option and would allow callers to restore fabricated labels.

### 5. Compute edge outcomes and node targets separately

Return one backup object containing:

```text
edge_outcomes: edge id -> BackupValue
node_targets:  node id -> BackupValue
```

The actor consumes edge outcomes. The critic consumes node targets.

#### AND-edge rules

For one tactic edge with several required children:

| Child evidence | Edge outcome |
|---|---|
| At least one child is a valid path failure | valid `0.0` |
| Every child has a valid numeric value | configured AND-combination of those values |
| Otherwise | unknown |

A child closed by `ELABORATION_ERROR` contributes unknown. A genuinely exhausted child
contributes valid path failure. A depth-limit or cycle child can make its incoming tactic
path fail, but it does not receive a standalone critic target because the same state
launched with a fresh root budget was not evaluated.

A childless edge is valid `1.0` only when Pantograph reported no remaining goals.

#### OR-node rules

| Edge evidence | Node target |
|---|---|
| At least one edge has hard, proof-derived `1.0` | hard `1.0`, source `LEAN_STATUS` |
| No hard success exists and at least one edge has estimated `1.0` | soft `1.0`, preserving the estimate source |
| The node is locally exhausted and every edge is valid `0.0` | valid `0.0` |
| Viable edges have valid estimates | search-value aggregation defined below |
| Any required alternative remains unknown and no success is known | unknown |
| The node is open or globally truncated before local resolution | unknown |

For hard status backup, OR remains the maximum over valid edge values. Unknown edges are
not silently inserted as zero. If a known success exists, it dominates unknown
alternatives. If no success exists and any viable alternative is unknown, the node remains
unknown.

Numeric backup never controls structural proof status. An estimated edge value of exactly
`1.0`, whether it originated from `CRITIC_BOOTSTRAP` or `VISIT_MEAN`, may supply a soft
critic target and MCTS action value, but it must not set `NodeStatus.SOLVED`, enter a proof
trace, or create an imitation sample. A direct childless edge becomes structurally solved
only when Lean returned no remaining goals. An edge with children becomes structurally
solved only when every required child is structurally solved through Lean-confirmed proof
steps.

### 6. Separate actor-edge rows from critic-node rows

Replace the combined `HarvestedTransition` contract with:

```text
ActorTransition
    node_id
    goal
    tactic
    reward
    successor_value
    return_
    edge_id

CriticSample
    node_id
    goal
    target
    source
```

Harvest one actor row per policy-produced edge whose edge outcome is valid. Harvest one
critic row per unique node whose node target is valid and whose closure reason permits a
state-level label.

Mixed-branch behavior:

- solved edge plus unknown edge: parent critic target `1.0`, actor row for the solved edge,
  no actor row for the unknown edge;
- failed edge plus solved edge: parent critic target `1.0`, low return for the failed edge,
  high return for the solved edge;
- only unknown edges: no actor-return row and no node critic target;
- known failed child plus unelaborated child on one AND-edge: the edge is a valid failure
  because one required child already failed, but the unelaborated child gets no critic row.

Executor-rejected tactics remain actor-only `FailureRecord` rows. Actions sampled for a
node that never elaborated must be discarded because they were never executed.

### 7. Make expansion completion explicit in both search modes

Add a base-class expansion-completion hook with a typed result:

```text
TACTICS_EXECUTED
NO_CANDIDATES
DEPTH_LIMIT
ELABORATION_ERROR
EXTERNAL_ABORT
```

Invoke it from the shared `_execute_and_link` path and every early exit in `_expand` and
`_expand_leaves`.

Migrate the RL pending stash:

- `TACTICS_EXECUTED`: accepted actions have already moved to `edge_actions`; flush remaining
  actions as executor failures;
- `ELABORATION_ERROR` or `EXTERNAL_ABORT`: discard pending actions for that node;
- `NO_CANDIDATES` and `DEPTH_LIMIT`: assert that no executable pending actions exist;
- normal `prove` completion requires an empty pending stash;
- cancellation cleanup discards unobserved actions rather than converting them to failure.

Do not keep the final-sweep failure conversion as a compatibility fallback. Every normal
node expansion must settle its own stash.

### 8. Make MCTS visit statistics validity-aware

Extend `EdgeVisitStats`:

```text
N       = completed traversals, used for exploration pressure
N_v     = traversals with a valid numeric backup
W       = sum of the N_v valid numeric backups
VL      = in-flight traversals
Q       = W / N_v when N_v > 0, otherwise first-play urgency
```

`_backup_simulation` returns `BackupValue` recursively:

- solved node -> valid `1.0`, source `LEAN_STATUS`;
- genuine path failure -> valid `0.0`, source `SEARCH_FAILURE`;
- ordinary unresolved leaf -> critic prediction, source `CRITIC_BOOTSTRAP`;
- unelaborated or infrastructure-failed leaf -> unknown;
- interior node -> validity-aware AND-combination of the chosen edge's children.

For every traversed edge, release `VL`. Increment `N` for the completed traversal. Increment
`N_v` and add to `W` only when the edge backup is valid. Unknown backup must not dilute
`Q` by entering its denominator.

PUCT exploration continues to use `N + VL`, because an unknown traversal still consumed
search work. Action-value exploitation uses `Q = W / N_v`.

### 9. Define MCTS soft critic targets from supported search values

The current `_soft_target` uses the maximum-prior edge's `W/N`. The policy prior says which
edge the actor initially preferred; it does not define the value that search discovered.

For this proof-search critic, choose the maximum `Q` among viable outgoing edges with
`N_v >= visit_threshold`. This estimates the strongest sufficiently supported action found
by the configured search. Record source `VISIT_MEAN`.

Rules:

- hard solved or validly failed status overrides a soft estimate;
- unknown and unelaborated nodes emit no critic sample;
- dead edges caused only by unknown evidence cannot supply a soft target;
- `visit_threshold` applies to `N_v`, not total `N`;
- soft targets retain their source in replay serialization and metrics.

Alternative semantics are possible: a visit-weighted mean would estimate the visit policy,
while maximum prior estimates neither the visit policy nor the best search action. The
chosen maximum-supported-`Q` target matches the existing search-for-a-proof objective.

### 10. Keep HTPS replay and imitation valid

The HTPS critic queue may contain hard status samples and soft `VISIT_MEAN` samples, each
with explicit source. It must never contain unknown samples.

`train_step_htps_style` continues to use its separate optimizer and may run several times
after the on-policy step. It must use only public actor-critic model operations so both
GraphSAGE and GATv2 follow the same path.

Minimal-hypertree imitation remains restricted to Lean-proved solved edges present in
`edge_actions`. Removing the PLN fallback ensures that a solved status cannot originate
from a heuristic pseudo-edge.

### 11. Keep one on-policy optimizer step with separate loss batches

Change `compute_onpolicy_loss` to accept actor transitions, deduplicated critic samples,
edge actions, and executor failures.

Use two forwards before one backward pass:

1. `model.evaluate_actions` over actor transitions and executor failures computes policy
   log-probabilities, entropy, and detached baselines;
2. the public model encode/value operation over unique critic samples computes
   `V_theta(s)` once per node.

If either batch is empty, construct differentiable zero losses without creating an empty
PyG batch. If both batches and executor failures are empty, take no optimizer step and do
not advance the BC anneal.

The complete on-policy update still performs exactly one `optimizer.step()`. HTPS replay
continues afterward through `optimizer_htps`.

Update-size validation counts the actual actor and critic graph batches. Report actor rows,
critic rows by source, executor failures, unknown edges skipped, unknown nodes skipped,
total graph nodes, and total graph edges.

### 12. Preserve every Lean subgoal

Remove `top_k_subgoals` from `HybridReasoner`, `RLTrainingConfig`, JSON, CLI wiring, tests,
and helper constructors. Migrate every caller in the same change.

When PLN is enabled, score and rank every subgoal for frontier order but attach every
subgoal to the tactic edge. When PLN is disabled, retain every subgoal in Lean's returned
order. `max_nodes` and `num_simulations` control later search work; they may not rewrite the
logical obligations produced by a tactic.

An expansion may add enough children to cross `max_nodes`. Preserve the complete edge and
stop before the next expansion. Treat `max_nodes` as a soft upper bound by at most one
complete tactic outcome rather than truncating that outcome.

### 13. Remove fabricated PLN success

Delete the branch that turns a PLN fallback score into a childless `PLN_fallback` solved
edge. PLN may rank nodes and provide potential-based reward shaping. It may not set solved
status, enter a proof trace, create imitation data, or create a hard critic target.

There is no compatibility flag. Retaining an opt-in pseudo-proof path would preserve the
invalid label source.

## Port strategy

### Stage 0: establish baselines and preserve branch-owned files

Before copying:

```bash
git rev-parse MCTS-AC-branch
git rev-parse AC-branch
git status --short
pytest maths_ai/gnn_inference/tests/
```

Record existing failures. Do not delete untracked documentation or user files. Do not
accept AC-side deletions of MCTS plans, selection policy, visit-stat tests, or HTPS tests.

### Stage 1: copy AC files with no MCTS divergence

The MCTS branch changed only its search, RL harvest/training/driver, related config/docs,
and tests after the common ancestor. Copy the current AC versions of the remaining shared
pluggable-model files, including:

- `atp_lean_gnn/architectures/`;
- `model_spec.py`, `model_factory.py`, `checkpointing.py`, `batching.py`, and
  `training_safety.py`;
- `actor_critic.py`, `argument_selector.py`, `model.py`, supervised training modules, and
  package exports;
- encoder-bound lemma, premise, inference, analysis, and artifact code;
- GraphSAGE/GATv2 supervised, pointer, and actor-critic configs;
- `scripts/migrate_model_checkpoint.py` and the updated inference/training/index scripts;
- pluggable architecture, checkpoint, batching, premise, and model-helper tests;
- `pantograph_env.py`, `state.py`, and the current AC `rl_smoke.py`;
- Pantograph environment, tactic-rendering, and graph-pipeline tests.

Use `git diff 5ffd525c..MCTS-AC-branch -- <path>` before each existing-file copy. The result
must be empty for a file treated as verbatim. If it is not empty, move that file into the
hand-port group instead of resolving by replacement.

### Stage 2: hand-port MCTS-diverged files

Hand-port these files against both branch tips:

| File | AC changes to import | MCTS behavior to preserve |
|---|---|---|
| `hypergraph.py` | closure reasons and shared status fixes | `EdgeVisitStats`, PUCT metadata, MCTS propagation |
| `joint_inference.py` | `PantographEnv`, restart logic, rendering, guard, all-subgoal rule | legacy and PUCT loops, batched leaves, deadlines |
| `pln_rl_training.py` | public pluggable model API, update-size guards | HTPS replay optimizer and imitation/critic step |
| `rl_reasoner.py` | pluggable model interface and completion semantics | batched proposal, per-node stash, critic leaf evaluation |
| `rl_training_driver.py` | manifest-driven construction, Pantograph fixes, counters | search-mode config, two optimizers, replay queues, MCTS metrics |
| `search_harvest.py` | actor/critic separation and validity types | visit soft targets and minimal hypertree mining |
| `rl_actor_critic.json` | environment and update budgets, no model fields | PUCT and HTPS settings |
| `rl_process_walkthrough.md` | pluggable model/checkpoint description | legacy plus PUCT/HTPS execution and validity semantics |
| RL/harvest tests | AC manifest and budget assertions | selection, visits, queues, batched stash, MCTS loss paths |

Keep `selection_policy.py`, `test_selection_policy.py`, `test_visit_stats.py`, and MCTS
integration plans unless a planned migration explicitly changes their APIs.

### Stage 3: port the Pantograph integration through shared seams

- Build one `PantographEnv` from configuration.
- Verify it before model construction in training and evaluation.
- Create the initial server and every restarted server through that environment.
- Use `_server_is_dead` to distinguish a crashed REPL from a live elaboration rejection.
- Reap through `_close()`, recreate the server, and reinstall it on the executor.
- Guard `_start_state` once inside `_execute_and_link`, shared by legacy and PUCT.
- Mark the node with `NodeClosureReason.ELABORATION_ERROR` and invoke the completion hook
  with `ELABORATION_ERROR`.
- Discard that node's pending actions.

Port case-label parsing, metavariable pool rejection, bracket-aware tactic rendering,
separate `rej`/`err` counters, `max_dead_rounds`, and optimizer-step-based BC annealing.

### Stage 4: port manifest-driven MCTS model construction

In `RLTrainingConfig`:

- retain selection policy, simulation, replay queue, optimizer, reward, environment, and
  search-budget fields;
- add `max_update_nodes` and `max_update_edges`;
- remove `hidden_dim`, `num_layers`, `dropout`, `max_args`, and `use_node_type`;
- remove `top_k_subgoals` under the all-obligations migration.

At warm start and evaluation:

- load the version-2 checkpoint;
- validate model kind and vocabulary fingerprints;
- reconstruct through `build_model_from_checkpoint`;
- retain the returned `ModelSpec` for checkpoint saves and resume validation;
- create the live reasoner through one helper that threads both Pantograph and MCTS search
  configuration.

At resume:

- reconstruct the model from the resume checkpoint manifest;
- require the resume `ModelSpec` to equal the warm-start `ModelSpec`;
- create both optimizers against the reconstructed parameters before loading their states;
- restore RNG, curriculum, anneal count, tactic queue, critic queue, and MCTS settings;
- save all future checkpoints with `checkpoint_payload` and the complete version-2 manifest.

### Stage 5: implement validity-aware hypergraph and MCTS backup

- Add closure and search-end reasons.
- Add `N_v` and migrate `Q`, PUCT tests, serialization, and summaries.
- Implement `BackupValue` and separate edge/node tables.
- Replace hard float recursion, simulation recursion, and soft target extraction.
- Add unknown counters and provenance to graph summaries and training metrics.
- Ensure virtual loss is released on valid, unknown, and exceptional simulation exits.

### Stage 6: split actor and critic harvesting and losses

- Replace combined transitions with actor transitions plus unique critic samples.
- Keep executor failures actor-only.
- Exclude unknown edges from actor, advantage, entropy-label, and BC rows.
- Exclude unknown nodes from on-policy and HTPS critic losses.
- Preserve one on-policy step followed by the configured number of HTPS replay steps.
- Serialize critic sample source in the replay queue.

### Stage 7: migrate logical graph construction

- Remove every `top_k_subgoals` caller and configuration key.
- Attach all Lean subgoals to each accepted tactic edge.
- Remove PLN-created solved edges.
- Restrict proof traces and minimal-hypertree imitation to Lean-confirmed solved edges.

### Stage 8: checkpoint migration policy

Runtime loading must not guess version-1 layouts or architectures. Extend the explicit
migration command for the pre-pluggable MCTS GraphSAGE actor-critic layout and verify model
outputs on a fixed batch before writing a version-2 checkpoint.

Pre-validity MCTS optimizer moments and critic replay queues contain gradients or targets
whose evidence provenance cannot be reconstructed. They are not valid resume state after
this semantic change. The migration command may produce a version-2 model warm start, but
must not claim an exact RL resume. Start a new RL run with fresh optimizers and empty HTPS
queues.

Do not add a runtime fallback, legacy flag, or silent queue conversion.

### Stage 9: update documentation

Update `rl_process_walkthrough.md` and `rl_training_setup.md` to describe:

- checkpoint-selected GraphSAGE/GATv2 construction;
- PUCT search under the same public model interface;
- the distinction between `V_theta(s)`, `y(s)`, `G(e)`, and value validity;
- `N`, `N_v`, `W`, `Q`, and unknown simulation backups;
- actor-edge versus critic-node harvesting;
- all-subgoal AND semantics;
- the fact that only Lean QED creates hard success.

## Test plan

### Pluggable architecture and checkpoint tests

- GraphSAGE and all four GATv2 readouts construct through the registry.
- MCTS reasoner and losses contain no concrete encoder imports.
- Version-2 warm start, save, resume, and evaluation validate model and vocabulary
  manifests.
- Encoder fingerprint mismatches reject lemma indices and premise scorers.
- RL update-size guards count actor and critic batches for both encoders.
- The MCTS checkpoint migration preserves public model outputs and refuses exact old-state
  resume.

### Pantograph and pool tests

- Environment path, REPL executable, import defaults, and toolchain agreement are checked.
- Crashed servers restart in the same environment; live elaboration errors do not restart.
- Case labels are removed without dropping a real hypothesis named `case`.
- Metavariable goals are rejected from dataset and file pools.
- Bracket-taking tactics render valid Lean syntax.
- Elaboration failure discards pending actions in legacy and batched PUCT expansion.

### Validity truth-table tests

- solved + solved -> edge valid `1.0`;
- solved + genuine failure -> edge valid `0.0`;
- solved + unelaborated -> edge unknown;
- genuine failure + unelaborated -> edge valid `0.0`;
- unknown edge + solved alternative -> parent valid `1.0`;
- unknown edge + failed alternative -> parent unknown;
- all known failed edges on an exhausted node -> parent valid `0.0`;
- open or globally truncated internal node -> unknown;
- unelaborated, cycle, and depth-limit nodes emit no standalone negative critic sample;
- clean root budget failure emits one root `0.0`, not zeros for untouched internal nodes.

### MCTS visit tests

- valid simulation increments `N`, `N_v`, and `W`;
- unknown simulation increments `N`, leaves `N_v` and `W` unchanged, and releases `VL`;
- `Q` uses `N_v` while PUCT exploration uses `N + VL`;
- soft targets require `N_v >= visit_threshold`;
- maximum supported `Q` is used instead of maximum prior;
- unknown and unelaborated edges never enter soft critic replay;
- an unknown child prevents AND multiplication unless another child already supplies valid
  zero.

### Actor and critic loss tests

- several accepted edges from one node produce several actor rows and one on-policy critic
  row;
- a solved alternative trains the parent critic toward `1.0` while an unknown alternative
  supplies no actor row;
- executor-rejected actions remain actor-only;
- elaboration-failed actions supply no actor, critic, BC, imitation, or replay row;
- actor-only, critic-only, HTPS-only, and empty updates handle batches without invalid PyG
  construction;
- one on-policy optimizer step precedes every configured HTPS step for GraphSAGE and GATv2.

### Graph and imitation tests

- a tactic returning more subgoals than the former cap retains every child;
- the parent cannot become solved until all Lean-returned children solve;
- PLN cannot create a proof edge or hard success target;
- minimal-hypertree imitation contains only policy edges in a Lean-confirmed proof;
- proof traces contain every required child.

### Driver tests

- BC annealing advances only after an on-policy optimizer step and restores on resume;
- `max_dead_rounds` counts zero-update rounds and reports whole-search errors separately
  from tactic rejections;
- manifest-driven GraphSAGE and GATv2 synthetic runs preserve selection-policy parameters;
- HTPS optimizer and replay queues save and resume under version-2 checkpoints;
- evaluation uses the same live-reasoner factory and respects `use_pln=False`.

## Verification

```bash
# Establish the pre-change baseline first.
pytest maths_ai/gnn_inference/tests/

# Focused shared infrastructure.
pytest maths_ai/gnn_inference/tests/test_pluggable_architectures.py
pytest maths_ai/gnn_inference/tests/test_pantograph_env.py
pytest maths_ai/gnn_inference/tests/test_tactic_rendering.py

# Focused MCTS and learning-value behavior.
pytest maths_ai/gnn_inference/tests/test_selection_policy.py
pytest maths_ai/gnn_inference/tests/test_visit_stats.py
pytest maths_ai/gnn_inference/tests/test_search_harvest.py
pytest maths_ai/gnn_inference/tests/test_pln_rl_training.py
pytest maths_ai/gnn_inference/tests/test_rl_reasoner.py
pytest maths_ai/gnn_inference/tests/test_rl_training_driver.py

# Full post-change suite.
pytest maths_ai/gnn_inference/tests/
```

Run four synthetic or live smoke combinations through collect, update, checkpoint save,
resume, and greedy evaluation:

```text
GraphSAGE + legacy
GraphSAGE + PUCT
GATv2 + legacy
GATv2 + PUCT
```

Then run the live Mathlib environment probe with `source_root=maths_ai/lean_mathlib`.
Expected observations:

- transitions are collected and whole-search errors remain zero;
- elaboration errors are counted separately and produce no learning rows;
- unknown simulations do not change `N_v` or `W`;
- critic rows do not exceed unique eligible graph nodes;
- no proof trace omits a subgoal or contains `PLN_fallback`;
- checkpoint manifests identify the active encoder and vocabulary fingerprints.

Do not run Lake setup, checkout, restore, or build against `maths_ai/lean_mathlib` while an
extraction process is using it. The RL driver treats the shared compiled environment as
read-only.

## Alternatives considered

### Merge `AC-branch` into `MCTS-AC-branch`

Rejected. AC lacks MCTS-owned files and its tip deletes MCTS plans, selection-policy code,
visit-stat tests, and HTPS documentation. The main RL files also diverge structurally.

### Cherry-pick `1476ca6` and the four pluggable commits

Rejected as the primary method. The new architecture modules can be copied exactly, but
`5905e0f` overlaps the MCTS driver and loss modules, while AC versions of search and harvest
do not contain PUCT or HTPS. Repeated conflict resolution can silently remove MCTS behavior.

### Port only the RL checkpoint loader

Rejected. It would make RL depend on a manifest and model factory while supervised,
inference, lemma, and premise callers remain on direct GraphSAGE construction. The improved
path would be exercised by only one caller and artifact fingerprints would be inconsistent.

### Keep an `unelaborated` boolean with float fallbacks

Rejected. `backup_values`, simulation backup, and visit means would still require every
caller to remember a special check. One missed check recreates the false zero.

### Use the critic prediction for an unelaborated simulation

Rejected. The prediction is the quantity being trained, not observed search evidence.
Backing it into `W` creates a self-target and changes PUCT statistics after an
infrastructure failure.

### Increment `N` and use `W/N` after unknown backups

Rejected. Unknown traversals would dilute the action value toward zero. Separate total
traversals `N` from valid-value count `N_v`.

### Keep critic targets inside edge transitions

Rejected. It duplicates one state target by the number of accepted tactics and leaves
state-value learning coupled to action storage. Actor rows must be edge-keyed; critic rows
must be node-keyed.

### Retain maximum-prior `W/N` as the soft target

Rejected. Maximum prior reports the actor's initial preference, not the value found by
search. Use maximum sufficiently supported `Q` for the search-for-a-proof value.

### Retain `top_k_subgoals`

Rejected. A search budget may stop expansion, but it cannot erase Lean proof obligations.

### Preserve version-1 checkpoint loading at runtime

Rejected. Architecture guessing and pre-validity replay queues make exact resume unsafe.
Use an explicit model migration and start a fresh RL optimization run.

## Completion criteria

The port is complete when:

- GraphSAGE and GATv2 use the same manifest-driven model construction in supervised,
  inference, legacy RL, and PUCT RL paths;
- both search modes create and restart Pantograph in one verified Lean environment;
- every actor return has valid edge evidence and every critic target has valid node
  evidence;
- unelaborated states and unknown simulations produce no numeric learning label;
- MCTS exploration counts unknown work without inserting it into action-value means;
- all Lean-returned subgoals remain attached to their tactic edge;
- only Lean-confirmed closure can create hard success, proof traces, or imitation samples;
- all callers, configurations, checkpoints, documentation, and tests use the new contracts
  with no legacy fallback path.
