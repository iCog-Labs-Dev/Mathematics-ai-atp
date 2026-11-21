# The RL process, end to end: Lean states → search → training → inference

This walkthrough describes `MCTS-AC-branch` after merge commit `f21d044`, which
merged [PR #59](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/pull/59).
It follows the implementation rather than assuming every available feature is
enabled in a particular run. Source paths below are relative to `maths_ai/`.

The actor proposes tactics and arguments. Monte Carlo tree search (MCTS)
allocates search effort among
those proposals. The custom Lean REPL checks transitions. The trainer learns
from the resulting proof-search graphs. MCTS value backup changes search
statistics; gradient backpropagation changes model parameters later.

## 1. Search settings and training settings are separate

`RLTrainingConfig` and `hybrid_reasoner/selection_policy.py` define two modes:

| Setting | Search behavior | Required simulation fields |
| --- | --- | --- |
| `selection_policy="legacy"` | Best-first AND–OR expansion using the graph frontier | `num_simulations`, `sim_batch_size`, and `puct_c` must all be null |
| `selection_policy="puct"` | Repeated partial-hypertree selection, batched leaf expansion, and value backup | Explicit `num_simulations`; omitted `sim_batch_size` resolves to 4 and omitted `puct_c` to 1.0 |

`"rp"` is reserved and raises `NotImplementedError`. The configuration class
defaults to `"legacy"`; a supplied JSON file can override it. Changing only the
policy to `"puct"` while leaving `num_simulations` null fails configuration
validation before training starts.

For example, this is a search-settings fragment, not a complete launch
configuration or a tuned budget:

```json
{
  "selection_policy": "puct",
  "num_simulations": 64,
  "sim_batch_size": 4,
  "puct_c": 1.0,
  "htps_steps_per_round": 0
}
```

`htps_steps_per_round` independently controls extra supervised replay updates.
MCTS can collect experience while this setting is zero. The driver also mines
replay under best-first search; that mode has no MCTS visit means to mine.

`use_pln` independently enables the Probabilistic Logic Network (PLN) scorer.
PLN scores influence best-first ranking and reward shaping. They are not a
direct term in PUCT selection and do not establish proof success.

## 2. Startup: model, graph contract, and the standalone client

### Checkpoint and prepared-data admission

The driver loads the actor–critic checkpoint and prepared node/tactic
vocabularies, validates their manifests and graph representations, and builds
the checkpoint-selected GraphSAGE or GATv2 encoder. GATv2's readout choice is
part of the model specification. Collection and training use the same public
encoder and graph-featurization interfaces.

The live representation is `MODEL_SEXPR_GRAPH_SPEC` in
`gnn_inference/atp_lean_gnn/graph_contract.py`:

- Representation: `lean-model-sexp-v2`.
- Model-expression wire version: `model_sexp_version=1`.
- Graph schema: `graph_schema_version=1`.
- Local-context schema: `fv-name-role-type-v1`.

The representation name and wire version are distinct identifiers. Pretty-text
graphs are not substituted when canonical model expressions are missing.

### Client and executable ownership

The application imports `pantograph_async` from the standalone
`pantograph-async-client` dependency. The root `pyproject.toml` pins client commit
`664bcaa10578cb933afc618b8e8cb20a1b751106`. The active path does not use
PyPantograph's `Server` or carry another transport implementation in this repo.

`create_model_sexpr_client` in `hybrid_reasoner/pantograph_model_sexpr.py`:

1. Checks that the application graph contract matches the client's model wire.
2. Adds `Mathlib` to requested imports without duplicating it.
3. Creates the client and runs model-state probes under one startup deadline.
4. Closes a partially started client if admission fails.

The client owns subprocess startup, GNU `stdbuf -oL` validation and launch,
protocol-descriptor admission, session-option setting and verification, request
serialization, response decoding, state handles, and subprocess shutdown.
Executable and Lean project paths are explicit environment settings. An old or
incompatible custom executable is not accepted merely because it starts.

`hybrid_reasoner/pantograph_adapter.py` projects immutable client execution
snapshots into application-owned `Goal` and `GoalLocal` values. Targets, local
types, local values, binder roles, and model expressions survive this boundary.
The client package does not depend on the GNN, search graph, or PLN.

### Environment setup and evidence

`gnn_inference/scripts/setup_sexpr_environment.py` obtains the pinned companion
source revision, patch, Mathlib revision, and toolchain from the client package.
It inspects both checkout destinations before installing a toolchain or
materializing either repository. Existing origins must match exactly. Reusing
an already patched companion requires its expected base, allowed status/path
set, and package-verified source digests. Setup no longer rewrites origins.

`core/git_worktree.py` supplies the shared NUL-delimited Git status parser used
by setup and MCTS diagnostics. Diagnostics record actual dirty-file hashes and
structured status metadata without stripping filename bytes. Historical port
evidence remains separate from the newer safety source manifest.

These checks are not a complete filesystem attestation: pre-existing Git
`assume-unchanged` or `skip-worktree` flags can conceal local edits from the
current status reader. Setup's early `stdbuf` check still checks existence only;
GNU identity is checked later during client startup. Both are remaining limits,
not guarantees supplied by PR #59.

## 3. Two graphs and one shared neural encoder

The proof-state graph describes one goal and its local context. The proof-search
hypergraph describes alternative tactics and their resulting subgoals. They
serve different purposes and have different node identifiers.

### Canonical state to neural tensors

Dataset text initially supplies a `LeanGoalSeed`, not authoritative neural
input. Before proposal, the reasoner replays the goal through Lean and serializes
the resulting canonical state. `make_dag_featurizer` then performs:

```text
Lean-canonical Goal
    → model_goal_to_dag
    → dag_to_pyg using prepared node vocabulary and reverse edges
    → premise mask, State-root index, and graph fingerprint
```

The DAG is a directed acyclic graph of the canonical model expressions. Sampled
argument indices refer to positions in this graph. Their names are resolved
against the same DAG and local-name mapping before tactic rendering.
Missing required structural vocabulary labels cause an error; semantic
out-of-vocabulary labels are counted in update metrics. The live path does not
rebuild a vocabulary from rollout states.

`ActorCriticWithArgsClassifier.encode` returns four tensors, not four heads.
For a neural batch, `total_nodes` means its total DAG-node count, `batch_size`
its number of proof states, `hidden_dim` the encoder output width, and
`num_tactics` the tactic-vocabulary size:

| Tensor | Shape | Producer |
| --- | --- | --- |
| Node embeddings | `[total_nodes, hidden_dim]` | Shared graph encoder |
| State embeddings | `[batch_size, hidden_dim]` | Encoder's configured state readout |
| Tactic logits | `[batch_size, num_tactics]` | Actor head |
| State values | `[batch_size, 1]` | Critic head |

The actor has a base classifier plus a residual classifier. Warm-start
conversion can initialize that residual to zero; an existing actor–critic
checkpoint instead supplies its saved actor and critic weights. Do not assume
that every RL launch initializes a fresh random critic.

The argument pointer uses the state embedding, tactic embedding, node
embeddings, and premise mask. Each selected argument's embedding conditions the
next pointer step. The shared encoder receives gradients from tactic prediction,
argument prediction, and critic regression.

### AND–OR proof structure

In `ProofHypergraph`, different outgoing tactic edges are OR alternatives: one
solved edge solves its parent. The children of one edge are AND obligations:
every child must be solved. A successful tactic returning no goals immediately
creates a solved edge.

For example, `constructor` on a conjunction creates one edge with two children.
Solving only one child does not solve that edge. Both search modes retain every
Lean-returned subgoal. PLN may reorder those children but does not select a
smaller subset to count as a proof.

Cycle and depth-limit closures describe failures of the bounded search path.
They are not mathematical proofs that the state is unprovable.

## 4. One search: materialization, proposal, and checked execution

### Scoped Lean state ownership

`HybridReasoner.prove` permits only one active search per reasoner and opens one
client `state_scope` for that search. The root is materialized before the search
graph is created. Later expansions replay each node into a fresh canonical
state; the cached root materialization avoids replaying the root unnecessarily.

`materialize_replay` starts the replay expression and applies the required local
introductions, or `skip`, to obtain serialized model expressions. Intermediate
start states are released. Each expanded materialization is released after use;
the enclosing scope cleans up remaining allocations on normal completion,
exceptions, and cancellation.

Every alternative tactic at a node is applied to that node's original
materialized state, not to the preceding alternative's successor.
`PantographExecutor` renders the command, asks the scope to apply it, converts
accepted successor goals to canonical application values, and releases the
temporary successor allocation.

A checked tactic rejection is negative action evidence. A transport, protocol,
or ownership failure is not. The latter propagates out of the search rather
than being converted into an ordinary rejected tactic. The client enforces the
companion transition-validation contract, including sorry/unsafe rejection.

Graph-based success is the implemented result. Separately replayed subgoal
closures do not assemble and kernel-check a complete theorem proof term.

### Action proposal and storage

Every search takes a `ProposalSpec` (`hybrid_reasoner/proposal.py`): a mode,
`sample` or `greedy`, and a seed. The reasoner seeds its own `torch.Generator`
from it on the model's device, so a search never draws from the global Torch
RNG. Training derives seeds from `cfg.seed`, the round and the input's replay
fingerprint; evaluation from `eval_seed`, the repeat and the input. Neither
depends on which worker ran the search. Draws repeat for a fixed seed, model,
input and device; equality across devices is not promised.

A proposal round calls `model.propose_actions` under `model.eval()` and
`torch.no_grad()`: it encodes each state once and draws exactly the requested
number of actions per state, in row-major order. Each action's arguments are
chosen against its state's cached node embeddings, conditioned on its own
tactic and earlier arguments. In `sample` mode it draws a tactic and then each
argument; in `greedy` mode it takes the argmax of the tactic and then of each
argument step, given the arguments before it.
Greedy is step-wise, not a search for the most probable argument sequence. A
multi-graph batch runs as many pointer steps as its largest arity, and each
row keeps only the steps its own tactic owns (`ActionSample.arities`). The
configured `top_k_tactics` is the number of draws in one proposal round, not
deterministic top-k enumeration.

Each decoded draw becomes a `ProposedAction`: the command, the raw action
(tactic id and argument indices) and its log-prior, the tactic's log-prob plus
the log-probs of exactly its own argument steps. A draw whose tactic needs
more arguments than the model chooses, or whose argument step had no legal
premise, is unrenderable. Identical raw draws are proposed once and keep a
multiplicity.

A command runs in Lean at most once per node: the node caches its checked
outcome (`ProofNode.command_outcomes`), and a later raw action that renders
the same command reuses it. The cache key is the final Lean command string
from the shared renderer in `data_models/tactic_commands.py`, including its
argument formatting and normalization. Graph edge uniqueness and alias
validation use that same identity. An unrenderable proposal never enters the
checked-outcome cache or creates a graph edge. Each raw action at a node becomes one
`PolicyObservation` with its multiplicity across all proposal rounds, its
outcome (accepted with an edge id, rejected, or unrenderable) and its prior.
Two raw actions rendering one command stay separate observations with their
own pointer tuples. Elaboration failures and external aborts discard the
round's draws rather than labeling untested actions as Lean rejections.

The result is `RLSearchResult(graph, observations)`. Each reasoner resets its
storage between searches. Edge IDs are local to one result; the trainer
offsets them before joining multiple results into a round batch.

### Best-first search

With `selection_policy="legacy"`, the reasoner expands the highest-ranked open
frontier node. Initial ranking combines incoming tactic probability with PLN's
strength-times-confidence score, or with 1 when PLN is disabled. Graph status
and rank propagation update ancestors after mutations. This is structural
propagation, not neural gradient backpropagation.

The loop stops on root success, root exhaustion, empty frontier, node budget,
or cooperative deadline. Depth limits close individual expansion paths.

## 5. MCTS: selection → batched expansion → value backup

With `selection_policy="puct"`, `_prove_mcts` repeats simulation batches over
the same growing hypergraph. A simulation selects a partial proof structure;
it does not necessarily perform a random rollout all the way to closure.

### Selection statistics and PUCT

At each expanded node, selection chooses one non-dead outgoing tactic edge.
It then enters every child of that edge. The following symbols refer only to
search statistics, not gradients or model parameters:

| Symbol | Meaning and implementation field |
| --- | --- |
| `e` | One tactic hyperedge in `ProofHypergraph.edges` |
| `N(e)` | Completed traversals, including unknown outcomes; `visit_stats.N` |
| `N_v(e)` | Completed traversals with valid numeric backups; `visit_stats.N_v` |
| `W(e)` | Sum of valid backup values; `visit_stats.W` |
| `VL(e)` | In-flight selection holds; `visit_stats.virtual_loss` |
| `P(e)` | Effective prior from `ProofHypergraph.puct_priors`: the edge's observed complete-action mass, normalized over the parent's non-dead edges |
| `Q(e)` | Valid-value mean `W(e) / N_v(e)`, or 0.5 when `N_v(e)=0`; `visit_stats.Q` |
| `total_visits` | Sum of `N + virtual_loss` over the parent's non-dead edges |
| `c` | Configured PUCT exploration constant, `puct_c` |

The score implemented by `puct_score` is:

```text
score(e) = Q(e) + c * P(e) * sqrt(total_visits) / (1 + N(e) + VL(e))
```

Each edge keeps the immutable log-priors of the distinct raw actions observed
to render its command; its mass is their log-sum-exp, so a repeated draw adds
no mass and a new alias adds its own once. `puct_priors` normalizes the masses
in log space over the node's non-dead edges on every selection, so an added
edge, a new alias or a dead edge renormalizes the priors without changing any
`N`, `N_v`, `W` or raw log-prior. This is mass over the raw actions the search
observed, not the exact probability of a command over every possible alias.
The tactic-family probability stays on `TacticCandidate.probability` for
best-first and PLN ranking only. The current `Q` property does not subtract
virtual loss from its mean. Virtual loss affects the exploration term through
the sibling total and edge denominator, never the prior.

### Progressive widening

A PUCT node's proposal source stays open across rounds. Under
`WideningPolicy` (`selection_policy.py`) a node with `v` completed simulation
visits may have attempted

```text
capacity(v) = min(widening_max_actions,
                  max(widening_initial_actions,
                      ceil(widening_coefficient * (1 + v) ** widening_exponent)))
```

distinct commands, accepted or rejected, and at most `widening_max_draws` raw
draws in total. Defaults are 4, 2.0, 0.5, 32 and 128; best-first rejects
explicit values. Selection stops at a node that may propose (capacity and
draws left, and its next round is due) and requests one proposal round there, sized before drawing as the minimum of
`top_k_tactics`, the remaining capacity and the remaining draws. Rounds at one
node are spaced by completed visits, with a gap that grows by one visit per
round, so a node keeps retrying while capacity and draws remain, including
after its capacity has saturated, and the visits in between descend through
its existing edges. Otherwise selection takes an edge by PUCT; with no
selectable edge the node is deferred. Every node a simulation selected gets
one visit when it backs up, so a rejected or deferred node's capacity can
grow. The generic engine is asked for each node's own count of ranked
candidates. A node is dead only when its source is
closed (`widening_bound`, a finite provider's single round, or a path limit)
and no edge survives; a closed node is never reopened. The generic engine and
the greedy actor are finite sources and close after one round.

Selection increments virtual loss immediately. Later selections in the batch
therefore see an altered exploration score. This encourages, but does not
guarantee, different branches. Simulations can still share leaves, which are
deduplicated before expansion.

### Batched expansion

`_expand_leaves` checks depth limits and materializes each open leaf before
proposal. `predict_next_tactics_for_nodes` preserves exact node ownership while
batching the canonical goals. The model runs one multi-state forward per draw
across the proposing leaves. It does not run just one forward for all draws.

The worker then executes candidates sequentially on its client and retains all
accepted subgoals. Independent workers can execute separate searches
concurrently; `sim_batch_size` does not create additional workers or REPLs.

### Backup and stopping

Backup combines the selected structure bottom-up. Solved nodes contribute 1;
recognized failed paths contribute 0; elaboration and infrastructure failures
are unknown. Unresolved leaves use `RLHybridReasoner._leaf_value`, which reads
the critic without gradients. MCTS clamps this estimate into `[0, 1]`.

The AND combination returns zero if a child has a valid zero, otherwise unknown
if any child is unknown, otherwise the product of all child values. This
product is the implemented heuristic, not a guarantee of calibrated independent
subgoal probabilities. The base `HybridReasoner` supplies a constant leaf value
of 0.5; selecting PUCT on that class alone does not load a learned critic.

Each traversed edge increments `N`. Only valid numeric outcomes increment
`N_v` and add to `W`. Virtual loss is released after backup and on exceptional
expansion exits. These operations never call an optimizer.

The loop checks root status, node budget, simulation count, and deadline
between batches. Graph status propagation also occurs during expansion.
Search returns a solved or partial graph; it does not repeatedly restart after
committing to the most-visited root action.

## 6. Collection workers and the fixed-model boundary

`LiveReasonerPool` in `gnn_inference/atp_lean_gnn/rl_live_collection.py` owns one
client/REPL and one reasoner per worker. Each worker runs one theorem at a time;
different workers take goals from a shared queue concurrently.

Models are shared per collection device. A worker on the update device uses
the canonical model; workers on another device share a read-only replica on
that device. Several independent REPL workers can use one GPU. Worker count,
GPU count, and MCTS simulation-batch size are different settings.

No optimizer runs until the round's collection has joined. The driver then
performs its on-policy update, optional supervised updates, and synchronization
of every distinct replica. A generation counter advances by the number of
actual optimizer steps. Pool generation assertions check the canonical and
replica generation before collection. Synchronization during collection is
rejected.

Each theorem gets a cooperative deadline. Clean deadline completion retains
the partial graph. An `asyncio.timeout` backstop at 1.25 times the theorem
timeout interrupts a search that has not completed cooperatively. Request
timeouts and transport failures retire that worker's client/reasoner binding;
the pool closes it and attempts a fresh admitted replacement for later work.
The interrupted theorem does not supply a completed search result.

Protocol, state-ownership, and configuration errors are fatal rather than
blindly retried. If every worker becomes unavailable, collection fails. The
single-reasoner collection path used by tests cannot replace its only client
and propagates failures that leave it unusable.

## 7. Harvest: graph evidence becomes actor and critic rows

`search_harvest.compute_backups` separates valid numeric outcomes from unknown
evidence and records each target's `BackupSource`. PLN scores are not directly
inserted into these critic labels.

| Evidence | Training treatment |
| --- | --- |
| Graph-solved node backed by checked transitions | State target 1, `LEAN_STATUS` |
| Closed proposal source with every alternative dead | State/path failure 0, `SEARCH_FAILURE` (a failure of this search configuration, not unprovability) |
| Open proposal source whose known alternatives all failed | Unknown; the node could still widen |
| Depth or cycle closure | Incoming path can fail; that node is omitted as a standalone fresh-root critic target |
| Elaboration or external failure | Unknown; no fabricated critic zero |
| Clean root node/simulation/deadline budget exhaustion | Root-only episode target 0, `ROOT_EPISODE` |
| Supported non-dead MCTS edge | Optional soft target from the maximum supported edge mean, `VISIT_MEAN` |

Root episode zeros mean failure under the configured search budget, not
unprovability. Soft means can contain earlier critic bootstrap estimates and
are not equivalent to Lean-established proof facts.

### Actor returns and PLN shaping

`extract_transitions` emits rows only for executed policy edges with valid
outcomes. It uses structural backups without a visit threshold. Unknown edges
are omitted. For each row, `reward` is the edge reward, `gamma` is
`RewardConfig.gamma`, and `successor_value` is the valid combined edge outcome:

```text
return_ = reward + gamma * successor_value
```

The edge outcome already includes deeper structural consequences. The code
applies this same return formula to closing edges as well as interior edges.

The terminal part of `reward` is success reward minus step penalty for a
closing edge, failure reward minus step penalty for a dead edge, and negative
step penalty otherwise. PLN adds a shaping term. Let `potential(node)` mean
the PLN strength stored on that search node, or strength times confidence when
`RewardConfig.use_score` is true. Solved/dead nodes and nodes without PLN
metadata have zero potential. The implemented shaping is:

```text
nonempty edge: sum(gamma * potential(child) - potential(parent) for each child)
closing edge:  -potential(parent)
```

With `use_pln=False`, all these potentials vanish. This describes the actual
branching reward; the standard single-trajectory potential-shaping theorem
alone does not establish policy invariance for this child-summed hypergraph
reward and product-valued backup.

### Separate critic and replay extraction

The on-policy trainer calls `extract_critic_samples` without a visit threshold.
Its state targets therefore do not automatically include MCTS visit means.
The driver separately calls that extractor with `cfg.visit_threshold` when
filling the critic replay queue. Support is measured by valid visits `N_v`, not
all completed visits `N`. Root episode overrides and validity rules still apply.

`extract_minimal_hypertree` chooses solved tactic edges minimizing total tactic
applications over their child proofs. It emits only edges joined to stored
policy actions. By default it mines all solved nodes, including solved
subproblems of an unsolved root; `mine_all_solved_nodes=False` restricts mining
to a solved root's selected proof. This is imitation of found proofs, not a
normalized visit-count distribution over every root action.

## 8. Gradient updates: recompute first, then update

### At most one on-policy update

`train_step_onpolicy` collects valid actor transitions, failure records, and
separate critic samples across the round's results. Re-featurization must match
each stored graph fingerprint. Configured update-node and update-edge limits
guard batch size; exceeding a positive limit raises rather than silently
training on only part of the collection.

`model.evaluate_actions` recomputes stored tactic log-probabilities and
teacher-forces the argument pointer through stored indices. Invalid or masked
argument positions contribute zero log-probability and a zero selected
embedding. No fresh action is sampled for the loss.

For an actor row, its advantage is its return minus its detached critic
prediction. Advantages are normalized jointly across accepted and rejected
rows using population standard deviation. Multiplicity weights the actor
loss after that normalization. A single actor row has zero normalized
advantage. Detaching the baseline blocks actor-loss gradients through that
value; it does not make a poor critic harmless.

The combined loss contains:

- Negative advantage-weighted tactic log-probability plus weighted argument
  log-probability, averaged with action multiplicity.
- Weighted mean-squared critic error on separate valid state rows.
- Subtracted positive tactic-distribution entropy, encouraging exploration.
- An annealed tactic cross-entropy anchor on accepted actor-transition rows.
  These rows need not all belong to a completed successful proof.

Rejected/unrenderable actions use failure reward minus step penalty and do not
create state-value targets merely because the action failed. Argument weight
`arg_loss_weight` scales the pointer part; at values other than 1 the actor
objective is not the unweighted full joint-action log-probability.

If no usable actor or critic rows exist, the step is skipped. Otherwise the
trainer clears gradients, computes the loss, calls `backward`, clips gradients,
and performs one optimizer step. There is no optimizer step between collection
and this recomputation.

This is the intended on-policy parameter boundary, not a claim that all
estimator assumptions are proven: collection uses `model.eval()` while updates
use `model.train()`. Nonzero dropout can change action probabilities despite
unchanged parameter values. Filtering unknown outcomes and the search-derived
returns also need to be considered when interpreting the estimator.

### Optional supervised replay updates

After the on-policy attempt, the driver mines both queues from every completed
graph. It does so in either search mode and even when
`htps_steps_per_round=0`. The second AdamW optimizer is also constructed
unconditionally; zero disables its update loop, not queue creation or mining.

Each enabled `train_step_htps_style` samples up to `htps_batch_size` rows from
each available queue. Imitation and critic rows use separate forwards. The
imitation loss is tactic cross-entropy plus weighted negative stored-argument
log-probability. The critic loss is mean-squared error against stored targets,
scaled by `w_critic_soft`.

`optimizer_htps` is a different optimizer instance over the same model
parameters. It maintains separate Adam states as an implementation choice;
changing parameters through either optimizer is visible to the other. The
ordering is collection → on-policy attempt → supervised replay steps → replica
synchronization. Replay begins from the parameters left by the on-policy step,
not from an unchanged copy of the collection model.

Proof labels remain observations of found proofs. Stored visit means are soft,
historical estimates that can become stale; neither type uses an on-policy
score-function ratio during these supervised updates.

## 9. The complete training round and checkpoint boundary

```text
Validate config, checkpoint, prepared graph contract, and vocabulary identity
Create the canonical model, both optimizers, replay queues, and admitted workers
Build the goal pool from Lean-admitted seeds and hold out the evaluation set

For each round:
    sample goals from the current curriculum window
    collect searches concurrently with one fixed model generation
    harvest structural actor/critic rows; attempt at most one on-policy update
    mine minimal solved proofs and threshold-supported critic rows into queues
    run the configured number of supervised replay updates
    advance model generation for actual updates and synchronize replicas
    update solve-rate history over searched seeds and grow the curriculum when its condition passes
    write periodic last.pt; periodically evaluate and save best.pt on improvement
```

Dataset mode starts from dataset proof states, which can be intermediate goals,
not necessarily complete theorem statements. File mode accepts supplied goal
seeds. Before the pool is built, `seed_admission.preflight` replays every seed
through Lean on a dedicated client and keeps only the ones that materialize. The
pool sorts by goal size and expands an accessible prefix as recent
solve rate reaches the configured threshold. A seed that fails to replay during
collection is counted as inadmissible, not as a failed search, and a round that
searched nothing leaves the window unchanged. Size is a curriculum heuristic;
the code does not guarantee a fixed distance-to-proof for dataset roots.

The tactic-anchor weight anneals by the count of actual on-policy optimizer
steps, even though its configuration field is named `bc_anneal_rounds`.
Consecutive rounds without either kind of optimizer update eventually raise
at `max_dead_rounds` rather than silently reporting productive training.

Checkpoints include the model manifest and weights, vocabularies, graph
representation, both optimizer states, round/curriculum state, on-policy step
count, search settings, Torch RNG state, and serialized replay queues.
`rl_target_schema_version=4` is required for resume, and the saved search
settings (including the widening bounds, proposal seed derivation and prior
contract) and evaluation contract must match the configuration. Replay rows retain canonical
goals and graph fingerprints; critic rows also retain target provenance.
Older incompatible optimizer/replay schemas are not silently resumed.
The saved Torch RNG state alone is not a claim that every asynchronous worker
schedule and every random source can be reproduced exactly.

## 10. Evaluation and inference

`evaluate_proof_rate` runs the held-out goals `eval_repeats` times with
`eval_proposal_mode` (`sample` by default, or `greedy`) and seeds derived from
`eval_seed`, without an optimizer update or a draw from the global RNG. It
retains the reasoner's configured search mode and budgets. Standalone
`--eval-only` and periodic evaluation read the same settings, and
`--eval-proposal-mode`, `--eval-seed` and `--eval-repeats` override them for
both. The result lists every repeat and the aggregate over searched seeds;
inadmissible seeds are reported separately. `--eval-report` writes a new
report with the search, prior, widening and dataset settings used. Proof rate,
rather than sampled training return, determines `best.pt`, and the checkpoint
stores the evaluation contract next to it: a resume with different
evaluation settings is refused rather than compared with an incomparable score.

Sampled evaluation proposes different tactic families and arguments at a
state. Greedy evaluation is deterministic for both and closes each node after
one round, so it measures the policy's step-wise argmax, not a diverse MCTS
candidate set or a model-only forward.

Inference with the trained actor and critic uses `RLHybridReasoner` with loaded
weights and the matching graph/vocabulary contract. Its name does not force a
training update: calling `prove` only runs search. A caller can choose sampled
or greedy proposals and best-first or PUCT search. No gradients or optimizer
steps are needed. The generic `HybridReasoner` instead uses its GNN-engine
proposal path and constant 0.5 leaf estimate unless explicitly specialized.

MCTS visit statistics are local to the current search. They update during
inference even though model parameters stay fixed. A solved graph records
checked local tactic transitions and AND–OR closure, not an independently
assembled theorem certificate.

## 11. Mechanism and source map

| Responsibility | Source |
| --- | --- |
| Model, sampling, stored-action evaluation | `gnn_inference/atp_lean_gnn/actor_critic.py`, `argument_selector.py` |
| Canonical graph admission and featurization | `gnn_inference/atp_lean_gnn/graph_contract.py`, `pln_rl_training.py` |
| Application/client projection and startup probes | `hybrid_reasoner/pantograph_adapter.py`, `pantograph_model_sexpr.py` |
| Scoped execution, best-first search, MCTS simulation loop | `hybrid_reasoner/joint_inference.py` |
| AND–OR propagation and visit statistics | `hybrid_reasoner/hypergraph.py` |
| PUCT formula and search-mode validation | `hybrid_reasoner/selection_policy.py` |
| Sample ownership, action decoding, learned leaf estimates | `gnn_inference/atp_lean_gnn/rl_reasoner.py` |
| PLN reward construction | `gnn_inference/atp_lean_gnn/pln_reward.py` |
| Structural/soft targets and minimal-proof mining | `gnn_inference/atp_lean_gnn/search_harvest.py` |
| On-policy and supervised replay updates | `gnn_inference/atp_lean_gnn/pln_rl_training.py` |
| Worker lifecycle and model generations | `gnn_inference/atp_lean_gnn/rl_live_collection.py` |
| Resource/device planning | `gnn_inference/atp_lean_gnn/rl_resources.py` |
| Curriculum, round ordering, evaluation, checkpoints | `gnn_inference/atp_lean_gnn/rl_training_driver.py` |
| Setup and shared Git inspection | `gnn_inference/scripts/setup_sexpr_environment.py`, `core/git_worktree.py` |
| Application and MCTS live diagnostics | `hybrid_reasoner/pantograph_diagnostics.py`, `mcts_diagnostics.py` |
| Launch and user settings | `gnn_inference/scripts/rl_train.py`, `gnn_inference/configs/rl_actor_critic.json` |

Historical live reports under `gnn_inference/tests/fixtures/pantograph/` identify
the source and executable tested at their capture time. They are not fresh
validation of a different local binary, this documentation update, or GPU
deployment. Relevant regressions live in `test_rl_reasoner.py`,
`test_search_harvest.py`, `test_pln_rl_training.py`, `test_rl_training_driver.py`,
`test_rl_live_collection.py`, `test_pantograph_model_sexpr.py`,
`test_git_worktree.py`, and `test_setup_sexpr_environment.py` under
`gnn_inference/tests/`.
