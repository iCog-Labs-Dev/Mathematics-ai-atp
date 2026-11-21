# Complete-action priors, progressive widening, and sampled MCTS evaluation

Status: implemented for issue #62. Stages 1–9 landed in order; what was
verified, offline and on the real REPL, and what is still open are in
[the handoff](../mcts_widening_evaluation_handoff.md). The plan below is kept
as written.

## 1. Objective and scope

Enable bounded exploration of additional tactic alternatives at previously
expanded MCTS nodes, and expose the same sampled proposal mechanism during
standalone evaluation, periodic training evaluation, and direct inference.
Use normalized complete-action probabilities in PUCT, including the tactic and
its conditional argument choices. Keep model weights fixed during every search
and evaluation pass.

The shared search, proposal, graph-status, and training-accounting mechanisms
must support repeated proposal rounds. Do not implement widening by restarting
a search, clearing a node's terminal flag, or invoking a separate reasoner.

Requested outcomes:

1. A heavily visited unresolved node can receive new complete actions later.
2. An initially rejected candidate set does not immediately terminate a node
   that still has a bounded opportunity to propose other actions.
3. Sampled evaluation explores tactic families and arguments, not only different
   arguments of one argmax family.
4. Evaluation proposal mode and seed reach every existing caller, including the
   worker pool and periodic checkpoint-selection evaluation.
5. Reports identify the proposal strategy, seeds, repeats, search budgets,
   dataset/admission identities, and widening behavior actually used.
6. Repeated samples remain learning observations without repeated Lean execution
   of the same command at the same canonical state.
7. PUCT priors distinguish argument choices as well as tactic families, with
   correct arity masking and numerically stable normalization.
8. Widening and newly observed command aliases update effective priors without
   resetting search statistics or changing best-first/PLN ranking semantics.

Related shared repair: the explicit `greedy` strategy must select both the
tactic and each argument by argmax. The current tactic-only greedy behavior is
not retained under another compatibility name.

Out of scope:

- Enumerating every action, deterministic beam search, policy-temperature
  changes, root Dirichlet noise, and transposition merging.
- Updating network weights inside a simulation or changing AND-value products.
- Complete-theorem proof assembly or certification.
- Retaining Lean snapshots between proposal rounds, changing the companion
  REPL/client, or implementing an untracked snapshot-retention plan.
- Repairing unrelated admission-cache, Git-inspection, or GNU-preflight issues.
- Claiming that this change proves the existing actor estimator unbiased.

## 2. Verified current branch and the recent feature

Plan baseline: `MCTS-AC-branch` at
`dd6eedc100321af5d8c39a97abfec4a7addf8fb7`, the merge of PR #61,
`feat/mcts-dataset-goal-reconstruction`. The earlier walkthrough commit is
`61f6d384730fca3dd7a56918377af4efb2c912cc`.

Read alongside:

- [Dataset replay contract](../dataset_goal_replay.md).
- [MCTS replay-port handoff](../mcts_dataset_replay_port_handoff.md).
- [Current walkthrough](../rl_process_walkthrough.md).

The recent feature is already implemented, not work to repeat:

| Current contract | Required preservation |
| --- | --- |
| `HybridReasoner.prove(seed)` and `RLHybridReasoner.prove(seed, ...)` accept `maths_ai.lean_replay.LeanGoalSeed` | Never reintroduce a goal/hypothesis-string search API or text fallback for structured rows |
| `build_theorem_pool(cfg, env, ...)` is asynchronous | Await it and preserve dedicated admission preflight before searches |
| `TheoremItem.seed` contains typed input and provenance | Keep this object through queues, seed derivation, and worker replacement |
| `materialize_seed` validates raw replay and normalized model input | Every widening round uses the same checked materialization path |
| Model identity differs from live execution identity | Cache and compare model fingerprints; render arguments through the current row's execution declarations |
| `_execute_and_link` inherits parent model identity onto successors | Keep that transformation before ranking, linking, and fingerprinting |
| Root replay rejection raises `ReplayInputError` before a graph exists | No visits, actor failures, or proof failure labels from that root |
| Descendant replay errors are unknown evidence | No zero target manufactured from failed reconstruction |
| Pool/single collection record inadmissible roots separately | Preserve searched-seed solve-rate denominators and curriculum handling |
| Client commit is `d1dac7e8a8ea296192e7a5b2616114b73e0812b7`, version 0.1.1 | Preserve explicit universe-level replay and the existing pinned dependency |

Current limitations directly relevant to this plan:

- `_expand_leaves` only proposes at `OPEN` nodes.
- `_execute_and_link` ends a candidate batch with `mark_node_exhausted`.
- A `DEAD` node is terminal; adding an edge later does not safely revive it.
- Node visits are not recorded independently of outgoing-edge visits.
- `_pending` is discarded after each node expansion. Later samples can create
  duplicate executions or overwrite the existing edge/action relationship.
- `RLSearchResult.edge_actions` holds one representative action per edge;
  `failure_actions` is a separate list. This is insufficient for repeated
  raw actions that map to the same executed command.
- `evaluate_proof_rate` hardcodes `greedy=True`; both CLI and periodic
  evaluation use it.
- `ActorCriticWithArgsClassifier.act` only applies `greedy` to the tactic.
  `ArgumentSelector.sample_step` still samples arguments.
- Collection samples currently use shared Torch random state.
- `_decode_row` sets candidate probability from `tactic_logp` alone;
  `ProofHypergraph.add_edge` copies it into `EdgeVisitStats.prior_prob`.
  Arguments therefore affect execution but not the stored PUCT prior.
- `ActionSample.arg_logp` sums pointer steps up to the batch's maximum arity.
  It cannot be used blindly for an individual shorter-arity action.
- `TacticCandidate.probability` also affects child ranking and PLN-related
  structural ranking. Reinterpreting that field as a joint or normalized
  PUCT probability would silently change another algorithm.
- Exact RL resume requires `rl_target_schema_version=3` and matching
  `search_settings`; evaluation semantics are not completely represented there.

### Baseline verification performed while writing this plan

Using the current `.venv/bin/python`, a focused offline run over actor,
argument, PUCT, visit, reasoner, scoped-client, driver, worker, harvest, loss,
dataset replay, model identity, and Lean replay tests produced:

**244 passed, 1 skipped, 9 deselected.** The deselected tests require the actual
REPL. This is current baseline evidence, not evidence that widening works.
The installed client distribution reports version 0.1.1 at the pinned commit.

## 3. Architectural decisions and alternatives

| Decision | Chosen approach | Alternatives and tradeoffs |
| --- | --- | --- |
| Evaluation exploration | Sample complete actions using the shared policy; keep explicit true-greedy diagnostics | Greedy-only restricts families. Beam proposals are deterministic but require a separate argument-sequence search |
| Search breadth | Visit-dependent widening with hard action-attempt and raw-draw bounds | One large initial batch spends Lean work on little-visited states. Restarting loses statistics. Unlimited resampling has no reliable cost bound |
| Widening capacity | Count distinct rendered commands attempted, whether accepted or rejected | Counting accepted edges alone is conventional but allows rejection-heavy nodes to issue too many Lean requests without a second bound |
| Repeated actions | Per-search execution cache plus raw-action observation ledger | Re-executing duplicates wastes Lean work. Dropping duplicates loses sampled-action credit. Keeping only one raw pointer tuple can train the wrong action |
| Randomness | Explicit per-search, per-device generators derived from stable input identity | Repeated global reseeding interferes with concurrent workers and training. Worker-ID seeds depend on scheduling |
| Exhaustion | Separate proposal-source closure from completion of one proposal batch | Clearing `DEAD` later invalidates ancestor status and earlier failure labels |
| Prior meaning | Arity-correct joint action log-probability, separate from tactic-family ranking confidence | Tactic-only priors cannot distinguish arguments. Reusing the ranking field changes best-first behavior unintentionally |
| Prior normalization | Normalize observed raw-action mass over accepted non-dead edges at selection time | Raw mass makes the exploration scale depend on admitted probability mass and action length. Normalizing before Lean admission leaves rejected actions in the denominator |
| Command aliases | Sum probabilities of distinct observed raw actions once per command | Multiplying by draw count adds mass repeatedly. Exact marginalization over all unobserved aliases requires enumeration not available here |
| API evolution | Migrate all callers to one proposal request and one observation schema | A `greedy` argument plus a second mode field leaves contradictory sources of truth |

The implementation must not retain a fixed-expansion PUCT compatibility mode.
PUCT uses the new bounded widening lifecycle. Best-first remains an explicitly
supported, different algorithm with a finite one-batch candidate source; it is
not an automatic fallback after MCTS budget exhaustion.

Decision for review: standard evaluation defaults to sampled proposals. True
greedy remains available as a separately labeled metric. Existing `best.pt`
scores must not be compared across these metric definitions.

## 4. Shared proposal policy and random-number ownership

Introduce a dependency-light immutable `ProposalSpec` in a shared search-policy
module. It describes `mode` (`sample` or `greedy`), a resolved integer seed,
and a seed-derivation version. It contains no live generator or client handle.

At the reasoner boundary, replace `prove(..., greedy=...)` with
`prove(seed, *, proposal=ProposalSpec(...), deadline=...)`. Migrate pool
callbacks, single collection, training, standalone evaluation, periodic
evaluation, smoke scripts, diagnostics, and tests together. Do not add a shim
that accepts both arguments.

The actor/pointer path receives the resolved mode and an explicit
`torch.Generator` on the model's device. A shared categorical-selection helper
performs argmax for greedy mode and generator-backed multinomial sampling for
sample mode. Preserve masks, all-invalid-row handling, returned log-probability,
entropy, and selected-embedding semantics. Migrate the old pointer sampling
entry point rather than keeping a duplicate implementation.

Arguments remain autoregressive. Greedy chooses each next argument by argmax;
it does not claim to find the globally most probable argument sequence.
Inactive arity rows must not contribute action probability or observation
arguments. Greedy with no usable premise still uses the explicit invalid-action
handling, not an invented valid pointer choice.

For evaluation, derive a search seed from the configured `eval_seed`, repeat
index, and stable admitted-input identity. Use a versioned cryptographic digest
of the replay fingerprint and source identity, not Python's randomized `hash`.
Do not include worker ID, task completion order, or checkpoint generation.
Repeated duplicate inputs must have a documented stable occurrence identity.

Training uses the same derivation helper with a training namespace, `cfg.seed`,
round identity, and input identity. Evaluation never reseeds or consumes the
global training RNG. Sampling is reproducible for a fixed execution/device
contract; cross-device bitwise equality is not promised.

Keep existing model-generation assertions, training/evaluation mode restoration,
and no-update-during-collection ordering. `model.eval()` and stochastic action
sampling are compatible operations.

## 5. Complete-action priors and normalization

This expands the shared candidate contract: PUCT candidates now carry the
probability of a complete policy action, not merely a tactic-family confidence.
Every search provider and graph/PUCT consumer must migrate together. There is
no optional joint-prior mode or fallback to a tactic-only prior.

### Arity-aware raw action probabilities

Arguments are conditional, not independent. For a sampled tactic and ordered
arguments, add the tactic log-probability and the log-probabilities of its
applicable autoregressive pointer choices. The pointer's condition includes the
state, selected tactic, and previously selected argument information.

The following named quantities refer only to proposal/prior computation:

| Quantity | Meaning and owner |
| --- | --- |
| `raw_action_log_prior` | Log-probability of one distinct tactic-ID/argument-index action under the fixed proposal model |
| `argument_arity` | Number of applicable argument steps for that raw action, resolved from its tactic and supported model contract |
| `edge_log_mass` | Log-sum-exp of priors of distinct observed raw actions mapped to one accepted command edge |
| `normalizer` | Log-sum-exp of the edge masses across the node's current accepted non-dead edges |
| `puct_prior` | Effective normalized probability supplied to the exploration term for one surviving edge |

Mechanism:

```text
raw_action_log_prior = tactic_logp + sum(applicable conditional argument logps)
edge_log_mass = logsumexp(distinct observed raw_action_log_priors for this edge)
normalizer = logsumexp(edge_log_masses of accepted non-dead sibling edges)
puct_prior = exp(edge_log_mass - normalizer)
```

The sum contains exactly `argument_arity` steps. Zero-argument tactics have an
empty argument sum and keep their tactic probability. Padded steps of other
rows and inactive steps beyond a shorter row's arity contribute nothing.

Extend the shared `ActionSample` contract with resolved per-row arities and
per-step argument log-probabilities or an equivalently auditable masked sum.
Compute the mask in one shared place and use it consistently during proposal,
decoding, and stored-action evaluation. Validate legal indices/masks and reject
unsupported required arities explicitly; do not invent missing arguments or
silently treat a truncated action as the complete requested tactic.

An all-invalid argument row remains unrenderable evidence; a zero placeholder
log-probability used for safe gradient handling is not a probability-one valid
argument for an executable edge. Only complete admitted actions get edge mass.

Use unrounded log-softmax outputs, not rounded display probabilities or
exponentiated products. Joint products can underflow even when their individual
terms are representable. Keep immutable raw logs and perform normalization
with log-sum-exp. Reject NaN, positive infinity, or inconsistent positive log
probabilities with an explicit contract error. Do not substitute uniform or
tactic-only priors for invalid metadata. A nonempty selectable set must have
finite normalizable mass; an empty set is handled by the proposal lifecycle.

`arg_loss_weight`, entropy weights, multiplicity, visit counts, rewards, critic
values, and PLN scores do not enter this probability calculation. Greedy mode
uses model probabilities of its selected action, not probability one merely
because selection was deterministic.

### Separate candidate metadata and migrate adjacent providers

Introduce a required scored-proposal record, such as
`ProposedAction(candidate, raw_action_id, raw_action_log_prior, prior_version)`.
`candidate` is the existing execution/ranking `TacticCandidate`;
`raw_action_id` identifies the provider's concrete complete choice. A search
proposal cannot omit its prior metadata. Keep `TacticCandidate.probability`
as tactic-family ranking confidence; do not overwrite it with a normalized
PUCT prior or infer an action log-prior from it downstream.

Migrate the actor provider, generic `GNNModelEngine` provider, scripted providers,
node-aware batch seams, shared execution/linking, graph construction, diagnostics,
and tests to this scored-proposal boundary. The executor can still consume the
unwrapped execution candidate. All graph insertion paths provide explicit raw
action mass; there is no graph default of `log(candidate.probability)`.

The generic provider is an adjacent shared caller requiring an explicit audit.
`inference.py` currently ranks retrieved/local arguments using `PremiseScorer`
scores rather than the RL pointer's categorical samples. It must define and
return probabilities for its actual supported choices. Use a documented
conditional softmax over the same eligible pool, excluding already selected
arguments where its selection is without replacement. Its ranked choices can
remain deterministic argmax under that distribution. Deterministic fresh-name
construction has a probability-one argument factor only where its provider
contract explicitly has one such choice. Missing argument probability is a
provider error, not an implicit factor of one. Do not replace the existing
provider with the RL component to avoid repairing its probability contract.

Provider law and support identity must be recorded in the prior version and
tests. They are not claims that the generic scorer and RL pointer implement
the same distribution. Preserve existing candidate ordering and best-first/PLN
ranking unless a separately reviewed bug makes that impossible.

### Rendered aliases and observed support

The execution cache coalesces identical rendered commands, but policy choices
are disjoint events identified by their raw tactic and argument IDs. Store an
immutable raw log-prior for each distinct observed raw ID under the node's
fixed model/graph/provider contract.

- Sampling the same raw ID again updates observation multiplicity, not mass.
- Discovering another raw ID that renders to the same command adds that new
  event's probability once to the command's `edge_log_mass`.
- Both raw observations retain their own training log-probability and pointer
  tuple; merging execution must not merge their gradients into one tuple.
- The command still has one accepted edge or checked rejection in the cache.
- Repeated evidence for the same raw ID must agree within documented numerical
  tolerance. A changed model/graph/provider identity is an error, not new mass.

This is normalization over the unique raw-action support actually observed and
admitted by this search. It is not the exact policy probability of a rendered
command after summing every possible unobserved alias. Reports must say so.
Multiplicity is neither an additional probability estimate nor an importance
correction for this prior. Investigating alternative sampled-search prior
corrections is a separate algorithmic extension.

### Dynamic normalization and PUCT integration

At selection, the graph supplies one coherent prior snapshot for the node's
accepted non-dead edges. Recompute it after accepted-edge admission, a new
distinct alias, or an edge becoming dead. Derive it on demand first; an optional
cache must have explicit invalidation for all three events.

Example: raw masses 0.32 and 0.08 normalize to 0.8 and 0.2. Adding an edge with
mass 0.10 changes them to 0.64, 0.16, and 0.20. The original raw logs and all
completed visits/value sums remain unchanged.

Migrate `puct_score` to consume the effective prior explicitly alongside edge
visit statistics. Remove `EdgeVisitStats.prior_prob` as a second mutable source
of truth; keep raw/alias prior ownership on the candidate/edge contract and
effective normalization at the parent-node boundary. Update summaries to
distinguish tactic confidence, raw action logs, alias mass, and effective prior.

Preserve `Q = W / N_v`, the existing first-play value, and the current virtual
loss behavior. Virtual loss does not alter the prior normalizer. A high prior
cannot admit a rejected action or re-enable a dead edge. Do not multiply the
prior into an AND backup or the empirical value average.

The effective prior can change as widening changes support. Do not reset
`N`, `N_v`, `W`, completed node visits, or raw action logs to compensate.
Treat this as a new search contract and validate its interaction with `puct_c`;
normalization changes its scale relative to the historical implementation.

## 6. Widening state, limits, and scheduling

### Definitions and widening rule

The capacity grows with visits to a node, independently of whether previous
attempts produced an edge. This prevents a rejection-only root from having no
way to become eligible for another proposal.

| Name | Meaning and owner |
| --- | --- |
| `v` | Completed simulation selections that include a particular node; new node-selection statistics |
| `a0` | Initial distinct-command capacity; `widening_initial_actions` |
| `k` | Capacity growth coefficient; `widening_coefficient` |
| `alpha` | Sublinear growth exponent; `widening_exponent` |
| `amax` | Hard cap on distinct commands attempted at that node; `widening_max_actions` |
| `dmax` | Hard cap on raw policy draws at that node; `widening_max_draws` |
| `capacity(v)` | Current allowed distinct-command attempts, computed by the shared widening policy |

Chosen rule:

```text
capacity(v) = min(amax, max(a0, ceil(k * (1 + v) ** alpha)))
```

Validate positive integer capacities and draw limits, `amax >= a0`, finite
positive `k`, and `0 < alpha < 1`. Validate positive finite simulation budgets,
batch sizes, and finite nonnegative PUCT exploration constants in the shared
resolver as well, rather than allowing zero-budget loops or nonfinite scores.

Illustrative resolved defaults are `a0=4`, `k=2.0`, `alpha=0.5`, `amax=32`,
and `dmax=128`. These are bounded starting values, not validated tuning results.
`top_k_tactics` remains the maximum number of raw draws in one proposal round,
not the number of guaranteed unique or accepted edges.

Before a round, choose its fixed draw count as the minimum of:

- `top_k_tactics`;
- remaining distinct-command capacity at the current visit count;
- remaining raw-draw budget.

Do not repeatedly sample until a desired unique count is reached. Duplicates
consume raw draws but not distinct-command capacity. A subsequent scheduled
round can try again, within the bounds.

### Per-node proposal state

Add one authoritative proposal-source state, containing draw/round counts,
distinct-command identities, closure reason, and reservation state. It is
separate from the node's solved/dead status and from its live materialization.

Completion of a round does not set permanent exhaustion. Permanent source
closure occurs when a hard attempt/draw bound is reached, the provider proves
its finite source exhausted, or a forced path limit prevents more work.
Deterministic greedy has one complete candidate at that state and closes its
source after that proposal; widening must not repeatedly ask for its same
argmax. The generic GNN-engine provider can report a finite ranked source as
closed. Do not pretend that it has a stochastic actor or a learned critic.

Retain `exhausted` only as a derived property of authoritative proposal/path
closure, or migrate all its readers to the new state. Do not store two mutable
closure flags. Distinguish candidate-budget closure from depth, cycle,
elaboration, and external-abort reasons.

### Scheduling in the shared MCTS loop

Extend `_Simulation` to record selected nodes and typed proposal frontiers,
not just outgoing edges and initially open leaves. A frontier can request
initial proposal, widening, or a deferred visit with no new execution.

For each selected unresolved node:

1. Respect solved/dead/path-closure status first.
2. If capacity and draw budget permit proposal, stop descent there and request
   a bounded proposal round.
3. Otherwise, select a surviving edge by the existing PUCT formula and descend
   into all its AND children.
4. If all known edges are dead but the source remains open, retain a deferred
   frontier. Complete a bounded simulation visit so capacity can grow later;
   do not report `FRONTIER_EMPTY` or close the node solely for this reason.

Node selection counts increment once per selected node per completed
simulation. Unknown numeric outcomes still count for this control purpose.
No edge visit is invented where no edge was selected.

Coalesce initial/widening requests by node ID across a batch. Reserve at most
one proposal round per node per batch. Compute that batch's capacity from
completed visits; in-flight reservations must not themselves inflate it.
Several simulations may reach the same frontier, but there is only one policy
batch contribution and one execution round for it. Release every reservation
and edge virtual loss on completion, cancellation, or exception.

An expanded node with remaining proposal opportunity is not permanently dead
when its known edges fail. A node becomes dead only when no viable edge remains
and its source/path is closed. Propagate those statuses through the existing
AND–OR rules. Do not revive terminal nodes by changing flags after propagation.

Check deadlines and remaining per-node budgets before additional replay and
execution work. Preserve all children of an admitted Lean transition; never
truncate an AND obligation to satisfy a node budget. State clearly that the
existing graph-size check is a search stopping condition, not permission to
partially insert a tactic's successors.

## 7. Execution identity and repeated-sample learning

### Two identities with different responsibilities

Use two distinct keys, both scoped to one search node and its stable model
graph fingerprint:

1. Execution identity: the actual normalized rendered Lean command. It prevents
   repeated Lean execution for the same canonical state. Do not use live state
   handles, which change at each materialization, as the permanent key.
2. Raw policy identity: tactic ID and the exact sampled argument-index tuple.
   It determines which neural action is evaluated during training.

The existing tactic renderer can merge superficially different argument lists
into one command. Such draws share an execution outcome but not necessarily
the same policy log-probability. Do not merge them into one arbitrary pointer
tuple or label one as rejected because it created no new edge.

Store one execution outcome per rendered command: checked rejection or accepted
edge ID. Protocol/transport/replay failures are not cached tactic rejections.
Keep the cache immutable after a checked outcome within a fixed model state.
Assert that repeated materialization preserves the model graph fingerprint.
Prior bookkeeping retains the distinct observed raw IDs sharing that outcome,
so a new alias can update mass without another execution, while a repeated raw
draw only changes its learning multiplicity.

Rebuild `local_declarations(canonical_goal)` for each materialized row, as in
PR #61. Resolve model context labels through that row's execution identities.
If replay produces a different rendered command for the same model action,
do not blindly reuse a string cache: validate the execution declaration
mapping or stop with an identity error. Do not silently apply a cached command
to another declaration.

### One observation schema

Replace the training-facing single-action-per-edge assumption with one
`PolicyObservation` schema. It contains node ownership, canonical goal,
graph fingerprint, raw tactic/argument IDs, multiplicity, proposal-round
provenance, immutable raw action log-prior and provider/version identity,
outcome category, and optional accepted edge ID. No autograd graph or Lean
handle survives in it. Recompute actor probabilities for the loss; do not
substitute stored or normalized search priors into the training objective.
Use explicit outcome variants: executable actions require valid full-prior
metadata, while an invalid-pointer/unrenderable variant can record that no
complete probability exists. That variant can never supply graph prior mass;
it is not an executable candidate with an optional-prior fallback.

Migrate `RLSearchResult`, proposal settlement, `compute_onpolicy_loss`,
`train_step_onpolicy`, minimal-proof mining, driver replay mining, diagnostics,
smoke scripts, and all fakes/tests in the same stage. Remove the old training
join/failure paths rather than interpreting both result formats.

Rules:

- Repeated accepted draws point to the existing edge; never create duplicate
  children or reset its visit statistics.
- Repeated checked-rejected draws retain failure observations without another
  Lean call.
- Unrenderable raw draws retain their explicitly classified actor-only outcome.
- Distinct raw pointer tuples remain separate even if they render identically.
- Identical raw actions at the same node can be grouped with multiplicity;
  total accounted multiplicity equals the number of settled policy draws.
- Incomplete execution, replay failure, and cancellation discard unobserved
  action outcomes; do not flush them as tactic failures.
- Repeated samples do not increment MCTS visits. Only selected simulations do.
- Harvest an accepted edge's return once, then join it to the matching raw
  observations. Do not duplicate the structural transition and multiply its
  multiplicity again.
- Deduplicate critic rows per graph node, not across unrelated theorem graphs.
- Minimal-proof imitation emits one observed raw action representative for each
  selected solved edge, not every duplicate draw. Choose the first successfully
  executed representative deterministically.

The model remains fixed throughout collection. Each scheduled proposal round
uses a fixed draw count chosen before its draws. Adaptive state visitation and
search-derived targets still require care when interpreting the actor
estimator; do not claim that multiplicity alone proves unbiasedness.

## 8. Backup and target validity with open proposal sources

Preserve the MCTS AND combination: valid zero dominates, otherwise unknown
dominates, otherwise multiply child values. Preserve `Q = W / N_v`, with 0.5
first-play urgency, and the current exploration-only virtual-loss effect.

The OR-target logic needs a shared correction for the widened scope:

- A checked solved alternative still establishes node success.
- All known edges failing is not a complete failure while new proposals remain
  possible. Structural harvest returns UNKNOWN for that incomplete OR set.
- An unresolved MCTS frontier may use a clearly tagged critic bootstrap during
  simulation; that estimate is not a Lean-confirmed failure or success.
- A closed, bounded candidate source with every alternative dead is a genuine
  failure of this search configuration, not proof of mathematical unprovability.
- A depth/cycle closure can invalidate its incoming action path but remains
  excluded as a standalone fresh-root critic target.
- Elaboration and external failures remain unknown. Root `ReplayInputError`
  still yields no graph or actor/critic observations.

Audit both `joint_inference._backup_simulation` and
`search_harvest.compute_backups`. In particular, the existing structural OR
code can return the maximum known zero even for an unexhausted node. A widened
node with incomplete alternatives must not acquire that zero through the
generic maximum-valid branch.

Keep the existing explicit root-only `ROOT_EPISODE` target for a clean global
budget failure, and label it with the effective widening/search contract.
Budget exhaustion and unknown internal alternatives are different evidence.
Do not reset `N`, `N_v`, `W`, or immutable raw action log-priors when adding an
alternative. Effective PUCT priors are deliberately renormalized over the new
selectable support, independently of value backup.

## 9. Evaluation configuration, reports, and checkpoints

### Public configuration

Add the following validated configuration fields and matching overrides where
specified:

| Field | Proposed default | Meaning |
| --- | --- | --- |
| `eval_proposal_mode` | `"sample"` | Complete-action proposal strategy for every evaluation entry point |
| `eval_seed` | `42` | Base for isolated evaluation search seeds |
| `eval_repeats` | `1` | Number of independently seeded passes over the admitted evaluation set |
| `widening_initial_actions` | null in the generic config | Resolve to 4 for PUCT |
| `widening_coefficient` | null | Resolve to 2.0 for PUCT |
| `widening_exponent` | null | Resolve to 0.5 for PUCT |
| `widening_max_actions` | null | Resolve to 32 for PUCT |
| `widening_max_draws` | null | Resolve to 128 for PUCT |

The existing shared search resolver owns effective widening settings. Reject
explicit widening settings under best-first mode, just as explicit simulation
settings are currently rejected there. Under PUCT, omitted values resolve to
the new lifecycle; there is no `enable_widening` compatibility switch.

Expose `--eval-proposal-mode {sample,greedy}`, `--eval-seed`, and
`--eval-repeats` through `driver_main`. These settings apply to periodic
evaluation as well as `--eval-only`. Training proposal mode remains sampled;
an evaluation override must not silently change collection policy.

Keep `num_simulations`, `sim_batch_size`, `top_k_tactics`, node/depth limits,
and timeouts distinct. `htps_steps_per_round` continues to control additional
supervised training only. No inference proposal flag enables optimizer steps.

### Evaluation behavior and reporting

`evaluate_proof_rate` accepts the resolved evaluation specification rather than
hardcoding a Boolean. Both the live-pool path and the single-reasoner path use
one shared request/seed derivation helper. Do not implement a separate
evaluation-only MCTS loop.

Run each repeat over `pool.eval_items` after the existing asynchronous admission
stage. Do not rerun admission for every repeat or reinterpret quarantined seeds
as proof failures. Reject an empty evaluation subset explicitly.

Report each repeat and an aggregate over searched attempts, retaining
`seeds_inadmissible` separately. Include:

- mode, base seed, derivation version, repeat seeds/identities;
- checkpoint/model generation and source revision;
- effective PUCT, widening, proposal, node/depth, and time budgets;
- complete-action prior version, provider laws, observed-alias aggregation,
  normalization domain, and raw/effective prior summaries;
- dataset name/revision/split, admitted subset identity, admission-report identity;
- searched, solved, unsuccessful, inadmissible, hard-timeout, and error counts;
- completed simulations and proposal rounds;
- raw draws, distinct command attempts, accepted edges, rejection/cache hits;
- widening rounds, source-closure reasons, and deferred-frontier counts;
- worker topology and the limits of the stated reproducibility contract.

Standalone evaluation currently prints an aggregate. Add an evaluation report
destination (`--eval-report`) that creates a new report rather than overwriting
historical evidence. Periodic evaluation writes the same result schema into the
run's evaluation/metrics outputs. Preserve existing proof-rate keys for current
consumers while adding explicitly named counters and metadata; do not relabel a
greedy score as sampled proof rate.

Report invariants include `settled raw draws = sum(observation multiplicities)`
and `new accepted edges <= distinct command attempts <= raw draws`.
Budget-deferred or interrupted observations are counted separately.

### Resume and model selection

Keep model tensor layout, prepared vocabulary contract, and canonical replay
queue labels unchanged. Existing compatible model weights remain warm starts.

Bump `rl_target_schema_version` from 3 to 4 when the new semantics first land.
Store a versioned effective search/proposal contract in `search_settings`,
including widening limits, proposal-ledger semantics, training seed derivation,
and the new prior semantics: `prior_semantics="joint-action-v1"`,
`prior_normalization="viable-observed-support-v1"`, and
`prior_alias_policy="unique-raw-action-sum-v1"`. No tactic-only compatibility
flag is added. Store the evaluation contract alongside
`best_proof_rate`: mode, seed, repeats, and metric version.

Reject an older exact resume or a changed search/evaluation contract with a
specific message directing the user to start a new run from model weights.
Do not import an old greedy `best_proof_rate` as the incumbent for sampled
evaluation. Do not add an old-schema interpretation branch. Model-only
evaluation remains available for admitted older weight manifests.

If restored queues remain unchanged in shape, document that explicitly. Search
observation ledgers are per-search state and are not silently added to durable
replay checkpoints. Save enough round/seed-derivation state to resume the new
request schedule without claiming deterministic OS or Lean timing.

## 10. Sequential stages and completion gates

Each stage migrates its affected callers and includes tests. Do not publish an
enabled widening configuration until the lifecycle, accounting, and loop stages
are complete. The complete-action-prior stage precedes widening and migrates
all its providers/consumers before activation. No stage edits the companion
repository or rebuilds its binary.

### Stage 0 — Record baseline and lock the replay contract

Files: current tests, `mcts_dataset_replay_port_handoff.md`, this plan, and a new
test fixture directory `tests/fixtures/pantograph/mcts_widening_evaluation/`.

1. Record HEAD, dependency identity, graph/checkpoint schemas, and actual
   available artifacts. Recheck for subsequent commits before implementation.
2. Characterize the tactic-only greedy behavior with an argument-bearing action.
   Do not rely on a zero-argument test to establish greedy determinism.
3. Add scripted fixtures with an initially rejected batch and a later successful
   action, an AND split, repeated commands, and uneven materialization batches.
4. Preserve PR #61 checks for typed roots, universe levels, duplicate display
   names, inaccessible locals, context-index gaps, and re-materialized identity.
5. Capture current worker and training-order behavior with HTPS steps off/on.
6. Characterize tactic-only priors, mixed-arity argument sums, rounded generic
   prediction probabilities, and every graph insertion/PUCT scoring caller.

Gate: fixtures expose the missing features and existing replay regressions pass.
Expected future-feature failures remain clearly marked tests, not claims of
implemented widening. Historical fixture reports/manifests remain unchanged.

### Stage 1 — Shared proposal strategy and explicit generators

Files: `actor_critic.py`, `argument_selector.py`, `rl_reasoner.py`, the shared
proposal-spec module, collection/driver callbacks, smoke scripts, diagnostics,
and their tests.

1. Introduce the single proposal-spec contract and migrate every `greedy`
   search caller, including fakes and helper collection functions.
2. Centralize categorical choice and pass an explicit device generator through
   tactic and autoregressive argument selection.
3. Correct true greedy arguments while preserving sampled argument embeddings
   and gradients in stored-action evaluation.
4. Expose actual row arities and per-step argument log-probabilities; mask sums
   for shorter rows and make unsupported/incomplete actions explicit.
5. Implement stable per-input seed derivation and deterministic ordering of
   node rows within a search proposal batch.
6. Keep global RNG untouched by inference proposals and restore original module
   training flags after evaluation.

Gate: both tactic and arguments are deterministic in greedy mode; sample mode
can produce different families/arguments; fixed search seeds reproduce draws
on the same device; model weights and buffers do not change during evaluation.
All-invalid masks, zero/mixed arities, and model-state identity tests pass.
An applicable argument sum is invariant to adding an unrelated higher-arity
row, apart from defined numerical tolerance; padding contributes no log-prior.

### Stage 2 — Wire sampled evaluation end to end

Files: `rl_training_driver.py`, `rl_live_collection.py`, `scripts/rl_train.py`,
evaluation/report types, config, checkpoint handling, and driver/pool tests.

1. Add the evaluation fields/CLI overrides and remove hardcoded greedy mode.
2. Apply the same settings to periodic evaluation and `--eval-only`.
3. Pass requests through pool replacement and single-reasoner paths without
   losing input identity or seed. Do not derive seeds from worker placement.
4. Implement repeats, searched-only aggregate rates, empty-subset rejection,
   and versioned evaluation reports.
5. Add evaluation contract checks to checkpoint selection/resume; introduce
   schema 4 at the first saved new-semantics checkpoint.
6. Correct documentation that describes sampled arguments as greedy, and label
   true-greedy versus sampled evaluation results explicitly.

Gate: CLI and periodic evaluation honor sample/greedy selection and seed
overrides; evaluation performs no optimizer steps; new reports identify actual
settings; inadmissible roots do not enter proof-rate denominators. Adding an
evaluation pass does not advance the global training RNG sequence.

This stage is a useful independently reviewable result. Widening is not yet
claimed or exposed as implemented at this gate.

### Stage 3 — Repeated-proposal execution and observation accounting

Files: `rl_reasoner.py`, shared action/proposal records, the tactic renderer,
`pln_rl_training.py`, `search_harvest.py`, driver mining, smoke scripts,
`mcts_diagnostics.py`, and all affected tests.

1. Introduce per-node execution caches and raw-action observations.
2. Replace the one-raw-action-per-edge and separate failure training interface;
   migrate all consumers rather than keeping both schemas.
3. Settle cache hits against known outcomes and attach accepted observations to
   existing edge IDs without adding children or resetting visits.
4. Keep round-local pending work separate from the search-lifetime ledger.
5. Make multiple manually scheduled rounds on a node pass before wiring visit
   scheduling into production MCTS.
6. Preserve graph fingerprints, live declaration resolution, rejected-action
   actor-only treatment, unique critic rows, and minimal-proof imitation.
7. Retain raw-action identities and unrounded log-probabilities separately from
   execution identity and sampling multiplicity, ready for edge-mass aggregation.

Gate: duplicates across rounds invoke Lean once, settle every raw sample, and
preserve correct pointer gradients. Different raw tuples rendering identically
remain distinguishable in training. Interrupted/unobserved actions never become
negative examples. All consumers use the new schema.

### Stage 4 — Complete-action prior contract and graph normalization

Files: `data_models/proof_components.py` or a dependency-light scored-proposal
module, `actor_critic.py`, `argument_selector.py`, `rl_reasoner.py`,
`gnn_inference/atp_lean_gnn/inference.py`, `gnn_inference/inference_engine.py`,
`hypergraph.py`, `selection_policy.py`, shared linking, summaries, diagnostics,
checkpoint metadata, and all producer/consumer tests.

1. Introduce the required scored-proposal boundary with explicit raw identity,
   arity-correct complete-action log-probability, and provider/version metadata.
   Migrate every search provider, scripted fake, graph insertion, and consumer.
2. Preserve tactic-family `TacticCandidate.probability` for existing ranking.
   Audit the generic premise-scorer provider and expose its documented
   conditional selection probabilities without rounding or implicit unit factors.
3. Store one immutable prior per distinct raw action. Sum distinct observed
   alias mass once per accepted command edge; retain independent learning
   observations and the single cached execution outcome.
4. Normalize in log space over accepted non-dead sibling edges. Invalidate or
   derive the snapshot after an added edge, newly observed alias, or dead edge.
   Rejected/unrenderable actions never receive selectable mass.
5. Pass effective priors into the shared PUCT formula and remove the old
   `EdgeVisitStats.prior_prob` authority. Migrate all direct constructors,
   diagnostics, summaries, and score tests rather than defaulting missing data.
6. Record the new prior/alias/normalization versions in saved effective search
   settings and reports. Keep schema 4 introduced with the new semantics;
   intermediate checkpoint settings must still reject an incompatible resume.
7. Add numerical, alias, ranking-isolation, and multi-provider regressions before
   integrating dynamic candidate growth into MCTS.

Gate: two actions of the same tactic family but unequal argument probabilities
receive unequal PUCT priors. Applicable pointer factors, masks, empty argument
products, and low-probability sequences are correct. Effective priors sum to
one for a nonempty selectable set within tolerance. Repeated identical draws
do not change priors; a distinct alias can change its edge mass once. Removing
a dead edge and adding an alternative renormalize siblings without changing
visits, value sums, actor log-probability recomputation, or best-first/PLN ranks.
There is no executable tactic-only or missing-prior fallback.

This stage can be reviewed with fixed candidate sets and direct graph
admission tests. Automatic progressive widening is not required to test the
normalizer or alias updates and is not claimed to work until its loop stage.

### Stage 5 — Proposal-source lifecycle and validity-aware graph closure

Files: `hypergraph.py`, shared proposal-state types, expansion completion hooks,
`search_harvest.py`, MCTS backup, and graph/harvest/reasoner tests.

1. Add the authoritative proposal-source state and independent node selections.
2. Replace batch-end permanent exhaustion with explicit batch settlement and
   source/path closure operations; migrate best-first and generic providers too.
3. Keep unresolved sources live when known alternatives fail. Never reopen a
   `DEAD` node or revive a dead ancestor after the fact.
4. Correct OR harvest to return unknown for incomplete all-failed alternatives.
5. Preserve forced-path closure, root episode targets, failure provenance,
   terminal success, and cycle safety.

Gate: a node remains unresolved after one failed round when its source is open;
closing that source correctly propagates failure. A solved edge still solves
its parent. Depth/cycle/replay/infrastructure failures keep their distinct
target-validity rules. Existing finite-source best-first behavior is preserved.

### Stage 6 — Integrate bounded widening into PUCT

Files: `selection_policy.py`, `joint_inference.py`, `hypergraph.py`,
`rl_reasoner.py`, config/resource wiring, and simulation tests.

1. Implement and validate the capacity rule and hard draw/attempt bounds.
2. Extend simulation selection to visit counts and typed initial/widen/deferred
   frontiers. Reserve and coalesce per-node work across a simulation batch.
3. Reuse checked materialization and row-aware policy batches for both new and
   previously expanded nodes. Keep one node's original state for its round's
   alternative executions and release it afterward.
4. Preserve all successor identities and AND obligations before linking.
   Admit each new edge through complete-action mass accounting and refresh the
   parent's effective-prior snapshot. Repeated raw draws leave mass unchanged.
5. Complete node/edge backup accounting and release every reservation/virtual
   loss on all exits, including failure during a later backup in the batch.
6. Handle rejection-only and duplicate-only sources without infinite retries or
   premature empty-frontier termination. A capacity-deferred simulation still
   consumes its simulation budget and advances defined node visits.
7. Activate widening for PUCT in training and inference, with no fixed-PUCT
   compatibility switch. Update effective saved search settings accordingly.

Gate: a later proposal round discovers a scripted solving action absent from
the first batch; old edges retain exact visit/value statistics and immutable
raw priors while effective priors renormalize; rejected commands are not
re-executed; node/action/draw/deadline bounds are enforced. Counts and cleanup
are correct for batch size one and larger uneven AND batches. A late-discovered
alias changes mass without a second Lean call or a second structural edge.

### Stage 7 — Driver, mining, metrics, and resume integration

Files: driver, live collection, harvest/loss, reports, config, checkpoint tests,
and end-to-end training fixtures.

1. Confirm actual driver rounds collect with widening, perform at most one
   on-policy step, mine valid replay, run configured HTPS steps, and synchronize
   only after all updates.
2. Add full counters and versioned contracts to reports/checkpoints.
   Include prior semantics, provider law, normalization support, and alias mass;
   reject attempts to compare or resume historical tactic-only search settings.
3. Preserve no-searched-root curriculum behavior and permit valid existing replay
   updates when a round's new roots are inadmissible.
4. Test exact new-schema resume and explicit rejection of old/different
   search/evaluation semantics. Preserve model-only warm starts.
5. Test standalone inference and periodic evaluation with identical search
   settings and no changes to model state or replica generation.
6. Update the shipped RL JSON example to a complete PUCT/sample configuration
   with finite widening bounds; do not overwrite a user's local edited config.

Gate: reproducible fixture reports show later OR alternatives, correct training
sample counts, searched-only evaluation rates, and correct optimizer/generation
counts with `htps_steps_per_round=0` and positive. Resume does not reuse an
incomparable historical best score.

### Stage 8 — Real REPL and checkpoint-backed acceptance

Files: live tests, `mcts_diagnostics.py`, `scripts/mcts_client_smoke.py`, new
evidence fixtures, and their provenance checker.

1. Run the admitted companion REPL with the existing pinned client and Mathlib.
   Do not substitute an unverified downloaded binary.
2. Use a scripted proposal provider through real scoped execution to force an
   initial rejected alternative and a later successful widening action.
3. Include real AND subgoals, repeated accepted/rejected commands, awkward
   local names, model/execution index differences, and multiple materializations.
   Assert complete-action prior values, exclusion of checked rejections, and
   live-set renormalization independently of AND backup and policy gradients.
4. Check state counts before/after repeated searches, cancellation during
   widening, transport retirement, and two workers sharing one model device.
5. Run short sampled and true-greedy evaluations with real prepared data and an
   admitted checkpoint, plus one bounded training round for each HTPS setting.
6. Preserve old source manifests/reports. Add a new provenance/source layer and
   migrate `test_owned_client_migration.py` to validate current hashes against
   that new layer without rewriting what earlier commits actually tested.

Gate: real transition execution, scoped cleanup, later-action discovery, and
short checkpoint-backed runs pass. Missing checkpoint/prepared/GPU resources
are explicit blocked acceptance items, never replaced by synthetic claims.
No guarantee of improved proof rate is inferred from functional fixtures.

### Stage 9 — Documentation and handoff

Files: `rl_process_walkthrough.md`, `owned_pantograph_client.md`, a new widening
handoff, config examples, CLI help, and this plan's stage status.

Document:

- Updated typed-seed/admission flow, not the removed string API.
- Progressive widening's attempt/draw/visit semantics and limits.
- Sample versus truly greedy evaluation, isolated seeds, repeats, and reports.
- Conditional complete-action probabilities, arity masking, observed raw-action
  aliases, live-set normalization, and the distinction from tactic-family ranks.
- CLI/file/dataset examples, proof-rate denominators, and checkpoint rules.
- Graph success versus complete-theorem certification.
- Actual validation commands, source/dependency/binary identities, and remaining
  limits. Preserve historical evidence as historical.

Gate: another developer can run sampled dataset evaluation, true-greedy
evaluation, and a bounded training round using the handoff without an ad hoc
notebook or private helper import.

## 11. Required regression matrix

| Area | Required cases |
| --- | --- |
| Proposal strategy | Tactic and argument sampling; true greedy; zero/mixed arity; all-masked pointer; finite probabilities |
| Complete-action probabilities | Same tactic/different arguments; conditional multi-argument product; zero arity; mixed batch arity; masked/unsupported steps; no display rounding; no `arg_loss_weight` in prior |
| Prior numerics | Stable log-sum-exp for tiny products; finite unit-sum effective priors; explicit invalid-metadata error; no uniform or tactic-only fallback |
| Aliases and normalization | Identical raw draw counted once in mass; distinct raw aliases summed once; observed-support labeling; accepted-edge admission; dead-edge removal; unchanged virtual-loss behavior |
| Shared-provider migration | Actor pointer, generic local/retrieved scorer, deterministic fresh-name rule, scripted providers, and all graph insertions provide explicit complete-action metadata |
| Ranking and training separation | Existing best-first/PLN confidence unchanged; no normalized prior in actor loss; original raw tuples/gradients preserved; no visits/value reset on renormalization |
| RNG ownership | Same search seed; different repeat seeds; no global-RNG consumption; worker replacement/order independence at the request level |
| Widening policy | Capacity growth/monotonicity; integer/finite validation; unique-attempt cap; draw cap; first visit; rejection-only and duplicate-only states |
| Simulation batching | One reservation per node; shared frontiers; visits without selected outgoing edges; no duplicate execution; cleanup on each interruption point |
| Graph semantics | Open source with all known edges dead; final source closure; solved alternative; complete AND children; depth/cycle closure; no terminal revival |
| Typed replay | Raw structured dataset seed; universe levels; model-index gaps; duplicate names; inherited successor identity; later materialization failure |
| Learning | Cross-round multiplicity; multiple raw actions per command; accepted/rejected cache hits; unknown-edge omission; no unobserved negative examples |
| Harvest | Incomplete OR set unknown; genuine closure zero; supported soft mean; root-budget episode zero; unique critic targets; minimal solved-proof representative |
| Evaluation | Standalone and periodic config/CLI wiring; empty subset; repeated rates; inadmissible denominator; no optimizer/model mutation |
| Lifecycle | Normal scope cleanup; cooperative deadline; hard timeout; transport replacement; fatal protocol/state errors; no stale materializations |
| Resume/evidence | New-schema round trip; old/incomparable resume rejection; model-only warm start; new source-hash layer; historical manifests unchanged |

A regression that asks only whether a new flag parses is insufficient. Tests
must observe proposal families/arguments, later edge creation, actual execution
counts, and the resulting training/accounting outcomes.

## 12. Verification commands and execution boundaries

Run all commands from the repository root in the pinned environment. Start with
the baseline-focused command, then expand to all GNN tests. Newly introduced
test files must be included explicitly in focused CI as they are added.
Add a dedicated `test_action_priors.py` for the raw/joint/alias/normalization
contract, and include `test_inference.py` and `test_tactic_rendering.py` in the
provider-migration gates. The baseline count above predates those new tests.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  maths_ai/gnn_inference/tests/test_actor_critic.py \
  maths_ai/gnn_inference/tests/test_argument_selector.py \
  maths_ai/gnn_inference/tests/test_selection_policy.py \
  maths_ai/gnn_inference/tests/test_visit_stats.py \
  maths_ai/gnn_inference/tests/test_rl_reasoner.py \
  maths_ai/gnn_inference/tests/test_mcts_owned_client.py \
  maths_ai/gnn_inference/tests/test_rl_training_driver.py \
  maths_ai/gnn_inference/tests/test_rl_live_collection.py \
  maths_ai/gnn_inference/tests/test_search_harvest.py \
  maths_ai/gnn_inference/tests/test_pln_rl_training.py \
  maths_ai/gnn_inference/tests/test_dataset_replay.py \
  maths_ai/gnn_inference/tests/test_model_identity.py \
  maths_ai/gnn_inference/tests/test_lean_replay.py \
  -m 'not pantograph_live'

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  maths_ai/gnn_inference/tests -m 'not pantograph_live'

git diff --check
```

For a required live run, set `PANTOGRAPH_TEST_SOURCE_ROOT` and
`PANTOGRAPH_TEST_REPL` to verified local artifacts, then use
`--run-pantograph-live -m pantograph_live`. Set
`PANTOGRAPH_TEST_PREPARED_ROOT` and `PANTOGRAPH_TEST_CHECKPOINT` for the existing
prepared/checkpoint admission gate. An absent resource is not a passing gate.

Existing full-suite failures must be reproduced on the recorded baseline
before being classified as pre-existing. The port handoff's old counts do not
replace running the current suite.

### Intended post-implementation launch example

This fragment is the target configuration, not something supported by the
current unmodified code:

```json
{
  "selection_policy": "puct",
  "num_simulations": 64,
  "sim_batch_size": 4,
  "puct_c": 1.0,
  "top_k_tactics": 4,
  "widening_initial_actions": 4,
  "widening_coefficient": 2.0,
  "widening_exponent": 0.5,
  "widening_max_actions": 32,
  "widening_max_draws": 128,
  "eval_proposal_mode": "sample",
  "eval_seed": 42,
  "eval_repeats": 3,
  "seed_cache_path": null
}
```

It augments a complete RL config with matching checkpoint/prepared paths,
verified REPL/project paths, dataset revision or theorem file, admission budgets,
evaluation subset size, and resource limits. Keep the known-incomplete admission
validation cache disabled unless separately repaired.

```bash
python -m maths_ai.gnn_inference.scripts.rl_train \
  --config /absolute/path/to/mcts_eval.json \
  --eval-only \
  --checkpoint /absolute/path/to/best.pt \
  --eval-proposal-mode sample \
  --eval-seed 42 \
  --eval-repeats 3 \
  --eval-report /absolute/path/to/new_evaluation_report.json
```

Use `--eval-proposal-mode greedy` for the corrected deterministic diagnostic.
For training, omit `--eval-only`; the configured evaluation strategy still
controls periodic evaluation, not the training proposal policy. JSONL theorem
input and structured dataset input both enter through the current asynchronous
admission path. `eval_pool_size=0` remains an empty evaluation subset, not an
instruction to evaluate the whole pool.

## 13. Final completion checklist

- [ ] Baseline is rechecked against HEAD before implementation.
- [ ] PR #61 typed replay, admission, identities, and accounting remain intact.
- [ ] Every proposal caller uses the shared mode/request contract.
- [ ] Sampled evaluation reaches both standalone and periodic execution.
- [ ] Explicit greedy mode covers arguments as well as tactic families.
- [ ] Joint priors use only applicable conditional argument factors, without
      loss weights, display rounding, or invalid-pointer placeholders.
- [ ] Every search provider supplies explicit complete-action log-priors.
- [ ] Selectable priors normalize over accepted non-dead observed support.
- [ ] Raw aliases contribute mass once; repeated sampling only changes learning
      multiplicity; best-first/PLN confidence remains separate.
- [ ] PUCT can add later OR alternatives without search restart or terminal revival.
- [ ] Rejections, duplicates, and deferred capacity cannot create unbounded work.
- [ ] Repeated draws preserve learning credit without repeated Lean execution.
- [ ] Unknown and incomplete alternatives do not become fabricated failure labels.
- [ ] Existing visits/value sums and raw priors survive widening; effective
      priors renormalize on support changes; cleanup releases all holds.
- [ ] No model update occurs during collection or evaluation.
- [ ] Reports and checkpoints identify actual proposal/widening/metric semantics.
- [ ] Existing compatible weights remain usable; incompatible exact resumes fail.
- [ ] Historical source/evidence manifests remain truthful and unchanged.
- [ ] Offline, real-REPL, and checkpoint-backed gates are separately reported.
- [ ] Documentation describes implemented behavior and remaining limitations.
