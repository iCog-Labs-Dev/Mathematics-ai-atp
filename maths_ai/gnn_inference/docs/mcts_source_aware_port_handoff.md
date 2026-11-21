# Source-aware actions on MCTS-AC-branch

This ports the new model architecture (#66: the source-aware action decoder,
candidate pools, query-only lemma retrieval and versioned checkpoints) from
AC-branch to the MCTS search, on top of the retained-snapshot port
([its handoff](mcts_snapshot_retention_port_handoff.md)). The architecture is
described in [SOURCE_AWARE_RL.md](SOURCE_AWARE_RL.md) and its validation in
[SOURCE_AWARE_RL_VALIDATION.md](SOURCE_AWARE_RL_VALIDATION.md). This file covers
what is different on MCTS: batched proposals, priors, widening and HTPS on the
new decoder.

## Identities

| | |
| --- | --- |
| Target base | MCTS-AC-branch `e96d2f7`: the retained-snapshot port (#69) with its replay-recovery fix `3d1899b` |
| Application source | AC-branch at `96f8bca` (#66, five commits from `680e58d`) |
| Client and companion REPL | unchanged from the retained-snapshot port |
| Source manifest | `tests/fixtures/pantograph/source_aware_actions/source_manifest.json`, the next lineage link |

The model files (`actor_critic`, `policy_model`, `source_aware_decoder`,
`policy_actions`, `retrieval_artifacts`, `argument_selector`, checkpointing and
the conversion scripts) are taken from the application source. The changes to
them are listed below.

## The action law on MCTS

An action is a tactic and a tuple of typed argument references, each a local
node of the goal's graph or a library lemma of its candidate pool. The decoder
draws the argument count, then each argument's source and identity.

| | Pointer decoder (before) | Source-aware decoder (now) |
| --- | --- | --- |
| Raw action id | `(tactic id, argument node indices)` | `(tactic id, ((source, identity), …))` |
| Search prior (`ProposedAction.log_prior`) | tactic + owned pointer steps | tactic + count + sources + identities: the complete action's exact log-probability |
| Argument count | the tactic's arity table, truncated at `max_args` | drawn by the decoder; an empty pool forces zero |
| Support | the premise mask | the node's candidate pool |
| Recorded action (`EdgeAction`) | integers | references, the pool and the graph fingerprint |

**One pool per node.** `RLHybridReasoner.prepare_node` retrieves a node's pool
before its first round and keeps it for the node's later widening rounds, so a
raw action means the same thing in every round and the command cache stays
valid. The seed's own theorem is excluded from retrieval, as on AC.

**Batched and seeded proposals.** `GraphActionPolicy.propose_actions(data,
counts, candidate_pools, generator)` encodes the batch once and draws
`counts[b]` complete actions for graph `b`. Each draw uses the search's own
generator, so a seeded search repeats exactly and the global RNG is untouched.
The decoder gained a `generator` argument for this; without one it samples as
before.

**Preparation before proposal.** Pools are retrieved asynchronously, so the
search prepares every node of a batch (`prepare_node`) before the synchronous
batched forward. A goal the model cannot observe fails its preparation or the
forward and is closed as unsupported, without failing the rest of the batch.
The generic engine follows the same order (`prepare_goal`, then `propose`), and
its prior is the greedy complete action's exact log-probability. Preparation
returns an explicit `PreparedObservation`; the predictor, engine and direct CLI
pass that record to synchronous scoring. The reasoner owns records only for its
active proposal batch and clears them before Lean execution, including after
errors or cancellation. There is no pipeline-owned prepared-pool cache.

**Updates.** The on-policy loss evaluates each observation's stored action
against its stored pool (`evaluate_actions`), with multiplicity-weighted
advantage normalization and AC's entropy surrogate, pool checks and candidate
budget. HTPS imitation does the same for actions on the certified witness;
`arg_loss_weight` now only weights the arguments' log-probability in
imitation. A checkpoint stores each queued imitation action with its pool
(without the library); resuming reattaches the open library and refuses a pool
drawn from a different one.

**Rendering.** An action whose bracketed rule list contains an argument that is
not a Lean name is rejected whole (`rw [h, ↑13]` is not run as `rw [h]`), as on
AC: dropping an argument would execute a different action than the one whose
prior was recorded.

**Local-only runs.** A model bound to no retrieval library
(`retrieval_binding == ""`) runs on `policy_actions.LocalCandidateProvider`;
the driver still refuses a library-bound model without retrieval artifacts.
`rl_smoke` and the MCTS diagnostics use the local provider.

## Fixed in the application source

`InferencePipeline.predict_from_goal` passed `top_tactic_predictions`
positionally to a keyword-only parameter in the pinned AC source, so every live
prediction through the generic engine raised `TypeError`. It is passed by
keyword here. This statement describes the source pin, not later AC revisions.

## Original port results

| Check | Result |
| --- | --- |
| Offline suite | 660 passed, 35 skipped, 5 failed; the 5 fail identically on their bases (the `stdbuf` path test and three Thompson-sampler tests on MCTS-AC-branch, `test_source_aware_policy` on AC-branch `96f8bca`) |
| Live suite, Mathlib | 32 passed, 1 skipped (prepared-vocabulary checkpoint gate); includes the replay-recovery live cases |
| `rl_smoke`, Mathlib, best-first and PUCT | both pass: root certified through replay, one admission, one on-policy update |

New offline coverage: batched seeded proposals on the new decoder (one encoder
pass per round, exact counts, a seed that repeats without touching the global
RNG), each proposed instance's prior equal to its recomputed probability, an
empty pool proposing no arguments, every argument inside its own pool, the
generic engine's prior, a library lemma through search and observation, typed
HTPS imitation rejecting an argument outside its pool, and the imitation queue
round trip with its pools.

## Limits

- The decoder's draws are sequential per instance inside one forward; batching
  shares the encoder pass, not the argument steps.
- Pools are retrieved once per node. A node expanded in many rounds does not
  see lemmas a later retrieval would return.
- No checkpoint, GPU or retrieval artifact was available here. Library lemmas
  were exercised offline, with an in-memory library, through the model tests
  and one search whose pool holds a lemma; the live runs are local-only.


## Shared validation and inference lifecycle follow-up (2026-11-01)

The follow-up starts from PR #70 head `8260a43`. Its architecture was explicitly
verified in `dev_plans/mcts_source_aware_validation_and_inference_lifecycle.md`.

Both on-policy and HTPS actions pass the shared `_check_action` validator:
recorded graph identity, pool graph and exact query identity, model/library
binding, selected-reference membership and argument-count budget. HTPS now
receives all four update limits from the driver and checks graph/candidate sizes
before device batching or library adaptation. Critic-only batches enforce graph
limits. Metrics record the admitted sizes. HTPS remains supervised replay with
its own optimizer; complete-action on-policy weighting is unchanged.

Generic inference uses explicitly owned prepared records instead of an
unbounded lookup dictionary. `GNNPredictor.prepare_goal` returns the record;
`predict_tactics_with_arguments` consumes that record synchronously. Preparation
and prediction can still surround a batch, but the reasoner releases its record
references on success, error and cancellation. Records are bound to the preparing
pipeline and checked for observation drift. RL pools remain theorem-local and
unchanged across widening rounds.

Each prediction records an `ActionRendering` outcome and optional command.
An unsupported highest-ranked action no longer aborts the candidate list. The
generic reasoner settles it individually, preserving later candidates and their
original probabilities. The direct CLI reports unsupported output explicitly.
No selected argument is dropped or replaced to make an action executable.

The controlled live test now adopts the admitted root into `SnapshotRegistry`,
applies commands to `ExecutionOccurrence`, checks selected completion and kernel
certification, releases leases and asserts zero remaining allocations. Additional
production-search cases certify library-only and mixed local/library actions
through replay.

The next source-lineage record is
`tests/fixtures/pantograph/source_aware_lifecycle/source_manifest.json`.
Historical manifests and their recorded results remain intact. No checkpoint
weight layout, client pin, REPL protocol or user configuration changed.

Validation used an isolated install of client `955a492` and a freshly built REPL
at `9ecaaea` in `/tmp`, the existing local Mathlib source root, and CPU execution.
The selected trained-artifact, GPU/JupyterHub and throughput gates remain
pending. The workspace's installed client was not upgraded; verification used
`PYTHONPATH=/tmp/pr70-fixes-client`.

Final verification:

| Check | Result |
| --- | --- |
| Complete application suite, including source lineage | 642 passed, 37 skipped, 35 subtests passed |
| Native local/library/mixed execution and production replay | 3 passed |
| Replay, recovery, widening and bounded HTPS update selection | 33 passed: 24 native cases and 9 offline cases |
| Native cases across the two sequential selections | 27 passed |
| Whitespace check | `git diff --check` passed |

The new offline regressions cover shared optimizer rejection of graph/query/
artifact drift, all four HTPS limits before forward execution, restored pool
validation, exact candidate-byte admission, critic-only limits, invalid-first
and entirely unsupported candidate lists, prepared-input ownership/mutation,
cleanup on preparation failure, scoring failure and cancellation, repeated
searches, and one retrieval/pool across RL widening rounds. Driver tests assert
that HTPS receives the actual configured limits.

The four reviewed findings are addressed locally. PR #70 was still at `8260a43`
when checked and GitHub reported `MERGEABLE`/`CLEAN` with no automated checks.
Recommendation: merge into MCTS-AC-branch after committing and pushing these
fixes and confirming the published head. No commit, push or merge was performed
by this implementation. The inherited imported-content environment identity and
hole-classification limitations remain separate work.
