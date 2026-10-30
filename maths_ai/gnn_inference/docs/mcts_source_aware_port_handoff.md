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
its prior is the greedy complete action's exact log-probability.

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
positionally to a keyword-only parameter, so every live prediction through the
generic engine raised `TypeError`. It is passed by keyword here; AC-branch still
has the bug.

## Results

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
