# Bounded RL update batches

RL collection can produce more training inputs than fit in one GPU forward and
backward pass. Each logical update now validates all eligible rows, plans
bounded computational batches, and accumulates gradients before taking one
optimizer step. This applies to complete-action on-policy updates, tactic-only
RL updates, and each HTPS supervised update.

## Configuration

The existing fields now limit **one computational batch**, rather than the sum
of graph inputs collected in the round:

```json
{
  "theorems_per_round": 16,
  "max_update_nodes": 50000,
  "max_update_edges": 250000,
  "max_update_candidates": 16384,
  "max_update_candidate_bytes": 67108864
}
```

All four limits must be positive integers. The default node and edge limits
are 50,000 and 250,000. Zero no longer disables a limit. The default candidate
limits remain 100,000 entries and 268,435,456 estimated bytes.

A round with 175,216 total graph nodes can update successfully under a 50,000
node batch limit when each individual row fits and its other costs also fit.
Actor and critic streams are batched separately. Repeated proof-state graphs
remain separate logical inputs when their observations are separate rows.

A single oversized graph or candidate pool raises `UpdateBudgetError` before
any backward pass or optimizer step. Its diagnostic identifies the phase,
role, row index, fingerprint, actual costs, and limits. No row is silently
dropped. Invalid graph/query/library bindings are still rejected.

Candidate bytes estimate the decoder's local vectors, gathered library
vectors, and adapted library vectors using the model's element size. Repeated
pools are conservatively charged per row even when adaptation is shared within
a batch. This is not a bound on total GPU memory: model weights, gradients,
optimizer states, attention intermediates, and backend workspaces also use
memory. Collection results and CPU metadata still grow with the round size.

## Learning behavior

On-policy updates first compute detached state values in bounded batches.
Returns minus those values are normalized across the entire actor stream,
with observation multiplicity weights. Each subsequent backward pass uses
those same global advantages and the global loss denominators. Actor,
entropy, critic, and tactic-anchor terms keep their existing weighting.

For numerical stability, advantage statistics use double-precision CPU scalar
reductions. The model's forward/backward dtype is unchanged. Splitting matrix
operations and reductions can introduce small floating-point differences;
results are numerically equivalent rather than bitwise identical.

Evaluation-mode forwards keep dropout disabled while gradient recording is
explicitly enabled during backward batches. The current actor-critic HTPS
forwards also use evaluation behavior through the shared encoder; this change
does not enable training-mode dropout for HTPS.

Parameters remain fixed throughout the value prepass and all on-policy
backward batches. Gradients are zeroed once, accumulated, checked for finite
values, clipped once, then applied with one optimizer step. Each configured
HTPS step retains its separate optimizer and queue sampling. Computational
batches do not increase logical optimizer-step counts.

A failure before the optimizer step clears partial gradients and ends the
update. Earlier completed logical updates are not undone. There is no claim
of transactional rollback if an optimizer itself fails during its step.

## Metrics

- `update_nodes`, `update_edges`, `update_candidate_count`, and
  `update_candidate_bytes` count logical-update inputs once. They can exceed
  the configured computational limits.
- `update_peak_nodes`, `update_peak_edges`, `update_peak_candidate_count`, and
  `update_peak_candidate_bytes` report the largest planned computational
  batch costs, verified again when inputs are materialized.
- `update_actor_batches` and `update_critic_batches` count backward batches;
  `update_backward_batches` is their sum.
- `update_value_batches` counts the detached RL value prepass. It does not add
  training examples, OOV counts, or optimizer steps.
- Existing losses and log-probability diagnostics retain their whole-stream
  reductions. They are not averages of equally weighted batch means.
- `onpolicy_optimizer_step` and `htps_optimizer_step` remain zero or one for
  each respective logical call.

The training driver prefixes HTPS resource metrics with `htps_`. Batch counts
sum across HTPS calls in a round; peak costs take the maximum across those
calls. Existing HTPS losses, logical-input totals, and OOV metrics continue
reporting the last HTPS call in the round. `htps_optimizer_steps` reports the
actual count of completed calls that stepped.

## Checkpoints and deployment

New training checkpoints store `training_update_contract`, including the
reduction version, forward-mode policy, and four computational limits. Exact
resume requires a matching contract. An older checkpoint without the contract
or a checkpoint recorded under different limits must be used as a weight-only
warm start for a fresh run. Model checkpoint format and weights are unchanged.

No custom REPL rebuild, Pantograph-client update, retrieval artifact export,
or environment rebinding is needed for this application change. Existing
positive-limit trial/full configurations use accumulation automatically.

For a first JupyterHub check, retain the intended 16 theorems per round, run
one or two rounds, and set `checkpoint_every` to 1. Check that logical totals
may exceed limits while all peak batch costs fit; check optimizer counts and
that `last.pt` exists after a completed round. CPU regression and native Lean
checks do not establish available memory for the deployed trained GPU model.

## Verification evidence

Both encoder families pass full-batch versus accumulated objective, gradient,
and one-step optimizer comparisons. A synthetic 16-theorem driver test also
covers checkpoint saving, contract validation, and periodic proof export.
Native Lean diagnostics passed with the pinned client/REPL and computational
node/edge limits of 10/22, including one on-policy step with HTPS disabled and
one on-policy plus one HTPS step with it enabled.

On an RTX 3050 Ti, a small synthetic HTPS model used exactly 17,193,984 peak
allocated CUDA bytes after optimizer warmup for 7, 140, and 1,400 total input
rows at an eight-node computational limit. Each update took one optimizer
step. This measurement does not establish peak memory for the deployed policy
or its larger proof-state graphs.
