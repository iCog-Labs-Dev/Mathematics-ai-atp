# MCTS RL collection and recomputation mode consistency

## Problem and scope

On `pre-model-update-MCTS`, collection disables dropout and gradient recording. The shared graph-harvest and on-policy update functions enable training-mode dropout before rebuilding action probabilities. Unchanged weights therefore do not guarantee that the recomputed probabilities describe the collection policy.

Port the approved AC correction to this branch's observation-based action records. Preserve sampled-action multiplicities, widening records, graph fingerprints, resource budgets, reward targets, and optimizer-generation accounting.

## Architecture

Both public RL loss builders, `compute_transition_loss` and `compute_onpolicy_loss`, enforce evaluation-mode forwards and explicitly enable gradient recording. Both optimizer entry points, `train_step` and `train_step_onpolicy`, do the same, including when called inside `torch.no_grad()`. The model remains in evaluation mode after these operations. An enclosing `torch.inference_mode()` is not supported for optimization.

Collection continues using evaluation mode under `torch.no_grad()`. Each update creates a fresh computation graph from recorded actions. Do not retain collection computation graphs, change the checkpoint specification, or add a configurable alternate mode.

This branch stores `PolicyObservation` objects rather than AC's failure/action tables. Adapt probability-consistency tests to the current observation API, including rejected observations and complete graph fingerprints. Preserve the existing `onpolicy_optimizer_step` metric and existing empty-update return schemas.

HTPS is a separate supervised imitation and critic-regression objective. Keep `train_step_htps_style` in training mode; it need not reproduce an unchanged collection policy. Do not put unconditional evaluation-mode mutation inside the actor's general-purpose encoder. Add a check that an HTPS update after an on-policy update explicitly restores training behavior. Ordinary offline supervised training remains unchanged.

## Alternatives considered

- **Evaluation-mode on-policy updates with gradients (chosen):** reproduces collection layer behavior without retaining search-time autograd graphs. Dropout is disabled for this objective.
- **Training-mode collection and recomputation:** retains dropout, but independent masks still change the policy. Correct reproduction requires recording stochastic execution and handling batching differences.
- **Disable dropout in every model forward:** also changes HTPS and offline supervised training. This exceeds the on-policy correction's scope.
- **Retain collection computation graphs:** increases search memory consumption and complicates worker ownership and cleanup.

## Sequential implementation

1. Update both loss builders and both on-policy/graph-harvest optimizer entry points together. Document evaluation-mode and gradient behavior in those functions and the stored-action evaluation contract.
2. Adapt the AC regression coverage to `PolicyObservation`, with a real GATv2 model using nonzero dropout. Compare individual collection with repeated batched recomputation across different graph sizes and argument counts. Verify gradient flow, actual parameter updates under an enclosing `torch.no_grad()`, critic-only losses, and empty updates.
3. Verify that the separately supervised HTPS path restores training mode after an on-policy update. Run focused loss, reasoner, driver, collection, and HTPS tests, followed by the broader inference suite when feasible.

## Acceptance and limitations

- Stored actions have matching collection and recomputed tactic/argument log probabilities before parameter updates, within numerical tolerance.
- Nonempty on-policy updates retain finite gradients and perform optimizer steps with dropout disabled.
- Direct loss callers share the same behavior; empty updates do not change parameters.
- HTPS retains its deliberate supervised dropout behavior.
- Matching probabilities still requires unchanged graph features, candidate masks, autoregressive inputs, and weights.
- This change does not repair an already degraded checkpoint or guarantee successful proof exploration.

## Implemented validation

Implemented locally on `pre-model-update-MCTS` at base `e96d2f7a733d1301e8167f52de2d988ea6eb28ff`, adapting AC commit `6b5ef6360ad2bed828698e8088a9b23c3da24a88` to observation-based action records. Added a new `mcts_policy_mode/source_manifest.json` verification link without rewriting prior source manifests.

- Focused RL loss, reasoner, and HTPS tests: 46 passed, with 4 passing subtests.
- Full inference suite: 611 passed, 34 skipped, with 39 passing subtests.
- Nonzero-dropout GATv2 regression verifies individual collection versus repeated batched recomputation, different graph sizes, variable argument counts, and pointer gradients.
- Both optimizer entry points update parameters inside `torch.no_grad()` while their encoder forwards run in evaluation mode with gradients enabled.
- An HTPS update after an on-policy update restores training mode and performs its separate optimizer step.
- `git diff --check` passed.

The final suite used an isolated install of the pinned client under `/tmp/mcts-policy-mode-client`. The project environment was not changed. These are CPU tests, not a JupyterHub GPU trial or a demonstration that an existing checkpoint has recovered its policy quality.
