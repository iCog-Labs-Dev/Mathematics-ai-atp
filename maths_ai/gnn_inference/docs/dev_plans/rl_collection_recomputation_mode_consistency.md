# Consistent model behavior during RL collection and updates

## Problem and scope

On `pre-model-update-AC`, `RLHybridReasoner.expand` collects actions with the model in evaluation mode and gradient recording disabled. Both RL update functions in `pln_rl_training.py` switch the model into training mode. With nonzero dropout, recomputing stored actions therefore changes the policy distribution before any optimizer step.

This establishes evaluation-mode forward passes as the shared contract for live RL collection and loss recomputation. Gradient recording remains enabled for loss construction and optimization. This change covers both graph-harvest and stored-action RL updates; offline supervised training retains its existing mode behavior.

## Chosen architecture

1. Make both public RL loss builders, `compute_transition_loss` and `compute_onpolicy_loss`, put the model in evaluation mode before their forwards. Direct callers must receive the same layer behavior as callers through the update functions. Leave the model in evaluation mode after RL computation.
2. Remove the training-mode switches from `train_step` and `train_step_onpolicy`. Explicitly enable gradient recording over each complete update so an enclosing `torch.no_grad()` does not silently disable learning. Ensure direct loss construction also records gradients. `torch.inference_mode()` is outside the supported update contract.
3. Keep collection under `torch.no_grad()`. Collection stores integer actions; updates perform fresh forwards rather than retaining or attempting to recover collection computation graphs.
4. Document that unchanged weights, graph features, argument masks, and autoregressive inputs are required for matching action probabilities. Evaluation mode corrects dropout behavior; it does not establish all other requirements automatically.
5. Audit the driver, smoke script, convenience wrappers, and direct loss callers. Migrate assumptions that RL updates leave the model in training mode. Preserve held-out evaluation's existing caller-mode restoration and ordinary supervised-training behavior.

No configuration flag, alternate update mode, checkpoint rewrite, loss clipping, or learning-rate change is required. This patch does not repair earlier offline actor training or promise to eliminate all extreme log probabilities.

## Alternatives considered

- **Evaluation-mode collection and recomputation, with independent gradient control (chosen):** avoids dropout-induced distribution changes and collection graph retention. RL updates do not receive dropout regularization.
- **Training-mode collection and recomputation:** retains dropout regularization, but independent masks do not reproduce the collection policy. Correct treatment would require recording and reproducing stochastic execution, including batching differences.
- **Retain collection computation graphs:** avoids recomputation but retains autograd allocations throughout Lean search, complicates worker boundaries, and increases memory use.
- **Set checkpoint dropout to zero:** avoids this particular mismatch but changes the deployed model specification and leaves shared RL update behavior inconsistent for other checkpoints.

## Sequential implementation and validation

### Stage 1: Shared forward and gradient contract

Update the two loss builders and both update functions together. Clarify their docstrings and the stored-action evaluation contract. Audit all current callers without changing the offline actor–critic objective.

### Stage 2: Probability and optimizer regression coverage

Use a real small GATv2 actor–critic with nonzero dropout. Collect sampled tactic and argument indices without gradients, then compare stored-action tactic and argument log probabilities before optimization. Repeat recomputation and compare individual-graph collection with batched recomputation using numerical tolerances. Include differing graph sizes and argument lengths.

Verify that both RL update paths still produce finite gradients and change parameters, including when invoked inside `torch.no_grad()`. Check that their model forwards disable dropout and enable gradients. Cover direct loss calls, critic-only updates, and empty updates. Preserve existing budget and failure-record tests.

### Stage 3: Integration checks

Run focused actor–critic, RL reasoner, graph-harvest training, live collection, and training-driver tests. Run the broader application suite if these pass and resources permit. Native Lean execution is not necessary to establish the dropout/autograd contract; a live smoke run is optional when a compatible executable is available.

## Acceptance criteria

- Recomputed tactic and argument log probabilities match collection before parameter updates within numerical tolerance with nonzero configured dropout.
- Both RL update functions can backpropagate and perform an optimizer step while model layers remain in evaluation mode.
- Collection does not retain autograd graphs.
- Existing callers use the same shared behavior without opt-in flags.
- Reports distinguish fixing this defect from demonstrating recovery of the user's trained checkpoint.

## Implementation and verification

Implemented on `pre-model-update-AC`. Both public loss builders enforce evaluation-mode layers. Both loss builders and both optimizer entry points explicitly enable gradient recording. The model remains in evaluation mode after RL computation, including empty updates. Collection retains its existing gradient-disabled behavior. Offline supervised training is unchanged.

Regression coverage verifies individual collection versus batched recomputation with a two-layer GATv2 model configured with dropout 0.4, different graph sizes, and zero, one, and two recorded arguments. Repeated recomputation matches, gradients reach the pointer, and both optimizer paths update parameters under an enclosing `torch.no_grad()`. Direct critic-only loss construction and empty updates are covered.

Validation used the pinned client installed separately at `/tmp/pr70-fixes-client`; the workspace Python environment was not modified.

- Focused training, reasoner, collection, and driver tests: 105 passed, 1 skipped.
- Full `maths_ai/gnn_inference/tests` suite: 428 passed, 28 skipped, 1 failed, with 17 passing subtests.
- The unrelated failure is `test_repository_config_omits_removed_resource_fields`: the committed test expects one worker and automatic collection devices, while the committed config specifies four workers and `cuda:0`. The mismatch was confirmed in `HEAD`; neither file was changed.
- `git diff --check` passed.

This verifies the mode and gradient contract on CPU. It does not establish that the user's previously trained checkpoint has recovered its accuracy or that every cause of extreme argument log probabilities is resolved. No checkpoint was rewritten and no JupyterHub training run was performed.
