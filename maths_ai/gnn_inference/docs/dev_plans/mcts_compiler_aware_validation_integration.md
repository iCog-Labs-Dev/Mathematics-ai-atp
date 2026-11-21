# Integrate compiler-aware Pantograph validation into MCTS actor–critic search

## Status and objective

This is a proposed application implementation and deployment plan. No runtime code is changed by this document. It coordinates [the REPL repair](pantograph_repl_compiler_aware_validation.md) and [the standalone-client migration](pantograph_client_compiler_aware_validation.md).

The objective is for best-first and PUCT search, periodic evaluation, and the API to use the repaired retained-state validator without concealing integrity failures or changing learned model inputs. Valid compiler-producing tactics must be accepted and replayed; reported success and terminal success rewards must still require original-root kernel certification.

Inspected application baseline: `MCTS-AC-branch`, commit `6f4d94df7343eef07d8843e048aa0c9974d7a8a9`. Current dependencies are client `053ff67dfc540391100c86e4deada06839904655` and REPL `ba2c3dd8383f90b4b5327e31a645a53411f385e4`. Recheck worktrees and revisions before implementation. A later AC-branch port requires its own caller audit; this plan does not assume those branches have identical files.

## Incident and uncertainty

The supplied JupyterHub traceback enters `evaluate_proof_rate()` from periodic RL evaluation, then fails in `RetainedTacticExecutor.apply()` while the client exchanges a `goal.tactic` request. The REPL reports:

```text
execution owner=declaration _private.0.Nat.pow.match_1.splitter._rarg._cstage2
path=environment audit/declaration-type/binder-type:
Missing proof dependency _obj
```

An earlier `simp` transition for `True ∧ True` was accepted and produced a QED candidate. The excerpt does not identify the failing command, establish a final certificate, or demonstrate a PLN translation failure. Treat compiler-data misclassification as the leading mechanism pending native reproduction.

This failure is not evidence that model weights, embeddings, or prepared vocabularies are malformed. Deployment identities can change when the repaired executable is installed; those bindings still need verified migration.

## Architectural decisions and alternatives

| Approach | Benefit | Cost or correctness problem | Decision |
| --- | --- | --- | --- |
| Catch the error in search and record a rejected tactic | Keeps the run moving | Conceals failed snapshot integrity and trains on an operational defect | Reject |
| Disable periodic evaluation, PLN, or the triggering tactic | May avoid the immediate trigger | Does not repair shared execution; another mode or tactic can encounter it | Reject as the fix |
| Reconstruct intermediate states instead of retaining snapshots | Avoids some saved data | Reintroduces context loss and diverges from the certification architecture | Reject |
| Refresh the matched companion pair, migrate all active consumers, and verify native search | Repairs the shared cause | Requires coordinated release and deployment binding | Choose |
| Re-export and reconvert every trained model | Can produce a fresh deployment | Duplicates large artifacts and can reset learned critic weights | Prefer existing tensor-preserving rebind when its preconditions hold |

The application must not implement compiler-expression classification. That belongs to the REPL. Application changes concern dependency migration, complete command diagnostics, native integration tests, and verified deployment.

## Shared invariants

- Both `selection_policy: legacy` for best-first and `selection_policy: puct` use the same retained executor and validator boundary.
- The target companion uses overall protocol 3, validation version 3, and scope `execution-closure-and-representation-aware-audit`.
- Genuine `validation` errors remain terminal. They are not converted to rejected-action actor samples, successful edges, or replay candidates.
- Preserve registry reservations, leases, replay resource limits, and worker/session cleanup. An integrity failure must not be caught and retried on the same session.
- Keep the existing root statement, normalized model expressions, local-argument identities, and source-aware candidate records. The repair does not justify changing checkpoint graph features.
- A QED candidate alone is not a solve. Keep sequential replay, original-root certification, API certification fields, and certified-success rewards.
- PLN produces guidance and shaped rewards according to the current validated rule-proof/STV contract. It does not supply the Lean kernel certificate.

## Stage 0: Capture a reproducible command history

Before replacing the executable, preserve its binary digest and the exact run config, client installation identity, Lean imports/toolchain, dataset inputs, proposal/evaluation seeds, and last saved checkpoint paths.

Provide a diagnostic capture using the shared typed-client boundary, not a best-first-only monkey patch. Record the command before awaiting its response, including the selected goal identity, tactic text, operation phase, and worker/session identity. Preserve the structured error payload, returned validation evidence, and bounded stderr tail. Use a bounded, incrementally flushed journal so a terminal error does not erase the last request.

Evaluate the installed client's public hooks before choosing an implementation. If no supported hook supplies the required records, place the diagnostic adapter in the application's shared retained-execution boundary and keep the client codec responsible for transport integrity. Do not depend on undocumented private reader attributes in production.

The reproduction must include preceding commands if compilation state accumulates across a session. Capture exploration versus replay versus inspection/certification distinctly. Avoid exporting large model tensors or unrelated environment contents into the journal.

**Exit gate:** the REPL implementer has the actual failing command sequence and immutable runtime identities. The supplied traceback is retained as evidence, not mislabeled as a complete reproducer.

## Stage 1: Consume the released client and repaired REPL

Depends on completion of the REPL native gate and standalone package release.

1. Pin the new standalone client to its full immutable commit in `pyproject.toml`.
2. Regenerate `uv.lock` and `requirements-mcts-ac.txt` through the repository's existing export workflow. Confirm the installed package's companion metadata identifies the exact repaired REPL commit.
3. Update `docker/Dockerfile` to the full repaired REPL commit and migrate build verification and companion-source checks together.
4. Update active protocol fixtures, fake servers, smoke scripts, diagnostic pins, examples, and tests to validation version 3. Keep historical captures immutable and extend their source-lineage record.
5. Audit `retained_executor.py`, replay, root materialization, admission, worker creation, evaluation runtime, and API runtime for direct version assumptions. All existing callers must use the updated client contract.
6. Preserve terminal exception handling. Add a regression if any caller currently reclassifies a protocol validation error as an ordinary tactic failure.
7. Inspect the new companion contract against `hasSorry`, `hasUnsafe`, provenance, and certification consumers. Do not weaken their checks to make a native control pass.

Primary application targets include dependency files, `maths_ai/hybrid_reasoner/retained_executor.py`, `joint_inference.py`, `pantograph_model_sexpr.py`, `maths_ai/gnn_inference/atp_lean_gnn/live_runtime.py`, `rl_live_collection.py`, `evaluation_runtime.py`, and the existing migration/lineage tests. Most shared execution files should need verification rather than speculative redesign.

**Exit gate:** the application requires the revised audit boundary on startup and imports the exact client release. Old validation-version-2 fakes cannot satisfy active tests.

## Stage 2: Native search, learning, evaluation, and PLN regressions

Use the minimal native compiler-producing sequence from the REPL handoff. Keep a deterministic scripted-proposal control separate from the real source-aware checkpoint trial so a random proposal miss cannot conceal execution correctness.

| Path | Required evidence |
| --- | --- |
| Direct typed-client sequence | Checked transition, retained continuation, original-root certificate |
| Best-first exploration and replay | Same valid command sequence accepted; certificate required; zero intermediate reconstructions |
| PUCT widening/batched expansion and replay | Same invariant through shared executor; no retained snapshot leak |
| Multi-goal continuation after a compiler-producing operation | Siblings, shared assignments, inspection, and replay remain correct |
| Periodic evaluation after at least one optimizer update | The exact driver-to-evaluation path completes and records evaluation metrics |
| Standalone `--eval-only` | Same contract, certification, and failure accounting as periodic evaluation |
| PLN enabled | Real provisioned PeTTa runtime, qualifying target-matching rule-proof/STV control, and certified Lean proof |
| PLN disabled | Compiler repair works independently of PLN |
| Injected genuine validation failure | Session retired; no proof success, actor rejection sample, or reused invalid state; owned processes and leases cleaned up |
| Transactional resource failure | Existing resource accounting and bounded recovery preserved |
| Prove API and evaluation-job API | Certification-aware response and existing lifecycle guarantees preserved |

Extend `test_scope_aware_validation_live.py` or add a clearly named successor test module, plus focused collection/evaluation and API tests. Keep the earlier `tauto`, variable-ownership, joint replay, and resource-limit regressions. A skipped native test is not a passing native acceptance gate.

The real checkpoint trial should use the current source-aware architecture and one GPU/one worker initially. Record actual optimizer counters. Do not require a sampled model to discover a particular tactic to establish the shared execution regression.

PLN setup is already provided by `scripts/provision_pln.sh`. Reuse that installation and activation environment. Do not rebuild PeTTa or change its parser merely because the Lean companion changed.

**Exit gate:** deterministic native regressions pass in both searches; periodic evaluation completes; PLN guidance and Lean certification are demonstrated separately; theorem-exit allocations and child processes return to their expected baseline.

## Stage 3: Rebind the existing JupyterHub deployment without resetting weights

Preserve the executable that actually matches the current inventory. The latest executable by filename is not necessarily that binary. Inspect the currently used configuration rather than assuming it still points to the original trial config.

Build the repaired companion in a new directory, for example `/home/jovyan/Pantograph-compiler-aware`, at the immutable commit supplied by the handoff. Install the matching client in `atp-mcts` and run `python -m pip check`.

Use the existing migration command with the actual active deployment config:

```bash
python -m maths_ai.gnn_inference.scripts.rebind_source_aware_environment \
  --config artifacts/source_aware_deploy_scope_aware/rl.json \
  --old-repl /home/jovyan/Pantograph-scope-aware/.lake/build/bin/repl \
  --repl /home/jovyan/Pantograph-compiler-aware/.lake/build/bin/repl \
  --output artifacts/source_aware_deploy_compiler_aware
```

The paths above assume the successful scope-aware bundle is still the active deployment. Replace them from the current configuration when different. The final output must be a new, nonexistent name on persistent storage.

The existing migration must verify unchanged imports/toolchain/compiled library and declaration availability, preserve every actor and critic tensor exactly, update the model/index/query/inventory binding chain consistently, and pass strict loaders. Preserve the persistent-publication implementation: preflight the actual filesystem, publish a completed backing directory with an atomic relative symlink, refuse overwrite, and retain validated bundles for supported retry. Keep the sibling backing directory after publication.

Add the minimized compiler-producing continuation/certification control to migration verification through a shared smoke helper if the existing trivial goal does not exercise the repaired path. Migrate normal smoke and release verification callers to that helper. Keep this bounded and independent of GPU weights or PLN availability.

When toolchain or compiled Mathlib contents actually differ, do not edit inventory hashes or force this REPL-only migration. Use the independently verified full deployment export route. The repair itself does not require retraining or a fresh critic.

The existing command migrates the checkpoint named by its input config. If preserving an RL-trained checkpoint, first provide a configuration that names that checkpoint and passes the old strict artifact bindings. This plan does not add automatic training resume across changed companion contracts or environments. Starting a new run from preserved weights and resuming optimizer/curriculum state are different operations; inspect the existing resume checks before proposing a separate migration.

**Exit gate:** the new persistent bundle loads, learned tensors are unchanged, the repaired continuation certifies, and the original deployment remains available.

## Stage 4: Trial and release handoff

1. Run the short best-first PLN trial using the new bundle's paths and the current small trial limits. Keep the trial's meaningful inputs and periodic evaluation enabled.
2. Verify admission, guidance status counters, accepted/rejected/resource outcomes, replay counts, certificates, optimizer steps, checkpoint writes, and periodic evaluation completion.
3. Run a no-PLN PUCT smoke to establish that the repair is shared rather than conditional on PLN.
4. Exercise the container build with the actual companion fetch/build and native CPU smoke. A Dockerfile syntax check is not an image-build result. State any unavailable model/GPU or deployment tests explicitly.
5. Record the final application/client/REPL commits, test commands, executable and artifact identities, native capture lineage, and JupyterHub activation/build/rebind commands.
6. Review the full-run configs against the newly rebound paths before launching long jobs. Do not start full training as part of merely publishing this patch.

## Cross-repository implementation and merge order

1. Capture and minimize the real failure; agree the representation classifier and version-3 contract.
2. Implement and natively verify the REPL repair, then commit it.
3. Build the standalone client against that immutable companion; test and publish its package/commit.
4. Integrate the exact client and REPL pins in the application, migrate fixtures/callers, and run native search/evaluation gates.
5. Merge/deploy the matched set, rebind the persistent deployment, and execute the JupyterHub trial before long runs.

Development can overlap once fixtures are agreed. Final companion hashes and application dependency pins follow the upstream commits they identify. Never deploy only the new REPL descriptor to a running client that still requires validation version 2.

## Completion criteria

The actual compiler-producing regression passes through both search modes and periodic evaluation, PLN returns separately validated guidance, and final solves have original-root certificates. Real validation errors remain terminal and excluded from policy rejection training. Dependency/fixture lineage and deployment bindings identify the matched components, while learned weights and persistent bundle publication guarantees are preserved.
