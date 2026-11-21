# PR #96 CI failure analysis and local repair

## What the GitHub run establishes

Reviewed [CI run 44493554017](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/actions/runs/44493554017), testing the merge of application head `8c4482222555269191ca2354d9ddc5c5d60ce929` into main `444e057df1b15c6a2be4afc5b7d9ae8b3cee3d7b`.

Dependency installation succeeded, including the pinned Pantograph client 0.7.0 at `2b3265083908b2d1e1e1c5698570d94bfb51c56c`. Pytest collected 1,344 cases. At approximately 53%, these tests failed:

- `test_factory_finishes_scoped_probe_or_cleans_up_failure`
- `test_factory_failure_reaps_started_client`

The next cancellation test waited for a probe-entry event without observing the startup task. If startup failed before setting that event, the waiter never completed. GitHub cancelled the job at its six-hour limit. Pytest never reached its failure-summary phase, so the downloaded log contains the failed test names but not their exception tracebacks.

## Confirmed defect versus unresolved cause

The unbounded startup-event wait is a confirmed test-harness defect. It can conceal an earlier startup exception indefinitely. The same waiting pattern existed in worker-pool cancellation/replacement tests.

The original two CI startup failures are not yet reproduced locally. The unmodified factory tests pass with the pinned client, both individually and within the inference suite. Do not infer a production REPL defect or claim that the original CI errors are fully diagnosed from this log.

The local virtual environment initially contained client 0.4.0 instead of the declared pin. That caused a separate, reproducible `observer` keyword error; it was not the GitHub failure, because CI installed 0.7.0. Local validation now uses the declared 0.7.0 revision.

## Changes applied

- Fake Lake/REPL executables use the Python interpreter running pytest instead of selecting an arbitrary host `python3` through PATH. A regression supplies a conflicting PATH interpreter and verifies that the diagnostic children still work. This addresses a portability risk; the archived CI log does not prove it caused the original failures.
- A shared startup helper observes either the entry event or operation completion, propagates early exceptions, bounds missing notifications, and cancels/awaits owned tasks. Factory and worker-pool cancellation tests use it.
- The shared fake-client runner has a finite test-body deadline. Existing child-process reaping assertions remain enabled.
- CI stops on its first test failure, writes JUnit results and a console log, uploads diagnostics, and requests faulthandler dumps after a stalled test. The job has an explicit 20-minute limit.
- A new source-lineage stage records the changed test files without rewriting earlier manifests.

The checked-out source branch was fast-forwarded from `3b1c50f889d85429b3ad1d63585a6dc37bc9f152` to the already-published PR head before applying the patch. Unrelated tracked and untracked local work was preserved. No production model/runtime code or artifact binding was changed by this repair.

## Local validation

- Clean PR head plus the CI-only patch: **1,022 passed, 70 skipped, 35 subtests passed** in the inference suite, using the installed pinned client. The original factory tests and new startup-failure/interpreter regressions pass.
- Source-lineage checks: **8 passed** after appending the new adaptation record.
- Workflow YAML parsing and `git diff --check` passed.
- The broader working tree, including existing uncommitted measurement work, produced **1,062 passed, 70 skipped**, and one source-lineage failure for its pre-existing `rl_training_driver.py` changes. Those changes were not reverted or included in the CI adaptation hashes. They need their own lineage record if committed separately.

This validation does not reproduce the complete GitHub runner environment or establish the cause of the original two startup errors. The full API/formal suite was not completed during this repair; the changes concern the shared diagnostic-child tests and workflow.

## Remaining release gate

Push the reviewed patch and rerun GitHub CI. A fresh run must establish whether the original two startup tests now pass. If either still fails, the fail-fast traceback and retained diagnostics are needed to identify its cause. This local repair is not evidence that PR #96 is already green or ready to merge.
