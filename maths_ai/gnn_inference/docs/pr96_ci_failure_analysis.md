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

- The original local validation snapshot produced **1,022 passed, 70 skipped, 35 subtests passed**, but omitted the changed workflow. It therefore did not validate the complete patch; the follow-up below corrects that limitation.
- Source-lineage checks: **8 passed** after appending the new adaptation record.
- Workflow YAML parsing and `git diff --check` passed.
- The broader working tree, including existing uncommitted measurement work, produced **1,062 passed, 70 skipped**, and one source-lineage failure for its pre-existing `rl_training_driver.py` changes. Those changes were not reverted or included in the CI adaptation hashes. They need their own lineage record if committed separately.

This validation does not reproduce the complete GitHub runner environment or establish the cause of the original two startup errors. The full API/formal suite was not completed during this repair; the changes concern the shared diagnostic-child tests and workflow.

## Remaining release gate

Push the reviewed patch and rerun GitHub CI. A fresh run must establish whether the original two startup tests now pass. If either still fails, the fail-fast traceback and retained diagnostics are needed to identify its cause. This local repair is not evidence that PR #96 is already green or ready to merge.

## Follow-up: workflow lineage mismatch

[Run 44890029659](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/actions/runs/44890029659), after commit `7c660bd0656d7138ed4462ff51cd3d331a25afce`, stopped after 647 passed and 51 skipped cases. Its first failure was the source-lineage assertion for `.github/workflows/ci.yml`: the actual hash was `60a690b2ccaac76244ae5d9721c2666a2b5292d183b01d497a4c84e95f2154c7`, while the last recorded hash was `43c2ecf990f897b298dea6b47a9d4e89fea9c6b85823a830ec32c301d1585270`.

The previous patch registered its changed test files but omitted the workflow from the new lineage stage. Its local validation snapshot also omitted the workflow change. This was a validation mistake, not an unexplained runner difference. The correction adds the workflow's prior and current hashes to that same adaptation stage without changing earlier records or disabling the check.

Follow-up verification uses a fresh archive of the exact pushed commit, reproduces the mismatch, and then applies only the manifest correction. All tracked files changed by the original CI commit must be present in that validation snapshot.

Result: the exact pushed snapshot reproduced the workflow mismatch before correction. With only the manifest correction applied, the inference suite completed with **1,022 passed, 70 skipped, and 35 subtests passed**. All seven other files from the original patch were verified unchanged against the pushed commit. `git diff --check` passed. A new GitHub run is still required, including the factory tests that this fail-fast run did not reach.

## Confirmed startup cause: non-GNU stdbuf on the runner

[Run 44891098225](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/actions/runs/44891098225) passes source lineage and reaches the original factory failure. The pinned client's startup check rejects the selected executable with `The configured stdbuf is not GNU coreutils stdbuf.` It stops after 668 passed and 53 skipped cases.

The runner is Ubuntu 26.04, whose default utilities are provided by uutils. GNU utilities remain available separately. See [Ubuntu's documented provider change](https://documentation.ubuntu.com/release-notes/26.04/summary-for-lts-users/) and the [GNU coreutils package](https://packages.ubuntu.com/resolute/gnu-coreutils). Inspection of that package confirms `/usr/bin/gnustdbuf` and `/usr/libexec/coreutils/libstdbuf.so`.

The workflow had declared neither the GNU dependency nor its selection. The client's strict check is intentional. A local non-GNU-banner launcher reproduces the same factory exception; selecting real GNU stdbuf ahead of it allows the factory tests to pass.

The correction selects Ubuntu 26.04 explicitly, installs `gnu-coreutils`, and prepends a job-local `stdbuf` symlink to `/usr/bin/gnustdbuf` through `GITHUB_PATH`. A version check runs before dependency builds. It preserves the host's coreutils provider and production validation. Docker's Debian-based runtime and the Conda environment already declare GNU coreutils; this change addresses the CI runner.

The workflow and lineage-test changes are recorded in a new `ci_gnu_launcher` adaptation stage based on `6a3b5a152b53c4f97ed02ff4080690d1604a0983`. Earlier lineage records are unchanged. Validation uses a fresh archive of that exact commit plus the complete patch, including the workflow.

Local results: **8 factory tests passed** with GNU selected ahead of a deliberately conflicting non-GNU launcher. The complete inference suite on the clean patched snapshot returned **1,022 passed, 70 skipped, and 35 subtests passed**. Workflow YAML, provisioning-shell syntax, manifest hashes, and `git diff --check` passed. Installing the package on the actual Ubuntu runner remains part of the next CI run.
