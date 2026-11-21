# Transactional Pantograph resource recovery in MCTS search

The transactional-resource integration originally used standalone client
`3300096484d95af070f2494eb42f5ab03a66450a` and custom REPL
`5cab95e132bcdd1d3cb715505d1694ac7bb0e0d6`. Both require protocol revision 3.
The client commit includes `bb22b22` and its corrected companion compatibility
resources. Docker builds the exact REPL revision; CLI setup consumes the
installed client's verified companion pins. For the current validation-version-2
pair and upgrade commands, use the [scope-aware application handoff](pantograph_scope_aware_application_integration.md).
The version-3 resource handling described below remains in force; the older
installation commands are retained as the original integration record.

## Behavior

Search records accepted commands, ordinary tactic rejections, and rolled-back
resource failures separately. A resource failure consumes its action budget
and settles its sampled aliases, but supplies no rejected-action actor target.
Other proposals can execute against the same retained input. Progressive
widening retains its existing bounds and command deduplication.

Resource-limited sources carry unknown proof evidence. Their failure does not
create a zero-valued critic target. Replay can try another recipe within its
existing budgets; a resource-limited combination is not cached as a logical
conflict. Kernel certification remains mandatory for success. Artifact
reverification surfaces `ReplayResourceExhausted` for a resource-limited tactic
instead of asserting `ArtifactMismatch`.

Post-tactic inspection is part of the operation: if it fails, the new result is
released before another proposal runs. Registry deletion failure invalidates
all leases and invokes the explicit session-retirement callback. A failed
deletion cannot leave an untracked allocation behind a usable registry.
`SnapshotRegistry` now requires `retire_session=client.aclose`. Artifact
reverification also requires that callback:

```python
async with client.state_scope() as scope:
    receipt = await reverify(scope, artifact, retire_session=client.aclose)
```

The collector distinguishes `resource_limit` from invalid initial input and
ordinary errors. Recoverable root failures retain a usable client; fatal
resource/internal errors use bounded worker replacement. Admission reports
operational `replay-resource` skips without quarantine or cache admission.

Metrics include `resource_limited_commands`, `resource_error_count`,
`replay_resource_errors`, and `searches_resource_limited`. Graph execution
records and API attempt records preserve bounded command, phase, limit and
recovery details. The proof-rate denominator excludes operationally aborted
attempts and reports them separately. Completed unsuccessful searches remain
in the denominator. An entirely operationally aborted round does not add a
curriculum failure or increment the no-update round counter. Valid historical
HTPS queue samples remain eligible for their ordinary supervised updates.

Search-evidence identity is now `kernel-certified-replay-resource-v2`, and the
evaluation metric is `searched-proof-rate-resource-v3`. Exact resume refuses
the older semantics. Start a fresh run from a compatible warm-start checkpoint;
do not change a saved run's evidence identifiers to bypass that check.

## Upgrade a JupyterHub installation

Pull the application commit containing this integration first. In the intended
Conda environment, update the client without replacing unrelated packages:

```bash
conda activate atp-mcts
python -m pip install --upgrade --no-deps \
  'pantograph-async-client @ git+https://github.com/123nol/Pantograph-Client@3300096484d95af070f2494eb42f5ab03a66450a'
python -m pip check
```

Build a separate executable so the previous installation remains available:

```bash
git clone --no-checkout https://github.com/123nol/Pantograph.git \
  /home/jovyan/Pantograph-transactional
git -C /home/jovyan/Pantograph-transactional checkout --detach \
  5cab95e132bcdd1d3cb715505d1694ac7bb0e0d6
cd /home/jovyan/Pantograph-transactional
lake build repl
```

Set `pantograph_repl` in the trial JSON to:

```text
/home/jovyan/Pantograph-transactional/.lake/build/bin/repl
```

Keep the existing compatible Mathlib `source_root`; the toolchain must match.
Verify artifact/environment bindings normally. Changed executable identities
invalidate relevant caches and can require legitimate deployment metadata
regeneration; do not edit hashes. Model tensors and graph vocabularies are not
changed by the application repair.

Use the ordinary module entry point for the trial:

```bash
cd /home/jovyan/Nol/Mathematics-ai-atp
python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/rl_mcts_joint_epoch5_trial.json
```

For Docker, rebuild the image and recreate the service with the existing
artifact/config mounts. Restarting an old image does not update its executable
or Python package.

## Validation boundary

The new regressions exercise controlled command errors, alias accounting,
snapshot cleanup, replay, admission and worker replacement. Native tests import
the companion's `Test.ContractFixtures` to trigger actual heartbeat exhaustion
and then certify a simple proof in the same session with both search modes.
The fixture module is used for tests, not added to production imports.

Run the resource tests with an explicit companion source root that contains
the built test fixture module:

```bash
PANTOGRAPH_RESOURCE_SOURCE_ROOT=/absolute/Pantograph \
PANTOGRAPH_TEST_REPL=/absolute/Pantograph/.lake/build/bin/repl \
python -m pytest maths_ai/gnn_inference/tests/test_transactional_resource_integration.py \
  --run-pantograph-live
```

Host-native certification is separate from full Docker image acceptance and
the trained-checkpoint GPU trial on JupyterHub. Report their results separately.

### Local verification record, 2026-11-04

- Combined inference/PLN/API suite: 773 passed, 43 skipped, 35 subtests passed.
  The skips include optional native/deployment checks without their configured
  environments. The client was installed in an isolated test directory; the
  developer's existing environment was not synchronized.
- Both search modes passed the genuine heartbeat-then-certification gate against
  the rebuilt companion, with zero allocations after each search.
- Six additional native tests passed for artifact reverification, conditional
  recipes, and production source-aware local/library argument execution.
- Fifteen operational recovery tests passed with
  `PytestUnraisableExceptionWarning` treated as an error. Remaining warnings
  were dependency deprecations and unavailable host NVML initialization.
- `uv lock --check --offline` passed using Docker's pinned uv 0.13.1. The only
  locked package version change is client 0.2.0 to 0.3.0.
- A wider native invocation reached all 24 successful test-call reports but hit
  its 180-second outer bound before a clean exit. It is not counted as a passed
  native acceptance run; the focused completed groups above are the evidence.
- Full Docker image execution and the user's GPU/trained-artifact JupyterHub
  trial were not performed here.
