# Pantograph failure evidence and the elaboration-abort repair

The application pins standalone client `e795e746abf72e325099bbc8206207e829302a4b`
(package 0.6.0) and custom REPL `661df4e32bb487f52e4bf229a34b155f9e69c76b`.
Protocol 3 and validation version 3 are unchanged.

## What the PUCT/no-PLN run showed

The 6 November run (50 rounds, 300 attempts) replaced its Pantograph worker 75
times: 74 `PantographRemoteInternalError`s and one hard timeout. The log names
only the exception class, so the failing requests were lost.

One cause is now reproduced and fixed. `exact`, `refine` and `have` with a
placeholder Lean cannot synthesize (for example a lemma applied with arguments
Lean cannot infer) log `don't know how to synthesize placeholder` and then throw
Lean's internal abort exception. The REPL treated every internal exception as
unexpected and ended the session. With only `Init` imported,
`exact (fun (x : Nat) => True.intro) _` reproduces it on the previously deployed
executable. The run's log fits: 2,478 failed `exact`, 394 failed `refine` and
106 failed `have` attempts, but no "don't know how to synthesize" message at
all. The repaired REPL returns an ordinary tactic rejection, so the search
simply tries the next candidate.

The log cannot say how many of the 74 failures had this cause. Every remaining
unexpected failure now leaves evidence (below), so the next run will tell.

## Application changes

- **Request history.** Every session from `create_model_sexpr_client` keeps a
  bounded `RequestHistory` (`hybrid_reasoner/failure_evidence.py`) of the
  client's request events: each request before it is written and its outcome
  after, labelled with the application phase (`operation`, `replay`,
  `artifact`, `certify`).
- **Command journal.** `MATHS_AI_COMMAND_JOURNAL` now records the same events
  for every operation, not only tactics. The executor no longer writes its own
  journal lines.
- **Failure archive.** When a collection worker fails, the attempt goes to a
  run-owned `FailureArchive`: `failures/ledger.jsonl` and
  `failures/attempts/<id>/failure.json`. A preliminary bundle is written before
  the failed session is cleaned up, then replaced by the final one once the old
  session's stderr is drained. Fatal integrity errors (protocol, validation,
  state) are archived and still propagate at once; a failed old-client close or
  a failed replacement start is kept as `cleanup_error` beside the primary
  error. A bundle holds the seed, the attempt context, the client's failure
  snapshot with the REPL record, the session's request history, the runtime
  identity (environment fingerprint over toolchain, Lake manifest, REPL,
  imports and compiled files; effective options; client and companion), and
  whether it is replayable. Reopening an archive restores its accounting, so a
  resumed run or restarted API never reuses an ID, overwrites a bundle or
  resets the bounds; the 256 MiB bound covers bundles and ledger, with room
  reserved for the single `capture_limit` line. Training writes to
  `<run>/failures`, standalone evaluation to `<eval-output-dir>-failures` (or
  `<run_root>/eval_failures/<time>`), and the API to `<state_dir>/failures`.
  Admission keeps a failed session's snapshot with the skipped row in its
  report.
- **Records and logs.** `CollectionRecord.failure` carries the failure ID,
  archive path, summary and timeout owner (`theorem-backstop`,
  `client-request-deadline`); `record_view` includes it. The console line now
  says what failed, for example
  `goal.tactic failed with PantographRemoteInternalError (REPL internal in execute: …); Pantograph worker replaced -> attempts/f000001-123/failure.json`.
- **Model-free replay.**
  `python -m maths_ai.gnn_inference.scripts.replay_pantograph_failure <failure.json> --source-root <root> --repl <repl> --imports <as recorded>`
  re-issues the session's history through typed client methods and maps
  recorded states to fresh ones. The target is the request the terminal
  snapshot names; earlier requests must end as recorded (success, tactic
  rejection, rejected start, resource rollback), otherwise the result is
  `diverged`. `reproduced` requires the same operation, error type, phase,
  category and exception identifier. A different runtime identity is reported
  as `environment-mismatch` without replaying; with `--allow-mismatch` the
  result is `reproduced-in-different-environment`.

Classifications are unchanged: rejections stay candidate outcomes, resource
rollback keeps the session, internal and validation failures retire it, and
operational failures never become training targets. Evaluation denominators
and checkpoint selection are not changed here.

## Update JupyterHub

```bash
conda activate atp-mcts
python -m pip install --upgrade --no-deps \
  'pantograph-async-client @ git+https://github.com/123nol/Pantograph-Client@e795e746abf72e325099bbc8206207e829302a4b'
git clone --no-checkout https://github.com/123nol/Pantograph.git /home/jovyan/Pantograph-failure-diagnostics
git -C /home/jovyan/Pantograph-failure-diagnostics checkout --detach 661df4e32bb487f52e4bf229a34b155f9e69c76b
cd /home/jovyan/Pantograph-failure-diagnostics && lake build repl && cd -
```

Rebind the active deployment to the new REPL with
`rebind_source_aware_environment` as in the
[compiler-aware handoff](pantograph_compiler_aware_application_integration.md),
using the compiler-aware REPL as `--old-repl`. No model tensor changes.

The same applies to a published serving release. Its declaration inventory is
bound to the REPL it was published with (its commit, or for older releases its
exact bytes), so an image built from this branch reports an environment
mismatch at readiness for a release bound to the previous REPL. Rebind the
deployment and publish it as a new version rather than editing the existing
one.

## Verification

Local, CPU, with the pinned client, the companion built from `661df4e` and the
pinned Mathlib `29dcec07`. Offline suite: 981 passed (two environmental
failures also fail on the base). Native live suite: 62 passed; four fail
identically on the base (no provisioned PeTTa or checkpoint artifacts). Not run
here: PLN-enabled native search and the GPU trial on JupyterHub.
Re-running the PUCT/no-PLN configuration there is what will show how many of
the historical failures this repair removes and what the rest are.
