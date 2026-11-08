# Pantograph failure evidence and the elaboration-abort repair

The application pins standalone client `985f936083f524619b757a2dcc0545ed82a82362`
(package 0.6.0) and custom REPL `d809a610a4cf24560ea836e8bcb879338e4f4797`.
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
- **Failure archive.** When a collection worker is replaced, the failed
  attempt is written to a run-owned `FailureArchive`:
  `failures/ledger.jsonl` and `failures/attempts/<id>/failure.json`, with the
  seed, attempt context (round index, worker, proposal, model generation), the
  client's final failure snapshot including the REPL's stderr record, the
  session's request history, client and companion identities, and whether the
  bundle is replayable and what is missing if not. Bundles are written
  atomically; the archive stops at 256 MiB or 2,000 bundles and says so in the
  ledger. Training writes to `<run>/failures`, standalone evaluation to
  `<eval-output-dir>-failures` (or `<run_root>/eval_failures/<time>`), and the
  API to `<state_dir>/failures`.
- **Records and logs.** `CollectionRecord.failure` carries the failure ID,
  archive path, summary and timeout owner (`theorem-backstop`,
  `client-request-deadline`); `record_view` includes it. The console line now
  says what failed, for example
  `goal.tactic failed with PantographRemoteInternalError (REPL internal in execute: …); Pantograph worker replaced -> attempts/f000001-123/failure.json`.
- **Model-free replay.**
  `python -m maths_ai.gnn_inference.scripts.replay_pantograph_failure <failure.json> --source-root <mathlib> --repl <repl>`
  re-issues the session's history through typed client methods, maps recorded
  states to fresh ones, expects recorded rejections again, and reports
  `reproduced` only if the final request fails with the same operation, error
  type, phase, category and exception identifier. It refuses a different
  companion unless `--allow-mismatch`.

Classifications are unchanged: rejections stay candidate outcomes, resource
rollback keeps the session, internal and validation failures retire it, and
operational failures never become training targets. Evaluation denominators
and checkpoint selection are not changed here.

## Update JupyterHub

```bash
conda activate atp-mcts
python -m pip install --upgrade --no-deps \
  'pantograph-async-client @ git+https://github.com/123nol/Pantograph-Client@985f936083f524619b757a2dcc0545ed82a82362'
git clone --no-checkout https://github.com/123nol/Pantograph.git /home/jovyan/Pantograph-failure-diagnostics
git -C /home/jovyan/Pantograph-failure-diagnostics checkout --detach d809a610a4cf24560ea836e8bcb879338e4f4797
cd /home/jovyan/Pantograph-failure-diagnostics && lake build repl && cd -
```

Rebind the active deployment to the new REPL with
`rebind_source_aware_environment` as in the
[compiler-aware handoff](pantograph_compiler_aware_application_integration.md),
using the compiler-aware REPL as `--old-repl`. No model tensor changes.

## Verification

Local, CPU, with the pinned client, the companion built from `d809a61` and the
pinned Mathlib `29dcec07`. Offline suite: 986 passed (two environmental failures also fail on the base). Native live suite: 54 passed; four fail identically on the base (no provisioned PeTTa or checkpoint artifacts). Not run here:
Docker image build, PLN-enabled native search, and the GPU trial on JupyterHub.
Re-running the PUCT/no-PLN configuration there is what will show how many of
the historical failures this repair removes and what the rest are.
