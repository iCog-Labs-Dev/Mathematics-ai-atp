# Recursion-depth exhaustion as a recoverable resource

Issue #91. Lean's `maximum recursion depth has been reached` used to end the
Pantograph session: the REPL answered it as an unexpected internal error, the
client raised `PantographRemoteInternalError`, and search abandoned the theorem
and replaced the worker. In the post-#88 PUCT miniF2F run this was 49 of the 60
archived failures (`norm_cast` 39, `trivial` 6, `rfl` 2, `decide` 1,
`apply Finset.sum_range_succ_comm` 1).

The limit says nothing about whether the tactic is valid or the theorem
provable, so it is now handled like heartbeat exhaustion: the command is rolled
back and search tries the other candidates.

## Runtime pair

- REPL `123nol/Pantograph` `91dbc0fb8b0af2d879abc51517693ec134343b33`: the phase
  boundary recognizes recursion exhaustion with Lean's
  `Exception.isMaxRecDepth` and reports it as a resource error,
  `kind: "recursion-depth"`, with the command's configured `maxRecDepth` as its
  limit. The transaction discards the staged state, so the answer is
  `command-rolled-back` (`frontend.process` stays `session-unusable`), and
  certification that exhausts it is `inconclusive`.
- Client `123nol/Pantograph-Client` `189a7b1ade1d1df88d3bab2108932d3f1297ba88`
  (0.7.0): resource details version 2, kinds `heartbeats` and
  `recursion-depth`. Both raise `PantographResourceError` after a verified
  rollback and `PantographSessionResourceError` otherwise. A companion that
  reports resource version 1 fails at startup.

## Application changes

- **Pins:** the client in `pyproject.toml`, `uv.lock` (client entry only;
  `uv lock --locked` passes), `requirements-mcts-ac.txt`, the diagnostics and
  conda setup, and the REPL in Docker.
- **Search, replay, admission and collection** already settle every resource
  outcome through `ResourceDiagnostic` and `CommandOutcome`, independent of
  the kind, so a recursion-depth failure reaches the same path as a heartbeat
  one: the input state is kept, the attempt counts against the existing
  action budgets, other candidates run, and the outcome is never a tactic
  rejection, a logical conflict or a learning target. The tests for these
  paths now run for both kinds.
- **API:** the response schema accepts `kind: "recursion-depth"`.

Training objectives, checkpoint format, evaluation denominators and the
recorded failure archives are unchanged.

## Update JupyterHub

```bash
conda activate atp-mcts
python -m pip install --upgrade --no-deps \
  'pantograph-async-client @ git+https://github.com/123nol/Pantograph-Client@189a7b1ade1d1df88d3bab2108932d3f1297ba88'
git clone --no-checkout https://github.com/123nol/Pantograph.git /home/jovyan/Pantograph-recursion-depth
git -C /home/jovyan/Pantograph-recursion-depth checkout --detach 91dbc0fb8b0af2d879abc51517693ec134343b33
cd /home/jovyan/Pantograph-recursion-depth && lake build repl && cd -
```

The REPL changes, so rebind the active deployment with
`rebind_source_aware_environment` (`--old-repl` the failure-diagnostics REPL
for an inventory that records only the executable's bytes). Declarations and
model tensors are unchanged; no retraining is implied. Published releases stay
as they are; a release for this runtime is published as a new version.

## Verification

Local, CPU, with the pinned client, the companion built from `91dbc0f` and the
pinned Mathlib `29dcec07`:

- **Offline:** the resource-integration tests run for both kinds, including
  continued search after a failed candidate in best-first and PUCT, failed
  post-tactic inspection, replay that tries another recipe without a conflict
  memo, RL aliases settled once without rejection records, admission skips,
  and collection that keeps resource metadata and replaces only unusable
  sessions.
- **Live:** in both search modes, `decide` on
  `(List.replicate 2000 0).length = 2000` exhausts the recursion depth, is
  recorded as a `recursion-depth` resource outcome with `command-rolled-back`,
  and the next candidate (`exact List.length_replicate 2000 0`) is
  kernel-certified on the same retained state in the same session, with no
  allocations left.

Not run here: replaying the archived miniF2F bundles (they need their recorded
runtime), PLN-enabled native search, and the GPU run on JupyterHub. The 11
`fun_prop` timeouts are a separate investigation.
