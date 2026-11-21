# Dataset goal reconstruction on MCTS-AC-branch

This is the MCTS port of the dataset goal reconstruction merged into AC-branch
by PR #60. The design and the shared rules are in
[dataset_goal_replay.md](dataset_goal_replay.md); AC's own results are in
[dataset_goal_replay_handoff.md](dataset_goal_replay_handoff.md). This file
covers what changed on MCTS and what was verified here.

## Identities

| | |
| --- | --- |
| Target base | MCTS-AC-branch `61f6d38` (walkthrough update after the PR #59 merge `f21d044`) |
| Application source | AC-branch `54d7cef` (merge of PR #60); reviewed feature commit `70f7b8d` |
| Client | `123nol/Pantograph-Client@d1dac7e`, version 0.1.1 (`start_goal(..., levels=...)`) |
| Dataset | `jajostrains/Mathlib-Normalized-Sexpr` at `f44841f9` |
| Mathlib | `29dcec07`, `lake exe cache get` |
| REPL | companion REPL built with `lake build repl` on `leanprover/lean4:v4.10.0-rc1`, sha256 `ffc2a919…` |

Every copied and adapted file is listed with its AC-source, MCTS-base and
current hashes in
[source_manifest.json](../tests/fixtures/dataset_replay/mcts_port/source_manifest.json)
(39 copied, 13 adapted, 9 MCTS-only). `test_owned_client_migration.py` checks
that chain on top of the two earlier MCTS client-port manifests, which are
unchanged.

## What changed on MCTS

The port is selective: files identical on both branches before #60 take the
merged version, and the rest re-apply #60's change on the MCTS code. The
search algorithm, PUCT selection, visit statistics, HTPS mining and the
optimizer order are unchanged.

- **One typed input.** `HybridReasoner.prove(seed, *, deadline=None)` and
  `RLHybridReasoner.prove(seed, *, greedy=False, deadline=None)` take a
  `LeanGoalSeed`; the goal/hypothesis string API is gone from every caller,
  script and fake. `_materialize_goal` calls the shared `materialize_seed` and
  wraps the result in the existing `MaterializedGoal`, so the root and every
  replayed node use the same checked path. The fingerprint is taken from the
  model-facing goal, after its source model identity is restored and checked.
  `_sanitize_replay_spec` is removed.
- **Successor identity.** `_execute_and_link`, which both search modes share,
  carries the materialized parent's model identity onto every successor before
  it is ranked, linked or compared.
- **Arguments.** In `_predict_next_tactics_batch` each row builds its own
  `local_declarations(goal)` map next to its DAG, pending owner and fingerprint,
  and `_decode_row` resolves `Hyp` and free-variable candidates only through
  that row's map.
- **Failures.** A root that cannot be replayed raises `ReplayInputError`
  before any graph exists, so it leaves no visits, failure records or
  samples. A descendant that cannot be replayed is closed as an elaboration
  error with UNKNOWN backup, as before: the traversal can count in `N`, never
  in `N_v` or `W`. Protocol, state and transport failures are unchanged.
- **Admission and accounting.** The theorem pool is built by
  `seed_admission.preflight` on a dedicated client. Both the single-reasoner
  path and the pool record an `inadmissible` outcome (`seeds_inadmissible`),
  separate from failed searches; the pool keeps the worker, since its client is
  still usable. Solve rates count searched seeds only, and a round that
  searched nothing leaves the curriculum window unchanged. AC still appends a
  zero for such a round; that needs the same fix there.
- **Learning.** No sample is mined from an inadmissible root. HTPS steps still
  run from replay entries validated in earlier rounds, so a round without a
  usable root can still update the model (tested).

## Results

All from this branch at `c59ad2e` with a clean worktree. Reports and their
commands are in [mcts_port/](../tests/fixtures/dataset_replay/mcts_port/)
(`provenance.json`). That commit was on `f21d044`; the branch was then rebased
onto `61f6d38`, which changes only `rl_process_walkthrough.md`, so the code
the reports cover is unchanged.

| Check | Result |
| --- | --- |
| Offline suite | 514 passed, 12 skipped (also through `uv run --locked --no-sync`); the 4 failures also fail on `f21d044` (GNU `stdbuf` path, three Thompson-sampler tests) |
| Live tests (`--run-pantograph-live`) | 35 passed, 1 skipped: the prepared-vocabulary/checkpoint gate |
| `mcts_client_smoke` | PUCT multi-leaf batches, checked rejection of alternatives, graph equivalence in both modes, cooperative-deadline scope closure, 20 repeated searches, two workers sharing one model device; learning rounds with HTPS off (1 step, generation 1) and on (2 steps, generation 2) |
| Admission, `train[0:500]` | 391 candidates, **319 Lean-validated**; quarantined: missing-context 46, unsupported-syntax 51, unresolved-universe 9, unresolved-term 3, model-mismatch 1; skipped: over-size 63, REPL failure 8 |
| Root graphs vs stored graphs | **319 of 319 identical**; 31 without the model-identity mapping |

The admission and graph numbers equal AC's for the same rows, but they were
produced here, not copied.

`scripts/rl_smoke.py --no-pln` passes in both selection modes (`legacy` and
`puct` with 8 simulations, batch 2): one on-policy step, no structural unknown
labels. Against `f21d044` on the same REPL its root expansion is identical
(same tactics and probabilities); the run differs from node 1 on, by design.
After `intro p` Lean shows the outer `p` as `p✝`. The removed sanitizer
renamed it to `p_` when replaying node 1, so the model saw a state Lean never
produced and proposed `intro p_`. Node 1 now keeps the model identity `p✝`.

## Resume and checkpoints

Model weights, vocabularies and their tensor schema are unchanged, so existing
checkpoints remain warm starts. Replay queues saved before the port still load
and featurize to the same graphs, because every new goal field is optional.
Those goals carry no raw execution expressions, so re-materializing one raises
`missing-context`; search never replays queue entries, so this does not affect
resume. The resume schema check and saved search settings are unchanged.

## Not done here

- The prepared-vocabulary/checkpoint gate (`test_dataset_replay.py`, needs
  `PANTOGRAPH_TEST_PREPARED_ROOT` and `PANTOGRAPH_TEST_CHECKPOINT`) is blocked
  on the real artifacts, which are on JupyterHub. It has not passed.
- No long checkpoint-backed RL run.
- The shared limitations listed for AC carry over unchanged: model-context
  metadata is not fully cross-validated, constant-application placeholders are
  accepted more widely than the producer contract, the validation cache identity
  is incomplete (keep it off), the admission deadline is checked only between
  candidates, malformed rows can escape quarantine, `validate_rl_seeds.py`
  exits 0 with no admitted seed, and text input does not cover explicit binder
  forms or multiline hypothesis lists. Each needs a shared fix ported to both
  branches.
- AC's zero-round curriculum update (see above).
