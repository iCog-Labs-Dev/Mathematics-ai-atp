# Dataset goal reconstruction (#58) — handoff, 2026-10-22

Operating details are in [dataset_goal_replay.md](dataset_goal_replay.md). This
note records what was built, the identities it was verified against and what is
still open.

## Inputs

| Input | Identity |
| --- | --- |
| Application base | `AC-branch` at `59148bbf495b0b0dc7fa303169c42c7c9633931e` |
| Client | `pantograph-async-client` 0.1.1, `123nol/Pantograph-Client@d1dac7e8a8ea296192e7a5b2616114b73e0812b7` (merge of PR #2, which adds `start_goal(levels=...)`) |
| Companion REPL | `jajos12/Pantograph@0dd5d222` plus the client's patch, binary sha256 `ffc2a91985137f9183c4cc1571c08ece182996484feecf44b5e5d22d932d7dc0` |
| Lean / Mathlib | `leanprover/lean4:v4.10.0-rc1`, Mathlib `29dcec074de168ac2bf835a77ef68bbe069194c5` |
| Dataset | `jajostrains/Mathlib-Normalized-Sexpr` (now `iCogLabs-Dev/Mathlib-Normalized-Sexpr`), revision `f44841f98193c09b20c6092198a22e98ff2cd7a8`, raw schema 4 |
| Raw serializer | `serializeExpressionSexp` in Pantograph `Serial.lean`; identical in the dataset producer `22ddfaaf` and the companion |

## Stages

| Stage | Done |
| --- | --- |
| 0 Evidence | Dataset revision pinned; compact original rows (0, 5, 7, 27, 41) and one synthetic `π` row frozen in `tests/fixtures/dataset_replay/rows.json`, each marked; producer grammar read from `Serial.lean` |
| 1 Client | `start_goal(expression, *, levels=())`, strict encoding, docs, 0.1.1; client suite 420 offline, 5 live (including a named-universe goal), sdist/wheel check and 427 from the sdist against the wheel |
| 2 Text | Lexical declaration parser shared by every `parse_state` caller; universe analysis by position; typed seeds; `levels` forwarded; sanitizer removed |
| 3 Structured | `DatasetRow.structured_seed`, checked raw decoder and explicit renderer, raw sexps kept on canonical goals, one renderer for initial and intermediate replay |
| 4 Admission | Preflight on a dedicated client, quarantine by reason, reports that are never overwritten, identity-keyed cache, inadmissible seeds counted apart from failed searches, `validate_rl_seeds.py` |
| 5 Verification | Results below |

## Results

Same first 500 train rows at the pinned revision, Mathlib session, request
timeout 60 s. Reports: [admission_train_500.json](../tests/fixtures/dataset_replay/evidence/admission_train_500.json)
(new path, written by `validate_rl_seeds.py`) and
[base_path_train_500.json](../tests/fixtures/dataset_replay/evidence/base_path_train_500.json)
(base path, a one-off comparison script, not part of the branch).

| | Base path (59148bb) | This branch |
| --- | --- | --- |
| Rows reaching Lean | 497 | 320 after quarantine and budget skips |
| Accepted by Lean | 8 | 320 |
| Checked against the source | no | 320 match the raw source; 319 also give the model exactly the source's normalized expressions |
| Lean rejections | 489 (279 unknown universe level, 197 parse, 11 unknown identifier, 2 other) | 0 |

319 rows are admitted. The other 181 are reported, not searched:

| Outcome | Rows |
| --- | --- |
| `unsupported-syntax`: private Mathlib constants, no source name | 51 |
| `missing-context`: `let` locals, schema 4 stores no value | 46 |
| `unresolved-universe` | 9 |
| `unresolved-term` | 3 |
| `model-mismatch`: metadata around a partial application (row 468) | 1 |
| skipped `over-size-replay` (> 50,000 characters) | 63 |
| skipped `replay-repl-failure` (REPL printed a heartbeat error) | 8 |

All 319 admitted goals printed the predicted local names (no name drift).
Admitted rows span 17 Mathlib areas (most from `Data`, `MeasureTheory`,
`RingTheory`, `Algebra`, `Topology`); every row over 1,600 printed characters
was quarantined, and the report lists coverage by area and size.

### Model input

[graphs_train_500.json](../tests/fixtures/dataset_replay/evidence/graphs_train_500.json)
(`scripts/compare_replay_graphs.py`) compares, for every admitted row, the
graph RL gives the model with the graph built from the row's stored model
columns, which is what the prepared data holds: **319 of 319 identical**, node
for node (labels, edges, roots, fingerprint). Without the model-identity
mapping only 31 would be, because 289 of the rows have context-index gaps from
hidden declarations and a replayed context is dense. Admission now enforces
this: a replay whose model expressions differ from the source's is rejected as
`model-mismatch` (row 468).

The mapping is carried to successor goals and through re-materialization, and
tactic arguments are translated back to live names; live tests cover gapped
rows (5, 7), the instance-heavy row 41, row 74 (its local `x` is renamed for
Lean because a nested `∀ x` refers to it, while the model keeps `x`), a successor with a new local, a
re-materialized successor, and a full search whose root and child nodes keep
the source identity.

Tactic arguments are resolved by model context label through a complete
label-to-live-declaration map, not by display name. The review case (outer `x`
aliased to `x_1` for Lean, then `intro x`) is covered offline for `Hyp` and
`FV` candidates through RL decoding and the inference resolver, after
re-materialization, and for shadowing twice, explicit renaming, removal and
alternative branches; live, `revert` of each resolved name removes exactly the
selected declaration, before and after re-materialization and on a second
branch. Offline, on 400 sampled rows, 20 of 348 renderable seeds
rename a local (21 locals in all), each to keep a nested binder's name, and
each keeps its source name for the model.

Existing graph and canonical goldens are unchanged, and no vocabulary or
checkpoint was modified. The check against the real prepared vocabulary and the
supervised checkpoint needs those artifacts, which are not on this machine:

```bash
PANTOGRAPH_TEST_SOURCE_ROOT=/path/to/mathlib4 \
PANTOGRAPH_TEST_REPL=/path/to/Pantograph/.lake/build/bin/repl \
PANTOGRAPH_TEST_PREPARED_ROOT=/path/to/prepared/sexpr_full_model \
PANTOGRAPH_TEST_CHECKPOINT=/path/to/best-v3.pt \
python -m pytest maths_ai/gnn_inference/tests/test_dataset_replay.py \
  --run-pantograph-live -k vocabulary_and_checkpoint

python -m maths_ai.gnn_inference.scripts.compare_replay_graphs \
  --source-root /path/to/mathlib4 --pantograph-repl /path/to/repl \
  --prepared-root /path/to/prepared/sexpr_full_model --report graphs.json
```

The test featurizes replayed fixture rows against the prepared vocabulary
(structural labels must all exist), requires their feature tensors to equal the
prepared rows' own, and runs the checkpoint on them. It was run here only
against a stand-in vocabulary and checkpoint built the same way, to check the
code path.

### Text parser

Printed text compared with the old parser on the 400 sampled source states:
329 identical, 22 where a multiline type had been split into extra locals,
27 where a `let` value had been folded into the type, 22 whitespace only.

Tests, run with GNU `stdbuf` first on `PATH`:

| Run | Result |
| --- | --- |
| `maths_ai` and `tests` | 413 passed, 9 skipped, 5 failed; the same 5 fail at the base (worker-count config test, `stdbuf` path test on this host, 3 Thompson-sampler tests) |
| Live with Mathlib (`--run-pantograph-live`) | `test_pantograph_client_live.py` 2 passed; `test_dataset_replay.py` 5 passed, 1 skipped without prepared artifacts (structured repairs and rejections, declaration resolution after intro, re-materialization and on a second branch, exact prepared graphs, successors and re-materialization, capture checks, a full search, universes, canonical, branch and cancelled replays, cleanup) |
| `rl_smoke` (live) | one on-policy optimizer step, finite losses |
| Translator `test_imports` | passed |
| Client suite | 425 with live tests (3 skipped without a companion checkout), sdist/wheel check, 427 from the sdist against the wheel |

The evidence run used the PR commit `455f8a3`; the merged pin `d1dac7e` adds only a CI
change, and the suites and live tests above were rerun with it installed from the pin.

## Decisions

- Universe names that the client can send are kept (`u_1` stays `u_1`); only
  names it cannot send get a `replay_uN` alias. The plan's example aliases every
  name, which would change every printed goal without changing what Lean
  checks; the mapping is recorded either way in `level_map`.
- Locals and printed names are written with their exact display names,
  quoted where needed (`«inst✝²»`), instead of being renamed. This keeps the
  names Lean shows, and it is what fixes the review's capture case: quoting
  changes no binder, so a nested `fun x_1` can no longer capture a reference.
  Finding a shadowed local by type (`‹Ring R›`) was tried first and rejected:
  on row 41 it left tactic metavariables Mathlib's unifier could not resolve
  within the heartbeat limit.
- When a nested binder and a local would collide, the local is renamed, not
  the binder, because binder names are part of the model S-expressions while a
  local's model name is kept by the mapping.
- Client callers that start fixed goals without universes (the client's own
  probes and example) keep the default `levels=()`; every application caller
  passes its levels explicitly.

## Model identity: alternatives considered

The model-identity mapping adds `execution_name` and `execution_index` to
`GoalLocal` and changes which identity the graph reads, so it needs an explicit
decision. The options were:

| Option | Effect | Status |
| --- | --- | --- |
| Map source model identity onto replayed goals (`model_identity.py`) | Prepared-data features unchanged: 319 of 319 admitted graphs identical to the prepared ones; tactics use live names | Implemented; needs approval and the prepared-artifact check |
| Accept Lean's dense indices and replayed names as the new model input | No mapping code, but `FV` labels shift on most rows (31 of 319 identical) | Would be a new feature contract: a separately reviewed prepared-data and checkpoint migration |
| Re-prepare the supervised data from replayed goals | Training and RL see the same dense contexts | Same migration, plus re-extracting every row; out of scope |
| Recreate the source's hidden declarations in the replay | Indices would match without a mapping | Not possible: the dataset does not record them |

## Validation limits

- Text seeds are checked for locals, roles and local definitions, not for the
  target expression: printed text has no raw form to compare against. Text
  replay renames nothing (it only quotes `✝` names), so the statement Lean
  elaborates is the input text read as Lean source. Dataset rows never use
  this path; they are validated against their raw expressions and, where
  recorded, their model expressions.
- The prepared-vocabulary and checkpoint check has only run against a stand-in
  vocabulary and checkpoint here; it has to be run where the real artifacts are.

## For the MCTS port

MCTS-AC-branch needs the same cutover, not a separate mechanism:

- the client pin with `levels`, and the `lean_replay` package;
- `prove(seed)` as the only entry point, so `SearchFactory` and the pool pass
  `TheoremItem.seed`;
- MCTS node re-materialization through `materialize_seed(goal.replay_seed())`
  (it replays canonical goals the same way `_materialize_goal` does here), and
  `carry_model_identity` on every successor so the model keeps the source
  identity;
- pool admission through `seed_admission`, and the `inadmissible` outcome kept
  out of `N`/`W`, HTPS data and the anneal clocks, as it is kept out of the
  solve rate here.

## Known, unrelated

Visible and unchanged: the worker-count config test (expects one worker, the
config has four), the `stdbuf` path test on hosts where `/usr/bin/stdbuf` is
uutils, three Thompson-sampler tests, and the stale `tests/test_protocol.py`
digest in the client's extraction manifest.

## Open

- Raw schema 4 records no `let` values, so rows with local definitions are
  quarantined (about one in ten of the sampled rows). Recording `value_sexp`
  in `local_context` or `raw_hyp_sexps` would admit them.
- About one row in ten refers to a private Mathlib constant
  (`_private.….0.…`), which no source name can reach; those are quarantined
  as `unsupported-syntax`.
- Universe level metavariables (`(:mv …)` in a sort) have no assignment in
  the dataset; those rows stay quarantined.
- Run the prepared-vocabulary and checkpoint test above where those artifacts
  are (JupyterHub).
- The companion REPL prints an uncaught Lean exception (for example a
  heartbeat timeout) as plain text on stdout instead of a JSON error; the
  client then closes the session as a protocol failure. This is a REPL fix.
- Very large raw goals (row 16 is 408 KB) can exceed the request timeout;
  admission skips them as `replay-timeout`.
- MCTS-AC-branch adopts this in a later port.
