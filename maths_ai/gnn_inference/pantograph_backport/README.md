# Repository-owned client: Stage 0 source and contract basis

Recorded on 2026-10-04, on `AC-branch` at
`216e074da75d8d5ca3011fd40146fd200f6a9715`. Only Stages 0 and 1 of
[the client plan](../docs/dev_plans/repository_owned_pantograph_async_client.md)
are implemented. No production factory is switched, no dependency is removed,
and no Lean validation patch exists yet. Do not launch the proposed client for RL.

## Verified source basis

The companion patch base is `https://github.com/jajos12/Pantograph.git` at
`73781c2d58456e4bf369dadd4a501e1b78a0b177`. It was checked out separately in
`/tmp/pantograph-owned-client-base`; the worktree was clean before capture.
Every tracked source/build file matches the user's downloaded archive at
`maths_ai/sexpr_environment/sexpr_environment/Pantograph`. The only difference
is the archive's omitted `.gitignore`, which is not a build input. Hashes of
all compared files are in
[provenance.json](../tests/fixtures/pantograph/baseline/provenance.json).

The verified checkout contains the required `modelSexp`, stable free-variable
context indices, instance and let metadata, and structured action-expression
extensions. There is no unresolved source mismatch for the companion patch.
The downloaded archive was not edited, checked out, or passed through setup.

Both local projects declare `leanprover/lean4:v4.10.0-rc1`. Lake reports Lean
4.10.0-rc1, compiler revision `3b58e0649156`. The deployment Mathlib pin remains
`29dcec074de168ac2bf835a77ef68bbe069194c5`. The downloaded Mathlib archive has no
Git metadata; its commit is an expected setup pin, not a locally verified HEAD.
It imports successfully in the live baseline captures. Reproducible release setup
must obtain the pinned Mathlib checkout rather than infer its revision from the
archive directory name.

`lake build repl` and `lake test` passed in the verified Pantograph checkout.
The existing archive's `.lake` build/dependency cache was copied to this temporary
checkout before these commands. This is a diagnostic baseline build, not the
fresh patched release build required by Stage 7. The captured executable SHA-256
is `511e32d79635a4fe55f711f6fc1d42c6d5044203829b57eb37ab72e8227570c3`.
The existing downloaded executable has the same observed hash.
Source equality alone does not establish the origin of a preexisting executable.
The recorded interpreter is `.venv/bin/python`, Python 3.11.15. Installed
PyPantograph is 0.3.15 at `2c9fad2727c405c9976b766088ca761f14733f95`.
The capture utility uses no PyPantograph imports or transport.

Existing changes, including `configs/rl_actor_critic.json`, were preserved.
The full worktree status at capture is recorded in the provenance descriptor.
No source patch is shipped in this stage. Stage 4 will create the single
`repository_client_contract.patch` over the verified base; it must not be
combined with the alternative modern-client backport.

## Evidence and capture procedure

[baseline.json](../tests/fixtures/pantograph/baseline/baseline.json) contains
31 raw request/response pairs, including original JSON response lines. Captures
cover readiness/options, identity proof, tactic and parse rejection followed by
recovery, two conjunction goals, explicit selection of goal index 1, local
instances and lets, admission, logged error, invalid identifiers, and deletion.
Both the initial and final `stat` responses report zero allocated states.

Deleting the identity proof's parent did not invalidate the introduced child.
Two alternative tactics independently succeeded from that retained child.
With two conjunction goals, `all_goals trivial` applied to explicit goal index 0
left one sibling. The owned API will preserve this focused selection contract.

The current executable does not implement `protocol.describe`. Both `sorry`
and `run_tac Lean.logError "capture error" <;> trivial` returned an empty goal list
without validation or messages. All baseline tactic responses are negative
inputs for the final decoder, even ordinary successful ones. The authored
[future_contract.json](../tests/fixtures/pantograph/future_contract.json) adds
expected descriptor/evidence/diagnostics for pure tests. These are not live
results from a patched executable.

After checking out the pinned base separately and building it, capture from the
repository root using a new output directory:

```bash
.venv/bin/python maths_ai/gnn_inference/scripts/capture_pantograph_fixtures.py \
  --source-root maths_ai/sexpr_environment/sexpr_environment/mathlib4 \
  --repl-source maths_ai/sexpr_environment/sexpr_environment/Pantograph \
  --verified-base /tmp/pantograph-owned-client-base \
  --repl /tmp/pantograph-owned-client-base/.lake/build/bin/repl \
  --output /tmp/pantograph-baseline-new
```

The utility refuses existing output directories, dirty/mismatched base sources,
source differences other than the omitted ignore file, and toolchain mismatch.
It resolves `LEAN_PATH` through Lake, launches `stdbuf -oL`, records exact wire
lines, and terminates/drains/reaps its diagnostic subprocess. It is not the Stage
2 production transport and is never imported by application code.

## Frozen wire contract for the companion extension

The machine-readable contract is [contract.json](contract.json). The pure codec
is `maths_ai/hybrid_reasoner/pantograph_protocol.py`. It supports one schema,
not baseline and modern-client modes.

Supported commands are `protocol.describe`, `options.set`, `options.print`,
`goal.start`, `goal.tactic`, `goal.delete`, and `stat`. Requests are compact UTF-8
JSON after the command and a space, terminated by exactly one newline. Tactic
requests always include a nonnegative integer `goalId`. Booleans are not IDs.
The numeric index selects an ordered execution goal, not its metavariable name.
The selected goal is focused; automatic mode resumes unresolved siblings.
The input allocation remains usable for alternative tactics.

Required options are `printJsonPretty=false`, `printExprPretty=true`,
`printExprAST=true`, `printExprModelAST=true`, `noRepeat=false`, and
`automaticMode=true`. The remaining three current options are typed Boolean
settings, not schema-selection switches. Every field of `options.print` is
required. `options.set` and `goal.delete` acknowledge with exactly `{}`.

The descriptor must match every field in `contract.json`. Successful tactics
have exactly `nextStateId`, `goals`, `validation`, and `messages`. The validation
version is 1 and the scope is `transition-expressions`. `checked=true` means the
checks ran. `hasSorry` and `hasUnsafe` are required Boolean results, not inferred
from the tactic text or empty goal list. Positive flags retain a decoded result
allocation for the future scope owner to delete before raising a domain error.
Missing evidence, a different scope/version, or `checked=false` is fatal.

Tactic rejection has either `parseError` (nonempty string) or `tacticErrors`
(nonempty string array), plus required `messages`. No success/failure indicators
may be mixed. Structured messages contain `severity` and rendered `data`.
Severity is exactly `information`, `warning`, or `error`. Optional `pos`,
`endPos`, `fileName`, and `kind` describe real diagnostics; missing positions are
omitted, not fabricated. Positions use one-based `line` and zero-based `column`.
An end position requires an earlier or equal start position. Optional fields are
omitted when unavailable, not populated with `null`. A success carrying an error
message is a protocol violation. No unknown additive fields are permitted in
version 1 beyond the explicitly named expression/message fields.

Command errors have exactly `error` and `desc`, both strings. Categories are
mapped by command in `contract.json`, based on `Repl.lean` and `Library.lean`.
For `goal.start`, `parsing` and `elab` become `PantographGoalRejected`.
For `goal.tactic`, `index` becomes `PantographStateError`; it is not an ordinary
failed policy action. `command`, `arguments`, and unsupported operation `invalid`
are protocol defects for this minimal typed API. The companion `validation`
category is a fatal failure to produce required evidence. Unknown categories
are fatal rather than mapped to a failed tactic.

Expressions require nonempty `pp`, raw `sexp`, exact `modelSexp`, and integer
`modelSexpVersion=1`. Optional `dependentMVars` is an array of names. No pretty
text is converted into model expressions. Goal `userName` is an optional case
tag; the baseline omits it for anonymous goals. `isConversion` is required and
preserved. Conversion search support is not introduced in these stages.

Locals require `name`, `userName`, `contextIndex`, `binderRole`, `isInstance`,
`isLet`, `isInaccessible`, and a complete `type`. Only local lets carry `value`,
and that value must be complete. Binder roles are `:explicit`, `:implicit`,
`:strict-implicit`, `:instance-implicit`, and `:let`. Instance/let flags must agree
with those roles. Local indices are unique and increasing; gaps are valid and
never renumbered. Raw names and raw expressions remain separate from canonical
free-variable labels such as `FV0`, which denotes context position 0.

The transition-validation boundary remains the plan's selected/coupled/reachable
changed assignments, relevant introduced dependencies/local values, and retained
obligations. Lean implementation must reject logged errors, restore failed
elaborator/environment state, detect hidden synthetic holes, and report missing
or indeterminate checks as errors. Pure fixtures do not implement those checks.
Imported Mathlib remains the fixed trust basis. Graph-based search success is
preserved; this does not certify an assembled complete theorem proof.

## Shared types and projections

Immutable `StateHandle` identifies session, lexical scope, and Lean allocation.
`StartedGoal` is an unmaterialized root and has no serialized goal list.
`TacticState` contains ordered execution goals, diagnostics, and transition
evidence. Wire results carry IDs separately until the future scope owner records
the allocation. The codec performs no allocation or deletion itself.

`ExecutionGoal.goal` and `execution_goal_to_goal` return fresh canonical `Goal`
and `GoalLocal` copies. The execution snapshot contains only immutable values.
Consumers can retain or mutate canonical copies without changing saved wire
data. `execution_state_to_goals` projects a materialized result. The new
`execution_goal_to_translator` projection explicitly contains `goal_name`,
`case_tag`, `is_conversion`, a target expression, and ordered `locals`. Expression
keys are `pp`, `sexp`, `model_sexp`, and `model_sexp_version`. Local keys are the
corresponding explicit metadata plus complete `type` and optional `value`.
Translator consumers will migrate to these names in Stage 6; there is no fake
upstream-shaped compatibility object.

Canonical fields, state fingerprints, DAG fingerprints, node features, roots,
and edges are frozen for identity, instance, let, and conjunction fixtures in
[canonical_graph_expectations.json](../tests/fixtures/pantograph/canonical_graph_expectations.json).
These expected canonical objects were independently assembled from the captured
wire data using the existing canonical models and graph builder.

## Caller and event-loop inventory

An entire-repository search of Python and notebook files found the following
live operation boundaries. No repository notebook files were found outside the
downloaded environment/dependencies. Remote JupyterHub notebooks are not locally
available and are a deployment-stage check, not an assumed migration success.

| Caller | Current operations and required migration |
| --- | --- |
| `hybrid_reasoner/pantograph_env.py` | Deferred upstream creation, Lake resolution, readiness/options and private close; Stage 2 owns these, Stage 6 switches the factory |
| `hybrid_reasoner/pantograph_model_sexpr.py` | Global parser patch, start/intro probe, canonical conversion; Stage 6 uses this codec and scoped probe |
| `hybrid_reasoner/joint_inference.py` | Start/replay/intro/skip, candidate tactics, cached state, internal restart, terminal `asyncio.run`; use theorem scopes and outer session replacement |
| `hybrid_reasoner/hypergraph.py` | Upstream types in executor interfaces, not an additional wire command; migrate interface types |
| `atp_lean_gnn/rl_reasoner.py` | Shared executor/search lifetime and action records; serialize overlapping search on one reasoner and propagate infrastructure failures |
| `atp_lean_gnn/rl_live_collection.py` | Worker startup, private close via threads, worker replacement; own awaited close/replacement on the collection loop |
| `atp_lean_gnn/rl_training_driver.py` | Factory/provenance and training/evaluation terminal loops; maintain one client-owner loop through collection |
| `atp_lean_gnn/pln_rl_training.py` | Concurrent single-reasoner helper and per-call `asyncio.run`; make live collection wrappers async and serialize same-reasoner searches |
| `gnn_inference/model.py` | Sync prediction creates a REPL, starts/skips, invokes nested `asyncio.run`; move materialization to a scoped async caller and make prediction canonical-only |
| `gnn_inference/inference_engine.py`, `atp_lean_gnn/inference.py` | Pass expression text or upstream states through inference; pass full canonical goals, preserving tensor gradient guards |
| `scripts/rl_smoke.py` | Start/intro plus collection/update on a top-level loop; scoped ownership and awaited cleanup |
| Translator `translator_modules/cli.py`, `extractor.py`, `parser.py`, package exports and `__main__.py` | Synchronous bundled server, start/tactic replay, generic dataclass projection; use explicit environment, async replay and own translator projection, validate before output writes |
| `scripts/setup_sexpr_environment.py` | Git pins/build/stdout buffering, existing-patch restore behavior; reproduce one verified patch and refuse unknown dirty sources in Stage 7 |
| Tests `model_helpers.py`, `test_model_sexpr_contract.py`, `test_pantograph_env.py`, `test_tactic_rendering.py`, `test_rl_reasoner.py`, `test_rl_training_driver.py`, `test_rl_live_collection.py` | Upstream parsers/errors or duck-typed async fakes; migrate together with production callers in Stage 6 |
| `pyproject.toml`, `uv.lock` | Upstream dependency and Git source; remove after atomic caller migration in Stage 7 |

The current callers need no environment mutation, save/load, or conversion
commands. Translator needs raw serialized expressions, not a separate wire API.
`stat` and deletion support diagnostics and explicit scopes. Readiness is a
transport event, not an additional command. The minimal proposed typed API is
sufficient for these callers.

Package initializers in `hybrid_reasoner` and `atp_lean_gnn` now resolve their
existing public exports lazily. This fixes the shared import boundary: loading
the protocol and graph contract no longer loads search, training, Torch, or
PyPantograph. No production backend-selection switch or renamed caller alias was
added. All exported names remain available.

## Stage gates and verification

Stage 0 source basis, operation inventory, schema omission rules, and final
contract are resolved above. Stage 1 provides the standalone types, typed errors,
strict JSON/request/response codec, canonical copies, and translator projection.
The following test command launches no Lean process:

```bash
.venv/bin/python -m pytest maths_ai/gnn_inference/tests/test_pantograph_protocol.py -q
```

Verification results:

- Pure codec suite: 159 passed.
- Codec plus existing model-expression/environment regression suites: 187 passed.
- Verified unmodified Lean checkout: `lake build repl` and `lake test` passed.
- Raw baseline: 31 exchanges, with allocation count restored from zero to zero.
- Formatting and `git diff --check` passed. Formatting used Black's single-process
  library entry point after its sandboxed process-pool invocation stalled.
- Broad GNN regression command:
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest maths_ai/gnn_inference/tests -q --durations=5`.
  Outside the sandbox it completed with 421 passed, 1 skipped (two-GPU topology),
  and 1 unrelated failure. `test_repository_config_omits_removed_resource_fields`
  expects `collection_workers == 1`; both baseline HEAD and the preserved current
  config set it to 4. Neither that test nor the user's configuration was changed.
  Sandboxed broader runs stalled in the existing worker-pool threaded cleanup;
  they were interrupted, not counted as passing full-suite runs.

A fresh-interpreter test actively forbids importing PyPantograph, Torch, and
model/training/search implementations. Baseline tactic responses lacking evidence
are rejected. Existing application factories and imports remain unchanged.
Stages 2–8 are not implemented. The next stage is owned subprocess transport;
actual transition-validation evidence depends on Stage 4 and live verification
on Stage 5. No full RL, GPU, checkpoint, Conda, or remote deployment run was made.
