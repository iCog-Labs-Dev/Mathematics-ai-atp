# Source-native gold-action replay

First integration milestone: check recorded actions and their flat rendering
against actual Lean tactic snapshots. This runs no GNN, retrieval, training
or proof search, and does **not** measure complete-theorem proof success.

## What is checked

1. Select theorem/file pairs deterministically from the requested split,
   before checking cache availability or action support. Include every
   recorded row of each selected theorem; failed/unsupported rows remain
   in the report denominator.
2. Validate original source-invocation caches against dataset state/tactic/
   target hashes. Require normalized and v3 source-syntax sidecars bound to
   those exact raw records.
3. Compile original source files using the pinned frontend. Locate each
   invocation by its original unit/index and UTF-8 byte range.
4. Capture `GoalState` from `mctxBefore` and all active goals, running inside
   original `ContextInfo` (environment, namespace, open declarations and name
   generator). Gold control executes captured `Syntax`, preserving layout
   and hygiene. Source-text reparsing and rendered text execute as separate
   branches restored from that snapshot. No pretty-printed goal
   reconstruction or `sorry` insertion is used.
5. Compare normalized input target/local types, indices, roles and names
   with cached model inputs. Compare all ordered resulting goals with the
   original invocation, including local let values. This is normalized
   model-input/transition parity, not equality of complete kernel states.

Renderer supports already-resolved flat local/global references, with the
same three-reference budget as current decoder. List tactics get brackets
(`rw [h]`, `simp [h]`); term tactics keep application syntax (`exact f x`).
Public global names are fully qualified; locals resolve by context index.
Ambiguous/inaccessible locals, private/numeric declaration names requiring
additional resolution, unannotated leaves, compound syntax and fresh-name
arguments remain explicit unsupported cases. Proof placeholders are not
rendered as proof actions. This is not a full source-syntax renderer.

An identifier annotation can name only the head of a derived expression:
`FiniteDimensional.finrank_pos.ne'` has head `LT.lt.ne'`, but replacing the
whole expression with that head loses its proof receiver. Both renderers
conservatively reject declaration-spelling mismatches, possible local
receivers and local-name mismatches. Renamed aliases may also be rejected
until a native resolver is available. This necessary spelling check is not
a full elaborated-term equivalence check.

Optional `--renderer template-v1` extends gold parsing/rendering for bounded
rw/simp directions, `only` and locations, while retaining flat rendering for
other families. See [V1 grammar and typed gold targets](tactic_templates_v1.md).
Default remains `--renderer flat`; existing model prediction is unchanged.

`--renderer template-v2` additionally supports bounded reference applications
inside rw/simp rules. All modes use validated source-or-exact-global-identity
lookup. See [V2 grammar and paired recovery procedure](tactic_templates_v2.md).

## Backend and scope

Use existing extraction environment:

- Mathlib `29dcec074de168ac2bf835a77ef68bbe069194c5`.
- Pantograph `73781c2d58456e4bf369dadd4a501e1b78a0b177`.
- Lean `leanprover/lean4:v4.10.0-rc1`.

The pinned protocol exports source invocation data but no command to restore
an arbitrary recorded invocation. `lean/ReplayGold.lean` uses its existing
frontend/GoalState APIs directly; no backend fork modification is needed.
Both projects' `lake env` search paths are resolved before running the Lean
helper. Source files and cached traces are read only; temporary request/
response files are deleted automatically. Original-source compilation
errors invalidate that file's replay results.

An invocation's active goals can be only part of a larger proof. Even an
empty resulting goal array is **not** counted as a completed theorem.
Full-proof reconstruction/rechecking and live model inference are later
milestones. Existing hybrid search is not changed by this runner.

## First server run

Pull published code and run from repository root in `atp`:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_smoke \
  --split val --num-theorems 3 --seed 42 --file-timeout 600
```

No environment rebuild, new dataset extraction or model checkpoint is
needed if the pinned extraction environment and sidecars already exist.
Missing dependencies or wrong/dirty source pins fail explicitly.

Report is `OUTPUT_DIR/report.json`, checkpointed after each source file.
Repeat identical options with `--resume` after interruption. Resume binds
selected rows, source/cache hashes, backend pins, helper/runner/renderer
code and timeout. Use a new directory for a changed cohort or implementation.

Read final summary counts:

- `selected_steps`: every selected dataset row.
- `renderable_steps`: renderer produced a candidate string, before Lean checking.
- `flat_renderable_steps`: flat grammar produced text for the same rows under the recorded lookup policy.
- `template_v1_parsed_steps`: V1 typed target extracted before name rendering.
- `newly_renderable_vs_flat`: chosen mode produced text where flat mode could not.
- `input_model_match`: live normalized inputs agree with cached model inputs.
- `original_transition_match`: captured syntax reproduces the source invocation's normalized after-goals.
- `original_text_executed` / `original_text_transition_match`: separate reparsed-text diagnostic.
- `rendered_transition_match`: rendered gold references reproduce that source transition.
- `original_control_modes`: recorded native control policies; v2 requires `captured_syntax` for captured invocations.
- `source_alignment_kinds`: exact versus branch-opener/unknown cache alignment.
- `dataset_exact_alignment_steps`: exact alignments with matching cached dataset/source targets.
- `dataset_aligned_original_matches` / `dataset_aligned_rendered_matches`: restrict those alignments to matching inputs and successful source control; rendered count additionally requires rendered match.
- `status_counts`: cache, capture, input-parity, original/rendered failures or unsupported syntax.
- `proof_success_measured`: always false in this milestone.

Exit 0 means every selected row either matches or is explicitly unsupported
by the renderer after successful original replay. Exit 2 means operational,
cache, parity or replay failures; inspect per-row details. Neither is a
proof-search score. First evaluate this smoke run before increasing cohort.

Report v2 fixes gold control at `original_replay_mode=captured_syntax`.
Branch-opener cache entries may describe a complete source branch block,
not the dataset's one-line opener or its immediate target. Their source
replay is diagnostic; it is not counted as an exact dataset transition.
Per-row `source_tactic` exposes the actual captured fragment. Training
export needs exact action alignment or separately versioned full-source
targets; a source-control hit alone is not a dataset-action hit.

## Verification

**38 focused tests passed**, including a real pinned-backend integration
check against existing extraction API coordinates. Native cases cover
local/global names, rewrites, function application, let bindings, local
instances, preserving two sibling goals and independent branch restoration.
A later theorem is unavailable in an earlier source snapshot. Unit checks
cover unsupported references, deterministic selection, digest binding,
failure denominators and report resume.

Native integration can be repeated with:

```sh
MATHS_AI_PANTOGRAPH_ROOT=/path/to/pinned/Pantograph \
  python -m pytest maths_ai/gnn_inference/tests/test_gold_replay.py -q
```

Without that variable, the real-backend integration test is skipped while
unit tests still run. Initial flat-renderer server results follow.

The initial server three-theorem Mathlib smoke subsequently matched **7/7**
cached normalized inputs and **7/7** original transitions; flat rendering
matched **1/1** renderable case, with six explicitly unsupported rows. This
does not establish a proof rate. Subsequent flat/V1 checks totaled **50
passed** before the first V1 server cohort. Changing code/renderer requires
a fresh report directory rather than resuming that older report.

## V2 same-cohort recheck

The first 20-theorem V1 server replay had **86/86** input matches, **85/86**
reparsed original transitions and **26/27** rendered matches. All **13/13**
newly renderable cases passed. The failures were an original multi-line
fragment reparsed as one tactic and an old-flat derived method reference
whose receiver was discarded. Fixes were verified on pinned native fixtures
and then on the same Mathlib cohort with commit `1071146`.

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_template_v1_recheck \
  --renderer template-v1 --split val --num-theorems 20 --seed 42 --file-timeout 600
```

Keep the same theorem count/seed; use a fresh directory because control
semantics, schema and implementation hashes changed. Text-only failures
stay visible without invalidating a successful syntax control. Derived
reference cases become explicit unsupported cases, not invented bare
reference targets. Original prepared labels are unchanged; the observed
method case exposes a legacy flat-label quality limitation to audit before
future export. Reclassifying support is not a model accuracy gain.

Current focused verification: **53 passed**, including actual multi-line
syntax control versus failed text diagnostic, method-receiver rejection in
flat/V1 paths, source-only alignment counters and obsolete-control rejection.

The server recheck completed with **86/86** normalized input matches and
captured-syntax original transitions. Of these, **85/85** are exact dataset
alignments; one branch-opener row is source-only. Rendered transitions match
**26/26**, all exact-aligned, including **13/13** newly renderable cases.
Current flat renderability is **13**, V1 typed parsing **19**; **60** rows
are explicitly unsupported, with no cache/native/replay failures. Reparsed
text still matches only **85/86**, confirming why it is diagnostic rather
than gold control. Report:
`maths_ai/gnn_inference/runs/native_gold_replay_template_v1_recheck/report.json`.

This clears the small-cohort replay gate. Next measure exact full-prepared
train/val grammar coverage before target export or template-head training.
This result measures neither model prediction nor whole-theorem proof success.

V2 subsequently passed on the same 86-step cohort: **86/86** normalized
input/original matches, **85/85** exact-aligned originals and **30/30**
exact-aligned rendered transitions, with **56** unsupported and no failures.
Two applied-rule targets are among the **21** V2 parsed targets. See
[completed V2 recovery and native results](tactic_templates_v2.md).
