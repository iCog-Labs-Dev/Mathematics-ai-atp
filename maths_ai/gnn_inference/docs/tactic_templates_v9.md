# V9: combined structural source programs

This batch uses the complete user-supplied V7 failure atlas rather than another
small feature-specific audit. Input: `v7_failure_atlas_20261109_085950.json`,
**51,412 train / 966 val exclusions**, grouped into **1,984 family/message groups**.
Those groups reduce to **21 top-level error codes**:

| Shared blocker | Train + val exclusions |
|---|---:|
| Flat grammar/modifier exclusions | 21,197 |
| Unsupported recursive syntax | 14,570 |
| Non-bare/unannotated references | 7,434 |
| Unsupported nested proof body | 2,324 |
| Multi-tactic wrapper | 2,255 |
| Invalid local-context trace (quarantined) | 840 |
| Other shape/binder/configuration/budget issues | 3,758 |

These are observed first blockers, not guaranteed recoveries. V9 combines V8's
math grammar with a bounded structural program codec for the larger declaration,
proof-block, combinator, alias and tactic-family gaps.

## Structural program representation

Select `--renderer template-v9`. Existing V8 typed targets are retained whenever
they render. Otherwise `source_program_tactic_v1` encodes the complete validated
syntax tree into structural nodes and typed leaves. Each node retains its syntax
constructor and child arity. Token adjacency is serialized without absolute file
positions. Only literal tokens and identifier leaves supply emitted text.

**Parent source fragments are never emitted as a proof-text fallback.** A separate
surface check verifies that the source leaves cover the complete captured action,
ignoring comments/formatting while preserving quoted literal contents. Missing
surface data, omitted tokens, truncated diagnostic previews and incompatible
choices remain failures.

The codec reconstructs layout for tactic sequences, nested `by` proofs, `<;>`,
`first` alternatives, case/match alternatives, bullets, calculations, structure
fields, `do` blocks and conversion sequences. Ordinary declarations and syntax
forms retain their structured token/child representation; they do not require a
separate hand-written parser for each tactic family.

Reference and name handling remains explicit:

- Input reference indices must exist in the validated original context.
- Direct bare input references render from their chosen context identity.
- Safe bare global references can render from their full declaration identity.
- Private aliases, derived names and method spellings retain their complete name
  syntax plus head annotations, preserving written receivers and implicit context.
- Scoped names and parameter/field labels remain name labels, not input premises.
- Unannotated names that cannot be grounded remain **deferred identifiers**, with
  no invented global/context identity. This is necessary for programs such as
  `first | exact missingName | exact hp`, whose first branch intentionally fails.

Deferred-name counts are reported separately. Their presence does not mean a
program is invalid, and their absence does not prove a complete semantic premise
dependency list. This is a broader source-program generation target, not proof
that the current flat argument decoder can predict these programs. Existing model
weights, prepared labels and serving defaults are unchanged.

Program targets record `requires_source_scope: true`: source-relative aliases,
notation and namespace lookup require the original frontend scope used in native
gold replay. No portability to a different source scope is claimed. All generated
programs still require native execution/goal validation. Static representation
support is distinct from model accuracy and whole-theorem proof success.

Bounds: **4,096 syntax nodes, depth 64, 4,096 characters per source leaf**. Explicit
`sorry`/`admit`, sorry-axiom references, missing syntax and unsafe source names are
rejected. Syntax quotation/antiquotation remains a distinct unsupported codec;
rewriting its syntax tree as ordinary executable tactics would change its meaning.
Invalid input traces stay quarantined.

## Replay context repairs

Native tests uncovered two replay-driver gaps, repaired without weakening checks:

1. Restore `invocation.ctx.parentDecl?` into the term elaboration context for all
   three branches. Pattern `let`/match elaboration creates auxiliary declarations
   and previously failed even for the captured original when the parent was absent.
2. Parse source/rendered text in the `conv` category when the captured main goal
   is a conversion goal. Captured syntax remains the control. Ordinary goals use
   the tactic category. This fixes inner conversion-step reparsing.

Original controls, input parity and all ordered normalized resulting goals remain
required. Preserve historical reports; helper/implementation hashes invalidate
native resume across this change.

## One combined coverage audit, directly against saved V7

An intermediate V8 full scan is not required. From the server repository root:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_selector_recovery \
  --renderer template-v9 \
  --previous-recovery-dir maths_ai/gnn_inference/runs/template_v7_recovery_20261108_113947 \
  --output-dir maths_ai/gnn_inference/runs/template_v9_recovery \
  --pilot-actions 24
```

This compares the combined implementation against **167,315 / 3,191** saved V7
support on **218,727 / 4,157** selected rows. It uses original raw/model/V3 caches,
does not load V2 sidecars or compile Lean files, and does not change old artifacts.
The same runner can also compare V9 against a completed V8 report if one exists.

Console prints brief progress and one table. V9 adds `program_gains` and
`deferred_names` columns to distinguish the new structural mode and the subset
containing deferred identifier resolution. Detailed metrics include:

- `template_v9_renderable_steps`, `v9_rescues_vs_v7`, `v9_losses_vs_v7` (or `vs_v8`).
- Exact-aligned support/rescues/losses; nonexact gains remain source-only.
- `source_program_steps`, `source_program_rescues`.
- `deferred_identifier_steps`, `deferred_identifier_rescues`.
- `no_deferred_identifier_program_rescues` (not a claim of universal grounding).
- Existing math/selector features, which can overlap the source-repair categories.

The report records schema `program_recovery_v1`. Static checkpoints do not resume;
use a fresh directory and let an advancing scan finish. No native job starts
automatically. A balanced new exact-val ID selection is written for a subsequent
native check using `--renderer template-v9` and a fresh native output directory.

## Verification and remaining uncertainty

The entire atlas's groups and constructor inventories were examined. After
normalizing diagnostic previews' omitted empty-child arrays, **2,201 complete
nonquarantined previews encoded/rendered structurally**. **10** previews were
truncated, **1** lacked part of its input context, and **4** contained syntax
quotation requiring a dedicated codec. These are representative-preview results,
not weighted row recovery or native validity of the whole population.

**200 focused regression tests pass**. New fixture checks **31** captured-original
and rendered all-goal transitions, covering declarations, nested proof programs,
parameterized helpers, instance declarations, replacement, both suffices forms,
pattern lets/lambdas, records, show/calc, combinators, alternatives, private/global
aliases, core induction/cases, configs, conversions, literals and `do` expressions.
It also verifies inner-conversion original-text execution. Existing V5–V8 native
fixtures and decoder/binding regressions remain covered.

Completed V9 static gains are recorded below; real-Mathlib native delta results
remain pending.
The 840 quarantined records cannot be made valid by inventing missing input
references. Preserve explicit residual/error categories and validate the combined
stage before supervision export or model training.

## Completed full-cohort server coverage

User-supplied final audit table, after all **218,727 train / 4,157 val** rows:

| Static representation support/gain | Train | Val |
|---|---:|---:|
| Saved V7 support | 167,315 | 3,191 |
| V9 support | **217,635** | **4,143** |
| V9 coverage | **99.50%** | **99.66%** |
| New versus V7 | **50,320** | **952** |
| Losses versus V7 | **0** | **0** |
| Exact-aligned new support | 47,411 | 930 |
| Structural-program gains | 48,324 | 926 |
| Rescued rows containing deferred identifiers | 12,330 | 227 |
| Remaining unsupported/invalid | 1,092 | 14 |

Structural-program gains plus retained typed-path gains **1,996 / 26** account
for the whole expansion. Deferred-identifier rescues are a subset of the
program gains; the console's `deferred_names` column counts **actions**, not
individual name occurrences. No missing identifier was assigned a guessed
premise/declaration ID.

The **2,909 train / 22 val** gains outside exact alignment remain source-only
candidates. Exact-aligned total support was not supplied with this compact
table; do not infer it from total source support alone.

Quarantine is unchanged by the paired audit: **831 / 9** invalid input traces.
After removing that quarantine from the residual population, **261 train /
5 val** renderer exclusions remain. Their current error distribution still
requires inspection; neither quotation nor another subtype is inferred from
the totals.

V9 alone exceeds the requested 50k train-row expansion relative to V7.
Combined gain versus frozen V5 support **127,245 / 2,438** is **90,390 train /
1,705 val**. This is broader source-program representation support, including
context-dependent/deferred names, not the current model's prediction accuracy,
complete semantic premise grounding or native-verified full-cohort validity.

Report: `maths_ai/gnn_inference/runs/template_v9_recovery/report.json`.
Audit generated a **24-action** new exact-val native pilot. Next check that
selection with `template-v9` against the pinned frontend snapshots, without
restarting the full static audit. Real-Mathlib native pilot results remain pending.

## First server native pilot and general term-body repair

The completed 24-row pilot reports **22 rendered transition matches**, **1 original
control mismatch**, and **1 rendered failure**. All 24 input models match and all
24 original tactics execute; only 23 reproduce their recorded next state. The
generated tactics execute on 23 rows and match on 22. All selected rows use the
source-program codec; **none contains a deferred identifier**.

This is a bounded family/feature-selected sample, not a random estimate of the
217,635-row population. It does not validate the deferred-name subset or establish
full-cohort execution reliability. Representation coverage and native validity
must remain separate. Broader stratified checks need to cover syntax/layout
families, deferred/nondeferred names and source scopes before supervision export
or calling the whole set reliable.

Compact diagnostics identify:

- **Val 2357**: a term-level `suffices … from let …` lost a declaration/body
  separator when original layout was flattened. Generated source failed parsing.
- **Val 1325**: `apply_rules [MeasurableSet.iUnion, MeasurableSet.iInter] <;> intro`
  leaves two goals on both original and rendered replay, versus one recorded goal.
  This remains an original replay/context mismatch, not evidence of a renderer
  transformation error or a valid certified label.

The layout repair handles the general pinned `optSemicolon termParser` pattern
for term-level `let`, `have`, `haveI`, `letI`, `suffices`, `let_fun`/`let_λ`,
`let_delayed` and `let_tmp`. It verifies the separator field, emits an explicit
semicolon, scopes the whole expression with parentheses, and gives `by` tails
an explicit braced sequence boundary. This prevents an outer term body from
being consumed as another tactic. Standalone tactic
declarations retain their existing behavior. This applies to every matching
term shape, not just val 2357; no static coverage rerun is needed to diagnose it.

**201 regression tests pass**, with **38/38** source-program native transitions.
Seven added native cases cover newline-separated let/have/pattern-let/suffices,
`by` tails, and the nested lambda/existential/pattern-let shape of 2357. Unknown separator syntax
is rejected rather than dropped. Actual Mathlib row 2357 recheck remains pending;
1325 requires separate control investigation. Preserve the original pilot report
and use a fresh native output for rechecks.
