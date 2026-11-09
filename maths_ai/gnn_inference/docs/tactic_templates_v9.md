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

No V9 full-dataset gain or real-Mathlib native delta result is claimed yet.
The 840 quarantined records cannot be made valid by inventing missing input
references. Preserve explicit residual/error categories and validate the combined
stage before supervision export or model training.
