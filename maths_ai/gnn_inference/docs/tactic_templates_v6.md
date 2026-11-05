# V6 lexical scopes, declaration patterns and owned semantic terms

V6 combines three recovery paths under `--renderer template-v6`:

- `scoped_tactic_v1`: V5 wrappers with lexical term binders and explicit mode.
- `scoped_pattern_tactic_v1`: branching/declaration patterns.
- Optional complete-expression recovery from digest-bound `action_trace_v2`.

The paired server audit measures net recovery against saved V5 support
**127,245 train / 2,438 val**, on the same **218,727 / 4,157** denominator.
The requested **>50,000 additional train rows** remains a target, not a measured
gain or a guaranteed consequence of the residual error buckets.

## Typed scopes and patterns

Terms support lambdas, forall, arrows, explicit `@`, substitution `▸`, unqualified
`Type`/`Prop`, and a bounded whitelist of atomic numeric-type notation. Binder
groups retain explicit/implicit/strict-implicit mode, optional types, order,
anonymous slots, dependent domains and trailing type annotations. Multiple
names sharing a type remain one binder group.

Every lexical declaration has an occurrence slot, separate from original input
context indices. Scoped uses resolve lexically, including nested shadowing.
Input references that would be captured are rejected. Renderer also rejects an
outer slot hidden by a later same-name binder. Domain/body rendering retains
scope; recursive V5 children use the same scoped rendering callback. Named
argument labels retain their separate role. Holes under a lambda retain their
pending local obligations.

Pattern actions support core `rcases`, `rintro`, `obtain`, `by_cases`, and `ext`:

- Ordered tuples, alternatives, grouping, typed patterns and explicit patterns.
- Named declaration slots, ignore `_`, clear `-`, and substitution pattern `rfl`.
- RCases target expressions and optional equation names.
- Obtain value/type clauses, including an explicit-type pending obligation.
- Named/default case hypotheses; function extensionality patterns.

RCases can retain an original fvar for a name-only pattern. Its declaration
position remains a declaration slot; a local annotation must still validate
against the exact input name/index. Pattern types referring to unsupported new
pattern scopes remain exclusions rather than guessed input references.

Existing limits remain 256 term nodes/action, depth 16 and 16 ordinary
arguments/rules. Scoped nodes allow at most 32 binders, with 256 lexical slots
per action. Pattern trees are bounded to 128 nodes, depth 16 and width 16.
Parameterized have/let declarations, arbitrary nested proof blocks, `cases'`/
`induction'`, explicit universe levels and explicit `ext` depth need additional
grammars. Unsupported syntax stays visible in per-family errors.

## Complete-expression recovery and cache binding

An identifier annotation may describe only an expression head. For example,
`h.symm` is not interchangeable with bare `Eq.symm`. V6 can recover the complete
owned expression, including `h`, from an optional V2 sidecar. It never infers a
receiver from a dotted spelling.

Production audit/replay load optional sidecars at
`{cache_root}/{split}/action_trace_v2/{row_index:09d}.json`. Loader verifies:

1. V2 schema/extractor and raw-record digest.
2. Dataset/split/row/theorem/tactic/file identity and pinned Mathlib/Pantograph.
3. State/tactic/target hashes and original input-context signature.
4. Exact agreement with V3's owned term metadata/ranges.

Recovery requires a unique expression for the failed node's exact byte range.
It is disabled inside a lexical binder environment. Missing, invalid or
conflicting sidecars never enable recovery; syntax-only support still works.

The V2 serializer omitted implicit/instance arguments, universes and binder
information. Decoder therefore accepts bounded globals/constructors, original
input locals, applications, natural literals and projections; it rejects
metavariables, bound variables, scopes, sorts and other insufficient payloads.
Reconstructed terms are **static candidates requiring native transition
validation**, not automatically accepted supervision. Complete-expression
reconstruction can fail to elaborate even when the source expression worked.

## Run paired coverage first

After syncing published code, from server repository root in `atp`:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_scoped_recovery \
  --previous-recovery-dir maths_ai/gnn_inference/runs/template_v5_recovery \
  --output-dir maths_ai/gnn_inference/runs/template_v6_recovery \
  --pilot-actions 24
```

Audit verifies saved V5/V4 lineage, original frozen baseline and row cache
bindings. It evaluates V6, without rerunning older renderer stages. Invalid
original contexts remain quarantined. Outputs are read-only with respect to
source artifacts; choose a fresh output directory. Checkpoints do not resume.

Inspect `report.json` split summaries:

- `template_v6_renderable_steps`, `v6_rescues_vs_v5`, `v6_losses_vs_v5`.
- Exact-aligned support, rescues and losses; source-only gains stay separate.
- `v6_syntax_renderable_steps` and `semantic_reconstructed_steps`.
- Exclusive rescue categories: semantic, pattern, scoped term, other syntax.
- `semantic_trace_status_counts`: bound, missing, invalid, quarantined.
- Per-family current errors. Rows retain both syntax-only and final errors,
  rendered text, term kinds, sidecar status/hash and reconstruction count.

No separate manual sidecar count is needed before this audit. If missing or
invalid sidecars dominate, use this evidence to select bounded semantic
extraction work. Do not interpret missing sidecars as semantic-decoder failure
or automatically launch full-cohort re-extraction. Per-row hashes and an
aggregate semantic-input digest bind the optional data actually observed.
Exit 2 with a complete report means paired losses require inspection.

## Focused native recovery pilot

Audit writes `new_val_pilot_rows.json`: at most 24 newly supported,
exact-aligned validation actions, round-robin by family then feature groups,
including whether semantic reconstruction was used. Skip an empty selection.

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --renderer template-v6 --split val \
  --row-indices maths_ai/gnn_inference/runs/template_v6_recovery/new_val_pilot_rows.json \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_template_v6_pilot \
  --file-timeout 600
```

Replay records optional semantic hashes in the resume manifest, plus scope,
pattern, semantic and S-expression parser fingerprints. A changed, added or
removed selected sidecar invalidates resume. Summary counts semantic cache
statuses and reconstructed actions. Captured original syntax remains the
control; all resulting goals must match. An unsupported fixed-row candidate
fails this targeted check. Keep audit/cache inputs fixed between audit and pilot.

This workflow is a bounded delta pilot. The prior interrupted exhaustive train
validation is not restarted. New row support, sample native validity, model
prediction and whole-theorem success remain distinct measurements.

## Verified locally

**153 focused tests pass** with pinned native backend; two existing Torch
deprecation warnings. New V6 fixture matches **38/38** captured originals and
rendered all-goal transitions: typed/implicit/dependent/multi-name binders,
shadowing, anonymous binders, pending scoped holes, forall/arrows, explicit `@`,
tuple/or/typed/clear/rfl/equation patterns, obtain obligations, by_cases and ext.
Four actions require complete-expression recovery (`h.symm`, transitivity,
refine and rewrite). V6 also matches all **27/27** existing V5 native fixture
actions, including pinned Mathlib `simp_rw`.

Unit/integration checks reject stale bindings, conflicting ranges, unknown
locals, insufficient semantic scopes, over-budget terms and changed resume
inputs. Paired synthetic cohorts verify full denominators, unchanged artifacts,
quarantine retention, syntax/semantic attribution and exact-only balanced
sampling. Regression checks cover V1–V5, inventory fingerprints and existing
joint/local decoders. Full server V6 row gains and real-Mathlib delta pilot
results remain unmeasured.
