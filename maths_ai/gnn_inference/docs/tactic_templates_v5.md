# V5 wrappers, locations, modifiers and simple declarations

V5 targets the large V4 fallback buckets using existing recursive terms.
Select `--renderer template-v5`; existing modes keep their previous behavior.
New gold schema: `extended_tactic_v1`. This is representation/rendering work,
not new model weights or exported supervision.

## Supported expansion

- `simpa` with ordered rules, `only`, and optional recursive `using` term.
- `simp_all` and `dsimp` with explicit simplifier arguments and `only`.
- `rewrite`, `rwa`, and pinned Mathlib `simp_rw`, preserving their distinct
  semantics rather than replacing them with `rw` or `simp`.
- Goal, multiple named hypotheses, combined hypotheses-plus-goal and wildcard
  `at *` locations, bounded to 16 explicit local targets.
- Simp lemma direction, pre/post traversal (`↓`/`↑`), erasure (`-lemma`) and
  context-star (`*`) arguments as separate modifier kinds.
- `unfold`, `delta`, `change`, `show`, `specialize`, and location-aware `symm`.
- Simple `have`/`let` declarations with a named or implicit binder, optional
  type and recursive value. Parameterized/pattern declarations remain excluded.
- Numeric projection selectors such as `h.1` or `(id h).1` as indices, not
  premise identities.
- Explicit core infix operators in bounded terms, preserving operator/grouping.
- Single atomic nested proofs such as `by rfl`, `by assumption`, `by ring`;
  no arbitrary proof-text copying, multi-tactic proof sequence or fresh proof scope.

Unknown configuration/discharger syntax, unsupported query/auto-unfold variants,
scoped declaration parameters, complex patterns, arbitrary notation, missing
value references and ambiguous method identifiers remain explicit failures.
Term budgets remain 256 nodes/action, depth 16 and up to 16 children/rules.

## Role and reference integrity

Every source tree first passes generic context/index validation. V5 then uses
exact syntax positions to distinguish declaration names, anonymous `have`
hygiene placeholders, parameter labels, locations and actual term references.
An unannotated value identifier is never guessed into a premise.

Earlier rewrites in a multi-location action can rebuild a later hypothesis;
the trace may mark its location identifier `scoped_local`. Only in location
positions, V5 maps a unique exact original input name back to its context ID.
It never applies this repair to a scoped identifier in a term value. Captured
gold names, ordered arguments, declaration types and proof terms are preserved;
types are not silently omitted to make a declaration render.

The location/lemma lists preserve order, wildcard operations are not fabricated
reference IDs, and erasure is not treated as a negative argument. V1–V4 APIs
and source caches are not rewritten by this expansion.

## Paired coverage against saved V4

After pulling published code, from server repository root in `atp`:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_wrapper_recovery \
  --previous-recovery-dir maths_ai/gnn_inference/runs/template_v4_recovery \
  --output-dir maths_ai/gnn_inference/runs/template_v5_recovery \
  --pilot-actions 24
```

The runner verifies completed V4 records/digests and the original frozen
baseline/cache bindings. It does not recompute V1–V4 renderers. Same denominator:
**218,727 train / 4,157 val**. It must reproduce the recorded V4 support
**107,913 / 2,096** before comparing V5. Invalid contexts remain quarantined.
No source graphs, caches, old reports or checkpoints are mutated.

Inspect `template_v5_renderable_steps`, `v5_rescues_vs_v4`, `v5_losses_vs_v4`,
exact-aligned counts and the exclusive family-based rescue categories.
`by_family` now includes each family's current error counts, making broad
fallback exclusions easier to prioritize. Counts are static support, not
native full-cohort validity or model accuracy. A complete report may exit 2
when paired losses occur; inspect the loss/alignment data. Use a fresh output
directory; this audit checkpoints but does not resume.

## Focused new-action native pilot

The audit writes `new_val_pilot_rows.json` with at most 24 newly renderable,
exact-aligned val actions. Selection round-robins families first and then
feature groups; no application-family starvation of rewrite/simp wrappers.
If `native_pilot.selected_steps` is zero, skip this check.

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --renderer template-v5 --split val \
  --row-indices maths_ai/gnn_inference/runs/template_v5_recovery/new_val_pilot_rows.json \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_template_v5_pilot \
  --file-timeout 600
```

This does not restart exhaustive train validation. Completed V3 val results,
including six original-control failures, and successful V4 application pilot
results remain separate evidence/diagnostics. Native resume still requires
unchanged implementation hashes; retain stopped old reports as historical data.

Verification: **134 focused tests passed**. Native fixture checks **27/27**
rendered transitions and original controls, including an exact test-only copy
of pinned Mathlib `simp_rw` (commit `29dcec074de168ac2bf835a77ef68bbe069194c5`),
combined/wildcard locations, rebuilt location hypotheses, modifiers, anonymous/
typed declarations, atomic proofs and numeric projections. Unit/paired checks
cover role conflicts, unknown values, immutable data, strict original-context
validation, complete denominators and exact-only balanced pilot selection.
Full server V5 gains and the focused new-action native result are recorded below.

## Completed paired server result

| Static support/gain | Train | Val |
|---|---:|---:|
| Frozen V4 support | 107,913 | 2,096 |
| V5 support | 127,245 | 2,438 |
| V5 rescues versus V4 | 19,332 | 342 |
| V5 losses versus V4 | 0 | 0 |
| Exact-aligned V5 support | 118,280 | 2,433 |
| Exact-aligned rescues | 18,015 | 342 |

Static coverage increases **49.34% to 58.18% train / 50.42% to 58.65% val**.
Rescues split into declarations **2,919 / 65**, simplifier wrappers **6,776 /
107**, rewrite wrappers **1,525 / 24**, existing-rule syntax **4,721 / 77**,
ordinary term syntax **2,946 / 56**, and other wrappers **445 / 13**. These
categories are exclusive; they sum to the total rescues.

The **1,317** train rescues outside exact alignment remain source-only/unknown
candidates, not ordinary dataset-step labels. Quarantine remains **831 / 9**.
Remaining unsupported/invalid population: **91,482 train / 1,719 val**. No model
accuracy or full-cohort Lean execution gain is implied by these static counts.

The generated pilot selects **24** new exact-aligned val actions from **71**
family/feature groups. It is a bounded sample, not coverage of all groups or
all 342 val rescues. Report:
`maths_ai/gnn_inference/runs/template_v5_recovery/report.json`.

The focused server pilot subsequently passed **24/24** selected exact-aligned
actions: cached input parity, captured-syntax original execution/transition,
reparsed original diagnostic, and rendered execution/transition all matched.
Every candidate uses `extended_tactic_v1`; flat grammar rendered none of them.
No operational, input, control or rendered replay failure occurred in this
sample. This supports the selected real-Mathlib instances, not universal
validity of every form/argument binding or completed theorem proofs. Report:
`maths_ai/gnn_inference/runs/native_gold_replay_template_v5_pilot/report.json`.
Next prioritize current per-family residual errors for further coverage.
