# V8: cached math notation and bounded quantifiers

V7 reached **167,315 train / 3,191 val** supported rows (**76.49% / 76.76%**).
Its remaining `have` errors exposed shared expression-grammar gaps:

| First unsupported constructor | Have rows | All-family rows |
|---|---:|---:|
| Bounded forall `∀ x …, …` | 308 | 333 |
| Norm `‖x‖` | 301 | 454 |
| Subset `⊆` | 248 | 281 |
| Scalar multiplication `•` | 212 | 504 |
| Negation `¬` | 172 | 203 |
| Inverse `⁻¹` | 167 | 481 |
| **Total** | **1,408** | **2,256** |

These are observed first blockers in distinct rows, not guaranteed rescues.
Parsing one constructor may expose another unsupported child. V8 handles this
shared family, including related notation, without new semantic extraction.

## Typed source grammar

Select `--renderer template-v8`. Schemas are `math_notation_tactic_v1` and
`math_notation_pattern_v1`, extending the existing V7 wrapper/pattern families.

- Unary forms: `¬ P`, `a⁻¹`, `‖a‖`, `‖a‖₊`, `|a|`.
- Infix forms: `⊆`, `⊂`, `⊇`, `⊃`, `∪`, `∩`, `\`, `•`, `∘`, `∉`, `>`, `≥`,
  `∣`, and `×`, alongside previously supported operators.
- Bounded `∀` and `∃` with the pinned binder predicates `<`, `≤`, `>`, `≥`,
  `≠`, `∈`, `∉`, `⊆`, `⊂`, `⊇`, `⊃`.
- Ordinary `∃` / `exists`: untyped, shared typed names, bracketed typed groups,
  dependent domains and anonymous declarations using the pinned core grammar.

Operator tokens are closed syntax labels, not invented library references.
Children retain exact local/global identities, source selectors, named args,
ascriptions and holes. Compound notation is parenthesized from its parsed tree;
operators are not replaced with expanded kernel applications.

Quantifiers retain declaration slots and type groups. Existential domains are
parsed before their declaration group; later groups can depend on earlier
binders. Bounded predicates differ: their bound expression is already inside
the new binder's scope (`∀ n ≤ n + 1, …`). Renderer preserves that scope,
including shadowing and anonymous binders. Environments restore on parse failure.

Math rendering is threaded through recursive applications, lambdas, binder
types, wrappers and typed patterns, so it is not limited to `have`. Existing
node/depth/binder limits and source/context validation remain active. Unknown
notation, token/constructor conflicts and invalid references remain failures.
V1–V7 keep their separate parser modes.

## Run full cached coverage

From server repository root, environment `atp` or the environment used for the
completed V7 audit, after syncing published code:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_selector_recovery \
  --renderer template-v8 \
  --previous-recovery-dir maths_ai/gnn_inference/runs/template_v7_recovery_20261108_113947 \
  --output-dir maths_ai/gnn_inference/runs/template_v8_recovery \
  --pilot-actions 24
```

The existing paired audit now supports both stages: default V7 versus saved V6,
or explicit V8 versus saved V7. It verifies immutable source reports, original
baseline and cache hashes. It reads raw/model/V3 caches, retains the full
**218,727 / 4,157** denominator and quarantine, and does not re-evaluate older
renderers. No Lean compilation, V2 sidecar loading or dataset download runs.

Console prints an initial status, short progress every 50,000 rows, and the
compact support/gain/loss/exact-gain/coverage table. Checkpoints update
`report.json` every 10,000 rows. Use a fresh output directory: static audits do
not implement `--resume`. Let an advancing audit finish rather than starting a
duplicate scan.

Inspect:

- `template_v8_renderable_steps`, `v8_rescues_vs_v7`, `v8_losses_vs_v7`.
- `exact_aligned_v8_steps`, `exact_aligned_v8_rescues`, `exact_aligned_v8_losses`.
- `math_notation_steps`, `math_notation_nodes`, `math_notation_rescues`.
- `other_source_repair_rescues`, which partitions total rescues with math rescues
  in V8; method-selector feature counts can overlap math features.
- Per-family residual errors and `semantic_cache_used: false` in the manifest.

Full V8 gains remain pending. The 2,256 first-blocked rows above are not a
prediction, and related grammar may change other rows. Dataset-step support
requires exact alignment; nonexact source-only gains remain separate.

## Native delta selection

The audit writes at most 24 newly supported exact val IDs to
`new_val_pilot_rows.json`, balanced by family then syntax features. It does not
execute them. A later native check uses the existing replay runner with
`--renderer template-v8`, those `--row-indices`, the frozen raw/model/V3 cache
root, and a fresh output directory. V8 does not accept a semantic overlay.

Native reports record `math_tactic_parsed_steps`, `math_pattern_parsed_steps`,
`math_notation_steps` and `math_notation_nodes`, alongside selector counters.
The new math parser hash binds resume. Captured original controls and all ordered
normalized resulting goals still gate acceptance.

## Local verification

**188 focused tests pass** with the pinned native backend; two existing Torch
deprecation warnings. New V8 fixture matches **50/50** original and rendered
transitions. It covers every new binary/unary operator, all 11 bounded predicate
kinds, anonymous/shadowed binders, dependent/multiple existential groups and
nested combinations across `have`, `exact`, `rw`, `refine`, `rintro`, `by_cases`,
`obtain`, `simpa` and scoped lambdas.

Test-only norm/inverse/scalar carriers verify the source notation grammar,
using the pinned Mathlib norm declarations and pinned Lean binder-predicate
grammar; they do not claim real-analysis theorem validation. V8 also passes
**44/44** shared scope/method fixture actions, **27/27** V5 wrapper regressions
and **5/5** chain/projection/name-resolution cases.

Paired tests verify immutable artifacts, unchanged populations/quarantine,
source-only versus exact gains, feature attribution, stale-input rejection,
V7/V8 stage binding and semantic-cache independence. Actual full-dataset V8
recovery and real-Mathlib native delta results remain unmeasured.
