# V3 coverage expansion: binders, parameter roles and reference applications

Next expansion: [V4 bounded recursive terms](tactic_templates_v4.md), including
nested applications, constructors and anonymous holes, with a focused delta pilot.

V3 targets three remaining gaps from the unsupported-action census:
simple introduction declarations, named-argument labels, and bounded
applications in `exact`/`apply`/`refine`. Select `--renderer template-v3`.
V3 also raises the rewrite/simp rule-list cap from three to sixteen.
This is gold representation/rendering work; these targets have not trained a
prediction head. Completed recovery results and broader validation follow.

## Introduction declarations

Schema `intro_binders_v1` represents `intro`/`intros` with at most **16 explicit
named/anonymous binder slots**. Slots are declaration positions, not pointers
to hypotheses already in the input graph:

```json
{
  "schema": "intro_binders_v1",
  "family": "intro",
  "binders": [
    {"slot": 0, "kind": "named", "gold_name": "b"},
    {"slot": 1, "kind": "named", "gold_name": "b_in_orbit"}
  ],
  "explicit_binder_count": 2,
  "mode": "explicit",
  "reference_count": 0
}
```

`intro` without arguments has mode `one`; `intros` without arguments has mode
`all`. The latter may introduce more than 16 declarations automatically: the
limit bounds the explicit output list, not Lean's inferred introductions.
An explicit `_` is an anonymous binder slot, not a proof argument or missing
reference. Original gold names are preserved for native goal/name parity;
future training must decide a binder-name/canonicalization policy separately.

Pinned `intro`/`intros` constructors, keyword, child counts and binder positions
are checked exactly. Scoped-local annotations and unannotated identifiers in
those declaration positions are accepted. Unannotated names elsewhere are
not guessed. Conflicting existing-premise annotations and invalid context
indices are rejected. Complex patterns, typed binder terms, match arms,
multi-tactic blocks and scoped `have`/`obtain`/`rintro` remain outside this grammar.

## Named parameter labels

Schema `rw_simp_template_v3` extends the V2 rule grammar with named reference
arguments, for example:

```lean
rw [Nat.add_zero (n := a)]
```

The argument is represented as:

```json
{"reference": {"source": "local", "identity": 1}, "parameter": "n"}
```

`n` is a syntactic parameter label, not a local/library premise. The value `a`
still requires a validated reference. Positional arguments use the same V3
shape with `parameter: null`; named and positional order is preserved.
V3 allows up to sixteen rules and up to three reference-valued arguments per
rule; V1/V2 retain three rules. Including a local location, the maximum V3
rule action has **65 reference occurrences**. Longer bare rule lists directly
target the original **6,517 train / 122 val** rule-budget exclusion bucket;
those are opportunities, not measured recovery counts.

`compile_source_syntax_trace(..., allow_parameter_labels=True)` recognizes
only exact five-child `Lean.Parser.Term.namedArgument` syntax and emits a
separate `PARAMETER_LABEL` terminal. It does not mutate source caches or
discard unresolved values. Wrong delimiters or conflicting label annotations
fail explicitly. The default compiler policy remains strict, preserving V1/V2
behavior and frozen census counts. This opt-in diagnostic role does not extend
the existing trained prefix decoder or alter prepared labels automatically.

## Bounded reference terms

Schema `reference_application_v1` represents a bare reference or a reference
head applied to zero to three positional/named reference arguments for
`exact`, `apply` and `refine`:

```lean
exact (f h)
apply (fpartial hp)
exact eqLift (a := a) (b := b) (h := h)
```

Transparent parentheses normalize; argument grouping/order and parameter labels
remain explicit. The target contains `reference`, ordered `arguments`, tactic
family and `reference_count`. Up to four references can occur in one such term,
so this is not legacy three-slot supervision. An `apply` can leave obligations;
native replay compares all resulting goals, not only successful closures.

Nested argument expressions, applied heads, literals, constructors, projections,
holes, fresh scopes and nested proofs remain unsupported. In particular,
`exact f (g h)` and `refine id ?_` are not silently flattened or turned into
stop-only targets. Existing receiver/alias/private-name guards remain in force.

## Measure V3 against V2 and the original frozen cohort

Pull published code, then run from server repo root in `atp`:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_template_recovery \
  --baseline-coverage-dir maths_ai/gnn_inference/runs/template_v1_coverage \
  --renderer template-v3 \
  --output-dir maths_ai/gnn_inference/runs/template_v3_recovery
```

Use the original coverage artifact, not a regenerated baseline. The audit
recomputes lookup-fixed V1 and V2 on the same bound snapshots, then adds V3.
It should reproduce V2 support **78,324 train / 1,531 val** before measuring
new gains. All **218,727 train / 4,157 val** rows remain in the denominator;
the **831 / 9** invalid baseline traces stay quarantined.

Inspect each split's:

- `template_v3_renderable_steps`, `v3_rescues_vs_v2`, `v3_losses_vs_v2`.
- `intro_rescues`, `named_rule_rescues`, `application_term_rescues`, `rule_list_rescues`, and
  `other_v3_rescues` (exclusive attribution of V3 gains).
- `exact_aligned_v3_*`: source-only/unknown alignments are not dataset-step labels.
- `overall_v3_rescues_vs_original` / `overall_v3_losses_vs_original`.
- `by_family`, `by_previous_error`, `v3_error_counts`, and diagnostic rows.

`by_previous_error` refers to the original frozen V1 result, not a new exclusive
V3 residual census. Parsed counts precede rendering. `named_argument_count`
in diagnostic rows includes rule and ordinary term arguments. Gains are
attributed to named-rule support before longer-list support when both are
needed; the latter bucket counts longer lists without named arguments.
`long_rule_list_parsed_steps` counts all parsed lists exceeding three rules.
These are static coverage counts, not model accuracy or full-cohort native success. Exit 2
after a complete report flags paired losses, including strict syntax rejection
of previously renderable source-only fragments; inspect their alignment.
Use a fresh directory; the audit has no resume support.

## Native same-cohort replay

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_template_v3 \
  --renderer template-v3 --split val --num-theorems 20 --seed 42 --file-timeout 600
```

Reports distinguish rw/simp, introduction and ordinary application schemas,
and bind all three parsers plus the syntax-role compiler by hash. Captured
syntax remains original control; reparsed text stays diagnostic. Native
validation specifically covering recovered rows is still a gate before export.

Verification: **116 focused tests passed**, including pinned native integration.
The new fixture matches **20/20 rendered transitions / 23/23 original controls**:
named/wildcard/automatic introductions, repeated/shadowed/escaped/Unicode names,
global-name collisions, named/mixed rule arguments, ordinary applications and
an `apply` preserving a new obligation, plus four- and sixteen-rule rewrites.
Three valid originals with nested
arguments or proof holes remain unsupported. Unit/paired checks preserve old
compiler policy, immutable caches, scope/parameter distinctions, exact versus
source-only alignment, loss accounting and renderer manifests. Full server
V3 recovery and native cohort results are recorded below.

## Completed V3 server result

V3 renders **93,113 train / 1,823 val** rows (**42.57% / 43.85%**), versus
V2 **78,324 / 1,531**. New V3 rescues **14,828 / 292**, with **39 / 0**
losses; net gains **14,789 / 292**. Rescues split into introductions
**6,630 / 139**, longer lists **7,684 / 142**, named rules **111 / 2**, and
ordinary reference terms **403 / 9**.

Exact-aligned support rises to **86,244 / 1,818**, with **14,047 / 292**
rescues and **zero exact-aligned losses**. The 39 train losses are outside
that exact-aligned cohort and still need individual error inspection.
Compared with the original V1 baseline, gross rescues are **26,239 / 501**
and losses **30 / 0** (nine V3 losses were earlier V2 rescues). Remaining
unsupported/invalid population: **125,614 train / 2,334 val**.

Same 86-step native cohort passes **86/86** input/original-source matches,
**85/85** exact-aligned original transitions and **34/34** exact-aligned
rendered transitions; **52** unsupported, no replay failures. Four intro,
five ordinary application and 21 rule targets parse. This does not establish
native validity for the whole static population.

Next: [complete extracted-form inventory and source-native validation plans](template_inventory_validation.md),
covering all exact-aligned validation actions and representatives of each
exact-eligible training form before further coverage expansion/export.

The remaining coverage program includes recursive terms/constructors/holes,
scoped proof blocks and extraction defects. Their original bucket counts are
opportunities, not guaranteed recovery or a promise to recover every row in
two days. Measured V3 residuals should choose the next bounded implementation.
