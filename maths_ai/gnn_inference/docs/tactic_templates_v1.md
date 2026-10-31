# Bounded rw/simp gold templates, V1

Current extension: [semantic lookup and bounded V2 applied rules](tactic_templates_v2.md).
V1 grammar/serialized targets stay bare-reference-only; current flat fallback
also benefits from semantic-name lookup. Historical coverage below predates
that lookup fix and is the frozen baseline for paired recovery measurement.

This milestone extracts typed **gold targets** and validates their rendering.
It does not add a prediction head, train a model, alter existing prepared
labels or measure proof-search success. The removed library-query experiment
is not part of this work.

## Scope fixed from training evidence

- `rw`: one to three bare local/library references, each with forward or
  backward direction (`←` and ASCII `<-` canonicalize to the same flag).
- `simp`: zero to three bare references, with optional `only`.
- Location: goal (default or explicit `at ⊢`) or one existing local declaration.
- Public global names resolve fully qualified; local names use context IDs.
- Single-child `tacticSeq`/`tacticSeq1Indented`/list wrappers are removed only
  along a single-tactic root path. Multi-tactic blocks remain unsupported.

Rule count, direction and location are distinct fields. Three rules plus a
local location can contain **four reference occurrences**; this gold schema
is not the legacy decoder's three-total-argument format. No trained decoder
has been extended to consume it yet. Repeated references remain legal.

Deferred: applied/parenthesized rules, field/method notation, tuples, fresh
binders, config/discharger syntax, wildcard or multiple local locations,
combined goal-plus-local locations, reverse/erasing simp arguments, and
other tactic families. Old flat rendering remains the default and is reused
for other families in template mode. Within rw/simp, unsupported structures
are rejected rather than silently dropping syntax.

## Target structure

`RwSimpTemplate.to_target()` emits a versioned target such as:

```json
{
  "schema": "rw_simp_template_v1",
  "family": "rw",
  "rules": [
    {"reference": {"source": "library", "identity": "Nat.add_comm"}, "backward": true}
  ],
  "only": false,
  "location": {"kind": "local", "reference": {"source": "local", "identity": 4}},
  "rule_count": 1
}
```

Library identities are fully qualified names; locals are source-context
indices, **not graph-node indices**. Future training conversion must map
them through the bound vocabulary/corpus and preserve scope. Modifier flags
are never relabeled as premise references. These targets use annotated
source-tree roles; unannotated names are not inferred from spelling.

Parsing first calls existing source-trace compiler to validate context
references and exclude missing/unannotated/new-scope syntax. It then checks
exact pinned syntax constructors, child counts, keywords and punctuation.
Template extraction and reference rendering are separate: a private global
may be a valid typed target but still need a resolver before rendering.

Global-head annotations alone are insufficient for bare-reference targets:
derived dotted expressions can elaborate to a function applied to a receiver.
Shared conservative source/declaration spelling checks reject those cases
in both flat and V1 paths instead of discarding receiver arguments. Some
genuine renamed aliases also need a native resolver and remain unsupported.
Existing training labels are not silently relabeled.

## Gold replay

Pull published code, then from server repository root in `atp`:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_template_v1_recheck \
  --renderer template-v1 --split val --num-theorems 20 --seed 42 --file-timeout 600
```

Same source-native snapshots, pinned toolchain and row-selection policy as
[gold replay](native_gold_replay.md); captured-syntax original control,
source-text diagnostic and rendered branches restore independently and
every active goal is compared. Report v2 separates source-block replay from
exact dataset-step alignment. Use a new report directory: control policy,
renderer, schema and parser-code hashes are resume-bound.

The report includes `template_target`, `template_error_code`, `render_path`
and `flat_renderable` per row. Summary adds:

- `template_v1_parsed_steps`: typed target extracted, before name rendering.
- `flat_renderable_steps`: old renderer could produce text on this cohort.
- `newly_renderable_vs_flat`: new mode produced text where old mode did not.
- `rendered_transition_match`: final rendered action reproduces reference
  normalized goal transition in native Lean.

Static renderability is not execution accuracy. All unsupported/failing
rows remain in the selected denominator. Typed gold structures are supplied
from recorded syntax, not predicted by a model. Full prepared-train V1
coverage remains unmeasured. First server results follow.

## Verification and next gate

**53 focused tests pass** with real pinned-backend integration enabled.
Native fixtures cover directions, local locations, `simp only`, empty simp
lists, mixed directions, repeated references, trailing commas and three
rules plus a fourth location reference. Combined locations, applied rules
and `simp [*]` are rejected; original source tactics still replay correctly.
Unit tests check wrapper boundaries, role/scope validation, serialized
targets, separate resolution failures and renderer-mode resume isolation.
Regressions additionally cover captured multi-line syntax that fails text
reparsing and receiver-bearing references rejected by both renderers.

First fixed 20-theorem server cohort: **86/86** input parity, **85/86**
original transitions, **27** renderable versus **14** old-flat, **26/27**
rendered matches. All **13/13** newly renderable cases passed. One original
text-reparse failure and one old-flat method-receiver failure motivated
report-v2 control/resolution fixes. The command above rechecks the same
cohort in a fresh directory. Recheck on `1071146` passed: **86/86** input
and original-control matches; **85/85** exact dataset-aligned original
matches; **26/26** exact-aligned rendered matches, including all **13** new
cases. Current flat renderability is **13**, V1 parsed targets **19**, and
**60** rows are explicitly unsupported. One branch-opener row is source-only;
source-transition totals must not be reported as exact dataset-step matches.
Text diagnostic remains **85/86**. No replay failures remain in this cohort.

Evaluate native replay and exact grammar coverage before exporting training
targets or adding a template-aware prediction head. Keep old flat cohort and
newly covered cases separate; no model/whole-proof improvement is claimed.

## Full prepared-cohort coverage gate

Run the read-only audit after pulling published code:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_template_coverage \
  --prepared-root maths_ai/_support_files/artifacts/prepared/sexpr_available_labels_v2 \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output-dir maths_ai/gnn_inference/runs/template_v1_coverage
```

Both full prepared **train and val** cohorts are fixed by the packed
manifest, including unsupported or invalid-cache rows. No test split is
read. The grammar is already fixed; the audit selects no new grammar from
validation results. It needs neither a GPU nor a running Lean backend.

`report.json` checkpoints after each packed chunk. `rows.jsonl` contains
diagnostics and factorized grammar features, not training targets. Use a
fresh output directory; partial runs are marked incomplete and can be
restarted in another directory. This audit does not support resume.

Bindings cover prepared graph dataset/split/row/theorem/tactic metadata,
the pinned raw extraction version/source commit, normalized/v3 raw-record
digests and the prepared trace digest when present. These check cached
provenance, not fresh dataset streaming or native state equivalence. Missing,
stale or invalid cache/trace rows stay in the denominator and are quarantined
from both renderer paths. The report records packed/chunk/code/label hashes
and a digest of per-row records including cache-file hashes.

Inspect each split's `summary` and `by_family`:

- `selected_steps` must equal the full prepared manifest population
  (expected **218,727 train / 4,157 val** for the accepted artifacts).
- `legacy_label_supported_steps` counts existing graph labels, distinct
  from the current conservative `flat_renderable_steps`.
- `template_v1_parsed_steps` precedes reference rendering; private names
  can parse but still be unrenderable.
- `template_mode_renderable_steps` includes V1 rw/simp plus flat fallback
  for other families. `newly_renderable_vs_flat` and
  `flat_only_renderable_steps` expose both support gains and losses.
- `exact_aligned_*` restrict static support to exact source alignment with
  matching dataset/source targets. Branch-opener support is source-only.
- `status_counts`, `error_counts`, `template_error_counts`,
  `flat_error_counts` and `trace_scope_counts` expose exclusions.
- `parsed_template_features` and
  `exact_aligned_renderable_template_features` count each family, rule
  count, local/library source pattern, direction pattern, `only` and
  location separately. They are not new template class labels.

All coverage counts are static syntax/renderability results. Full-cohort
Lean execution, model predictions, retrieval reachability, graph-index
conversion and whole-proof success are not measured. Review the coverage
and exclusions before designing a separate versioned target export.

Verification: **58 focused tests pass**, including pinned-native regressions
and coverage checks for stale bindings, unchanged input artifacts, invalid
traces, unrenderable private targets, source-only/target-mismatched alignment
and duplicate/incomplete prepared populations.

Full-cohort server scan completed on `5335427`:

| Static coverage stage | Train | Val |
|---|---:|---:|
| Selected prepared rows | 218,727 | 4,157 |
| Existing supported labels | 61,853 | 1,207 |
| Current conservative flat rendering | 43,864 | 862 |
| V1 parsed targets | 38,325 | 775 |
| Combined V1 + flat rendering | 66,904 | 1,322 |
| Newly renderable versus current flat | 23,048 | 460 |
| Current flat only, lost in V1 mode | 8 | 0 |
| Exact-aligned combined rendering | 61,836 | 1,317 |
| Exact-aligned rendered V1 targets | 35,019 | 773 |
| Unsupported rendering | 150,992 | 2,826 |
| Invalid context traces | 831 | 9 |

The net change versus existing labels is only **+5,051 train / +115 val**:
new support also excludes some old labels under stricter checks. Existing
label support and executable reference rendering are different policies.
Do not classify every exclusion as an incorrect label without inspection.
Train alignment includes **12,425 unknown / 1,752 branch openers**; val has
**27 branch openers**. Invalid trace reason is `unknown_local_reference`.

There are **151,823 train / 2,835 val** unsupported-or-invalid rows. Existing
scope/flat/template counters overlap and do not form an exclusive failure
census. Inspect primary blockers, secondary syntax flags and representative
examples across all tactic families before choosing the next extension.
These are unsupported formats/data, not measured Lean execution failures.

Next: [exclusive unsupported-action census](unsupported_action_census.md),
reusing this report to inspect all tactic families before choosing new support.
