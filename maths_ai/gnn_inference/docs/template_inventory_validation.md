# Extracted template inventory and native validation

## Complete implemented formats

These are gold action schemas/grammars, not trained template classes.
Current `template-v3` uses three structured schemas plus flat fallback;
the older rule schemas remain compatibility modes.

| Format | Supported shapes |
|---|---|
| `rw_simp_template_v1` | `rw [r₁, …]`, `simp [r₁, …]`, `simp only […]`; bare references, up to 3 rules, rw directions, goal/one-local location |
| `rw_simp_template_v2` | Same modifiers/locations; each rule can be a reference head with up to 3 positional reference arguments; up to 3 rules |
| `rw_simp_template_v3` | Up to 16 rules; positional or named reference arguments, preserving order; same bounded directions/only/locations |
| `intro_binders_v1` | `intro`/`intros`, 0–16 explicit named/anonymous declaration slots; zero-argument one/all behavior |
| `reference_application_v1` | `exact t`, `apply t`, `refine t`; bare reference or head with up to 3 positional/named reference arguments |
| Flat fallback | Other admitted tactic names with 0–3 resolved references; list versus term syntax emitted by the flat renderer |

References distinguish input-local identities from library declarations.
Binder names and parameter labels are not premise references. Transparent
parentheses normalize. Nested applications/constructors/holes/scoped proofs
beyond these grammars remain excluded.

The earlier **30,043 train raw syntax shapes** were an exploratory catalog
including unsupported syntax and duplicate wrappers. They are not 30,043
implemented or trained templates.

## What “valid” means

An abstract `rw [L0]` is not valid for every goal or every lemma assigned to
`L0`. Verification concerns an extracted instantiation in its original Lean
context. Distinguish:

1. Typed syntax extraction and role/context integrity.
2. Reference resolution and renderability.
3. Source-native execution reproducing the input and ordered goal transition.
4. Whole-theorem kernel-checked proof success, which this harness does not measure.

Existing fixtures and the **34/34** rendered server smoke do not certify every
extracted form or every one of the **93,113 train / 1,823 val** renderable rows.
The next validation cohorts explicitly cover all **1,818 exact-aligned val**
actions and examples from each exact-eligible train form. Nonexact source
fragments and parsed-but-unrenderable instances stay visible as unverified.

## Build inventory from the completed recovery report

From server repo root in `atp`, after pulling published code:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.inventory_tactic_templates \
  --recovery-dir maths_ai/gnn_inference/runs/template_v3_recovery \
  --output-dir maths_ai/gnn_inference/runs/template_inventory_v3
```

The CPU-only inventory verifies recovery/baseline digests, renderer fingerprints
and candidate cache snapshots. It re-extracts each typed target or rendered
fallback from the same source-bound trace. It includes parsed targets that
cannot yet render, with explicit errors; it does not treat them as native-ready.

`catalog.json` contains:

- `schema_registry`: implemented formats/bounds, with active V3 modes marked.
- `forms`: every observed extracted form, train/val counts, renderable and
  unrenderable counts, exact-aligned renderable counts, and concrete examples.
- `losses_vs_v2`: the 39 train losses with alignment and recorded V3 errors.
- Per-split actual rendered schema/family populations and parsed-target counts.
- Forms with no renderable or no exact-aligned candidates, which cannot be
  marked validated by the default plans.

Forms abstract local/library identities while preserving their repeated-use
relationships, ordered arguments, named parameter spellings, directions,
`only`, locations, binder modes/kinds, repeated names and detected shadowing.
Absolute premise identities remain in concrete examples. Goal types and all
possible declaration semantics are not abstract-form validity guarantees.

Two immutable plans are written:

- `plan_val_all_exact.json`: every exact-aligned rendered validation instance.
- `plan_train_forms.json`: deterministic lowest-hash examples per exact-eligible
  training form (one by default).

Use `--train-examples-per-form 3` for broader form samples. To select every
exact-aligned rendered train instance, use a value at least the whole train
population, e.g. `--train-examples-per-form 218727`, in a new inventory directory.
The resulting train plan should select **86,244** rows for the current artifacts;
native runtime scales with selected source files/actions. This checks extracted
instances, not all hypothetical instantiations of the grammar.

## Execute validation plans

All exact validation actions:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --renderer template-v3 --split val \
  --validation-plan maths_ai/gnn_inference/runs/template_inventory_v3/plan_val_all_exact.json \
  --output-dir maths_ai/gnn_inference/runs/template_validation_v3_val \
  --file-timeout 600
```

Every exact-eligible train form's selected examples:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --renderer template-v3 --split train \
  --validation-plan maths_ai/gnn_inference/runs/template_inventory_v3/plan_train_forms.json \
  --output-dir maths_ai/gnn_inference/runs/template_validation_v3_train_forms \
  --file-timeout 600
```

Plan mode replaces theorem sampling. It verifies the original dataset row,
source coordinates, cache hashes, action hash and form ID before Lean runs.
Changed inputs or renderer policies require a fresh inventory. Reports are
checkpointed per file; repeat identical arguments with `--resume` after an
interruption. Existing source/toolchain pins and captured-syntax controls apply.

`template_validation` reports selected/checked/pending rows, form-level statuses,
forms whose selected instances all matched, and forms with failures. A form with
one passing sample is not universally certified; the report explicitly sets
`universal_template_validity_proved: false`. Input/control/serialization failures
remain distinct from rendered-action failures. In plan mode only an actual
`rendered_transition_match` passes; newly unsupported rows are failures rather
than silently accepted exclusions. All failures stay in the selected denominator.

Verification: **121 focused tests passed**, including pinned-native fixtures.
Inventory tests cover identity abstraction versus repeated references, named
labels, escaped separators, binder shadowing, parsed-but-unrenderable targets,
loss diagnostics, immutable snapshots and exhaustive val/train-form selection.
Runner tests cover plan execution/resume, stale-action rejection, complete
denominators and failed-candidate/form accounting. Full server inventory and
expanded native validity results remain pending.
