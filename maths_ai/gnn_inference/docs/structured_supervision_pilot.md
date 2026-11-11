# First structured-decoder supervision pilot

## Implemented preparation stage

`scripts/prepare_structured_supervision.py` consumes the completed immutable V9
recovery, its original baseline/source-cache bindings, and the accepted joint
bundle's corpus/index identity tables. It writes a separate candidate package;
legacy prepared labels and frozen caches are read only.

Defaults select up to **5,000 train / 500 validation** candidates, balanced across
tactic-family/schema groups, then ordered by seeded row hashes within each group.
The seed is 42. Limits of zero explicitly request all typed candidates. Train and
validation selection remain separate; official test rows are not read.

Inputs after deployment:

- `--recovery-dir`: completed server `maths_ai/gnn_inference/runs/template_v9_recovery`.
- `--bundle-dir`: accepted server `maths_ai/_support_files/artifacts/hf_joint_model`.
- `--output-dir`: a new directory, for example
  `maths_ai/_support_files/artifacts/prepared/structured_v3_pilot`.

This stage does not compile Mathlib, run native replay, train a model, or repeat
an exhaustive train-native validation. Only selected candidate caches are read
after the frozen row populations/digests are checked.

## Eligibility and bindings

The first package requires exact source alignment and matching dataset/source
successor state. It excludes quarantined traces, unrenderable actions, all V9
source-program fallbacks (deferred or source-scope-dependent), references outside
the matched library corpus, and inconsistent normalized/source local contexts.

Typed targets retain their existing schema. Flat fallback actions receive an
explicit typed reference target and must reproduce the checked flat rendering.
Local identities remain **Lean context indices**; library identities include both
the fully qualified declaration name and matched corpus lemma ID. They are not
interchangeable with graph node IDs or retrieved candidate positions.

Cache byte hashes, raw/normalized/source-sidecar bindings, corpus/index IDs/names,
frozen baseline/V9 row populations, prepared manifest/label hashes, and renderer/
builder hashes are checked or recorded. Source changes during export abort the
package instead of quietly mixing versions. Exclusion counts remain visible.

## Outputs

- `train.jsonl`, `val.jsonl`: typed-IR candidate records and source-cache hashes.
- `train_row_indices.json`, `val_row_indices.json`: selected dataset row IDs.
- `native_val_pilot_rows.json`: first up to 24 family-balanced validation IDs for
  a bounded next native check.
- `report.json`: limits, selection policy, source bindings, populations, exclusion
  counts, emitted counts, family/schema histograms and output digests.

Schema: `structured_supervision_candidates_v1`. Records explicitly say
`native_validation="not_measured"`, `retrieval_reachability="not_measured"`, and
`neural_target_compiled=false`. Gold declaration names are retained for renderer
checks. These records are **not inputs to the old flat GRU trainer** and do not
certify every emitted target as natively executable or retrievable.

## Next training steps

1. Run this bounded preparation on the server caches and inspect actual emitted
   counts/exclusions; no full-cohort exported counts have been measured locally.
2. Check selected native validation transitions, preserving original-control
   failures as failures rather than changing goal parity.
3. Canonicalize declaration slots with capture-avoiding fresh names, keeping
   input pointers and lexical references distinct. Preserve syntax labels such
   as method selectors and named parameters as labels, not invented premises.
4. Build the structured action codec and neural vocabulary from **train only**;
   validation-only payloads remain explicit unknowns/unrepresentable cases.
5. Align local references with the normalized graph and library references with
   the frozen matched retrieval pool. Retrieval misses remain misses in reported
   end-to-end evaluation, never silently injected validation candidates.
6. Train the new decoder first with the accepted GNN/retriever frozen. Compare
   old and new policies through the same best-first/verifier boundary.

Seven regression checks cover reference ID spaces, binder-vs-input distinctions,
nonexact/context exclusions, frozen-source preservation, stale cache/index
rejection, split isolation, and deterministic bounded selection. The full server
package and new decoder training have not yet run.
