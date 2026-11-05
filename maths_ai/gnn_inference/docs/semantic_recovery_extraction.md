# Isolated V6 semantic-recovery extraction

Completed syntax-only V6 audit recovered **21,624 train / 382 val** rows, with
zero losses. Its real-Mathlib delta pilot passed **24/24** input, original and
rendered transition checks. Semantic cache inspection found **no bound data**:
train **5,189 invalid / 212,707 missing / 831 quarantined**; val **4,148 missing /
9 quarantined**. This explains why that run could not exercise semantic recovery;
it does not measure the decoder's yield on usable expressions.

`extract_semantic_recovery.py` captures a bounded sample into a separate
overlay. It reads frozen raw/model/V3 records and old reports, writing only its
new output directory. Old V2 payloads are not re-stamped with newer raw digests.

## First server probe

From repository root, environment `atp`, after pulling published code:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.extract_semantic_recovery \
  --recovery-dir maths_ai/gnn_inference/runs/template_v6_recovery \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --split val --max-actions 64 --max-files 16 --pilot-actions 24 \
  --output-dir maths_ai/gnn_inference/runs/template_v6_semantic_val_probe \
  --file-timeout 600
```

Selection is frozen before extraction: exact-aligned, valid original contexts,
V6-unsupported actions whose syntax error is `non_bare_reference`, whose
semantic data was missing/invalid, and whose V3 trace has owned term ranges.
It round-robins families with deterministic row hashes, bounded to **64 actions
and 16 source files**. Actual selected count may be lower due to the file cap.
This is a targeted opportunity sample, not random dataset accuracy. Official
test data is excluded. No exhaustive train replay or full re-extraction runs.

The script prints and saves `prior_semantic_binding_errors`, aggregated from
the completed V6 row records across train/val. Those are the original observed
failures, not a claim that the old payloads can be safely rebound.

## Binding and resource rules

- Verify immutable V6 rows and original baseline hashes before selection.
- Verify Mathlib/Pantograph checkout pins, toolchains and clean tracked sources.
- Match selected dataset rows to original state/tactic/target hashes and source
  identity; preserve recorded unit/invocation coordinates without reselection.
- Re-capture the same source invocation; require exact annotated syntax parity,
  exact normalized input model parity, and matching V3 owned ranges/context.
- Validate new V2 payload in memory against the original raw digest before
  writing `{output}/{split}/action_trace_v2/{row_index:09d}.json`.
- Keep the conservative decoder's lexical-scope, receiver and payload guards.
- Run one frontend process per file to bound retained Lean state; checkpoint
  each completed file. Per-file compilation limit is 600 seconds in the command.

Use the identical command plus `--resume` after interruption, with unchanged
code, source files, selection and frozen cache hashes. Completed sidecar hashes
are checked on resume. Failed completed files remain failures; inspect their
diagnostics and use a fresh output for a changed/fixed run. Exit 2 indicates
capture/binding failures, not that every bound expression should have rendered.

## Read the extraction report

`report.json` records:

- Actual selected/processed steps and per-file capture failures.
- `bound_sidecars`: newly captured data passes original source bindings.
- `semantic_rescues`: syntax-only V6 failed, overlaid V6 now renders.
- Per-row render errors, reconstruction counts and emitted action text.
- `native_transition_measured: false`: extraction is not Lean replay validity.

A bound sidecar may still contain unsupported scopes, metavariables or other
lossy V2 nodes. Preserve those decoder failures for the next implementation
decision. Do not extrapolate a small val sample into a >50k train recovery.

## Native-check only newly recovered semantic actions

Extraction writes `native_pilot_rows.json`, at most 24 recovered actions from
the selected split, balanced by family. If `native_pilot.selected_steps` is
zero, inspect binding/render errors instead of invoking an empty pilot.

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --semantic-cache-root maths_ai/gnn_inference/runs/template_v6_semantic_val_probe \
  --renderer template-v6 --split val \
  --row-indices maths_ai/gnn_inference/runs/template_v6_semantic_val_probe/native_pilot_rows.json \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_template_v6_semantic_probe \
  --file-timeout 600
```

An explicit semantic overlay replaces original V2 lookup; it does not silently
fall back to invalid old sidecars. Raw/model/V3 caches still come from the frozen
cache root. Overlay location and content hashes bind native resume. V6 summaries
now include `scoped_tactic_parsed_steps` and `scoped_pattern_parsed_steps`; the
older-schema-only counters in the first V6 pilot did not count these targets.

After native checking, a full read-only paired audit can use the same overlay:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_scoped_recovery \
  --previous-recovery-dir maths_ai/gnn_inference/runs/template_v5_recovery \
  --semantic-cache-root maths_ai/gnn_inference/runs/template_v6_semantic_val_probe \
  --output-dir maths_ai/gnn_inference/runs/template_v6_semantic_probe_coverage \
  --pilot-actions 24
```

Only sampled val rows have fresh sidecars in this overlay, so train semantic
recovery remains unmeasured. Audit adds `semantic_trace_error_counts` alongside
status counts and retains syntax-only support separately.

## Local verification

**156 focused tests pass** with pinned native backend. New end-to-end native
test starts from a frozen unsupported method expression, performs isolated
re-capture, recovers its receiver-preserving application, matches original and
rendered transitions, and successfully reuses a completed extraction checkpoint.
It rejects changed annotated syntax, changed normalized input and changed
completed sidecars. Paired overlay tests preserve original artifacts and verify
that missing overlay rows do not fall back to original V2 data. Server sample
yield and semantic native pilot remain pending.
