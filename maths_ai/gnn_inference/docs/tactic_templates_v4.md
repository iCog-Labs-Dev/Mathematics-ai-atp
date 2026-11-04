# V4 bounded recursive terms

`--renderer template-v4` extends gold representation beyond flat reference
arguments. Existing V1/V2/V3 modes retain their grammar; introduction targets
are reused unchanged. This does not train a new model or rewrite labels.

## New support

- Nested applications and applied heads: `exact f (g h)`, `exact (f hp) hq`.
- Positional/named recursive arguments, including rewrite/simp rule terms.
- Anonymous constructors: `exact ⟨hp, hq⟩`, `refine ⟨?_, ?_⟩`.
- Anonymous implicit `_` and explicit `?_` holes, with distinct goal slots.
- Numeric literals, negation and explicit type ascriptions using supported terms.
- Core `<|` application syntax, preserving its grouping.
- Structural projections with the receiver retained: `exact (id h).left`.
- `exact`, `apply`, `refine`, `refine'`, `rw`, `simp`, and existing introductions.

Schemas are `recursive_application_v1` and `recursive_rw_simp_v1`. Each tree
node has an explicit kind, typed references, ordered children and bounded
scalar payloads. No uninterpreted source-text leaf is accepted. Limits:
256 term nodes per action, depth 16, up to 16 children/arguments, 16 rules,
and numeric tokens up to 256 characters.

References remain source-bound and pass existing spelling/context/private-name
guards. A projection must have explicit `Lean.Parser.Term.proj` structure;
the receiver is not inferred by splitting a dotted identifier. `exact h.left`
may therefore remain unsupported while `exact (id h).left` is representable.
Lambda/let scopes, arbitrary nested tactic proofs, unrecognized notation,
unresolved value identifiers, and named holes remain unsupported. Anonymous
holes are new obligations, not premise IDs, and native checks preserve all goals.

## Measure actual recovery

After pulling published code, from server repo root in `atp`:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_template_recovery \
  --baseline-coverage-dir maths_ai/gnn_inference/runs/template_v1_coverage \
  --renderer template-v4 --pilot-actions 24 \
  --output-dir maths_ai/gnn_inference/runs/template_v4_recovery
```

The original frozen cohort stays fixed. V3 should reproduce **93,113 train /
1,823 val** renderable rows. Inspect `template_v4_renderable_steps`,
`v4_rescues_vs_v3`, `v4_losses_vs_v3`, exact-aligned counterparts,
recursive-rule/application rescues, `by_family` and `v4_error_counts`.
Constructor/hole/numeral/projection rescue counters overlap and must not be
summed. Completed server recovery follows.

### Completed paired server result

| Static support or recovery | Train | Val |
|---|---:|---:|
| V3 renderable | 93,113 | 1,823 |
| V4 renderable | 107,913 | 2,096 |
| New recursive-rule actions | 4,019 | 58 |
| New recursive-application actions | 10,781 | 215 |
| Total rescues versus V3 | 14,800 | 273 |
| Losses versus V3 | 0 | 0 |
| Exact-aligned V4 support | 100,265 | 2,091 |
| Exact-aligned rescues | 14,021 | 273 |

Coverage rises **42.57% to 49.34% train / 43.85% to 50.42% val**. The
remaining unsupported-or-invalid population is **110,814 train / 2,061 val**.
The **779** new train actions outside exact alignment remain source-only or
unknown-aligned candidates, not dataset-step labels. Quarantine remains
**831 train / 9 val**; earlier V3 losses remain separately recorded.

Among rescued rows, constructor features occur **2,395 / 43**, holes
**9,397 / 181**, numerals **578 / 4**, and structural projections **3,066 /
69**. These feature counts overlap. A represented hole denotes a pending
obligation; it does not count as a completed proof. This result is static
representation recovery, not full-cohort native validity or model accuracy.
Report: `maths_ai/gnn_inference/runs/template_v4_recovery/report.json`.
The focused new-action native pilot remains pending.

The audit also writes `new_val_pilot_rows.json`: at most 24 newly renderable,
exact-aligned val actions, chosen deterministically across family/feature groups.
Check `native_pilot.selected_steps` in the report; if zero, there is no pilot
to run. This is a targeted sample, not another exhaustive train-form pass.

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --renderer template-v4 --split val \
  --row-indices maths_ai/gnn_inference/runs/template_v4_recovery/new_val_pilot_rows.json \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_template_v4_pilot \
  --file-timeout 600
```

Fixed-row mode validates every selected dataset ID, binds the selection-file
hash, and fails on missing/duplicate IDs. Only rendered transition matches
pass; unsupported candidates remain failures. Native reports checkpoint per
file and support `--resume` with identical inputs. Use fresh output directories.
The six original-control failures from exhaustive V3 val validation remain
separate diagnostic work; V4 does not relabel them as renderer successes.

Verification: **126 focused tests passed**. New pinned-native fixture verifies
**19/19 rendered transitions and 22/22 original controls**, including two pending
constructor-hole goals, inferred rewrite holes, explicit type ascriptions,
recursive named arguments and a constructor member whose receiver is not its
first explicit argument. Three valid originals with ambiguous dotted syntax,
lambda scope or nested tactic proofs remain excluded. Paired tests check
losses, exact versus nonexact recovery, immutable inputs and bounded pilot
selection. Full-cohort native validity and model accuracy are not measured.
