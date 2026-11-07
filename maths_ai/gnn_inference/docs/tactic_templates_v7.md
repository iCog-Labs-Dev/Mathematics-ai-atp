# V7: method selectors from existing source caches

V7 targets method calls such as `h.trans h₂`, `hu.fderiv_right (m := 0) le_rfl`,
and `specialize hn (n + 1) (n.le_add_right 1)` without a new semantic extraction
pass. Select `--renderer template-v7`. This is a source action representation;
full-dataset coverage is measured by the paired audit below.

## Representation

An unannotated method head no longer requires a guessed global declaration:

```text
application
  method_selector
    receiver: local context index of h
    label: trans
    style: dotted_identifier
  argument: local context index of h₂
```

Selector names are syntax labels, like named-parameter labels, not library
premise pointers. The receiver remains an exact input context pointer or a
lexical binder slot. Already-resolved declarations retain their existing
reference representation and spelling guards.

V7 preserves positional/named arguments, types, holes, arithmetic and binder
scopes through the existing typed term grammar. It supports bounded selector
chains and explicit projection syntax on parsed receiver terms. Original
dotted identifiers render as dotted identifiers; explicit `(receiver).field`
renders with its receiver grouping. The distinction is serialized and retained
in canonical forms. This preserves original Lean name lookup rather than
forcing a different method elaboration shape.

Schemas: `source_selector_tactic_v1` and `source_selector_pattern_v1`, reusing
wrapper and pattern families from V6. Each new `method_selector` term carries
one receiver, a bounded name label and its source style, with no fabricated
global identity. Target/reference counters count the receiver separately;
`method_selector_count` records the additional syntax nodes.

Unknown bare names, unknown receiver roots, duplicate receiver names,
inaccessible locals, conflicting annotations and malformed syntax remain
exclusions. Exact lexical names take precedence over selector interpretation.
Explicit `_root_` names retain global lookup. Existing node/depth budgets still
apply. Lexical shadowing uses binder slots, not the similarly named input
declaration.

V7 does not load V2 sidecars, use semantic whole-expression fallback or consume
an optional semantic overlay. Passing semantic data directly does not enable
it. V1–V6 remain separate modes, including V6's optional semantic recovery.
All V7 counts are static support candidates until native transition checking.

## Measure all frozen train/val rows first

From the server repository root in `atp`, after syncing published code:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_selector_recovery \
  --previous-recovery-dir maths_ai/gnn_inference/runs/template_v6_recovery \
  --output-dir maths_ai/gnn_inference/runs/template_v7_recovery \
  --pilot-actions 24
```

Use the original completed V6 report with **148,869 train / 2,820 val** support
on **218,727 / 4,157** selected rows. The runner verifies its immutable report/
rows and original baseline/cache digests, then evaluates V7 once per valid row.
Quarantine remains part of the denominator. No earlier renderer stages, dataset
download, Lean source compilation or semantic extraction run.

Console output is one compact old/new/gain/loss/exact-gain/coverage table.
Detailed rows, current errors and per-family summaries go into `report.json`
and `rows.jsonl`. Output checkpoints every 10,000 rows; choose a fresh directory
because this static audit does not resume.

Important fields:

- `template_v7_renderable_steps`, `v7_rescues_vs_v6`, `v7_losses_vs_v6`.
- Exact-aligned support/rescues/losses for dataset-step eligibility.
- `method_selector_steps`, `method_selector_nodes`, `method_selector_rescues`.
- `other_source_repair_rescues`: changes since the saved V6 version that do not
  require a new selector node, including the prior source-preserving repair.
- `coverage_percent`, quarantine and current per-family error counts.

Manifest explicitly records `semantic_cache_used: false`. Optional V2 files
can be absent or stale without affecting this audit. A completed report exits
2 if paired losses occur; inspect those rather than silently dropping rows.
If the supplied V6 report included semantic-only support, V7 may lose those
actions because this comparison deliberately measures source-only support.

## Native sample after measuring coverage

The audit writes at most 24 newly supported exact val IDs in
`new_val_pilot_rows.json`, round-robin by family then feature group. This is a
selection file only; the audit does not start native replay. For a nonempty
selection, use the existing replay runner with `--renderer template-v7` and
those `--row-indices`, the frozen raw/model/V3 cache root, and a fresh output
directory. Do not supply `--semantic-cache-root` for V7.

Native summaries include `source_selector_tactic_parsed_steps`,
`source_selector_pattern_parsed_steps`, `method_selector_steps` and
`method_selector_nodes`. Source/cache/implementation hashes bind resume as in
earlier versions. Exact all-goal transition parity remains the acceptance test;
successful execution alone is insufficient.

## Verified locally

**172 focused tests passed** with the pinned native backend; two existing
Torch deprecation warnings. V7 source-only replay matches **44/44** shared
scope/method fixture actions, **27/27** V5 wrapper regression actions, and
**5/5** additional chain/projection/pattern/name-resolution cases. These checks
include the previously sidecar-dependent named/type/specialize patterns with
zero semantic reconstructions. V6's own semantic mode remains tested separately.

Synthetic paired cohorts verify immutable artifacts, full denominators,
quarantine retention, exact-only delta sampling, stale-input rejection and
semantic-cache independence. Tests also distinguish selector labels from global
references, cover lexical shadowing, quoted labels, ambiguous/missing receivers,
rooted globals, namespace conflicts, budgets and ignored semantic input.

Actual V7 full-dataset gains and real-Mathlib native delta results remain
pending. Do not extrapolate fixture success into a row-count or >50k gain.
