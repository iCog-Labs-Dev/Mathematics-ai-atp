# Semantic-name lookup and bounded applied rules

This extends gold parsing/rendering, before prediction-head training or target
export. Two fixes follow the completed unsupported-action census:

1. Match qualified dataset names to exact validated declaration identities.
2. Preserve bounded applications inside `rw`/`simp` rules.

## Reference lookup

Dataset `Polynomial.degree_erase_lt` can match a trace leaf whose original
source is `degree_erase_lt` and whose resolved name is
`Polynomial.degree_erase_lt`. Matching uses original source spelling or the
complete resolved global identity, normalizing root/escaped spelling. It
does not infer identities from matching suffixes. Conflicting source and
semantic matches are rejected, not resolved by arbitrary precedence.

Every annotated leaf still passes the existing bare-reference check before
indexing. Receiver-bearing methods, unknown aliases and possible local
receivers do not become bare global references merely because the dataset
contains the annotated head name. Local references still require their bound
context identity/name. Renamed aliases needing native resolution remain
unsupported. The check is necessary, not proof of elaborated-term equivalence.

This lookup is used by flat fallback in all renderer modes. Python callers
can use `reference_lookup="source"` to reproduce strict spelling-only lookup.
Current audits record the new lookup policy; use the previously generated
coverage artifact for paired recovery rather than regenerating the baseline.

## V2 grammar and targets

Select `--renderer template-v2` for schema `rw_simp_template_v2`:

- Same rw/simp families, three-rule limit, rw directions, optional simp `only`
  and goal/one-local location as V1.
- Each rule has a local/library reference head and **zero to three ordered
  local/library reference arguments**.
- Exact `Lean.Parser.Term.app` structure is required. Transparent term
  parentheses around the rule, head or bare argument can be removed.
- Nested argument applications, applied heads, named arguments, literals,
  holes, arbitrary expressions, binders, nested proofs, config/dischargers,
  combined locations and reverse/erase simp arguments remain unsupported.

For example, `rw [if_pos h]` has one rule, not two:

```json
{
  "schema": "rw_simp_template_v2",
  "family": "rw",
  "rules": [{
    "reference": {"source": "library", "identity": "if_pos"},
    "arguments": [{"source": "local", "identity": 30}],
    "backward": false
  }],
  "only": false,
  "location": {"kind": "goal", "reference": null},
  "rule_count": 1,
  "reference_count": 2
}
```

References use source-context indices or full declaration identities, not
graph/corpus indices. Three heads, three arguments per rule and one location
allow **13 reference occurrences**; this is not the legacy three-total-argument
decoder. V1 remains bare-reference-only and preserves its existing serialized
target format. Model weights and prepared labels are unchanged.

## Measure actual recovery on the frozen cohort

After pulling published code, run from server repo root in `atp`:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_template_recovery \
  --baseline-coverage-dir maths_ai/gnn_inference/runs/template_v1_coverage \
  --output-dir maths_ai/gnn_inference/runs/template_v2_recovery
```

This CPU-only audit reuses the original **218,727 train / 4,157 val** rows.
It verifies baseline row digests/populations and every recorded cache hash,
including renderable rows. Invalid baseline traces remain quarantined. Input
graphs/caches/reports are read only. Output checkpoints are incomplete until
finished; use a fresh output directory, with no resume support.

Each split compares three stages:

1. Frozen V1 baseline, originally **66,904 train / 1,322 val** renderable.
2. V1 grammar with fixed semantic-name lookup.
3. V2 grammar with the same fixed lookup.

Inspect `lookup_rescues/losses`, `v2_grammar_rescues/losses`, and
`combined_rescues/losses`. `applied_rule_rescues` separates actual applications
from `bare_rule_rescues` due to parenthesis-only normalization. Exact-aligned
counts separate potential dataset-step support from source-only cases.
`by_family`, `by_previous_error`, `v2_error_counts` and diagnostic `rows.jsonl`
expose what recovered and what remains excluded. Parsed targets can still
fail reference rendering. Exit 2 after a complete report means a paired loss
was detected; inspect it rather than treating positive net coverage as enough.

The targeted census buckets contained **20,817 train / 378 val** rows, not a
promised gain. Bounded argument shapes and secondary blockers limit recovery;
parenthesis normalization can additionally recover other categories. All
these coverage numbers are static, not full-cohort Lean execution or model
accuracy. Completed server recovery counts follow.

### Completed paired server result

Same **218,727 train / 4,157 val** cohort on `63a359b`:

| Static renderability stage | Train | Val |
|---|---:|---:|
| Frozen V1 baseline | 66,904 | 1,322 |
| Lookup-fixed V1 | 72,837 | 1,436 |
| V2 grammar + lookup | 78,324 | 1,531 |
| Lookup rescues | 5,933 | 114 |
| Applied-rule rescues | 5,482 | 95 |
| Parenthesis-only bare-rule rescues | 5 | 0 |
| Combined rescues | 11,420 | 209 |
| Combined losses | 0 | 0 |

Both intermediate stages also have **zero losses**. Static coverage rises
**30.59% to 35.81% train / 31.80% to 36.83% val**. These are increases of
**17.07% / 15.81% relative to baseline renderable populations**, not model
accuracy gains.

Exact-aligned combined support rises **61,836 to 72,197 train / 1,317 to
1,526 val**, recovering **10,361 train / all 209 val** exact-aligned rows
with no losses. The other **1,059 train rescues** have nonexact/unknown
alignment and cannot automatically become dataset-step training labels.
Quarantine remains **831 train / 9 val** rows.

V2 typed parsing succeeds on **43,837 train / 870 val** rows. Applied-rule
targets parse on **5,507 / 95**; **5,482 / 95** yield newly renderable
actions. Thus **25** parsed train applications still fail rendering.
Report: `maths_ai/gnn_inference/runs/template_v2_recovery/report.json`.

## Same-cohort native check

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_template_v2 \
  --renderer template-v2 --split val --num-theorems 20 --seed 42 --file-timeout 600
```

Captured-syntax control and exact-alignment separation stay in force. Reports
bind lookup policy, schema and implementation hashes. V2 parsed counts are
reported separately from V1 and applied-rule counts. Use a new report directory.
This small theorem cohort is not a guarantee that every newly recovered full-
cohort row has been exercised; native validation of recovered cases remains
a gate before training/export.

Server V2 replay on the same **86-step / 20-theorem** cohort completed:
**86/86** input and captured-syntax original matches, **85/85** exact-aligned
original matches and **30/30** exact-aligned rendered matches. V1 recheck
had 26 rendered matches on this cohort. V2 parses **21** typed targets,
including **2** applied-rule targets; current flat rendering supports **17**
rows. Remaining **56** rows are explicitly unsupported; no replay failures.
The single branch-opener row remains source-only. Reparsed-text diagnostic
still matches **85/86**. Whole-proof success is not measured. Report:
`maths_ai/gnn_inference/runs/native_gold_replay_template_v2/report.json`.

Next gate: native replay specifically covering the **209 newly renderable
exact-aligned val rows**, with separate lookup/application results, followed
by checks on recovered exact-aligned train cases. The 30 small-cohort native
matches do not certify all static rescues. Versioned graph/corpus reference
mapping and target export follow that validation.

Verification: **83 focused tests passed**, including the rebuilt pinned Lean
backend. New native fixture verifies **9/9** rendered source transitions and
**14/14** original controls, covering qualified dataset/source spellings,
`if_pos h`, local heads, global arguments, parentheses, three arguments and
13 reference occurrences. Valid source examples with nested/named/literal/
hole arguments or four arguments remain rejected by V2. Unit/paired tests
check ambiguity/method guards, V1 serialization, bounds, changed snapshots,
exact versus source-only recovery, loss detection and renderer resume isolation.
