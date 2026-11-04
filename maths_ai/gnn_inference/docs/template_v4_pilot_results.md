# V4 application pilot and remaining rule coverage

The first focused server pilot passed **24/24** rendered transitions, input
matches, captured-syntax controls and exact dataset alignments. All 24 are
`recursive_application_v1`; zero are `recursive_rw_simp_v1`. There are no
failures in this sample, but it does not exercise the new recursive rule path.
No whole-theorem proof success or model accuracy is measured.

Inspection found a selection imbalance: feature groups were sorted by family,
and a 24-group cap could fill entirely with application groups before reaching
rw/simp. Future audits now round-robin families first, then feature groups.
Selection is still deterministic and independent of Lean outcomes. Existing
successful results remain evidence for their selected application instances.

## Complementary rule-only pilot, using saved rows

Pull published code, then run from server repo root in `atp`. This reads the
existing audit and writes only a new selection file; no full audit rerun:

```sh
python - <<'PY'
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from maths_ai.gnn_inference.scripts.audit_template_recovery import select_native_pilot

root = Path("maths_ai/gnn_inference/runs/template_v4_recovery")
report = json.loads((root / "report.json").read_text())
rows_path = root / "rows.jsonl"
assert report["complete"] and hashlib.sha256(rows_path.read_bytes()).hexdigest() == report["row_records_sha256"]
groups = defaultdict(list)
with rows_path.open() as handle:
    for line in handle:
        row = json.loads(line)
        if (row["split"] == "val" and row.get("v4_schema") == "recursive_rw_simp_v1"
                and row["v4_renderable"] and not row["v3_renderable"]
                and row["alignment_kind"] == "exact" and row["dataset_target_matches_source"]):
            rank = hashlib.sha256(str(row["row_index"]).encode()).hexdigest()
            groups[(row["family"], tuple(row["v4_term_features"]))].append((rank, row["row_index"]))
selected = select_native_pilot(groups, 6)
output = root.parent / "template_v4_rule_pilot_rows.json"
with output.open("x") as handle:
    json.dump(selected, handle)
print("Rule-only pilot rows:", selected)
print("Selection:", output)
for split in ("train", "val"):
    print(split, "V4 residual errors:", json.dumps(report["splits"][split]["v4_error_counts"]))
PY
```

Run native replay on those at most six actions:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --renderer template-v4 --split val \
  --row-indices maths_ai/gnn_inference/runs/template_v4_rule_pilot_rows.json \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_template_v4_rules \
  --file-timeout 600
```

The residual errors guide the next scope/term/extraction expansion. The six
earlier V3 reference-control failures remain separate diagnostics. Full train
validation was stopped at user request; this pilot does not restart it.
