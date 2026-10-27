# LeanDojo Data Preprocessing

This directory contains data preprocessing scripts for filtering and formatting formal theorem proving datasets.

## Scripts

### `filter_local_leandojo.py`
Filters theorems from LeanDojo (Lean 4 / Mathlib4) that can be proven using **purely local context declarations and structural tactics** with zero calls to external lemmas or automated global solvers (`simp`, `linarith`, `aesop`).

#### Usage:
```bash
python data_preprocessing/filter_local_leandojo.py --output_dir ./data_preprocessing/output --split train --max_samples 2000
```

#### Outputs:
1. `local_theorems_{split}.jsonl`: Complete theorem objects and proof steps.
2. `local_state_tactic_pairs_{split}.jsonl`: Formatted `[GOAL] state -> tactic` training pairs.
