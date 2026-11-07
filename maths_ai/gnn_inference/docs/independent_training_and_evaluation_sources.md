# Independent RL training and evaluation inputs

Training and evaluation now admit separate sources. The driver no longer removes an evaluation subset from the training pool.

For the pinned Mathlib-Normalized-Sexpr revision `f44841f98193c09b20c6092198a22e98ff2cd7a8`, use `train` for training, `val` for periodic validation/model selection, and `test` for final evaluation. The spelling `validation` is accepted and resolves to the published split `val`.

## Configuration

The old flat data/admission fields are retired. Define `training_data` and `evaluation_data` instead:

```json
{
  "training_data": {
    "data_source": "dataset",
    "dataset_name": "jajostrains/Mathlib-Normalized-Sexpr",
    "dataset_revision": "f44841f98193c09b20c6092198a22e98ff2cd7a8",
    "split": "train",
    "max_pool_size": 500,
    "max_scan_rows": 2500,
    "max_state_chars": null,
    "max_replay_chars": 50000,
    "admission_time_budget_s": 600,
    "seed_cache_path": "runs/train-admission-cache.json"
  },
  "evaluation_data": {
    "data_source": "dataset",
    "dataset_name": "jajostrains/Mathlib-Normalized-Sexpr",
    "dataset_revision": "f44841f98193c09b20c6092198a22e98ff2cd7a8",
    "split": "val",
    "max_pool_size": 20,
    "max_scan_rows": 1000,
    "max_state_chars": null,
    "max_replay_chars": 50000,
    "admission_time_budget_s": 600,
    "seed_cache_path": "runs/val-admission-cache.json"
  }
}
```

These blocks accompany the existing checkpoint, retrieval, Lean runtime, search, and optimizer fields. Keep each source's cache path separate.

`max_pool_size` is the number of Lean-validated states requested from that source. In this example, training can use 500 states and evaluation can use 20 additional validation states. Admission stops on its pool, scan, or time bound; undersized pools are reported. An empty required source fails rather than switching to another split. States are selected by bounded source scanning, then ordered deterministically by size and input identity. They are not a uniform random sample of the full split.

For file inputs, replace dataset fields with `data_source: "file"` and `theorem_file`. Training and evaluation must identify different effective sources. Explicit file examples use separate JSONL files; callers no longer request a held-out partition from one file.

## Launching training and evaluation

The updated `configs/rl_puct_pln_large.json` retains the previous `best.pt` warm start and training/search budgets, but trains on `train` and evaluates on `val`. It now requests 500 training states rather than the former 480 training plus 20 train-split evaluation states.

```bash
conda activate atp-mcts
cd /home/jovyan/Nol/Mathematics-ai-atp
source "$CONDA_PREFIX/opt/pln/activate.sh"

python -u -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/rl_puct_pln_large.json
```

Use a fresh run when adopting these source semantics. Do not resume a run using the old partition.

For standalone validation, only `evaluation_data` is admitted:

```bash
python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/eval_puct_pln_epoch5.json \
  --eval-only \
  --eval-output-dir reports/puct-pln-validation
```

For final test evaluation:

```bash
python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/eval_puct_pln_test_epoch5.json \
  --eval-only \
  --eval-output-dir reports/puct-pln-test
```

Both example evaluation configs use the previously supplied `best.pt`. To evaluate newly trained weights, pass `--checkpoint /absolute/path/to/new/run/best.pt`. Output directories must be new. Keep test inputs out of routine checkpoint selection.

If overriding dataset identity from the CLI, use `--training-dataset-name`, `--training-dataset-revision`, `--evaluation-dataset-name`, or `--evaluation-dataset-revision`. The corresponding source block must already be a dataset source. Split selection and admission limits live in the source block.

## Reports, archives, and resume

Training writes admission reports under `<run_dir>/admission/training/` and `<run_dir>/admission/evaluation/`. Each role records its source contract, actual ordered pool identity, environment, counts, and report digest. Known admitted-state/theorem overlaps are reported without silently removing records.

Standalone evaluation admits no training data. Its proof archives record the actual evaluation source, and every attempt preserves its original dataset split or file provenance. Successful traces and kernel receipts continue to be exported normally.

A checkpoint now binds exact resume to configured sources and the actual admitted pools, including labels and ordering. A changed split, revision, file content, filter contract, or resulting pool requires a fresh run. Cache/report timestamps do not redefine the input pool. A cache path alone is not part of the source's semantic identity.

Older checkpoints lacking this data contract remain usable as model-weight warm starts if their model/retrieval contracts match. They cannot resume exactly under the new source semantics. Existing historical reports and archived run configs are not rewritten. Comparisons against the former train-heldout results must acknowledge the changed evaluation inputs.

## API and deployment

Each server-owned `dataset_profiles` entry now uses the same source fields above. Its `max_pool_size` is the number of evaluation inputs to admit; remove the old `eval_pool_size`. API evaluation directly uses that admitted list, with the same deterministic ordering as the CLI. Requests still select an allowlisted profile name. `/prove` does not admit a dataset.

New generated deployments use source-scoped cache paths. Previously published serving configs with retired flat fields must be migrated or republished as configuration; do not edit files inside a hash-verified deployment bundle in place. The model weights, retrieval artifacts, REPL, and client do not need conversion or rebinding solely for this change.

Distinct split names establish source separation, not that a checkpoint never saw related examples during supervised training. Admission diagnostics inspect only the admitted subsets and do not prove full-dataset theorem disjointness.

## Implementation verification

Verification used the repository-pinned client from an isolated temporary installation and the matching custom REPL build.

- The focused CLI, admission, evaluation-export, deployment, analysis, and API regression invocation passed 214 tests; six opt-in tests were skipped.
- Final source/deployment/lineage checks passed 59 tests; one native test was skipped in that offline invocation. These suites overlap and their counts must not be added.
- The separately enabled native test passed independent admission of two JSONL sources, source-specific reports/caches, and kernel-certified evaluation with proof exports under both best-first and PUCT search.
- All authored runtime configs parsed, and enabled training/evaluation phases have explicit sources. Docker service configuration parsed; `git diff --check` passed.
- Full live HTTP verification was attempted twice. Both attempts failed before dataset admission because the Lean worker exceeded its startup timeout. API unit/lifecycle tests and direct native admission/certification passed, but the HTTP end-to-end result is not claimed as verified.

The actual Hugging Face validation/test pools and trained GPU checkpoint were not run in this local verification. Their paths and runtime artifacts remain JupyterHub-specific. No image build or new model-quality result is implied.
