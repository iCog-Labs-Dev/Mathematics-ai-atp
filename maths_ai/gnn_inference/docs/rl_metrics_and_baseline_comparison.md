# Analyze RL runs and compare checkpoints

Run these commands from the repository root in your activated JupyterHub environment. The analyzer reads saved files; it does not load checkpoints, start Lean, or run optimization. It uses the project's evaluation metric definitions and therefore needs the project's Python dependencies installed.

## Inspect a completed training run

Set `RUN` to the timestamped directory printed as `Run dir:` during training. Choose the actual timestamp, rather than the parent containing several runs:

```bash
conda activate atp-mcts
export RUN="/home/jovyan/Nol/Mathematics-ai-atp/runs/rl_pln_best_first_full/REPLACE_WITH_RUN_TIMESTAMP"

python -m maths_ai.gnn_inference.scripts.analyze_rl_run \
  --run-dir "$RUN" \
  --output "$RUN/analysis_training" \
  --window 20
```

For the run without PLN, use its timestamped directory under `runs/rl_puct_no_pln_full`. For the short trial, use the corresponding trial run directory and `--window 2`.

The new output directory contains:

- `report.md`: readable training, optimizer, PLN, and evaluation summaries.
- `summary.json`: machine-readable summaries and comparison checks.
- `training_rounds.csv`: one row per training round, including the original scalar metrics.
- `evaluations.csv`: periodic held-out evaluation results, separated from training rows.

Existing output directories are refused. Choose a different output name when analyzing again. An incomplete final JSONL line is ignored with a note so a run can be inspected while writing. For final analysis, wait until training has finished. Duplicate rounds are rejected instead of silently counted twice.

## Evaluate the base and trained checkpoints

Training round zero already follows an update. It is not an untrained baseline. Comparing its actor loss with a final proof rate does not measure improvement. Evaluate the converted version-4 warm-start checkpoint and the trained checkpoint using the same saved configuration.

This reads the baseline path recorded by that run:

```bash
export BASE="$(python - <<'PY'
import json
import os
from pathlib import Path

config = json.loads((Path(os.environ["RUN"]) / "config.json").read_text())
print(config["warmstart_checkpoint"])
PY
)"
```

If the baseline checkpoint was subsequently replaced, restore the original bytes before comparing. Its path alone cannot establish its identity.

For a PLN-enabled run, activate the same PLN installation used during training:

```bash
source "$CONDA_PREFIX/opt/pln/activate.sh"
```

Skip that activation for a run with `use_pln: false`. Run both evaluations with the same configuration, devices, deployed artifacts, and PLN runtime:

```bash
python -m maths_ai.gnn_inference.scripts.rl_train \
  --config "$RUN/config.json" \
  --eval-only \
  --checkpoint "$BASE" \
  --eval-repeats 1 \
  --eval-report "$RUN/base_eval.json"

python -m maths_ai.gnn_inference.scripts.rl_train \
  --config "$RUN/config.json" \
  --eval-only \
  --checkpoint "$RUN/last.pt" \
  --eval-repeats 1 \
  --eval-report "$RUN/trained_eval.json"

python -m maths_ai.gnn_inference.scripts.analyze_rl_run \
  --run-dir "$RUN" \
  --baseline-eval "$RUN/base_eval.json" \
  --trained-eval "$RUN/trained_eval.json" \
  --output "$RUN/analysis_vs_base" \
  --window 20
```

`last.pt` measures the final saved policy. Substitute `best.pt` to evaluate the checkpoint selected by periodic evaluation, and label that choice when reporting results. Evaluating on the same set used to select `best.pt` is not an independent test of generalization.

Both commands perform real searches and can take substantial time. They use the configured evaluation pool and rebuild admission; they do not train. Increase `--eval-repeats` equally for both evaluations to average more sampled searches. A configuration with `eval_pool_size: 0` needs a nonempty evaluation split before this comparison can run. Do not alter only one evaluation's split, search budget, or proposal mode.

The analyzer checks the ordered admitted-input digest, evaluation contract, repeat proposal seeds, search settings, budgets, dataset settings, worker count, admission environment, and PLN runtime when applicable. It suppresses the proof-rate difference if recorded conditions are missing or differ. Different admitted subsets require rerunning evaluations under matching conditions.

## Interpret the measurements

**Proof rate** divides kernel-certified solves by searches with usable evidence. The denominator removes inadmissible starting states, operationally failed searches, and resource-limited searches using the shared evaluator's definition. Hard timeouts are already included in failed searches and are not removed twice. Cooperative deadline results remain searched. Zero eligible searches give an undefined rate, displayed as unavailable.

**Coverage** is searched states divided by attempted states. **Solved per attempt** includes operational failures in its denominator. Read both alongside proof rate so resource failures do not disappear from the comparison. With repeats enabled, these counts include repeated attempts on the same admitted inputs.

**Optimizer counters** distinguish on-policy and HTPS update steps. A zero actor loss does not imply that no update occurred. Loss and return summaries show means of finite reported round values, not means weighted by training sample counts. The HTPS losses recorded by the driver describe the last HTPS step in each round; their window means do not average every HTPS minibatch.

**Training solve rates** reflect changing weights and curriculum states. First/last windows describe training progress, rather than a controlled checkpoint comparison. Short runs can have overlapping windows. Reported round durations exclude periodic evaluation and do not measure the total job duration.

**PLN counters** separate outcomes such as `pln_ok` from unsupported inputs or fallback results. Only successful rule-proof/STV validation counts as a successful PLN result. Status keys are sparse in current driver output; absent rounds contribute zero to a status that was recorded elsewhere. If no numeric PLN counters occur at all, the analyzer reports no recorded counters rather than asserting that PLN ran successfully.

**Proposal and rejection counters** are not interchangeable: raw draws, distinct rendered commands, rejected commands, successful transitions, and training observations count different events. Repeated or unrenderable actions prevent a simple one-to-one interpretation.

Current aggregate evaluation reports do not bind checkpoint bytes, device, retrieval artifact bytes, or every resource limit. Keep those fixed externally. They also do not contain per-state outcomes, so the analyzer cannot identify individual newly solved/regressed states or perform a paired significance test. A reported percentage-point difference is descriptive.

Compare each trained policy with its own base checkpoint using its own fixed profile. Comparing a PLN best-first run with a no-PLN PUCT run changes both search configuration and weights; that comparison does not isolate the effect of RL training. These proof-search measurements are also distinct from supervised tactic/argument accuracy.
