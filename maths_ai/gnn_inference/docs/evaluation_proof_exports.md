# Certified proof exports from evaluation

Both periodic evaluation during RL training and standalone `--eval-only` now save evaluation evidence automatically. This applies to best-first and PUCT search and to greedy and sampled proposals. Training exploration is not exported by this feature.

For a standalone evaluation from the repository root:

```bash
python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/rl_mcts_joint_epoch5_trial.json \
  --eval-only \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --eval-output-dir reports/eval-final
```

The output directory must not already exist. Without `--eval-output-dir`, the driver creates a unique directory under the configured `run_root/evaluations/`. The optional `--eval-report` argument still writes the aggregate report to its requested location.

Periodic evaluations create unique directories under the training run's `evaluation/` directory. Their names include the zero-based training round. The console prints the archive location; the corresponding evaluation metrics include `archive` and `exported_proofs`.

Each archive contains:

- `evaluation.json`: evaluation provenance and status (`running`, `completed`, or `failed`).
- `attempts.jsonl`: one record per finished attempt, with input identity, source, repeat, proposal seed, outcome, and a relative `proof_file` for certified successes.
- `proofs/*.json`: starting state, successful nested proof trace, replay artifact, and certification receipt for each certified attempt. Failed exploratory alternatives are excluded.
- `summary.json`: the aggregate report, written when evaluation completes.

Input indices are positions in the admitted evaluation subset, not necessarily source-dataset row numbers. Use `source` to locate the dataset record. Repeats preserve separate attempt and proof records, even for identical starting states. Worker completion order may differ from input order.

Proof files are written before their ledger references, and complete ledger records are flushed and synchronized as attempts finish. After interruption, retain the entire archive. `running` indicates that evaluation never completed its final manifest update. A reader should reject or explicitly report a truncated final JSONL line. An unreferenced proof file can exist if interruption happened between proof publication and ledger append; it must not be counted as a completed ledger attempt. Archive write failures stop evaluation rather than silently losing evidence.

The successful trace is a nested obligation tree; the replay artifact also contains the ordered executable witness steps. The certification receipt contains statement/proof digests and environment evidence. It is not a serialized Lean proof term or a standalone `.lean` theorem. Rechecking the replay artifact requires a compatible Lean environment and the existing replay verifier.

Standalone provenance identifies the checkpoint by path and SHA-256. Periodic provenance identifies the run, round, model generation, and initial checkpoint. The initial checkpoint is not the updated model evaluated in that round. Trace export does not create an additional weights checkpoint for each evaluation; retain the run's checkpoints if you also need those weights.

No additional REPL rebuild, client installation, artifact rebinding, or model conversion is needed for this application change. Earlier aggregate-only runs cannot supply missing tactic traces; rerun evaluation on their saved checkpoints to export proofs.
