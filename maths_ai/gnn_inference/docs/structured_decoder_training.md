# Structured-decoder pilot training

## Implemented learner

`atp_lean_gnn/structured_decoder.py` implements `structured_source_decoder_v1`:

- A typed-token embedding/output head and GRU command writer.
- Input-local and library pointer scoring, initialized from accepted joint
  decoder projections, GRU, tactic embeddings and text-vector adapter.
- A new lexical-declaration memory/key projection for bound references.
- Row-balanced token cross-entropy plus reachable pointer cross-entropy.

The accepted GNN backbone/tactic classifier, text encoders and lemma index stay
frozen. New decoder modules are independent trainable copies; the optimizer has
no encoder parameters. GRU memory is within one action, not across proof steps.
Native search remains responsible for cross-step proof state.

## Exact retrieval and graph alignment

`structured_features.py` verifies the original candidate-cache manifest hash
against bundle provenance, and checks array hashes and index compatibility.
The original epoch-5 dataset-state query results are reused unchanged. A human
model-path alias may differ after clean-root export; vector/corpus identities
and the provenance-bound original manifest must still match.

Normalized input graphs are rebuilt from the bound model S-expressions, using
the accepted node vocabulary/edge policy. Context indices map to their actual
FV nodes, including when vocabulary labels are OOV. Only one input FV key per
context identity is used by the new local head. Operation-specific masks remove
captured/inaccessible locals and out-of-scope lexical declarations.

Frozen state/local vectors and target arrays are saved in bounded CPU shards.
Library vectors remain in one shared, hash-bound matrix; each row stores only
retrieved IDs/scores/positions and teacher-feedback identities. Materialization
is resumable, and completed caches do not re-encode graphs or re-query retrieval.
Unbound/tampered shards, arrays, source caches and vocabulary changes are rejected.

Graph batches default to 16 graphs / 30,000 total nodes. Individually oversized
graphs are explicit feature exclusions, retained in validation denominators.
Training batches default to 16 actions / 8,192 padded tokens, using deterministic
shard-aware length bucketing to avoid padding every action to 595 tokens.

## Missing premises and teacher forcing

Missing gold library declarations **never enter the retrieved candidate pool**.
Their library identity loss is skipped and counted as unreachable. Token and
other reachable-reference losses still train on that row. For later teacher-
forced steps, the previous gold reference can supply its frozen matched-index
vector, separately from candidate keys. The current target ID is not supplied
before its token/pointer prediction.

Validation uses gold tactic conditioning and previous gold tokens/references.
Metrics explicitly report teacher-forced token/byte/structure accuracy, pointer
accuracy given source/retrievability, missing-premise counts and exact-label hits.
Every unreachable action is an exact miss. The **500 original candidate rows**
remain the denominator even though only 496 compiled; resource exclusions and
unvisited rows remain visible too. Full-data runs inherit their own candidate
denominators, rather than hard-coding 500.

These diagnostics do not measure free-running action execution or theorem
success. Autoregressive structured inference and integration of trained weights
into best-first search are the subsequent model-evaluation milestone.

## Training runner and checkpoints

Entry point: `scripts/train_structured_decoder.py`.

Required inputs:

| Argument | Existing input or new output |
| --- | --- |
| `--compiled-dir` | Compiled typed-action package |
| `--bundle-dir` | Accepted `artifacts/hf_joint_model` |
| `--candidate-cache` | Existing `runs/sexpr_text_library_epoch_5` |
| `--feature-cache` | New frozen-feature cache, outside source roots |
| `--run-dir` | New model/checkpoint output, outside data/cache roots |

Defaults: CUDA device 0, 10 epochs, batch size 16, 8,192 padded tokens, AdamW
learning rate `3e-4`, weight decay `1e-4`, gradient clipping 1, seed 42, atomic
checkpoint every 25 updates. Frozen features are prepared automatically before
the optimizer starts.

`--max-updates 100` performs an actual bounded training smoke, then validates and
saves. It is a **global completed-update budget**, not a per-epoch sample limit.
`--resume --max-updates 0` continues from the saved epoch/batch cursor without
resetting optimizer/RNG/update counts. Requested epochs may be extended. Data,
vocabulary, feature bindings, architecture and optimization hyperparameters must
match; changed compiled/full-data packages require new cache/run directories.

`last.pt` stores new decoder parameters, optimizer, cursor, history and RNG state.
`best.pt` uses teacher-forced exact-label rate with validation loss as a tie-break,
not a theorem solve score. `report.json` records current counters and diagnostics.
Interrupted feature writes adopt only provenance-bound orphan shards. Training
interruptions retain the last completed atomic checkpoint.

## Verification and outstanding server run

Combined focused suite: **126 tests pass**. New checks exercise real optimizer
updates and loss reduction, frozen parent weights, legal scopes, unreachable
library misses, original retrieval reuse, shard integrity/recovery, unchanged
validation denominators, CLI optimizer/checkpoint creation, and resumed updates
matching uninterrupted updates. Local learning checks use small controlled CPU
fixtures. The actual 5,000-example server training run has not yet been measured.
