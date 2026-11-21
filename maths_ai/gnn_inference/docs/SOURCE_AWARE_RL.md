# Source-aware actor-critic migration and deployment

The AC model now predicts a tactic, an argument count, and an ordered sequence of
local or library arguments. One shared GRU decoder supplies supervised scoring,
sampling, greedy decoding, and forced probability evaluation. The canonical graph
encoder remains shared with the tactic actor and critic.

The implementation ports the architecture pinned at
`b8a3de7f450c2103a04f2b7a37ca85b113c45c96`, rather than merging its experimental
training scripts into AC. No trained checkpoint was selected for this migration.

## Changes and contracts

- `ArgumentReference` distinguishes graph-local node indices from stable library
  IDs. Local execution resolves through the model's declaration identity to Lean's
  live name. Library execution resolves through the verified ID/name table.
- `CandidatePoolSnapshot` retains ordered IDs, scores, optional tactic-specific
  type cues, graph/query fingerprints, and retrieval-policy provenance. Actions
  retain their collection pool; updates do not retrieve replacement candidates.
- The actor likelihood includes tactic, count, every source choice, and every
  within-source identity choice. The selected local vector or adapted library
  vector updates the GRU before the next choice.
- The encoder, tactic actor, count/source heads, recurrent pointer, lemma adapter,
  retrieval-score temperature, and contextual library residual can receive policy
  gradients. The critic keeps its separate validity-aware targets.
- Collection and likelihood recomputation disable dropout. Forced evaluation
  retains autograd. A collection round still completes before its single optimizer
  step, and existing replica-generation checks remain in effect.
- Argument-count supervision excludes unsupported tactic syntax. Missing retrieved
  gold lemmas do not become zero-argument targets: count supervision remains, while
  source/identity supervision stops at the unreachable prefix.
- The live retriever owns only the frozen query encoder. Lemma embeddings are
  precomputed and read-only. A shared bounded service handles requests from pool
  workers, uses a dedicated native executor, and joins active work during shutdown.
- Library adaptation is shared within one model forward and recomputed on the next
  forward. Full-corpus vectors are not copied into each policy replica.

The full-action entropy objective uses the sampled joint log probability and the
score-function surrogate specified in the plan. Metrics distinguish the sampled
entropy estimate from that surrogate. Gradient norms are reported by component.
Multiplicity weights apply to the actor objective and advantage normalization.
For singleton or constant-advantage batches, centering yields zero actor gradient;
critic, entropy, and the configured tactic anchor remain separate objectives.
The tactic anchor uses accepted sampled tactic IDs, not independently supplied
corpus labels or full-action behavioral-cloning targets.

Graph-based search success is unchanged. Complete-proof replay, retained Lean
snapshots, and kernel certification belong to the separate snapshot plan.

## Artifacts required before an RL launch

Select these together:

1. A complete source `joint_source_flat_decoder_v2` or
   `frozen_graph_local_identity_v1` checkpoint.
2. Its exact prepared root and candidate-cache manifest, including unchanged
   vocabularies, `argument_labels.json`, and packed bidirectional manifest.
3. Its complete pretrained text dual-encoder checkpoint and paired lemma index,
   including `manifest.json`, `faiss.index`, `lemma_vectors.npy`, `lemma_ids.json`,
   `lemma_names.json`, and the original corpus JSONL.
4. A compatible custom REPL and compiled Lean/Mathlib project.
5. An explicit list of benchmark-disallowed declaration names. The inventory
   filters these, and collection also excludes the current theorem by name.

The deployment fingerprint binds the original text checkpoint, query weights,
lemma-encoder identity, tokenizer policy, corpus, index files, and compiled Lean
environment. A changed checkpoint or environment requires explicit re-export and
conversion. Runtime loaders reject disagreement rather than initializing missing
policy heads or choosing another retriever.

## Export and conversion order

Run commands from the repository root using the Python environment that contains
the locked application dependencies, including Transformers. The exporter uses
local Hugging Face files; acquire the selected tokenizer/config before exporting.
The variables below refer to selected deployment artifacts, not files included in
this repository.

```bash
export SOURCE_PREPARED=/absolute/path/to/source/prepared
export SOURCE_ACTION_MODEL=/absolute/path/to/joint-source/best.pt
export SOURCE_CANDIDATES=/absolute/path/to/candidate-cache/manifest.json
export SOURCE_TEXT_MODEL=/absolute/path/to/text-retriever/best.pt
export SOURCE_INDEX=/absolute/path/to/text-index
export SOURCE_CORPUS=/absolute/path/to/lemmas.jsonl
export LEAN_SOURCE_ROOT=/absolute/path/to/mathlib4
export CUSTOM_REPL=/absolute/path/to/Pantograph/.lake/build/bin/repl
export DEPLOY_ROOT=/absolute/path/to/new-deployment
mkdir -p "$DEPLOY_ROOT"
```

If the source prepared root does not yet have AC's graph metadata, attach it with
this explicit migration. The command verifies source prepared hashes and the
expected vocabulary schema. It adds only `metadata/graph_representation.json`.

```bash
python -m maths_ai.gnn_inference.scripts.attach_source_graph_contract \
  --prepared-root "$SOURCE_PREPARED" \
  --candidate-manifest "$SOURCE_CANDIDATES"
```

If the contract already exists, verify it through the conversion loader; the
attachment command deliberately refuses to overwrite it.

Create `excluded-declarations.json` as a JSON array containing the declarations
excluded by the evaluation policy. An empty array is an explicit choice. Export
from the same imports and compiled environment that RL will use:

```bash
python -m maths_ai.gnn_inference.scripts.export_declaration_inventory \
  --source-root "$LEAN_SOURCE_ROOT" --repl "$CUSTOM_REPL" \
  --imports Init Mathlib \
  --exclude-file excluded-declarations.json \
  --output "$DEPLOY_ROOT/declarations.json"

python -m maths_ai.gnn_inference.scripts.export_query_artifact \
  --checkpoint "$SOURCE_TEXT_MODEL" --index-root "$SOURCE_INDEX" \
  --corpus "$SOURCE_CORPUS" --inventory "$DEPLOY_ROOT/declarations.json" \
  --output "$DEPLOY_ROOT/query"

python -m maths_ai.gnn_inference.scripts.convert_source_aware_checkpoint \
  --checkpoint "$SOURCE_ACTION_MODEL" --prepared-root "$SOURCE_PREPARED" \
  --index-manifest "$SOURCE_INDEX/deployment_manifest.json" \
  --candidate-manifest "$SOURCE_CANDIDATES" \
  --output "$DEPLOY_ROOT/actor-critic-v4.pt"
```

The converter copies all trained encoder/decoder tensors strictly, copies the
supervised tactic classifier into the actor base, initializes the actor residual
to zero, and declares the newly initialized critic. It discards only the source's
auxiliary stop diagnostic. `--type-bonus` explicitly records the optional soft
local-type cue strength; it defaults to zero.

The query exporter reads the full text checkpoint on CPU during conversion.
Serving then loads only the state tower from its exported artifact. It does not
construct a lemma tower to verify its hash.

## RL configuration migration

Copy `configs/retrieval.example.json` to a deployment-specific file. Replace all
four paths with the exported query directory, paired index directory, original
corpus, and declaration inventory. Choose retrieval placement independently of
policy placement; CPU retrieval is supported. Start with one FAISS thread and
ensure its configured count does not exceed the resolved CPU quota.

```bash
python -m maths_ai.gnn_inference.scripts.migrate_source_aware_rl_config \
  --input maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --retrieval-config "$DEPLOY_ROOT/retrieval.json" \
  --checkpoint "$DEPLOY_ROOT/actor-critic-v4.pt" \
  --output "$DEPLOY_ROOT/rl-trial.json"
```

The command removes the old RL `arg_loss_weight`, embeds retrieval settings, and
adds candidate-count and candidate-byte update budgets. Update these trial fields:

```json
{
  "prepared_root": "/absolute/path/to/source/prepared",
  "source_root": "/absolute/path/to/mathlib4",
  "pantograph_repl": "/absolute/path/to/custom/repl",
  "collection_workers": 1,
  "collection_devices": ["cuda:0"],
  "update_device": "cuda:0",
  "num_rounds": 2,
  "theorems_per_round": 2,
  "eval_every": 0,
  "checkpoint_every": 1
}
```

Use `cpu` for both collection and update devices when testing without a GPU.
The paths and trial fields above must be combined with the migrated full config.
The old local-only actor checkpoint cannot supply trained count/source/library
heads. The repository's existing RL config still names its previous checkpoint;
it must be migrated before launch.

```bash
python -m maths_ai.gnn_inference.scripts.rl_train \
  --config "$DEPLOY_ROOT/rl-trial.json"
```

Keep the policy/index/query/inventory pairing fixed when resuming. Check the run's
recorded config, sampled entropy, per-head likelihoods, candidate budgets,
gradient norms, Lean rejection categories, and optimizer-step count. Then verify
checkpoint resume and increase workers while retaining the collection barrier.

## Supervised callers

Epoch-level pointer and premise training accept an explicit candidate provider.
`local_pools` supplies intentionally local-only supervision; it cannot teach a
missing library identity. `OfflineCandidateProvider` accepts a verified table of
`CandidatePoolSnapshot` objects keyed by original `(split, row_index)` and checks
the prepared-artifact identity. It removes PyG's row-index batch offset before
lookup. This row-keyed provider is for offline data only.

`train_pointer` requires its provider explicitly. Mixed-source supervised callers
must supply pools bound to the same retrieval artifact as the model and retain
corrected `arg_count`, local/library target arrays, and
`action_target_supported` metadata. Existing local-only command-line training
supplies `local_pools` explicitly. Regenerate older argument targets before using
those training commands; absent support metadata is rejected.

## Verification and remaining deployment gates

The implementation test report is in `SOURCE_AWARE_RL_VALIDATION.md`. It includes
pinned-source numerical fixtures, full-action probability tests, dropout-disabled
recomputation, checkpoint conversion, retrieval queue shutdown, driver/barrier
regressions, and real controlled Lean execution.

Selection and testing of an actual trained joint checkpoint/index pair remain
pending. GPU parity, generated-state retrieval coverage with the trained model,
one-/multi-worker trained RL trials, checkpoint resume on JupyterHub, and measured
startup/RAM/GPU/throughput comparisons also remain pending. The synthetic artifact
fixtures and controlled Lean tests establish software behavior; they do not
establish trained proof-search quality or deployment performance.
