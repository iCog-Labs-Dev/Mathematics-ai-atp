# Clean-root pre-template joint-model release

Target: `iCogLabs-Dev/Mathlib-Sexpr-GNN` (the old jajostrains URL redirects
there). Versions live in Hugging Face commits/tags, not release/legacy folders.
One new commit replaces the current file tree; older revisions remain accessible.
The previously published revision was `700c48cf1b741e59d39fa29d105cd2d2f77fdb9f`.

## Contents

```text
README.md, LICENSE, NOTICE, bundle.json
policy/{model.safetensors,config.json,inference.json}
vocab/{node_vocab.json,tactic_vocab.json}
retriever/{model.safetensors,config.json,tokenizer/}
retriever/index/{faiss.index,lemma_vectors.npy,lemma_ids.json,lemma_names.json,manifest.json}
retriever/corpus/lemmas.jsonl
evaluation/{metrics.json,protocol.json}
provenance/{training.json,versions.json}
```

Policy weights contain the entire GNN/tactic/GRU/count/source/local/library
system. Architecture comes from the joint epoch-3 checkpoint, full refined
weights from the local epoch-8 checkpoint, and retrieval from its matched
epoch-5 snapshot/index. Epoch-7 standalone retrieval is not substituted.
The corpus is included for reproducible ID/name/type lookup; large prepared
graphs and extraction caches remain outside the model package.

Model artifacts retain the existing HF Apache-2.0 license, selected by the
user's delegated license decision. The implementation keeps its separate
GPLv2 source-code license; NOTICE identifies this distinction and component
origins. No source-code relicensing is performed.

## Export on server

Run from repository root in `atp` after pulling published code. The existing
cached tokenizer/encoder configuration is used offline. Dependencies include
PyTorch, PyG, Transformers, safetensors, FAISS and huggingface_hub.

```sh
ROOT=/home/jovyan/yonatan/Mathematics-ai-atp
RUNS="$ROOT/maths_ai/gnn_inference/runs"
ARTIFACTS="$ROOT/maths_ai/_support_files/artifacts"
BUNDLE="$ARTIFACTS/hf_joint_model"

python -B -u -m maths_ai.gnn_inference.scripts.export_joint_bundle \
  --joint-checkpoint "$RUNS/sexpr_joint_source_flat_epoch_5_count_pilot/best.pt" \
  --local-checkpoint "$RUNS/sexpr_local_identity_full_frozen_v2/best.pt" \
  --text-checkpoint "$RUNS/premise_retriever_text_full/epoch_5_snapshot.pt" \
  --text-index "$RUNS/premise_retriever_text_full/index_epoch_5" \
  --prepared-root "$ARTIFACTS/prepared/sexpr_available_labels_v2" \
  --candidate-dir "$RUNS/sexpr_text_library_epoch_5" \
  --corpus-path "$ARTIFACTS/lemmas/v2/corpus/lemmas.jsonl" \
  --output-dir "$BUNDLE"
```

Output must be a new directory outside source roots. Export stages in a
sibling `.building` directory and renames it only after strict offline reload
and tensor/retrieval/decoder round-trip checks. An interrupted `.building`
directory is retained for inspection; use a fresh output name to retry.
Source checkpoints, dataset and indexes are read only. Optimizers/pickles
and training-machine paths are not exported.

The exporter verifies:

- Local checkpoint's exact base-checkpoint hash and declared local-only updates.
- Shared candidate, vocabulary, corrected-label and packed-population bindings.
- Exact retriever snapshot, tokenizer vocabulary, corpus and lemma-tower hashes.
- Every FAISS row equals the normalized stored vector at the same ID position.
- Unique corpus/index ID/name mappings, dimensions and inner-product metric.
- Exported model tensors remain identical, with full strict loading.
- Offline query embeddings, retrieved IDs/scores and joint count/GRU decoding
  reproduce the pre-export components.

## Independent verification and loading

```sh
python -B -u -m maths_ai.gnn_inference.scripts.verify_joint_bundle \
  --bundle-dir "$BUNDLE" --device cpu
```

```python
from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import load_joint_source_bundle
system = load_joint_source_bundle("/path/to/downloaded/model", device="cpu")
ids, vectors, scores = system.retrieve(["n : Nat\n⊢ n = n"])
actions = system.predict(batch, original_state_texts, graph_format="lean-model-sexp-v2")
```

`batch` is a normalized prepared PyG Batch with node types, State root,
edges and premise mask; gold labels are not required. This loader does not
construct compatible live graphs from printed goals. Default `base` inference
keeps local soft cues off. The explicit `accepted_local_cues` profile requires
matching normalized goal/hypothesis records. Runtime settings are separate
from tensor weights. Repeated arguments are allowed; explicit count prediction
determines length, and the stop head remains auxiliary.

`evaluation/metrics.json` contains distinct stage diagnostics, not a new full-
system evaluation or a cross-cohort comparison with the old model. No template
prediction training or whole-theorem solve-rate claim is made by this release.

## Inspect and publish one HF commit

Authentication uses the server's stored HF credentials (`hf auth login` if
needed); no token belongs in code or chat. The account needs write access to
the org model repo.

```sh
python -B -u -m maths_ai.gnn_inference.scripts.publish_joint_bundle \
  --bundle-dir "$BUNDLE" --repo-id iCogLabs-Dev/Mathlib-Sexpr-GNN \
  --release-tag v2.0.0 --previous-tag v1.0.0 --dry-run
```

After checking that plan, repeat without `--dry-run` to publish. It pins
`v1.0.0` to the current old HF commit if absent, creates one parent-bound
commit adding/replacing the clean-root files and deleting obsolete paths from
the new revision, checks the committed tree, then creates immutable `v2.0.0`.
`.gitattributes` remains for Hub file tracking. Existing release tags are never
moved. This workflow does not purge old Git/LFS history. Code and HF artifact
commits are distinct: source code commit goes into provenance; the release tag
identifies the eventual HF commit without a self-referential manifest hash.

Consumers should pin the tag or full commit:

```python
from huggingface_hub import snapshot_download
path = snapshot_download("iCogLabs-Dev/Mathlib-Sexpr-GNN", revision="v2.0.0")
system = load_joint_source_bundle(path)
```

The loader supports Hub snapshot file symlinks while validating logical member
paths and content hashes. All tokenizer files must be manifested; local-only,
non-remote-code loading is enforced. The standalone package loads after the
training data/checkpoints are removed.

Verification: **62 focused tests passed**, including seven new joint-bundle
cases using a real tiny BERT/tokenizer, FAISS index and joint policy without
downloads. Tests cover independent loading after source removal, prediction
without gold labels, optional cues, Hub symlinks, wrong epoch/checkpoint/vocab/
head/vector bindings, tampering, clean-tree replacement and immutable tags.

## Published release

Server export and independent offline verification completed successfully:
22 hash-bound component files, **3,869,544 policy parameters**, **45,426,432
retriever parameters**, and **318,953 indexed lemmas**. The package's source
code revision is `d9e09cc31682ee6da9a032b1e3f788a343954b9a`.

Published and remotely verified:

- `v2.0.0`: `8aba62625dfeb833e4591498700f351f3bc60af9`.
- `v1.0.0`: `700c48cf1b741e59d39fa29d105cd2d2f77fdb9f` (previous model).
- Release: https://huggingface.co/iCogLabs-Dev/Mathlib-Sexpr-GNN/tree/v2.0.0
- Commit: https://huggingface.co/iCogLabs-Dev/Mathlib-Sexpr-GNN/commit/8aba62625dfeb833e4591498700f351f3bc60af9

Remote API confirms clean-root files and preserved old-tag contents. Large-file
SHA256 metadata for policy/retriever weights, corpus, FAISS index, vectors and
lemma names matches `bundle.json`. Published provenance confirms joint epoch 3,
local refinement epoch 8 and matched retrieval epoch 5. These remote checks
plus the server load-back do not claim a separate fresh-download inference
test or new whole-proof performance evaluation.
