# Accuracy Analysis: Mathematics-ai-atp GNN Pipeline

**Date**: 2026-06-29  
**Status**: Active analysis — fixes applied incrementally

---

## Executive Summary

The GNN pipeline has **3 critical bugs** (P0), **5 significant architectural limitations** (P1), and **5 moderate issues** (P2). The most damaging is a **train/inference edge mode mismatch** — the model trains on bidirectional edges but infers on forward-only edges.

---

## P0 — Critical Bugs (fix immediately)

### Bug 1: Inference uses forward-only edges; training uses bidirectional

**Files**: `inference.py:162`, `pyg.py:41`, `training.py:411-422`

**What happens**:  
The DAG stores edges as `(child_id, parent_id)` — directed child→parent (`graph.py:113`). During training, `transform_edge_index()` doubles these to bidirectional:

```python
# training.py:418-422
forward = edge_index.to(dtype=torch.long)
reverse = forward[[1, 0], :]
combined = torch.cat([forward, reverse], dim=1)
```

But at inference (`inference.py:162`):
```python
data = dag_to_pyg(dag, self.node_vocab)  # add_reverse_edges=False (default)
```

No `transform_edge_index` is ever called. The GNN sees a fundamentally different graph topology at inference.

**Impact**: Train-test distribution mismatch. The model's node embeddings at inference are computed on a graph it has never seen during training. This is one of the most damaging bugs in ML.

**Fix**: Apply `transform_edge_index(batch.edge_index, edge_mode="bidirectional")` in `inference.py` after creating the batch.

---

### Bug 2: Premise mask overridden to ALL nodes during argument selection

**Files**: `argument_selector.py:243-247`, `pyg.py:89-107`

**What happens**:  
`build_premise_mask()` carefully filters structural nodes:
```python
# pyg.py:100-107
_PREMISE_SELECTABLE_TYPES = {"var", "predicate", "type"}
_PREMISE_SELECTABLE_META_LABELS = {"Hyp"}
```

But `argument_selector.py:243-247` throws this away:
```python
# Overwrite the cache's premise_mask if it's too restrictive
node_types = data.node_type.to(device=node_embeddings.device)
premise_mask = (node_types >= 0) & (node_types <= 5)
```

Since `NODE_TYPE_TO_ID` maps types to integers 0–5, this condition is **always True**. Every node becomes a valid candidate.

**Impact**: The pointer network trains on ALL nodes including `App`, `Arrow`, `State`, `Goal`. At inference, the model may select structural nodes as arguments.

**Fix**: Remove lines 243-247. Use `data.premise_mask` directly.

---

### Bug 3: Score index mismatch for local-only tactics

**File**: `inference.py:259-277`

**What happens**:  
```python
local_sorted = local_scores.argsort(descending=True)[:arity]
local_indices = torch.where(local_mask)[0][local_sorted]
...
for idx in local_indices.tolist():
    ...
    score=float(local_scores[local_indices.tolist().index(idx)].item())
```

`local_indices.tolist().index(idx)` finds the position of `idx` in the **global** `local_indices` tensor, but `local_scores` is indexed by position in the **filtered** local set. These index spaces are different.

Example:
- `local_indices = [3, 7, 12]` (global positions)
- `local_scores = [0.9, 0.3, 0.7]` (scored in filtered order)
- For `idx=7`: `local_indices.tolist().index(7)` returns 1, so `local_scores[1] = 0.3`
- But the correct score depends on the candidate's rank in `local_sorted`, not its position in `local_indices`

**Impact**: Wrong confidence scores. The candidate identity is correct, but the reported score is scrambled.

**Fix**: Use `enumerate` to track the rank position:
```python
for rank, idx in enumerate(local_indices.tolist()):
    ...
    score=float(local_scores[local_sorted[rank]].item())
```

---

## P1 — Significant Architectural Limitations

### Issue 4: No residual connections in 4-layer GraphSAGE

**File**: `model.py:75-80`

```python
for index, conv in enumerate(self.convs):
    x = conv(x, data.edge_index)
    x = F.relu(x)
    if index < len(self.convs) - 1:
        x = self.dropout(x)
```

With `num_layers=4`, deep GraphSAGE without residual connections suffers from **oversmoothing**: all node embeddings converge to similar values, losing discriminative power.

**Impact**: The GNN cannot distinguish between structurally similar but semantically different nodes.

**Fix**: Add skip connections: `x = x + conv(x, data.edge_index)`.

---

### Issue 5: Readout is single State node only

**File**: `model.py:82-99`

```python
def readout(self, node_embeddings, data):
    return node_embeddings.index_select(0, state_node_index)
```

Only the `State` node embedding is used for classification. The entire graph's information must be aggregated into this single node through message passing. After 4 GNN layers, only nodes within 4 hops of State contribute.

**Impact**: Information loss for complex proof states with deep expression trees.

**Fix**: Add global mean/max pooling alongside State node readout, or use attention-based readout.

---

### Issue 6: Single pool built for all tactic candidates

**File**: `inference.py:190-198`

```python
pools = build_unified_pools(state_emb, node_embeddings, batch.premise_mask, batch.batch, ...)
pool = pools[0]  # ONE pool used for ALL tactic candidates
```

Different tactics need different premises:
- `rw` needs rewrite lemmas (equality-related)
- `exact`/`apply` needs exact-match hypotheses or function lemmas
- `cases`/`rcases` needs inductive type hypotheses
- `simp` needs simplification lemmas

Building one pool with one tactic-agnostic query means the same candidate set is scored identically for every tactic type.

**Impact**: Suboptimal argument selection for each tactic.

---

### Issue 7: Argument ground-truth uses first-match label heuristic

**File**: `preprocess.py:107-119`

```python
label_to_id = {}
for node in dag.nodes:
    if node.label not in label_to_id:
        label_to_id[node.label] = node.id  # FIRST match wins
```

In a hash-consed DAG, multiple nodes can share the same label. The ground-truth argument may reference a different node with the same label. The training signal tells the model to select the wrong node.

**Impact**: Noisy ground-truth argument labels degrade argument selection accuracy.

---

### Issue 8: Inference bypasses TacticWithArgsClassifier.forward

**File**: `inference.py:178-182`

```python
node_embeddings = self.model.backbone.encode_nodes(batch)
state_emb = self.model.backbone.readout(node_embeddings, batch)
tactic_logits = self.model.backbone.classifier(state_emb)
```

This bypasses `TacticWithArgsClassifier.forward` which has the autoregressive argument selection logic. Any improvements made to `forward` will not benefit inference. The inference code reimplements the pipeline manually.

**Impact**: Code duplication, maintenance burden, and the two paths can diverge silently.

---

## P2 — Moderate Impact

### Issue 9: No edge-type encoding

**File**: `pyg.py:37-81`

All edges are treated identically. The DAG has semantically distinct edge types (parent-child, hypothesis name→type, goal→expression, state→hypothesis/goal, binder→variable). These are not encoded.

---

### Issue 10: No hard negative mining in premise ranking

**File**: `premise_scoring.py:202-298`

Cross-entropy is computed over the entire candidate pool. When the pool is large (k=500) and only 1-2 are positive, the loss is dominated by easy negatives. Hard negative mining would provide stronger training signal.

---

### Issue 11: Missing tactics from arity registry

**File**: `labels.py:46-112`

Missing from `TACTIC_ARITY`: `split`, `exists`, `injection`, `clear`, `revert`, `rename`, `swap`, `rotate`, `done`, `iterate`, `repeat`, `all_goals`, `any_goals`, `first`, `try`, `rwa`. Default arity is 1, which is wrong for 0-arg or multi-arg tactics.

---

### Issue 12: No proof history or next-state context

**File**: `preparation.py:14-19`

`PreparedExample` stores only the current proof state. Missing: previous proof states (tactic history), whether this is the first tactic applied, success/failure status, depth in the proof tree. Without proof context, the model cannot learn sequential tactic patterns.

---

### Issue 13: Training data prepared with old parser

The dataset at `artifacts/prepared/v1/` was prepared before the parser comma fix (the fix that separates `q:` into `q` and `:` tokens, and recognizes `forall` text in addition to `∀` symbol). The trained models (baseline, pointer) were trained on old DAG topology. They need retraining.

**Impact**: The model sees different graph patterns at inference than during training, contributing to wrong predictions like `constructor` for `∀ (p q : Prop), Or p q → Or q p`.

**Fix**: Re-run `--stages prepare,baseline,pointer` to retrain with the fixed parser.

---

## Recommended Fix Priority

### Immediate (P0 — biggest accuracy gains):

1. **Bug 1**: Add `transform_edge_index` in inference.py
2. **Bug 2**: Remove premise_mask override in argument_selector.py
3. **Bug 3**: Fix score index mapping in local-only tactic scoring

### Soon (P1 — architecture improvements):

4. **Issue 4**: Add residual connections to GraphSAGE
5. **Issue 5**: Add multi-head attention readout
6. **Issue 6**: Build tactic-aware candidate pools
7. **Issue 13**: Retrain models with fixed parser

### Later (P2):

8. Add edge-type encoding to `dag_to_pyg`
9. Add hard negative mining to premise ranking loss
10. Complete the arity registry
11. Add proof history context to training examples
12. Unify inference path with `TacticWithArgsClassifier.forward`
