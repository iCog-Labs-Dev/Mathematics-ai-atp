# Accuracy Analysis: Mathematics-ai-atp GNN Pipeline

**Date**: 2026-06-29  
**Status**: Findings document -- fixes not yet applied

---

## Executive Summary

The GNN pipeline has **3 critical bugs** that cause immediate accuracy loss, **5 significant issues** that degrade training quality, and **several architectural limitations** that cap the system's potential.

The most damaging bug: **inference uses forward-only edges while training uses bidirectional edges**. The model operates on graph topologies it has never seen during training.

---

## P0 -- Critical Bugs

### Bug 1: Inference uses forward-only edges; training uses bidirectional

**Files**: `inference.py:162`, `pyg.py:41`, `training.py:411-422`

**What happens**:
The DAG stores edges as `(child_id, parent_id)` -- directed child to parent (`graph.py:113`). During training, `transform_edge_index()` in `training.py:411-422` doubles these to bidirectional:
```python
forward = edge_index.to(dtype=torch.long)
reverse = forward[[1, 0], :]  # flip child/parent
combined = torch.cat([forward, reverse], dim=1)
```
The GNN backbone learns to aggregate information from both directions -- children influence parents AND parents influence children.

But in `inference.py:162`:
```python
data = dag_to_pyg(dag, self.node_vocab)
```
This does NOT pass `add_reverse_edges=True` (default is `False` at `pyg.py:41`). The inference code never calls `transform_edge_index`. So at inference, the GNN only sees forward edges (child->parent), meaning information flows only upward.

**Why this kills accuracy**: The node embeddings at inference are fundamentally different from training. The tactic classifier and argument selector both operate on these degraded embeddings. This is a **train-test distribution mismatch** -- one of the most damaging bugs in ML.

**Evidence**: The model predicts `constructor` (95% confidence) for `forall (p q: Prop), Or p q -> Or q p` when it should predict `intro`. The model's own predictions remain wrong because it sees a different graph than what it was trained on.

---

### Bug 2: Premise mask overridden to ALL nodes during argument selection training

**Files**: `argument_selector.py:243-247`, `pyg.py:89-107`

**What happens**:
`build_premise_mask()` in `pyg.py:89-107` carefully filters structural nodes:
```python
_PREMISE_SELECTABLE_TYPES = {"var", "predicate", "type"}
_PREMISE_SELECTABLE_META_LABELS = {"Hyp"}
```
Only `var`, `predicate`, `type` nodes and `Hyp` (hypothesis) nodes are valid argument candidates. Structural nodes like `App`, `Arrow`, `Forall`, `State`, `Goal` are excluded.

But in `argument_selector.py:243-247`:
```python
# Overwrite the cache's premise_mask if it's too restrictive
node_types = data.node_type.to(device=node_embeddings.device)
premise_mask = (node_types >= 0) & (node_types <= 5)
```
Since `NODE_TYPE_TO_ID` maps types to integers 0-5, this condition is **always True**. Every node becomes a valid candidate.

**Why this kills accuracy**: The pointer network trains on ALL nodes including `App`, `Arrow`, `State`, `Goal`. The ground-truth argument is a meaningful variable, but the model learns structural nodes are equally valid. At inference, the model may select `App` or `Arrow` nodes as arguments, producing invalid tactic strings.

The comment says "Overwrite if too restrictive" -- the right fix is to improve `build_premise_mask`, not bypass it.

---

### Bug 3: Fresh name generation returns ALL bound variables instead of first

**Files**: `inference.py:42-49`

**What happens**:
`_extract_fresh_names_from_dag()` returns ALL forall-bound leaf variable names:
```python
return [
    node.label for node in dag.nodes
    if node.is_bound == 1 and node.binder_kind == BINDER_KIND_FORALL
    and not node.children
]
```
For `forall (p q: Prop), Or p q -> Or q p`, this returns `['p', 'q']`.

When the tactic is `intro` with arity 1, the model generates `intro p q` -- valid Lean syntax, but wrong for the prover loop. The prover expects each iteration to introduce ONE binder.

**Impact**: Multi-argument `intro` when only one is expected. The proof tree depth is reduced, but subgoal ranking and hypothesis tracking become misaligned with the prover's loop structure.

---

## P1 -- Significant Accuracy Loss

### Issue 4: Score index mismatch for local-only tactics

**File**: `inference.py:259-277`

When scoring local-only tactics (`cases`, `rcases`, `obtain`):
```python
local_sorted = local_scores.argsort(descending=True)[:arity]
local_indices = torch.where(local_mask)[0][local_sorted]
...
score=float(local_scores[local_indices.tolist().index(idx)].item())
```

`local_indices.tolist().index(idx)` finds the position of `idx` in the **global** `local_indices` tensor, but `local_scores` is indexed by position in the **filtered** local set. These index spaces are different.

Example:
- `local_indices = [3, 7, 12]` (global positions)
- `local_scores = [0.9, 0.3, 0.7]` (scored in filtered order)
- For `idx=7`: `local_indices.tolist().index(7)` returns 1, so `local_scores[1] = 0.3`
- But the correct score for candidate 7 depends on its rank in `local_sorted`, not its position in `local_indices`

**Impact**: Wrong confidence scores. Potential `IndexError` crash for certain tactic/candidate combinations.

---

### Issue 5: Argument ground-truth uses first-match label heuristic

**File**: `preprocess.py:107-119`

`_resolve_arg_node_indices()` builds:
```python
label_to_id = {}
for node in dag.nodes:
    if node.label not in label_to_id:
        label_to_id[node.label] = node.id  # FIRST match wins
```
In a hash-consed DAG, multiple nodes can share the same label. The ground-truth argument may reference a different node with the same label. The training signal tells the model to select the wrong node.

**Impact**: Noisy ground-truth argument labels degrade argument selection accuracy.

---

### Issue 6: No residual connections in 4-layer GraphSAGE

**File**: `model.py:75-80`

```python
for index, conv in enumerate(self.convs):
    x = conv(x, data.edge_index)
    x = F.relu(x)
    if index < len(self.convs) - 1:
        x = self.dropout(x)
```

With `num_layers=4` (default), deep GraphSAGE without residual connections suffers from **oversmoothing**: all node embeddings converge to similar values, losing discriminative power. After 4 layers of mean aggregation, every node's embedding is a weighted average of its 4-hop neighborhood -- nodes that started very different end up nearly identical.

**Impact**: The GNN cannot distinguish between structurally similar but semantically different nodes. The classifier receives embeddings where all nodes look alike.

---

### Issue 7: Readout is single State node only

**File**: `model.py:82-99`

```python
def readout(self, node_embeddings, data):
    return node_embeddings.index_select(0, state_node_index)
```

Only the `State` node embedding is used for classification. The entire graph's information must be aggregated into this single node through message passing. This creates a bottleneck:

1. After 4 GNN layers, only nodes within 4 hops of State have influenced its embedding
2. Distant but relevant nodes (deep in expression subtrees) may not contribute
3. No attention mechanism to weight important neighbors differently

**Impact**: Information loss for complex proof states with deep expression trees.

---

### Issue 8: Single pool built for all tactic candidates

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

## P2 -- Moderate Impact

### Issue 9: No edge-type encoding

**File**: `pyg.py:37-81`

All edges are treated identically. The DAG has semantically distinct edge types:
- Parent-child in expressions
- Hypothesis name to type
- Goal to expression
- State to hypothesis/Goal
- Binder to variable

These are not encoded. A typed edge representation would provide more structural information.

---

### Issue 10: No hard negative mining in premise ranking

**File**: `premise_scoring.py:202-298`

Cross-entropy is computed over the entire candidate pool. When the pool is large (k=500 from lemma index + local nodes) and only 1-2 are positive, the loss is dominated by easy negatives. Hard negative mining (selecting high-scoring negatives) would provide stronger training signal.

---

### Issue 11: Missing tactics from arity registry

**File**: `labels.py:46-112`

Missing from `TACTIC_ARITY`:
- `split` (0 args)
- `exists` (1 arg)
- `injection` (0 args)
- `clear` (1 arg)
- `revert` (1 arg)
- `rename` (1 arg)
- `swap` (0 args)
- `rotate` (0 args)
- `done` (0 args)
- `sorry` (0 args)
- `iterate` (0-1 args)
- `repeat` (0-1 args)
- `all_goals` (1 arg)
- `any_goals` (1 arg)
- `first` (multiple args)
- `try` (1 arg)
- `rwa` (1 arg)

Default arity is 1, which may be wrong for 0-arg or multi-arg tactics.

---

### Issue 12: No proof history or next-state context

**File**: `preparation.py:14-19`

`PreparedExample` stores only the current proof state. Missing:
- Previous proof states (tactic history)
- Whether this is the first tactic applied
- Success/failure status
- Depth in the proof tree

Without proof context, the model cannot learn sequential tactic patterns (e.g., `intro` is often followed by `simp` or `cases`).

---

### Issue 13: Inference bypasses TacticWithArgsClassifier.forward

**File**: `inference.py:178-182`

Inference manually calls:
```python
node_embeddings = self.model.backbone.encode_nodes(batch)
state_emb = self.model.backbone.readout(node_embeddings, batch)
tactic_logits = self.model.backbone.classifier(state_emb)
```

This bypasses `TacticWithArgsClassifier.forward` which has the autoregressive argument selection logic. Any improvements made to `forward` will not benefit inference.

---

## Recommended Fix Priority

### Immediate (do first -- biggest accuracy gains):

1. **Fix Bug 1**: Add `transform_edge_index(batch.edge_index, edge_mode="bidirectional")` in `inference.py:162` or pass `add_reverse_edges=True` to `dag_to_pyg`.

2. **Fix Bug 2**: Remove the premise_mask override in `argument_selector.py:243-247`. Use `data.premise_mask` directly.

3. **Fix Bug 3**: Change `_extract_fresh_names_from_dag` to return only `[:arity]` names.

### Soon (significant gains):

4. **Fix Issue 4**: Fix the score index mapping in the local-only tactic scoring loop.

5. **Fix Issue 6**: Add residual connections to GraphSAGE: `x = x + conv(x, edge_index)`.

6. **Fix Issue 7**: Add multi-head attention readout or global mean pooling alongside State node readout.

### Later (architecture improvements):

7. Add edge-type encoding to `dag_to_pyg`
8. Add hard negative mining to premise ranking loss
9. Complete the arity registry
10. Add proof history context to training examples
