# End-to-End RL Flow
*Reference: "end to end flow RL.pdf"*

The live RL search uses three different representations of a proof state:
1. Pantograph's live `GoalState`, which Lean can execute.
2. The project's textual `Goal`, containing an expression and hypothesis strings.
3. The graph/DAG representation used by the GNN.

The DAG is not converted back into a Lean expression. The textual `Goal` is the bridge used to reconstruct a fresh Pantograph state.

## End-to-end flow

```text
dataset text
  | parse_state
  V
internal Goal(expression, hypotheses)
  | make_dag_featurizer
  V
DAG -> PyTorch Geometric graph -> actor-critic action
  | decode DAG node indices into names
  V
TacticCandidate("exact", ["h"])
  | render tactic
  V
Pantograph GoalState + "exact h"
  | goal.tactic
  V
Pantograph response
  | GoalState / Goal / Variable parsing
  V
new internal Goal objects for subgoals
  |
  +--> next DAG for the next policy decision
  +--> later reconstruction into another Pantograph GoalState
```

## 1. Initial textual state

A dataset row might contain:
```lean
p: Prop
h: p
⊢ p
```

[`build_theorem_pool` (line 208)](/home/nolawi/new-maths/maths_ai/gnn_inference/atp_lean_gnn/rl_training_driver.py:208) calls [`parse_state` (line 46)](/home/nolawi/new-maths/maths_ai/gnn_inference/atp_lean_gnn/state.py:46). That produces:

```python
Goal(
    expression="p",
    hypotheses=["p: Prop", "h: p"],
)
```

The `parse_state` function parses text. It does not parse Pantograph JSON.

## 2. Textual state to DAG

[`goal_to_state` (line 83)](/home/nolawi/new-maths/maths_ai/gnn_inference/atp_lean_gnn/pin_rl_training.py:83) reconstructs the textual form:
```lean
p: Prop
h: p
⊢ p
```

Then [`make_dag_featurizer` (line 113)](/home/nolawi/new-maths/maths_ai/gnn_inference/atp_lean_gnn/pin_rl_training.py:113) performs:
`Goal` -> `goal_to_state` -> `proof_state_to_dag` -> `dag_to_pyg`

[`proof_state_to_dag` (line 314)](/home/nolawi/new-maths/maths_ai/gnn_inference/atp_lean_gnn/graph.py:314) creates graph nodes such as:
```text
State
Hyp
h
p
Goal
p
```

[`dag_to_pyg` (line 42)](/home/nolawi/new-maths/maths_ai/gnn_inference/atp_lean_gnn/pyg.py:42) converts that DAG into tensors:
- `x`: node-label vocabulary IDs.
- `edge_index`: graph edges.
- `node_type`: coarse node categories.
- Binder features.
- `premise_mask`: nodes eligible as tactic arguments.
- `state_node_index`: the node used for the state embedding.

The DAG is a model representation. It is not intended to be valid Lean syntax.

## 3. DAG to a sampled tactic

[`RLHybridReasoner.predict_next_tactic` (line 105)](/home/nolawi/new-maths/maths_ai/gnn_inference/atp_lean_gnn/rl_reasoner.py:105) passes the PyTorch Geometric graph to `model.act`.

The actor samples:
- A tactic ID.
- Argument node indices from the DAG.

For example:
- tactic ID: `exact`
- argument node index: node representing `h`

[`RLHybridReasoner._decode` (line 145)](/home/nolawi/new-maths/maths_ai/gnn_inference/atp_lean_gnn/rl_reasoner.py:145) converts the tactic ID to `"exact"` and resolves the selected DAG node to `"h"` using `_resolve_local_node_name`.

The result is:
```python
TacticCandidate(
    tactic_name="exact",
    arguments=["h"],
)
```

The stored integer node index is retained for the later policy-gradient recomputation.

## 4. Reconstructing a Pantograph state

The model does not send the textual state or DAG directly to Pantograph. Before applying tactics, `_start_state` [(line 493)](/home/nolawi/new-maths/maths_ai/hybrid_reasoner/joint_inference.py:493) rebuilds a Lean goal.

For:
```python
Goal(
    expression="p",
    hypotheses=["p: Prop", "h: p"],
)
```
it constructs:
```lean
(p: Prop), (h: p), p
```

It sends that expression through:
```python
server.goal_start_async(expression)
```

Pantograph creates a root goal. The reasoner then sends:
```lean
intro p h
```

This restores the local context that the original proof state had:
```lean
p: Prop
h: p
⊢ p
```

This reconstruction is why the hypothesis strings must be valid Lean binder declarations. The case-label and inaccessible-name fixes are needed at this boundary.

## 5. Applying the tactic

`expand` [(line 558)](/home/nolawi/new-maths/maths_ai/hybrid_reasoner/joint_inference.py:558) applies each sampled candidate to the same reconstructed Pantograph state.

[`PantographExecutor.apply` (line 140)](/home/nolawi/new-maths/maths_ai/hybrid_reasoner/joint_inference.py:140) renders the candidate. For example:
```python
TacticCandidate("exact", ["h"])
```
becomes:
```lean
exact h
```

For rewrite tactics, the renderer produces the required syntax:
```lean
rw [h]
```

Then it calls:
```python
await server.goal_tactic_async(state, tactic_cmd)
```

## 6. Pantograph response back to internal goals

Pantograph returns a `GoalState`. PyPantograph parses the JSON into:
`GoalState` -> `Goal` -> `Variable`

The executor converts each returned Pantograph goal into the project's internal form:
```python
Goal(
    expression=str(g.target),
    hypotheses=[str(v) for v in g.variables],
)
```

For example, a returned Pantograph goal with:
- target = `q`
- variables = `[h: p, hpq: p -> q]`

becomes:
```python
Goal(
    expression="q",
    hypotheses=["h: p", "hpq: p -> q"],
)
```

If Pantograph returns no goals, the tactic is treated as a QED branch. Otherwise, these internal `Goal` objects become child nodes in the proof hypergraph.

## 7. The next tactic

When the search frontier later selects one of those child nodes, the same process repeats:
1. The child `Goal` is converted to text.
2. The text is parsed into a new DAG.
3. The actor samples a tactic and argument node.
4. The argument node is decoded into a Lean name or expression.
5. The child `Goal` is independently reconstructed with `goal_start_async`.
6. Its hypotheses are restored with `intro`.
7. The next tactic is applied.

The previous Pantograph `GoalState` is not retained as the canonical state for future search. The canonical search state is the internal textual `Goal`; Pantograph states are temporary execution states reconstructed when a node is expanded.

## Important distinction

The current live RL path is text-based:
`Pantograph GoalState` -> `internal Goal text` -> `text parser` -> `DAG`

The S-expression path in [`graph.py` (line 14)](/home/nolawi/new-maths/maths_ai/gnn_inference/atp_lean_gnn/graph.py:14) is a separate path used by S-expression-aware inference. It requires `printExprAST` and `patch_pantograph_for_sexp()`. The current RL reasoner does not use that path; it uses `Goal.target` and `Variable` pretty-printed strings.

The training phase repeats the same `Goal` -> `DAG` -> `PyG` conversion using the same featurizer instance. That is necessary because the stored argument indices refer to positions in the exact DAG node order used during collection.
