# Training-mode expansion: closing the on-policy collect → harvest → train loop

## Purpose and the exact gap

The PLN actor-critic RL machinery is implemented and tested (Stages 0–1c): `act()` samples
`(tactic, args)` with log-probs and value; `pln_reward` + `search_harvest` turn a
`ProofHypergraph` into shaped rewards and AND-OR value targets; `pln_rl_training.train_step`
takes the gradient step. What is **missing** is the *bridge* from the live search into that
machinery — the "training-mode `_expand`".

Verified against the code: `model.act()` is called **nowhere** in the search/collect path,
and `pln_rl_training.collect_batch` calls the existing `HybridReasoner.prove`, whose `_expand`
proposes tactics via `predict_next_tactic` (GNN top-k, **deterministic**). Consequently the
current collect path would harvest from the deterministic GNN search — the *off-policy*
alternative Stage 1c explicitly rejected — and the argument-level policy gradient never fires
(the harvested transitions carry no log-probs, so `compute_transition_loss` only does the
tactic level).

This plan specifies the bridge. It needs **no trained weights** — a fresh policy simply samples
mostly-rejected tactics, which validates control flow and reward plumbing; proving quality
waits on real checkpoints.

## What the bridge must do

Four responsibilities, none currently implemented:

1. **Sampling expansion** — propose a node's tactics by sampling from the actor-critic policy
   (`model.act`) instead of `predict_next_tactic`.
2. **Action decoding** — turn a sampled `(tactic_id, [arg_node_idx…])` into a `TacticCandidate`
   the Pantograph executor can apply: `tactic_id → name` via the tactic vocab, and each sampled
   argument node index → a Lean argument string.
3. **On-policy stashing** — record the `ActionSample` (`tactic_logp`, `arg_logp`, `value`)
   against the hyperedge it produced, so the harvest can join log-probs and the **argument**
   policy gradient actually flows.
4. **Wiring** — point `collect_batch` at the sampling expansion and have `train_step` consume
   the stashed log-probs.

## Symbol / seam table

| Seam | Where | Role in the bridge |
|------|-------|--------------------|
| `HybridReasoner.predict_next_tactic(goal_expr) -> [TacticCandidate]` | `joint_inference.py:200` | the proposal seam we override to sample |
| `HybridReasoner._expand(graph, node)` | `joint_inference.py:361` | applies candidates, PLN-ranks, `add_edge` |
| `ProofHypergraph.add_edge(source_id, tactic, ranked_subgoals) -> ProofHyperedge` | `hypergraph.py:220` | returns the edge (carries `edge.id`) |
| `_resolve_local_node_name(node, dag)` | `inference.py:228` | renders a DAG node → Lean argument string |
| `make_featurizer(node_vocab)` | `pln_rl_training.py` | `Goal → PyG Data` (string path, OOV→UNK) |
| `ActorCriticWithArgsClassifier.act(...) -> ActionSample` | `actor_critic.py` | on-policy sample (tactic+args, log-probs, value) |
| `extract_transitions(graph, …, edge_ids=…)` | `search_harvest.py` | already accepts an edge-id filter for on-policy selection |

## Design

### Placement — a subclass, not a flag on the production path

Add `RLHybridReasoner(HybridReasoner)` in a new `atp_lean_gnn/rl_reasoner.py`, overriding only
the proposal + stashing. Production `joint_inference.py` stays byte-for-byte unchanged.

Rationale: the deterministic top-k search and the on-policy sampling search are genuinely
different modes with different needs (top-k for coverage vs. single-sample with recorded
log-prob for unbiased policy gradient). A subclass is a clean separation, not a dual-mode flag
threaded through shared code. One small refactor to the base is required, below.

### The one base refactor: a proposal hook

`_expand` currently calls `self.predict_next_tactic(sanitized.expression)` to get candidates.
Keep that. The subclass overrides `predict_next_tactic` to:

1. featurize the goal (`make_featurizer`) → `Batch` of size 1,
2. call `self.model.act(batch, id_to_tactic=self.id_to_tactic)` → `ActionSample`,
3. decode into `TacticCandidate`s (below),
4. stash the `ActionSample` keyed by a **candidate fingerprint** `(goal_key, tactic_name,
   tuple(args))` in `self._pending: dict[str, ActionSample]`.

Then wrap edge creation so the stash migrates from fingerprint → `edge.id`. Because `_expand`
calls `add_edge` right after a successful apply, override a thin hook
`_record_edge(edge, tactic)` (called by the subclass's own `add_edge` wrapper, or by overriding
`_expand` to call `super().add_edge`-equivalent) that moves `self._pending[fingerprint]` into
`self.edge_actions[edge.id]`. The cleanest seam: have the subclass override `_expand` to call a
tiny `self._link(graph, node, tactic, ranked)` that does `graph.add_edge(...)` and then
`self.edge_actions[edge.id] = self._pending.pop(fingerprint)`.

### How many tactics per node?

On-policy policy gradient wants the log-prob of the action actually taken. Two options:

- **Sample `k` tactics per node** (`k` independent draws from `act`, each its own `ActionSample`
  and log-prob). Preserves AND-OR branching while staying on-policy; every edge is a genuine
  sample with a correct log-prob. **Recommended** (`k = top_k_tactics`).
- **Sample 1 tactic per node**, rely on many rollouts for coverage. Simpler, lower branching,
  slower coverage. Fallback if duplicate-sample handling is a nuisance.

With `k` draws, dedup identical `(tactic_name, args)` samples per node (Lean would reject the
duplicate edge anyway); keep the first's `ActionSample`.

### Action decoding

- **Tactic:** `tactic_name = self.id_to_tactic[int(sample.tactic_action[b])]`. `id_to_tactic`
  is the inverse of the tactic vocab (the same map `inference.py:126` builds).
- **Arguments:** `act()` returns `arg_actions` as **padded per-graph node indices** in the
  pointer's node space. For a batch of size 1 the padded index equals the node's offset within
  its DAG, and `dag_to_pyg` preserves node order, so `padded_idx → dag.nodes[padded_idx]`. Render
  each with the existing `_resolve_local_node_name(node, dag)` (reused from inference) → the Lean
  argument string. Drop indices that fall on non-premise/padding nodes (the pointer masks these
  to `-inf`, but guard anyway). Assemble `TacticCandidate(tactic_name, arguments, probability=
  exp(tactic_logp))`.
  - **Open decision:** lemma-corpus arguments (non-local premises) are out of the pointer's
    current node space (the pointer scores DAG nodes only). Stage 1 restricts arguments to local
    DAG nodes; lemma arguments remain future work, consistent with the pointer's design.

### On-policy harvest join

`train_step` today calls `extract_transitions` then `compute_transition_loss`, which recomputes
`log π(τ)` from a fresh encode. To use the **stashed** log-probs (and get the argument-level
gradient), extend the harvest join:

- `extract_transitions(graph, …, edge_ids=list(reasoner.edge_actions))` — restrict to the edges
  the policy actually produced (on-policy).
- Add `attach_action_samples(transitions, reasoner.edge_actions)` mapping each transition (by its
  edge) to its `ActionSample`, so the loss can use `sample.tactic_logp + sample.arg_logp` weighted
  by the AND-OR advantage `Â = return − V_pred(s)`, instead of recomputing only the tactic term.
  - **Decision — recompute vs. stashed log-probs.** Stashed log-probs are exactly on-policy and
    give the argument gradient for free, but hold a slice of the autograd graph across the whole
    (async) search, which is memory-heavy for long searches. Recomputing (a second encode in the
    train phase) is the standard A2C/PPO pattern and bounds memory. **Recommend recompute for the
    tactic term** (as now) **plus a stashed-index path for arguments** — store only the sampled
    `arg_actions` (ints, no graph) per edge, and recompute `log π(arg)` in the train phase by
    re-running the pointer on the featurized state with those indices. This gives the
    argument-level gradient without holding the search-time graph.

## File-by-file changes

**New**
- `atp_lean_gnn/rl_reasoner.py` — `RLHybridReasoner(HybridReasoner)`: `__init__(… , model,
  node_vocab, tactic_vocab)`, overridden `predict_next_tactic` (sample + decode + stash),
  `_link` (add_edge + migrate stash), `edge_actions: dict[int, EdgeAction]` where `EdgeAction`
  holds `tactic_id` and `arg_indices` (ints) for train-phase recompute.
- `tests/test_rl_reasoner.py` — mock-executor unit tests (below).
- `scripts/rl_smoke.py` — live smoke harness (real `Server` + `petta`, trivial seed goal).

**Modified**
- `hybrid_reasoner/joint_inference.py` — factor the single `graph.add_edge(...)` call in
  `_expand` behind `self._link(graph, node, tactic, ranked)` (default just calls `add_edge`), so
  the subclass can migrate the stash. No behavior change to production.
- `atp_lean_gnn/pln_rl_training.py` — `extract_transitions(edge_ids=…)` on-policy path; a
  `recompute_arg_logp(model, featurize, transition, arg_indices)` helper; fold the argument term
  into `compute_transition_loss`.

## Alternatives considered

1. **Training-mode boolean flag on `HybridReasoner._expand`.** Rejected — a dual-mode branch on
   the shared production method, exactly the pattern the working-style guidance says to avoid.
2. **Full `_expand` override in the subclass.** Rejected — duplicates the PLN-fallback and
   exhaustion logic; the `_link` seam is a far smaller surface.
3. **Off-policy harvest from the existing GNN search (no bridge).** Rejected in Stage 1c —
   requires importance weighting; the whole point is an unbiased on-policy gradient.
4. **Hold autograd graph across the search (stash live log-prob tensors).** Rejected as default —
   memory-heavy across an async multi-node search; recompute in the train phase instead.

## Test plan

| Test | How |
|------|-----|
| decode tactic id → name | fake `ActionSample` → `TacticCandidate` with expected name |
| decode arg index → Lean string | tiny DAG, known node → `_resolve_local_node_name` output matches |
| stash migrates to edge id | after `_link`, `edge_actions[edge.id]` holds the sampled action |
| on-policy edge filter | `extract_transitions(edge_ids=…)` returns only policy-produced edges |
| arg log-prob recompute | `recompute_arg_logp` gradient flows into `argument_selector` |
| mock-executor rollout | `NullTacticExecutor`-style fake → search runs, harvest→train step finite |
| live smoke (manual) | `scripts/rl_smoke.py`: real `Server` + `petta`, seed `∀ p, p → p`, bias toward `intro`/`exact` so the QED branch fires; assert a `ProofHypergraph` builds and one train step runs |

Unit tests run under `uv run python -m pytest`; the live smoke is a manual script (needs the
Lean/petta toolchain), not part of the CI unit suite.

## Staging

```
1. Base refactor: extract self._link seam in _expand (no behavior change) + its test
2. RLHybridReasoner: sample + decode + stash, mock-executor rollout test
3. On-policy harvest join: edge_ids filter + arg log-prob recompute, gradient test
4. Live smoke script: real Server + petta on a trivial seed goal (manual)
5. (later, needs weights) full validation with trained GNN checkpoints
```

## Refinements (agreed in review)

1. **Multiplicity-weighted dedup.** The k draws are i.i.d. samples from π(·|s); an action drawn
   `m` times must contribute its gradient term `m` times or high-probability actions are
   under-weighted. Dedup for the **executor** (apply each unique action once) but record
   `multiplicity = m` on the `EdgeAction` and weight that transition's actor term by `m`.
2. **One-step-per-collect invariant.** Recomputed log-probs are exactly on-policy only because
   no optimizer step occurs between collect and train (θ identical in both phases). Exactly one
   `optimizer.step()` per collect round; multiple steps per batch would need a PPO-style ratio
   clip. Stated as an explicit constraint, enforced by the loop structure.
3. **`predict_next_tactic` takes the sanitized `Goal`, not a string.** The RL override needs the
   hypotheses (the pointer's argument candidates live in the hypothesis nodes of the state DAG).
   Migrate the base signature to `Goal` and its one caller (`_expand`); the base implementation
   still forwards `goal.expression` to the GNN engine — no behavior change.
4. **Failure transitions.** `_expand` records **no edge** for an executor-rejected tactic, so
   rejected samples would produce no training signal — yet with an untrained policy they are
   most of the signal. The RL reasoner keeps every sampled action in a pending stash; entries
   still pending after the node's expansion (rejected or never applied) are flushed to
   `failure_actions` and enter the loss as actor-only transitions with
   `return = terminal_failure − step_penalty` and no critic target (the *state* may still be
   provable via another tactic; only the *action* failed).
5. **Engine construction hook.** `HybridReasoner.__init__` unconditionally builds
   `GNNModelEngine` from checkpoint paths; the RL subclass samples from the actor-critic and
   needs no engine (and no checkpoints exist yet). Factor engine creation behind
   `self._build_gnn_engine(...)`; the RL subclass overrides it to return `None`.
6. **Per-search result, sequential collect.** `edge.id` is unique only within one
   `ProofHypergraph`, so `edge_actions` keyed by bare edge id collide across searches. The RL
   reasoner's `prove` resets its stashes per call and returns an `RLSearchResult(graph,
   edge_actions, failure_actions)`. On-policy collect runs theorems **sequentially** on one
   reasoner (PLN concurrency *within* a search still applies); cross-theorem concurrency needs
   per-search reasoner instances and is deferred.

## Out of scope

- Lemma-corpus (non-local) arguments in the pointer's action space.
- Batched GNN inference across concurrent frontiers (throughput).
- Trained-checkpoint validation and any proving-quality claims (waits on weights).
