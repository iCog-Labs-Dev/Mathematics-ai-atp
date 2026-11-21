# Implementation Plan: Integrating HTPS-Style MCTS Search and Training into the Existing AND/OR RL Architecture

## 0. Scope and non-goals

**In scope.** Port five mechanisms from HyperTree Proof Search (HTPS) into the existing
system described in `rl_process_walkthrough.md` (`atp_lean_gnn/`, `hybrid_reasoner/`):

1. Repeated MCTS-style simulation against a persistent hypergraph, with `N(g,t)`/`W(g,t)`
   visit statistics and PUCT/RP-style selection, in place of the current single best-first
   pass per theorem.
2. Virtual loss, so simulations can be batched for GPU efficiency instead of run one at a
   time.
3. A soft critic-regression target (`W(g,t*)/N(g,t*)`, threshold-gated) as an addition to
   the existing hard structural `backup_values`.
4. Minimal-hypertree tactic mining, feeding a supervised imitation objective, as an
   addition to the existing on-policy actor loss.
5. Per-attempt search hyperparameter randomization (sample count, temperature, exploration
   constant, depth penalty) instead of fixed values.

**Out of scope.** The model architecture itself does not change. This system's GraphSAGE
encoder, factored (tactic-head + pointer-head) action space, and PLN-based reward shaping
are not being replaced with HTPS's seq2seq transformer or its token-level tactic decoding —
those are unrelated design axes. This plan ports HTPS's *search and training-data*
mechanics onto the existing model, not the existing model onto HTPS's.

**Guiding principle.** Every new mechanism should degrade gracefully to current behavior
at its default/disabled setting, so the migration can be validated incrementally rather
than as one large rewrite. This is threaded through the phased rollout in Section 9.

---

## 1. Design decisions (resolve before writing code)

### Decision 1 — Replace or augment the training objective?

| Option | Description | Risk |
|---|---|---|
| A. Full replacement | Drop `L_actor` (policy gradient), `Â`, PLN shaping entirely. Train the actor by pure cross-entropy imitation on minimal-hypertree pairs, as HTPS does. | High — discards a working, validated training loop in one step; loses the failure-record negative signal HTPS has no equivalent for. |
| **B. Augment (recommended)** | Keep the existing on-policy `L_actor + L_critic + L_entropy + L_bc` untouched. Add two new, independently-sourced loss terms — `L_tactic_imitation` and `L_critic_soft` — trained through a **separate, decoupled step** that does not need to obey the on-policy invariant. | Low — additive, individually ablatable, nothing existing has to change to land it. |

**Recommendation: B.** The two objectives have genuinely different validity requirements
(Section 3 explains why), so keeping them as separate loss pathways rather than merging
them into one `total` expression is not just lower-risk, it's the architecturally correct
choice.

### Decision 2 — How many simulations per theorem?

> **Superseded** by `htps_integration_addendum_legacy_selection.md`. `num_simulations`
> does not select the search mode: one simulation expands every unexpanded leaf of the
> selected partial hypertree, while the legacy loop runs until root resolution or the
> `max_nodes` budget, so no simulation count reproduces the legacy process. The mode is
> selected by `selection_policy` (`"legacy"` — the default — runs the best-first loop;
> `"puct"` runs the simulation loop and requires an explicit `num_simulations`).

Introduce a config value `num_simulations` (default candidates: 1 for legacy-equivalent
behavior, up to several thousand for full HTPS-style search). At `num_simulations = 1`,
the new search loop must reduce exactly to today's single best-first pass — this is the
regression test for Phase 1 (Section 9).

### Decision 3 — Synchronous or asynchronous infrastructure?

**Recommendation: stay synchronous for now.** HTPS's asynchronous prover/trainer fleet
exists because its tactic objective *requires* full proof successes, which are rare early
on, so it needs many searches running continuously to produce enough data. This system's
new `L_tactic_imitation` and `L_critic_soft` have the same "doesn't go stale" property
HTPS's objectives have (Section 3), so they can be trained from a **replay buffer** fed by
the existing synchronous `collect_round`, without needing a separate prover fleet. Revisit
async infra only if single-machine throughput becomes the bottleneck (Phase 5).

### Decision 4 — What does the critic regress toward?

**Recommendation:** generalize `backup_values` rather than replace it. Currently:

```
value(SOLVED)     = 1.0
value(DEAD)       = 0.0
value(unexpanded) = 0.0                                    # hard default
value(interior)   = max over edges of AND-combine(children)
```

New version: for a node with status OPEN (neither SOLVED nor DEAD) whose visit count
`N(g, t*)` has crossed a configurable threshold, use `W(g, t*)/N(g, t*)` in place of the
hard `0.0`. Below the threshold, fall back to the existing `0.0`. At `num_simulations = 1`
no node will ever cross the threshold, so this exactly reduces to current behavior —
consistent with the Decision 2 regression test.

---

## 2. New components to add

### 2.1 `EdgeVisitStats` — visit-count and total-value tracking

New data structure attached to each hyperedge (not each node — tactics are the thing being
selected, mirroring HTPS's `N(g,t)`/`W(g,t)`):

```python
@dataclass
class EdgeVisitStats:
    N: int = 0                 # completed-simulation visit count
    W: float = 0.0             # total backed-up value across those visits
    virtual_loss: int = 0      # simulations currently in flight through this edge

    @property
    def Q(self) -> float:
        # First-play-urgency, adapted from HTPS Appendix A.2
        C = self.N + self.virtual_loss
        if C == 0:
            return 0.5
        return self.W / max(1, C)
```

Location: `hybrid_reasoner/hypergraph.py`, alongside the existing edge representation.

### 2.2 PUCT / RP selection policy

> **Superseded in part** by `htps_integration_addendum_legacy_selection.md`. The prior
> is stored once on `EdgeVisitStats.prior_prob` at edge creation (by
> `ProofHypergraph.add_edge`) and read — never recomputed — inside `puct_score`, whose
> signature is `puct_score(stats, total_node_visits, c)`. The config flag takes three
> literals, `selection_policy: "legacy" | "puct" | "rp"`, with `"legacy"` as the
> default (today's default behavior is the legacy loop) and `"rp"` reserved
> (`NotImplementedError`).

```python
def puct_score(stats: EdgeVisitStats, total_node_visits: int, c: float) -> float:
    return stats.Q + c * stats.prior_prob * math.sqrt(total_node_visits) / (1 + stats.N + stats.virtual_loss)
```

Location: new file `hybrid_reasoner/selection_policy.py`. Support both PUCT and the
regularized-policy (RP) variant behind a config flag (`selection_policy: "legacy" | "puct" | "rp"`),
matching HTPS's own finding that the better choice is environment-dependent (RP performed
better on Equations, PUCT on Metamath/Lean in their experiments) — this system should not
assume PUCT is universally correct without testing.

### 2.3 Multi-simulation search loop with virtual loss

Replaces a single `frontier().pop()` → expand → done with a loop of simulations against
one persistent hypergraph:

```python
def run_search(root_goal, num_simulations, sim_batch_size):
    graph = ProofHypergraph(root_goal)
    for batch_start in range(0, num_simulations, sim_batch_size):
        partial_trees = []
        for _ in range(sim_batch_size):
            tree = select_partial_hypertree(graph, puct_score)   # applies virtual_loss along the path
            partial_trees.append(tree)
            if graph.root.status in (SOLVED, DEAD):
                break
        unexpanded_leaves = collect_unexpanded_leaves(partial_trees)
        expand_batch(unexpanded_leaves)          # one batched GNN forward pass for the whole batch
        for tree in partial_trees:
            backup_simulation(graph, tree)        # updates N, W bottom-up; clears virtual_loss
        _propagate(graph, touched_nodes=...)      # existing status/rank maintenance, now covering new edges
        if graph.root.status in (SOLVED, DEAD):
            break
    return graph
```

Location: `hybrid_reasoner/joint_inference.py` (base loop) with the RL-specific seams still
overridden in `atp_lean_gnn/rl_reasoner.py`, matching the existing inheritance pattern.

`select_partial_hypertree` is new: unlike the current single-path `frontier()` pop, it must
recursively descend through **every** child of an AND-edge (not just the top-ranked node
globally), exactly as HTPS's Section 4.1 selection phase does, since a simulation has to
reach a full set of leaves consistent with one candidate proof, not one leaf in isolation.

`backup_simulation` is new and distinct from `_propagate`: it walks the *specific simulated
partial hypertree* bottom-up, using the AND-combine product rule to compute a value for
each touched node (using `V(s)` from the critic head for any leaf that's still unexpanded
after this batch's expansion, mirroring HTPS's `v_T(g) = c_θ(g)`), and increments `N`/`W`
on every edge in the tree. `_propagate` still runs afterward for status/rank maintenance —
these are two separate walks over possibly-overlapping node sets, doing different jobs.

### 2.4 Search hyperparameter sampler — DEFERRED (out of scope for Phases 0–3)

Per-attempt hyperparameter sampling (`SearchHParams`: sample count, temperature,
exploration constant, depth penalty; HTPS Appendix C) is a locked out-of-scope decision
for the current integration. HTPS's ablation (Section 7.2.3) shows sampling beats fixed
parameters, so this remains Phase-4 work — sweep it only after `num_simulations > 1` has a
measured proof-rate-vs-compute baseline. No `search_hparams.py` file is created now.

### 2.5 Minimal-hypertree tactic-imitation mining

```python
def extract_minimal_hypertree(graph: ProofHypergraph) -> list[TacticImitationSample]:
    """Only call if graph.root.status == SOLVED.
    Walk down from root, at each SOLVED node picking whichever SOLVED edge minimizes
    total downstream step count (or CPU time, if available from the executor),
    recursively. Emit one TacticImitationSample(goal, tactic) per edge on that path."""
```

`TacticImitationSample(goal, tactic)` is a new, deliberately minimal record — just enough
to reconstruct a supervised (input, target) pair, analogous to HTPS's `(goal, tactic)`
pairs mined from its own minimal proof hypertree.

The minimality criterion is step count only: `TacticOutcome` records no timing, so HTPS's
`tactic_cpu_time` criterion (their choice for Lean) would need executor instrumentation
that does not exist today. If timing is added to the executor later, revisit the criterion
then.

Also expose a `mine_all_solved_nodes` config flag: HTPS's own ablation found that mining
*every* solved node (not just those on the root's minimal path) outperformed root-only on
one of its three environments and lost on another — this should be a config knob to sweep,
not a hardcoded assumption (Section 10, Ablation 1).

Location: `atp_lean_gnn/search_harvest.py`, alongside the existing `extract_transitions`.

### 2.6 Soft critic-target extraction

```python
def extract_critic_samples(graph, visit_threshold) -> list[CriticSample]:
    samples = []
    for node in graph.nodes.values():           # graph.nodes is a dict keyed by node id
        if node.status == SOLVED:
            samples.append(CriticSample(node.goal, 1.0))
        elif node.status == DEAD:
            samples.append(CriticSample(node.goal, 0.0))
        else:
            # Unresolved: EXPANDED (has edges, children unresolved) or OPEN
            # (no edges yet — then there is no t_star and no sample).
            t_star = max(node.outgoing_edge_ids,
                         key=lambda eid: graph.edges[eid].tactic.probability,
                         default=None)
            if t_star is not None and graph.edges[t_star].visit_stats.N >= visit_threshold:
                stats = graph.edges[t_star].visit_stats
                samples.append(CriticSample(node.goal, stats.W / stats.N))
            # else: excluded — insufficient evidence, do not fabricate a label
    return samples
```

This runs over **every** finished search, regardless of whether the root ultimately
resolved — consistent with HTPS's finding that critic data does not require the root to
solve (Section 5.2 of the HTPS paper; also the reasoning we walked through earlier in this
conversation). Note this deliberately never touches PLN — it is pure visit-statistics and
status, same principle as the existing `backup_values`'s "no PLN in the value target"
invariant.

Location: `atp_lean_gnn/search_harvest.py`.

### 2.7 Replay queues (Phase 3+)

Two new finite-size FIFO queues, directly analogous to HTPS's tactic/critic queues:

```python
tactic_imitation_queue: deque[TacticImitationSample]  # maxlen configurable
critic_soft_queue: deque[CriticSample]                # maxlen configurable
```

Populated at the end of every `collect_round`, sampled uniformly to build batches for the
new decoupled training step (Section 2.8). Discard-oldest on overflow, exactly as HTPS
does — this is what lets the training distribution track the model's *current* best-known
proofs over time (a theorem solved with a shorter proof later naturally evicts the older,
longer-proof samples once the queue cycles).

### 2.8 Decoupled HTPS-style training step

```python
def train_step_htps_style(model, optimizer, tactic_queue, critic_queue, batch_size, w_critic):
    tactic_batch = sample(tactic_queue, batch_size)
    critic_batch = sample(critic_queue, batch_size)
    tactic_logits = model.tactic_head(encode(tactic_batch.goals))
    L_tactic = cross_entropy(tactic_logits, tactic_batch.tactics)
    values = model.critic_head(encode(critic_batch.goals))
    L_critic_soft = mse(values, critic_batch.targets)
    loss = L_tactic + w_critic * L_critic_soft
    loss.backward(); optimizer.step()
```

Unlike `train_step_onpolicy`, this can run **multiple times per round** — its data does not
go stale the way recomputed log-probs do, because it's ordinary supervised regression on
stored (input, label) pairs, not a score-function estimator. This is the direct payoff of
Decision 1: by keeping it decoupled, none of the "exactly one optimizer step per collect"
machinery has to be touched or reasoned about for this new pathway.

Location: `atp_lean_gnn/pln_rl_training.py`, as a new function alongside
`compute_onpolicy_loss`.

---

## 3. Why the decoupling in Decision 1 is not optional

This is worth stating explicitly since it drives several of the choices above. The
existing `L_actor` is a policy-gradient term — its correctness depends on
`log π(a|s)` being evaluated under the *exact* parameters that generated the action, which
is why the codebase enforces "exactly one optimizer step per collect round." `L_tactic_imitation`
and `L_critic_soft` are both ordinary supervised losses over (input, label) pairs that
remain correct regardless of how stale the parameters that *generated* the label are — the
label is either "this tactic was verified to work by Lean" or "this state's provability
evidence accumulated to this ratio," neither of which depends on which θ produced the
underlying search. Merging these into the same `total` loss and the same one-step
constraint as `L_actor` would import a restriction they don't need and gain nothing.

---

## 4. Existing components to modify

### 4.1 `hybrid_reasoner/hypergraph.py`

- Add `EdgeVisitStats` to the edge representation (Section 2.1).
- Extend `_propagate` — no change to its status/rank logic, but it must now be callable
  with a *set* of touched nodes (batched simulations touch many nodes per round-trip)
  rather than assuming a single mutation, since Section 2.3's loop calls it once per
  simulation-batch rather than once per single edge addition.

### 4.2 `hybrid_reasoner/joint_inference.py`

- `HybridReasoner.prove` gains the `num_simulations` / `sim_batch_size` parameters
  (Section 2.3). At `num_simulations=1` this must be behavior-identical to today —
  this is the Phase 1 regression test.
- New method `select_partial_hypertree` (Section 2.3), replacing the current single-node
  `frontier().pop()` call site when `num_simulations > 1`.

### 4.3 `atp_lean_gnn/rl_reasoner.py`

- The existing RL seams are `predict_next_tactic` (proposal), `_link` (stash migration on
  edge creation), and `_build_gnn_engine` (checkpoint-engine bypass); `rank_subgoals` is
  never overridden — PLN scoring stays on the base class. Under multi-simulation search
  these seams are called once per simulation-batch leaf rather than once per frontier pop.
- One new seam IS needed: the pending action stash must be re-keyed **per proposing node**
  (`goal_key → {fingerprint → EdgeAction}`), with a hook that flushes exactly one node's
  still-pending samples to failure records when its expansion finishes. Batched expansion
  proposes for several leaves before any of them links, so a single flat stash would
  misattribute one leaf's rejected samples to another leaf's flush.

### 4.4 `atp_lean_gnn/search_harvest.py`

- `backup_values`: add the threshold-gated soft-target branch (Decision 4). Signature
  gains an optional `visit_stats` and `visit_threshold` argument; both default to `None` /
  `inf` so existing callers (and Phase 0/1) are unaffected.
- Add `extract_minimal_hypertree` (Section 2.5) and `extract_critic_samples` (Section 2.6)
  as new, additive functions — `extract_transitions` (the existing on-policy harvest) is
  untouched.

### 4.5 `atp_lean_gnn/pln_rl_training.py`

- Add `train_step_htps_style` (Section 2.8) as a new function.
- `compute_onpolicy_loss` is untouched — no new terms added to its `total` expression, per
  Decision 1.

### 4.6 `atp_lean_gnn/rl_training_driver.py`

- New config fields: `num_simulations`, `sim_batch_size`, `selection_policy`, `puct_c`,
  `visit_threshold`, `tactic_queue_size`, `critic_queue_size`, `htps_steps_per_round`,
  `htps_batch_size`, `htps_learning_rate`, `w_critic_soft`, `mine_all_solved_nodes`.
  (`search_hparam_ranges` deferred with Section 2.4; virtual loss is a fixed +1 per
  traversed edge, no `virtual_loss_amount` knob.)
- Driver loop gains a second call after the existing on-policy step:

```
for round in 0..num_rounds:
    batch    ← sample from curriculum window
    results  ← collect_round(reasoner, batch)                       # now runs multi-sim search
    metrics  ← train_step_onpolicy(..., bc_weight=anneal(round))    # UNCHANGED, one step
    tactic_queue.extend(extract_minimal_hypertree(...) for graphs in results)            # 2.5, 2.7
    critic_queue.extend(extract_critic_samples(...) for all graphs in results)           # 2.6, 2.7
    for _ in range(htps_steps_per_round):
        train_step_htps_style(model, optimizer_htps, tactic_queue, critic_queue, ...)   # 2.8, can run N times
    window widens when the recent solve rate ≥ threshold             # unchanged
    checkpoint / eval as before                                       # unchanged
```

Note: use a **separate optimizer instance** (`optimizer_htps`) for the decoupled step, or
at minimum separate Adam moment buffers, so the two training signals don't fight over
shared optimizer state in ways that are hard to reason about.

### 4.7 `atp_lean_gnn/actor_critic.py`

- No structural change required. A `temperature` parameter on `act()` becomes relevant
  only when the deferred hyperparameter sampler (Section 2.4) lands in Phase 4.

---

## 5. Inference-time changes — DEFERRED (out of scope for Phases 0–3)

The deployment path (`evaluate_proof_rate`, greedy/argmax policy, single best-first pass)
stays the only inference mode; no `prove_mcts` is added (locked decision). The notes below
are kept for the eventual Phase-4+ revisit.

```python
def prove_mcts(reasoner, goal, num_simulations, sim_batch_size, selection_policy="puct"):
    """High-effort inference mode (NOT IMPLEMENTED — deferred). Would reuse the exact same
    simulation loop, PUCT scoring, and virtual-loss machinery built for training
    (Section 2.3) — no separate implementation to maintain. Sampling defaults to greedy at
    the leaves (argmax tactic, argmax pointer) since exploration noise is not wanted in a
    deployed proof attempt, but PUCT still explores across *tactics* via the
    visit-count/prior tradeoff."""
```

Practical notes for this mode (when revisited):

- **Cost scaling.** Every additional simulation costs one batched GNN forward pass
  (cheap relative to HTPS's 600M-parameter transformer) plus, for any newly-created
  subgoal, one PLN subprocess call (`evaluate_async`) and one Lean tactic execution. The
  GNN side scales gracefully; PLN and Lean-kernel cost do not disappear just because the
  network itself is cheap — profile before defaulting to large simulation budgets in
  production.
- **No pass@k needed.** HTPS's pass@k exists because each attempt is a single,
  comparatively shallow best-first-ish process with high variance; deep PUCT search over
  many simulations against one root is closer to running pass@k *inside* a single attempt
  already, via the exploration term. Treat `num_simulations` as the primary knob to tune
  for inference quality, not repeated independent attempts, unless evidence says
  otherwise.
- **Virtual loss still needed at inference** if `sim_batch_size > 1`, for the same reason
  as training — without it, batched simulations would all greedily pick the identical
  top-PUCT path.

---

## 6. Compatibility with the existing invariants checklist

| Existing invariant | Status after this change |
|---|---|
| one optimizer step per collect | **Unchanged**, still applies to `train_step_onpolicy` only. Explicitly does **not** apply to `train_step_htps_style` (Section 3) — document this exception directly next to the invariant in code comments. |
| featurizer identity collect↔train | Unchanged, still required for the on-policy path. The new imitation/critic samples are goal-string-keyed and re-featurized fresh at each `train_step_htps_style` call, so this invariant does not apply to them. |
| vocabs from `prepared_root` only | Unchanged. |
| Φ(terminal) = 0 | Unchanged — PLN shaping is untouched by this plan. |
| critic sees success rows only | **Unchanged** for the existing `L_critic` (structural backup, on-policy rows). The new `L_critic_soft` has its own, separately documented rule (Section 2.6: SOLVED / DEAD / high-visit-OPEN) — do not conflate the two masks. |
| dedup with multiplicity | Unchanged, applies only to the on-policy actor loss. |
| value target from backup, not PLN | **Preserved and extended**, not violated: the new soft target is still built purely from `N`/`W` (search statistics), never from PLN. |
| sequential collect per reasoner | Unchanged at this phase — batching in Section 2.3 is *within* one theorem's search (batched GNN calls across simulations), not across theorems. Revisit only in Phase 5. |

**New invariants to add:**

| New invariant | Enforced by | Broken consequence |
|---|---|---|
| soft critic target only used above `visit_threshold` | `extract_critic_samples` | low-N noise gets treated as a reliable label |
| virtual loss fully cleared after every `backup_simulation` call | `backup_simulation` | `Q` estimates drift, PUCT selection degrades over the course of a search |
| `num_simulations=1` reduces exactly to legacy single-pass search | regression test in Section 9, Phase 1 | silent behavior change for anyone running default config |
| `train_step_htps_style` never touches `optimizer` (only `optimizer_htps`) | separate optimizer instances | shared Adam moments couple two unrelated gradient signals |

---

## 7. Data flow (end to end, one round)

```
sample_search_hparams()
        │
        ▼
collect_round(reasoner, batch, hparams)
        │
        ├─ per theorem: run_search()  ── select_partial_hypertree × sim_batch_size
        │                              ── expand_batch()  (GNN fwd, PLN, Lean exec)
        │                              ── backup_simulation()  (N, W updates)
        │                              ── _propagate()  (status, combined_rank)
        │                              … repeat until num_simulations exhausted or resolved
        │
        ▼
results (finished hypergraphs, one per theorem)
        │
        ├──────────────────────────────┬───────────────────────────────┐
        ▼                               ▼                               ▼
extract_transitions()          extract_minimal_hypertree()     extract_critic_samples()
(existing, on-policy)          (new, solved roots only)        (new, any resolved node)
        │                               │                               │
        ▼                               ▼                               ▼
train_step_onpolicy()          tactic_imitation_queue.extend()  critic_soft_queue.extend()
(UNCHANGED, one step)                    │                               │
                                          └───────────────┬───────────────┘
                                                           ▼
                                          train_step_htps_style() × htps_steps_per_round
```

---

## 8. Model/inference call-count comparison, before and after

| Quantity | Current (K=1) | After (K=`num_simulations`) |
|---|---|---|
| GNN forward passes per theorem attempt | k single-graph forwards per expanded node (one per i.i.d. draw from `model.act`), one node per frontier pop | k multi-graph forwards per simulation batch (one per draw, batched across the batch's deduplicated leaves) × `num_simulations / sim_batch_size` batches |
| PLN calls per theorem attempt | 1 per newly-created subgoal | scales with total subgoals discovered across all simulations — grows with K |
| Lean executions per theorem attempt | 1 per unique candidate tactic tried | scales similarly — same growth driver as PLN |
| Training data per round | on-policy transitions only | on-policy transitions **+** tactic-imitation samples (only from solved roots) **+** critic-soft samples (from any resolved-enough node, solved or not) |

This table should be re-measured empirically once Phase 1 lands (Section 9) — treat these
as expected-direction estimates, not committed numbers, since actual PLN/Lean cost per
simulation depends heavily on how much subgoal reuse occurs within one theorem's search.

---

## 9. Phased rollout plan

| Phase | Content | Exit criterion |
|---|---|---|
| **0** | Add `EdgeVisitStats` to the graph; track `N`/`W` but do not use them for selection or training. | Stats accumulate correctly; zero change to existing metrics. |
| **1** | Add `select_partial_hypertree`, PUCT scoring, `backup_simulation`, virtual loss. Wire `num_simulations` through the driver, default to `1`. | At `num_simulations=1`, proof rate and wall-clock are statistically indistinguishable from the pre-change baseline (regression test). |
| **2** | Enable `num_simulations > 1` on a subset of the curriculum. Add soft critic target (Decision 4) and `L_critic_soft` via the decoupled step. | Ablation 2 (Section 10) shows soft critic ≥ current hard-only critic; no regression in `L_critic` (on-policy) behavior. |
| **3** | Add `extract_minimal_hypertree` and `L_tactic_imitation` via the decoupled step. | Ablation 1 (Section 10) run; pick default for `mine_all_solved_nodes` based on results, not assumption. |
| **4** | Sweep `num_simulations`, `sim_batch_size`, and search-hyperparameter ranges (Section 2.4); recalibrate curriculum timeouts for the new per-round wall-clock cost. | Proof-rate-vs-compute curve plotted; pick a production default `num_simulations`. |
| **5 (optional)** | Only if Phase 4's throughput is insufficient: decouple generation from training with an async prover/trainer split, mirroring HTPS's architecture. | Not pursued unless single-machine synchronous throughput is demonstrated to be the bottleneck. |

---

## 10. Testing and ablation plan

**Unit tests**

- `EdgeVisitStats.Q` returns `0.5` when `N=0` (first-play-urgency), and `W/N` otherwise.
- `backup_simulation` correctly zeroes `virtual_loss` after completion; a second
  simulation through the same edge sees `virtual_loss=0`, not a residual value.
- `select_partial_hypertree` descends into **every** child of an AND-edge, not just the
  highest-PUCT one — a single-child bug here would silently degrade to plain best-first.
- `extract_minimal_hypertree` only returns non-empty output when `root.status == SOLVED`.
- `extract_critic_samples` never emits a sample for an OPEN node below `visit_threshold`.
- At `num_simulations=1`, `run_search` produces a graph identical (same nodes, same edges,
  same statuses) to the current `HybridReasoner.prove` on the same seeded inputs.

**Ablations (run once Phases 2–3 land), mirroring the HTPS paper's own methodology**

1. **Tactic mining source** — root-only minimal hypertree vs. all-solved-nodes minimal
   hypertrees. Measure proof rate on held-out eval pool. (Mirrors HTPS Table 4; note the
   HTPS paper itself found the winner is environment-dependent — do not assume Lean will
   match Metamath's or Equations' result.)
2. **Critic target** — no `L_critic_soft` (current) vs. soft `W/N` vs. a "hard" variant
   that treats every OPEN node as `0.0` regardless of visit count. (Mirrors HTPS Table 5;
   watch specifically for the failure mode HTPS found on Equations, where hard labeling
   performed *worse* than no added signal at all.)
3. **Search hyperparameters** — fixed vs. sampled-per-attempt. (Mirrors HTPS Section
   7.2.3.)
4. **Simulation count sweep** — `num_simulations ∈ {1, 8, 64, 512, 4096}`, plotting proof
   rate against total GNN + PLN + Lean call count, to find the marginal-value curve before
   committing to a production default.

---

## 11. Open risks

- **PLN cost may dominate at high `num_simulations`.** Unlike HTPS, this system has a
  heuristic-scoring step (PLN) with no analogue in HTPS's cost model. Profile Phase 4
  before assuming GNN cheapness translates into overall search cheapness.
- **Two optimizers touching the same parameters.** Even with separate optimizer instances,
  `L_actor`/`L_critic` (on-policy) and `L_tactic_imitation`/`L_critic_soft` (decoupled) are
  both updating the *same* underlying weights. Watch for interference during Phase 2–3
  validation, not just correctness of each pathway in isolation.
- **Curriculum timeout invalidation.** `collect_round`'s existing per-theorem timeouts were
  presumably tuned against single-pass search cost; multi-simulation search will need new
  timeout budgets, or theorems will start timing out mid-search for reasons unrelated to
  their actual difficulty.
- **Environment-dependent ablation outcomes.** Flag explicitly to whoever reviews Phase 3
  results: HTPS's own paper found root-only vs. all-solved-nodes mining, and PUCT vs. RP
  selection, both flip winner depending on the proving environment. Do not port HTPS's
  Lean-specific numbers as an assumption — this system's Lean setup differs enough
  (GNN vs. transformer, PLN-guided vs. pure learned critic) that it needs its own ablation
  run, not an inherited conclusion.

---

## 12. File map (new and modified)

| File | Status | Content |
|---|---|---|
| `hybrid_reasoner/hypergraph.py` | modified | `EdgeVisitStats`, `_propagate` batched-touch support |
| `hybrid_reasoner/selection_policy.py` | **new** | PUCT scoring (RP deferred) |
| `hybrid_reasoner/joint_inference.py` | modified | `_select_partial_hypertree`, multi-simulation loop in `prove` |
| `atp_lean_gnn/rl_reasoner.py` | modified | per-node pending stash, batched proposal, critic leaf value |
| `atp_lean_gnn/search_harvest.py` | modified | soft-target branch in `backup_values`; new `extract_minimal_hypertree`, `extract_critic_samples` |
| `atp_lean_gnn/pln_rl_training.py` | modified | new `train_step_htps_style`; `compute_onpolicy_loss` unchanged |
| `atp_lean_gnn/rl_training_driver.py` | modified | new config fields, second training call, replay queues |
| `maths_ai/gnn_inference/configs/rl_actor_critic.json` | modified | add all Section 4.6 config fields |
| `maths_ai/gnn_inference/tests/test_selection_policy.py` | **new** | PUCT unit tests |
| `maths_ai/gnn_inference/tests/test_search_harvest.py` | **new** | soft-target and minimal-hypertree extraction tests |
| `maths_ai/gnn_inference/tests/test_rl_training_driver.py` | modified | default-off regression + decoupled-step and checkpoint tests |

Deferred with Sections 2.4 and 5: `atp_lean_gnn/search_hparams.py`, the `temperature`
parameter on `act()`, and `prove_mcts`.
