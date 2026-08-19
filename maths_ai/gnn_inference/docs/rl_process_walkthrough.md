# The RL process, end to end: sampling → search → storage → gradient step

This document walks one training round through every mechanism, in execution order, with
the exact tensors and file locations. It describes the system as implemented in
`atp_lean_gnn/` (`actor_critic.py`, `rl_reasoner.py`, `pln_reward.py`,
`search_harvest.py`, `pln_rl_training.py`, `rl_training_driver.py`) and
`hybrid_reasoner/` (`hypergraph.py`, `joint_inference.py`,
`selection_policy.py`).

Two search modes are available, selected by `selection_policy` in `RLTrainingConfig`:

- **`"legacy"`** (default) — the original best-first AND-OR loop, unchanged in behavior.
  An explicit simulation budget under this mode raises `ValueError` at config construction.
- **`"puct"`** — HTPS-style repeated simulation: PUCT-guided partial-hypertree selection
  with virtual loss, batched leaf expansion, validity-aware per-edge `N/N_v/W/Q` statistics, and a decoupled
  imitation + soft-critic step fed from the accumulated visit data.

Sections 2–6 describe both modes side by side. The legacy path is the original document's
text; the HTPS additions are clearly delimited.

## Symbol table

| Symbol | Meaning | Where it lives |
|---|---|---|
| `s` | a proof state: goal expression + local hypotheses | `Goal(expression, hypotheses)` |
| `τ` | a tactic family (e.g. `intro`, `exact`) | index into the tactic vocab |
| `u_k` | the k-th pointer-selected tactic argument (a DAG node) | index into the state DAG's node list |
| `a = (τ, u_1…u_K)` | one full action | `EdgeAction(tactic_id, arg_indices, multiplicity)` |
| `π(a\|s)` | the policy: tactic head × autoregressive pointer | `ActorCriticWithArgsClassifier` |
| `V_θ(s)` | the current critic prediction for state `s` | `CriticHead` on the state embedding |
| `y(s)` | a numeric critic learning target, with explicit validity and provenance | `CriticSample` from `compute_backups` |
| `Φ(s)` | shaping potential = PLN strength σ of `s` (0 at terminals, 0 when stv is None) | `pln_reward.potential` |
| `r` | per-edge shaped reward `r_term + Σ_j(γΦ(child_j) − Φ(parent))` | `pln_reward.edge_shaped_reward` |
| `G` | per-edge return `r + γ·AND-combine(children's backup values)` | `search_harvest.extract_transitions` |
| `Â` | advantage `G − V(s)`, batch-normalized | `pln_rl_training.compute_onpolicy_loss` |
| `m` | multiplicity: how many of the k i.i.d. draws produced this action | `EdgeAction.multiplicity` |
| `γ` | discount (default 0.99) | `RewardConfig.gamma` |
| `N(e)` | completed traversals of edge `e`, including unknown backups | `EdgeVisitStats.N` |
| `N_v(e)` | traversals of edge `e` with a valid numeric backup | `EdgeVisitStats.N_v` |
| `W(e)` | accumulated value over those `N_v` valid backups | `EdgeVisitStats.W` |
| `VL(e)` | virtual loss: in-flight simulations holding edge `e` | `EdgeVisitStats.virtual_loss` |
| `P(e)` | policy prior: tactic probability stamped at edge creation | `EdgeVisitStats.prior_prob` |
| `Q(e)` | mean valid action value `W/N_v`, first-play urgency 0.5 when `N_v=0` | `EdgeVisitStats.Q` |
| `c` | PUCT exploration constant (default 1.0) | `puct_c` in `RLTrainingConfig` |
| `v_T(s)` | leaf value estimate for unresolved simulation leaf | `RLHybridReasoner._leaf_value` |
| `B` | simulation batch size | `sim_batch_size` in `RLTrainingConfig` |

## 0. The model: one encoder, four heads

`ActorCriticWithArgsClassifier` runs the checkpoint-selected encoder once per state and
feeds four heads from its outputs (`encode`, actor_critic.py). The version-2 checkpoint
manifest selects GraphSAGE or GATv2, including the GATv2 readout mode; RL search and loss
code use the common encoder interface and do not import either architecture directly.

1. **Node embeddings** `[N, H]` — one vector per node of the state DAG (the proof state
   parsed into a hash-consed graph by `proof_state_to_dag`).
2. **State embedding** `[B, H]` — readout at the `State` root node.
3. **Tactic logits** `[B, T]` — the actor head: `actor(h) = base(h) + residual(h)`, where
   `base` inherited the supervised classifier at warm start and `residual`'s output layer
   was zero-initialized, so the RL policy starts exactly at supervised behavior and RL
   gradients grow the deviation.
4. **Value** `[B, 1]` — the critic head, a small MLP on the state embedding. Random at
   warm start (nothing supervised estimates values); harmless because the advantage uses
   `V.detach()`.

The pointer (`ArgumentSelector`) is the argument half of the policy: query =
`proj([state_emb; tactic_emb])` (or `proj([state_emb; tactic_emb; prev_arg_emb])` from the
second argument on — autoregressive), keys = node embeddings, scaled dot-product scores
over the state's nodes, non-premise/padding positions masked to −inf.

Under the HTPS mode the critic head gains a second role: `_leaf_value` queries it at
every unresolved simulation leaf as `v_T(s) = c_θ(s)`, providing the value estimate that
`_backup_simulation` propagates back up the partial hypertree. A well-trained critic
therefore steers simulations toward promising branches even before the leaf is expanded.

## 1. Action sampling: `model.act` (i.i.d.) vs. `greedy`

`act(data, id_to_tactic, greedy)` (actor_critic.py) produces one full action per call:

- **Tactic**: build `Categorical(logits=tactic_logits)`; **training** draws
  `tactic_dist.sample()` — an i.i.d. sample from π(τ|s); **evaluation** takes `argmax`.
  Either way `tactic_logp = log π(τ|s)` and the distribution entropy are recorded.
- **Arguments**: the sampled tactic's arity (via `get_tactic_arity`) fixes the number of
  pointer steps. Each step calls `ArgumentSelector.sample_step` — again
  `Categorical(logits=scores).sample()` under training, and the **sampled** node's
  embedding (not the argmax's) feeds the next step's query, so the trajectory through
  argument space is itself on-policy. `arg_logp` accumulates `Σ_k log π(u_k|s,τ,u_<k)`;
  a state with no valid premise (all-−inf row) contributes exactly 0 to it.

Why i.i.d. draws and not top-k? The policy-gradient estimator
`∇J = E_{a~π}[∇log π(a|s)·Â]` is unbiased only when the actions are *drawn from* π.
Deterministic top-k picks the k highest-probability actions with probability 1, which is a
different sampling distribution; its log-prob weights would systematically misweight the
gradient. So the RL search draws `top_k_tactics` independent samples instead — a
high-probability action may appear several times, and that repetition is *information*
(section 3 keeps it as `multiplicity`).

**Batched proposal (HTPS mode)**: `predict_next_tactics_batch` replaces the per-node loop
with one `Batch.from_data_list` across all leaves of a simulation batch, runs
`top_k_tactics` draws through `model.act` on the joint batch, and splits results back per
goal. The policy math is identical — `model.act` sees the same batch size per draw; only
the graph-construction overhead is amortized.

Greedy mode exists solely for evaluation: `evaluate_proof_rate` measures the deterministic
policy — the thing that will be deployed — not the exploration distribution around it.

## 2. Search: the AND-OR hypergraph as the environment loop

`RLHybridReasoner.prove` (rl_reasoner.py) inherits from `HybridReasoner` (joint_inference.py)
and swaps in the sampled policy at specific seams. The active search mode is resolved at
construction time by `resolve_search_params` (selection_policy.py) and stored on
`self.selection_policy`.

### 2a. Legacy best-first search (`selection_policy="legacy"`)

The original best-first AND-OR loop, unchanged in behavior:

1. **Root**: `ProofHypergraph(Goal(...))` creates node 0, `status=OPEN`.
2. **Frontier pop**: `graph.frontier()` returns OPEN nodes sorted by `combined_rank`
   descending; the loop expands the top one. `combined_rank` starts as
   `gnn_probability × STV.score` (or `gnn_probability × 1.0` when `stv is None`) and is
   continuously revised by backprop (step 6 below).
3. **Proposal** (seam 1, `predict_next_tactic`): featurize the node's sanitized goal —
   `Goal → goal_to_state → proof_state_to_dag → dag_to_pyg` under the **fixed prepared
   vocab** — then draw k i.i.d. actions from `model.act` under `torch.no_grad()`. Each
   draw is **decoded**: `tactic_id → name` via the inverted tactic vocab; each sampled
   pointer index → the DAG node at that offset → a Lean argument string via
   `_resolve_local_node_name`. Identical decoded actions are deduplicated for the executor
   but counted (`multiplicity`).
4. **Execution** (`_execute_and_link`): for each unique candidate,
   `PantographExecutor.apply` sends `"{tactic} {args}"` to the live Lean server.
   - Rejected → no edge; stays pending → failure record (section 3).
   - Accepted, no subgoals → QED: `add_edge(node, tactic, [])`, immediately SOLVED.
   - Accepted, with subgoals → PLN scores each subgoal concurrently (`rank_subgoals`);
     the top-k `(Goal, STV)` pairs become the children of one AND-edge.
5. **Cycle guard**: `add_edge` fingerprints every child against the node's ancestors; a
   child matching an ancestor is force-marked DEAD.
6. **Propagation**: every mutation triggers `_propagate` — a bottom-up fixpoint that
   re-derives AND-OR statuses and re-ranks every ancestor's `combined_rank`. This is what
   makes the search best-first in provability: a new leaf score immediately re-orders the
   global frontier.
7. **Termination**: root SOLVED, root DEAD, `max_nodes` budget, or the `deadline`
   timestamp (`time.monotonic()`). The `deadline` check runs between expansions, so the
   partial graph — with all experience gathered so far — is always returned intact.
8. **Expansion hook**: `_on_expansion_complete(node)` fires when a node's expansion
   fully finishes (all candidates executed, or the node was exhausted without a proposal).
   The base class no-ops it; `RLHybridReasoner` flushes that node's still-pending sampled
   actions to failure records there (section 3).

### 2b. HTPS simulation loop (`selection_policy="puct"`)

Replaces the single frontier-pop loop with `_prove_mcts`: repeated batches of B
simulations, each simulation being a selection + expansion + backup cycle over the same
growing hypergraph.

#### 2b-i. Selection: `_select_partial_hypertree`

Descend from the root. At each **EXPANDED** node, choose the non-DEAD outgoing edge that
maximizes the PUCT score:

```
puct_score(e) = Q(e) + c · P(e) · sqrt(total_visits) / (1 + N(e) + VL(e))
```

where `total_visits = Σ_{e' viable} (N(e') + VL(e'))` over the node's non-DEAD edges,
`P(e)` is the tactic probability stamped at edge creation time (`EdgeVisitStats.prior_prob
= tactic.probability`), and `Q(e) = W(e) / N_v(e)` with first-play urgency 0.5 when no
valid numeric backup exists.

The AND semantics require entering **every child** of the chosen edge — all subgoals
must close for the edge to count. The descent therefore pushes all children of the chosen
edge onto the stack, not just one. Each traversed edge receives `virtual_loss += 1`
immediately, so the next selection in the same batch sees an inflated visit count and
steers toward a different branch (the batching effect: B simulations explore B distinct
paths without waiting for each other's backups).

An **OPEN** node is a leaf: append to `simulation.leaves` and stop descending through it.
A **SOLVED** or **DEAD** node is a terminal for this simulation: stop, but the backup
will read its status.

Returns `_Simulation(chosen_edges={node_id: edge_id}, leaves=[...])` or `None` when no
path exists (root already resolved, or all edges under it are DEAD).

#### 2b-ii. Batched leaf expansion: `_expand_leaves`

Collect all unique leaf node ids across the batch (two simulations can meet at a shared
leaf after virtual-loss steering). For each:
- Skip if already non-OPEN (an earlier leaf's propagation may have resolved it).
- Apply depth limit → `mark_node_exhausted` + `_on_expansion_complete`.
- Otherwise: add to `to_propose`.

Call `predict_next_tactics_batch(sanitized_goals)` — one multi-graph policy forward
across all proposing nodes. Then for each node, call `_execute_and_link` sequentially
(one Pantograph server; Lean execution cannot be parallelized). The `_on_expansion_complete`
hook fires per node when its execution finishes.

#### 2b-iii. Backup: `_backup_simulation`

Walk each simulation's `chosen_edges` and assign values bottom-up:

```
value(SOLVED node) = valid 1.0 from Lean
value(genuine path failure) = valid 0.0
value(unelaborated or infrastructure-failed node) = UNKNOWN
value(unresolved leaf) = critic prediction with CRITIC_BOOTSTRAP provenance
value(interior node) = validity-aware AND-combination of every chosen child
```

The product over children is the AND-combine: all subgoals of an edge must close. For
each `(node_id, edge_id)` in `chosen_edges`:

```
edge_value = AND-combine(value(child) for child in edge.child_ids)
stats.N += 1
if edge_value is valid:
    stats.N_v += 1
    stats.W += edge_value
stats.virtual_loss = max(0, stats.virtual_loss - 1)   # release the in-flight hold
```

`_leaf_value` in the base class returns 0.5 (uninformed). `RLHybridReasoner` overrides it
with the critic head's value estimate `c_θ(s)` — the same head used for the advantage
baseline. A well-trained critic here focuses simulations toward branches the model already
believes are promising.

#### 2b-iv. Iteration

After each batch of B simulations, the loop checks: root solved/dead, `max_nodes` budget,
`num_simulations` counter, `deadline`. Because status propagation runs inside `add_edge`
during `_execute_and_link`, the graph's structural state is up-to-date after every
expansion — the next selection batch immediately sees the resolved subtrees.

So under HTPS the AND-OR graph plays three roles simultaneously: **control structure**
(PUCT selection decides where the policy is asked next), **experience accumulator** (edge
visit statistics record what the search has learned about each branch), and **training
data source** (its topology and terminal statuses drive rewards, critic targets, and
imitation samples — see sections 4 and 4b).

## 3. Storage: what survives the search, and why only integers

The search-time forward passes ran under `no_grad` — nothing differentiable survives, by
design: holding autograd graphs for every sampled action across a multi-node async search
would pin the memory of every intermediate activation for minutes. What is stored instead
(rl_reasoner.py):

- **Per-node pending stash** — during a node's expansion, every decoded draw sits in
  `self._pending: {node_id → _PendingNode}`, where `node_id` is the unique graph-node
  identifier and `_PendingNode.actions = {fingerprint → EdgeAction}`. `fingerprint =
  (tactic_name, args)`. `EdgeAction` is three integers-worth of data: `tactic_id`,
  `arg_indices` (the raw sampled pointer positions — the recompute must evaluate what was
  actually sampled), and `multiplicity`.

  The per-node keying (not a single flat dict) means that a batched proposal across
  several simulation leaves cannot misattribute one leaf's rejected samples to another
  node's goal. The flush is scoped to exactly the finished node.

- **Edge join** (seam 2, `_link`) — when `_execute_and_link` links an accepted tactic
  into the graph, the subclass moves the stash entry from the pending sub-dict to
  `edge_actions[edge.id]`. This is the join key between graph structure and policy
  actions. Under HTPS, the same `_link` override fires for every leaf of every simulation
  batch — the join table accumulates across the entire search.

- **Failure flush** (`_on_expansion_complete`) — entries still in the node's pending
  sub-dict when `_on_expansion_complete` fires were rejected by Lean. They move to
  `failure_actions` as `FailureRecord(goal, action)`. Under the legacy loop this fires at
  the end of each `_expand` call; under HTPS it fires per leaf immediately after that
  leaf's `_execute_and_link` finishes.

- **Per-search container** — `prove` returns
  `RLSearchResult(graph, edge_actions, failure_actions)` and resets stashes. `edge.id` is
  unique only within one `ProofHypergraph`; collect therefore runs theorems sequentially
  on one reasoner (`collect_round`, rl_training_driver.py), with per-theorem
  timeout/exception isolation so one dead search cannot kill the round.

- **Visit statistics** (HTPS mode only) — `EdgeVisitStats.N`, `N_v`, `W`, `Q`,
  `virtual_loss`, and `prior_prob` are stored on every `ProofHyperedge` inside the graph
  (hypergraph.py). `N` counts all completed traversals, including unknown outcomes;
  `N_v` counts only traversals with valid numeric backups; `W` sums those valid values;
  and `Q = W/N_v`, with first-play urgency when `N_v = 0`. Unknown outcomes therefore
  affect exploration through `N` but never add a numeric value to `W`. These statistics
  survive in `RLSearchResult.graph` and are consumed after the search by
  `extract_critic_samples` and `extract_minimal_hypertree` (section 4b).

## 4. Harvest: turning the finished graph into on-policy training targets

`train_step_onpolicy` (pln_rl_training.py) first harvests each result's graph. This
section describes the on-policy A2C path that runs every round regardless of search mode.

**Value backup** (`search_harvest.compute_backups`) — a validity-aware AND-OR recursion,
memoized and cycle-safe. In the legacy mode:

```
value(SOLVED)      = valid 1.0 from Lean
value(genuine DEAD) = valid 0.0
value(unelaborated) = UNKNOWN
value(interior)    = validity-aware OR over tactic edges
AND-combine        = product (default) or min of valid children
value(unexpanded)  = UNKNOWN
```

Under HTPS, `compute_backups` accepts an optional `visit_threshold`. A supported outgoing
edge with at least that many valid backups contributes a soft `VISIT_MEAN` target from
the maximum supported `Q`. Unknown and unelaborated nodes emit no critic sample. Under
`selection_policy="legacy"` no edge accumulates visits, so no soft target is available.

The backup is the **critic's regression target**. It contains no PLN number — only
demonstrated solvability (and, in HTPS mode, accumulated simulation consensus) enters the
value target, so an unreliable PLN or an untrained critic cannot poison this signal. In an
AND-edge with multiple subgoals, one solved child contributes `1.0`, but the edge remains
`UNKNOWN` when another child is unelaborated or infrastructure-failed; it does not become
`0.0` merely because a complete proof was not observed. A hard `0.0` requires a genuine,
fully searched failure boundary.

**Per-edge reward** (`pln_reward.edge_shaped_reward`) — where PLN enters, in the only
provably-safe slot:

```
r = r_term + Σ_j ( γ·Φ(child_j) − Φ(parent) )
r_term = +terminal_success − step_penalty   if the edge is QED (no children, SOLVED)
         terminal_failure − step_penalty    if the edge is DEAD
         −step_penalty                      otherwise
Φ(s)   = PLN strength σ of s
         0 for SOLVED/DEAD nodes, and 0 when node.stv is None
```

`Φ` returns 0 when `node.stv is None` — so in any context where PLN did not score a
node (the kill-switch path on AC-branch, or any node expanded under HTPS after PLN was
not called), the shaping term vanishes automatically and `r = r_term`. The shaping term
is potential-based (Ng–Harada–Russell): because Φ is a function of state alone and
Φ(terminal) = 0, the shaping telescopes along every path to `γ^T·0 − Φ(s_0)` — a
constant offset per trajectory — so it can bias learning speed toward states PLN likes
but cannot change which policy is optimal.

**Actor transitions** (`extract_transitions`) — one `ActorTransition` per policy-produced
edge with a valid edge outcome. Unknown edges are omitted. Critic targets are separate
`CriticSample` rows, deduplicated by node and carrying their `BackupSource` provenance.

```
ActorTransition(node_id, goal, tactic, reward=r,
                successor_value = valid AND-combined child outcome,
                return_        = r + γ·successor_value,
                edge_id)
```

`return_` is a one-step bootstrapped target: the immediate shaped reward plus the
discounted validity-aware AND-OR value of what the action produced. Because
`successor_value` comes from the backup — which already OR-maxes over everything the
return sees arbitrarily deep consequences without an explicit T-step rollout sum.

## 4b. HTPS-mode additional harvest: imitation + soft-critic queues

After each search in HTPS mode, two more extraction passes run before the on-policy step,
filling rolling deques that are consumed by `train_step_htps_style` later in the round.

**Critic samples** (`extract_critic_samples`) — one `CriticSample` per unique node with a
valid target and an explicit source:
- Lean-confirmed SOLVED node: `1.0`, source `LEAN_STATUS`.
- Genuine search failure: `0.0`, source `SEARCH_FAILURE`.
- Clean root budget failure: root-only `0.0`, source `ROOT_EPISODE`.
- Supported MCTS action: `Q`, source `VISIT_MEAN`.
- Unknown or unelaborated evidence: no row.

These samples bypass the on-policy recompute entirely: they are supervised regression
targets whose validity does not depend on the current parameters — any parameters that
encode "this goal was proved" should output value near 1.0.

**Minimal-hypertree imitation samples** (`extract_minimal_hypertree`) — for each SOLVED
node, find the **step-minimal** proof path: the Lean-confirmed SOLVED outgoing edge whose subtree
requires the fewest total tactic applications (memoized over the shared subgraph). Emit
one `TacticImitationSample(goal, hypotheses, tactic_id, arg_indices)` per edge on that
minimal tree, restricted to edges present in `edge_actions`. PLN ranking cannot create a
proof edge or hard success target.

The minimal hypertree is the proof the search actually found expressed as the cheapest
path to closure. Imitating it is more targeted than imitating every edge in the SOLVED
subtree, which might include exploratory detours the search later proved suboptimal.

## 5. The on-policy gradient step: recompute, then one update

`compute_onpolicy_loss` (pln_rl_training.py) assembles one batch from all of the round's
results (edge ids re-keyed per graph to avoid collisions), each row being either a
successful transition or a failure record:

**Log-prob recompute** — `model.evaluate_actions(batch, tactic_ids, arg_indices)` re-runs
the encoder **with gradient** under the current parameters and evaluates the log-probs of
the *stored* actions: `tactic_logp = log_softmax(logits)[τ]`, and the pointer is
teacher-forced through `forced_step` — at step k the stored index `u_k`'s log-prob is
gathered and `u_k`'s embedding (not a fresh sample's) feeds step k+1. Invalid/-1 indices
contribute log-prob 0 and a zero embedding. This is the standard A2C evaluate-actions
pattern, and it is exactly on-policy **only because no optimizer step ran between collect
and train** — θ is identical in both phases. Hence the hard invariant: exactly one
`optimizer.step()` per collect round.

**Returns per row type**:
- valid edge row: `G` from the actor harvest;
- failure row: `G = terminal_failure − step_penalty`, and **no critic target** — the
  action failed, but the state may still be provable by another tactic.

The critic is evaluated in a separate forward over unique `CriticSample` rows. Unknown
nodes do not enter that batch. Actor-only and critic-only updates therefore avoid empty
PyG batches while still contributing to one combined backward pass and optimizer step.

**Advantage** — over all rows jointly:

```
Â_raw = G − V(s).detach()
Â     = (Â_raw − mean) / (std + 1e-8)        # population std; batch of 1 → 0, not NaN
```

Joint normalization is deliberate: the failure rows' low returns and the success rows'
bootstrapped returns supply the contrast the near-constant PLN shaping lacks — without
variance in Â, the actor gradient is zero regardless of how many transitions were
collected. `detach()` on V keeps the critic out of the actor's gradient path.

**The four loss terms**:

```
L_actor   = − Σ_i m_i · (log π(τ_i|s_i) + w_arg·Σ_k log π(u_k|s_i,τ_i)) · Â_i  /  Σ_i m_i
L_critic  = MSE(V_θ(s), y(s))                # unique valid critic rows only
L_entropy = − mean H(π(·|s))                 # subtracted: rewards exploration
L_bc      = CE(tactic_logits, τ)             # success rows only, annealed weight
total     = L_actor + 0.5·L_critic − 0.01·L_entropy + w_bc(t)·L_bc
```

`L_actor` mechanically: a positive advantage increases the joint log-probability of the
executed tactic and its arguments; a negative advantage decreases it. The multiplicity
`m` restores the i.i.d. weighting that executor-side dedup removed. The BC anchor is
annealed from `bc_anneal_start` to `bc_anneal_end`; it lives in the loss, not the
reward, so it cannot contaminate the value target.

Then: `loss.backward()`, `clip_grad_norm_(grad_clip)`, **one** `optimizer.step()`.

## 5b. HTPS decoupled step: imitation + soft-critic regression

`train_step_htps_style` (pln_rl_training.py) runs after the on-policy step, using the
queues filled in section 4b, through a **separate optimizer** (`optimizer_htps`). The
one-step-per-collect invariant belongs to `compute_onpolicy_loss`'s score-function
estimator only; supervised regression on stored (input, label) pairs has no
importance-ratio to invalidate and may run many times per round.

`htps_steps_per_round` controls how many times the decoupled step fires per round.
Setting `htps_steps_per_round=0` (the default) disables it entirely — the queues never
fill, no second optimizer is created, and existing behavior is unchanged.

Imitation rows and critic rows use separate forwards through the same public encoder
interface. This keeps actor-only and critic-only batches well-defined for GraphSAGE and
GATv2.

```
L_tactic_imitation = CE(tactic_logits[imitation rows], tactic_id)
                   + arg_loss_weight · Σ_k forced_step_logp(arg_indices_k)
L_critic_soft      = MSE(V_θ(s), stored Q-or-status target)
loss               = L_tactic_imitation + w_critic_soft · L_critic_soft
```

`ArgumentSelector.forced_step` teacher-forces the argument pointer through the stored
`arg_indices` — the same path `evaluate_actions` uses for the on-policy step. Invalid/−1
argument positions contribute 0.

`optimizer_htps` must be a separate instance from the on-policy optimizer (the driver
asserts identity inequality). The two Adam moment buffers must not mix: the on-policy
estimator's gradient signal comes from return–baseline contrasts in the search; the
imitation estimator's signal comes from direct cross-entropy toward proved actions. Using
one optimizer would corrupt both sets of moments.

**Why the decoupled step does not invalidate the on-policy invariant**: the on-policy step
uses log-probs recomputed under parameters θ that are identical to the collect-time
parameters. The decoupled step runs on the *same* θ (it updates through `optimizer_htps`,
not `optimizer`). After the decoupled step, θ has moved — but the on-policy step has
already completed, so there is no stale-log-prob issue. The sequence per round is:
collect → on-policy step (one `optimizer.step()`) → N decoupled steps
(`optimizer_htps.step()` × N). The on-policy invariant is satisfied because the single
`optimizer.step()` runs before any decoupled updates.

## 6. The driver loop around it

`run_rl_training` (rl_training_driver.py) repeats sections 1–5 as rounds:

```
warm start (strict load of the supervised actor-critic best.pt)
pool ← LeanDojo proof states (parse_state), size-sorted, eval set held out
for round in 0..num_rounds:
    batch    ← sample from the curriculum window (easy prefix of the sorted pool)
    results  ← collect_round(reasoner, batch)        # sequential, fault-isolated
    metrics  ← train_step_onpolicy(..., bc_weight=anneal(round))   # ONE optimizer.step()
    [HTPS mode] extract_minimal_hypertree + extract_critic_samples → fill queues
    [HTPS mode] train_step_htps_style × htps_steps_per_round
    window widens when the recent solve rate ≥ threshold
    last.pt every checkpoint_every rounds (model+optimizer+RNG+round → resumable)
    every eval_every rounds: greedy proof rate on the fixed eval pool → best.pt on improvement
```

**Deadline and timeout** — `prove` receives `deadline = time.monotonic() + timeout_s`
and checks it between expansions (legacy) or between simulation batches (HTPS). The
partial graph is returned cleanly with all experience gathered so far. A hard
`asyncio.wait_for(timeout=timeout_s * 1.25)` remains as a backstop for a hang inside a
single Lean call, where the event loop is blocked and the deadline check cannot fire;
that case loses the search's experience.

**Checkpoint format** — `save_checkpoint` writes the version-2 model manifest, both
optimizer states, the optimizer-step anneal count, fixed search settings, and both replay
queues. Critic rows serialize `(node_id, goal, hypotheses, target, source)`. Resume
requires the validity-aware schema and identical search semantics; pre-validity optimizer
and replay state must start a fresh RL run.

Two data-side choices matter for why this learns at all:

- **Rollout roots are interior proof states, not theorem statements.** The terminal reward
  fires only when a rollout reaches QED, and the probability of that scales like `p^k` in
  the distance-to-closure `k`. Dataset proof states sit 1–3 tactics from closure; whole
  statements sit 10–30. Starting near the terminal boundary makes `terminal_success`
  actually occur, and the critic then carries that signal one step outward per curriculum
  widening (value bootstrapping = dynamic programming from the base case inward).
- **Model selection is greedy proof rate on a fixed held-out pool**, never training
  return: return is measured on a sampled policy over a shifting curriculum window, so it
  is neither comparable across rounds nor robust to entropy collapse.

## 7. Invariants checklist

| Invariant | Enforced by | Broken consequence |
|---|---|---|
| one `optimizer.step()` per collect | loop structure of `train_step_onpolicy` + driver | recomputed log-probs off-policy; biased gradient |
| `optimizer_htps` is a separate instance | driver assertion at construction | Adam moments from on-policy and imitation gradients corrupt each other |
| on-policy step fires before decoupled steps | driver round ordering (collect → on-policy → decoupled) | decoupled update shifts θ before on-policy recompute; log-probs stale |
| featurizer identity collect↔train | `reasoner.dag_featurize_data` passed to the trainer | stored arg indices point at wrong DAG nodes |
| vocabs from `prepared_root` only | `_load_vocabs`; never `build_vocab` on rollout states | warm-started embeddings mean the wrong tokens |
| Φ(terminal) = 0 | `potential` returns 0 for SOLVED/DEAD and for `stv is None` | shaping stops telescoping; terminals get biased by `γ^T Φ(s_T)` |
| critic sees unique valid node rows only | `extract_critic_samples` + separate critic batch | failed actions or unknown states poison state values |
| dedup with multiplicity | `EdgeAction.multiplicity` weighting | high-probability actions under-weighted |
| value target from validity-aware backup, not PLN | `compute_backups` emits hard/root/soft-Q targets with provenance | unreliable PLN trains the critic |
| per-node pending stash | `_pending: {node_id → _PendingNode}` in `RLHybridReasoner` | batched proposals misattribute one leaf's failures to another goal |
| edge-id uniqueness per search | sequential collect; `RLSearchResult` resets stashes | edge-id collisions corrupt the `edge_actions` join table |
| virtual loss released by backup | `stats.virtual_loss = max(0, vl - 1)` in `_backup_simulation` | in-flight selections never clear; future PUCT scores permanently inflated |
| prior_prob stamped once at `add_edge` | `EdgeVisitStats(prior_prob=tactic.probability)` | `puct_score` reads a stale or recomputed prior; selection not reproducible |
| imitation filter excludes PLN-fallback edges | `extract_minimal_hypertree` restricted to `edge_actions` keys | fabricated QED edges contaminate the imitation target distribution |

## 8. File map

| Step | Files |
|---|---|
| model + sampling + recompute | `atp_lean_gnn/actor_critic.py`, `atp_lean_gnn/argument_selector.py` |
| PUCT scoring + search-mode validation | `hybrid_reasoner/selection_policy.py` |
| search + graph + visit stats | `hybrid_reasoner/joint_inference.py`, `hybrid_reasoner/hypergraph.py` |
| RL search overrides + batched stash | `atp_lean_gnn/rl_reasoner.py` |
| reward + shaping | `atp_lean_gnn/pln_reward.py` |
| backup + transitions + critic/imitation harvest | `atp_lean_gnn/search_harvest.py` |
| on-policy loss + decoupled HTPS step | `atp_lean_gnn/pln_rl_training.py`, `atp_lean_gnn/actor_critic_loss.py` |
| driver, curriculum, eval, checkpointing | `atp_lean_gnn/rl_training_driver.py`, `scripts/rl_train.py`, `configs/rl_actor_critic.json` |
| tests | `tests/test_actor_critic.py`, `tests/test_pln_reward.py`, `tests/test_pln_rl_training.py`, `tests/test_rl_reasoner.py`, `tests/test_rl_training_driver.py`, `tests/test_search_harvest.py`, `tests/test_selection_policy.py`, `tests/test_visit_stats.py` |
