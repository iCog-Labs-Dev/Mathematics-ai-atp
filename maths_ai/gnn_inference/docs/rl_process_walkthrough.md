# The RL process, end to end: sampling → search → storage → gradient step

This document walks one training round through every mechanism, in execution order, with
the exact tensors and file locations. It describes the system as implemented in
`atp_lean_gnn/` (`actor_critic.py`, `rl_reasoner.py`, `pln_reward.py`,
`search_harvest.py`, `pln_rl_training.py`, `rl_training_driver.py`) and
`hybrid_reasoner/` (`hypergraph.py`, `joint_inference.py`).

## Symbol table

| Symbol | Meaning | Where it lives |
|---|---|---|
| `s` | a proof state: goal expression + local hypotheses | `Goal(expression, hypotheses)` |
| `τ` | a tactic family (e.g. `intro`, `exact`) | index into the tactic vocab |
| `u_k` | the k-th pointer-selected tactic argument (a DAG node) | index into the state DAG's node list |
| `a = (τ, u_1…u_K)` | one full action | `EdgeAction(tactic_id, arg_indices, multiplicity)` |
| `π(a\|s)` | the policy: tactic head × autoregressive pointer | `ActorCriticWithArgsClassifier` |
| `V_theta(s)` | the critic's current prediction; its target meaning is the probability that the configured search proves `s` within its tactic, depth, node, and time budgets | `CriticHead` on the state embedding |
| `y(s)` | a valid numeric critic target for `s`, emitted only when search evidence is conclusive | `CriticSample.target` |
| `e` | one tactic hyperedge from a parent state to every subgoal that tactic produced | `ProofHyperedge` |
| `E(e)` | the validity-aware outcome of edge `e`: known `0`, known `1`, or unknown | `BackupTables.edge_outcomes` |
| `Φ(s)` | shaping potential = PLN strength σ of `s` (0 at terminals) | `pln_reward.potential` |
| `r` | per-edge shaped reward `r_term + Σ_j(γΦ(child_j) − Φ(parent))` | `pln_reward.edge_shaped_reward` |
| `G(e)` | per-edge return `r(e) + γ·successor_value(e)`, present only when `E(e)` is known | `ActorTransition.return_` |
| `Â` | advantage `G(e) − V_theta(s)`, batch-normalized over actor rows | `pln_rl_training.compute_onpolicy_loss` |
| `m` | multiplicity: how many of the k i.i.d. draws produced this action | `EdgeAction.multiplicity` |
| `γ` | discount (default 0.99) | `RewardConfig.gamma` |

## 0. The model: one encoder, four heads

`ActorCriticWithArgsClassifier` runs one registered state-graph encoder **once** per
state and feeds four heads from its outputs (`encode`, actor_critic.py). The checkpoint
manifest selects `graphsage` or `gatv2`; the actor, critic, pointer, search, and loss paths
do not branch on that choice.

1. **Node embeddings** `[N, H]` — one vector per node of the state DAG (the proof state
   parsed into a hash-consed graph by `proof_state_to_dag`).
2. **State embedding** `[B, H]` — GraphSAGE reads the `State` root. GATv2 can use the
   same root or fuse it with state-conditioned attention and global mean/max summaries.
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

## 1. Action sampling: `model.act` (i.i.d.) vs. `greedy`

`act(data, id_to_tactic, greedy)` (actor_critic.py) produces one full action per call:

- **Tactic**: build `Categorical(logits=tactic_logits)`; **training** draws
  `tactic_dist.sample()` — an i.i.d. sample from π(τ|s); **evaluation** takes
  `argmax`. Either way `tactic_logp = log π(τ|s)` and the distribution entropy are
  recorded.
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

Greedy mode exists solely for evaluation: `evaluate_proof_rate` measures the deterministic
policy — the thing that will be deployed — not the exploration distribution around it.

## 2. Search: the AND-OR hypergraph as the environment loop

`RLHybridReasoner.prove` (rl_reasoner.py) inherits the base best-first loop
(`HybridReasoner.prove`, joint_inference.py) and swaps in the sampled policy at exactly
two seams. One search proceeds:

1. **Root**: `ProofHypergraph(Goal(...))` creates node 0, `status=OPEN`.
2. **Frontier pop**: `graph.frontier()` returns OPEN nodes sorted by `combined_rank`
   descending; the loop expands the top one. `combined_rank` starts as
   `gnn_probability × STV.score` and is continuously revised by backprop (step 6).
3. **Proposal** (seam 1, `predict_next_tactic`): featurize the node's sanitized goal —
   `Goal → goal_to_state → proof_state_to_dag → dag_to_pyg` under the **fixed prepared
   vocab** — then draw k i.i.d. actions from `model.act` under `torch.no_grad()` (no
   autograd graph is held across the search; see section 3). Each draw is **decoded**:
   `tactic_id → name` via the inverted tactic vocab; each sampled pointer index → the DAG
   node at that offset → a Lean argument string via `_resolve_local_node_name` (a `Hyp`
   node renders as its bound name, e.g. `h`). Identical decoded actions are deduplicated
   for the executor but counted (`multiplicity`).
4. **Execution**: for each unique candidate, `PantographExecutor.apply` sends
   `"{tactic} {args}"` to the live Lean server.
   - **Rejected** → no edge is created; the candidate stays in the pending stash and
     becomes a failure record (section 3).
   - **Accepted, no subgoals** → QED: `add_edge(node, tactic, [])`, immediately SOLVED.
   - **Accepted, subgoals** → PLN scores each subgoal concurrently
     (`rank_subgoals` → `evaluate_async`, the petta subprocess dispatched off the event
     loop), and every `(Goal, STV)` pair becomes a child of one **AND-edge**:
     all Lean-returned obligations must eventually close for this tactic to count.
5. **Cycle guard**: `add_edge` fingerprints every child (`expression::sorted(hyps)`)
   against the node's ancestors; a child identical to an ancestor is force-marked DEAD —
   otherwise a tactic that rewrites a goal into an earlier one loops forever.
6. **Propagation**: every mutation triggers `_propagate` — a bottom-up fixpoint walk that
   re-derives the AND-OR statuses (node SOLVED ⇐ any edge SOLVED; edge SOLVED ⇐ all
   children SOLVED or no children; edge DEAD ⇐ any child DEAD; node DEAD ⇐ exhausted and
   all edges DEAD) and re-ranks every ancestor (`combined_rank = max(local_score,
   max_edge tactic_prob × min_child rank)` — OR-max over edges, pessimistic AND-min over
   children). This is what makes the search *best-first in provability*: a PLN score
   assigned to a new leaf immediately re-orders the global frontier.
7. **Termination**: the graph records `ROOT_SOLVED`, `ROOT_DEAD`, `MAX_NODES`,
   `FRONTIER_EMPTY`, or `EXTERNAL_ABORT`. A clean `MAX_NODES` result is a root-level
   finite-budget failure only; untouched internal nodes remain unknown.

So during the search the AND-OR graph plays two roles at once: it is the **control
structure** (frontier ordering decides where the policy is asked next) and it is
accumulating the **experience** the trainer will read (its topology and terminal statuses
determine every reward and value target).

## 3. Storage: what survives the search, and why only integers

The search-time forward passes ran under `no_grad` — nothing differentiable survives, by
design: holding autograd graphs for every sampled action across a multi-node async search
would pin the memory of every intermediate activation for minutes. What is stored instead
(rl_reasoner.py):

- **Pending stash** — during a node's expansion, every decoded draw sits in
  `self._pending: {fingerprint → EdgeAction}` where the fingerprint is
  `(goal_expr, hyps, tactic_name, args)` and `EdgeAction` is three integers-worth of data:
  `tactic_id`, `arg_indices` (the raw sampled pointer positions, including out-of-range
  ones — the recompute must evaluate what was actually sampled), `multiplicity`.
- **Edge join** (seam 2, `_link`) — when the base `_expand` links an accepted tactic into
  the graph, the subclass moves the stash entry to `edge_actions[edge.id]`. This is the
  join key between graph structure and policy actions.
- **Failure flush** — entries still pending after a normal tactic-execution expansion were
  rejected by Lean: they move to `failure_actions` as `FailureRecord(goal, action)`. Entries
  discarded because the goal failed to elaborate, the backend failed, or the search was
  cancelled are not failures because Lean never observed those actions.
- **Per-search container** — `prove` returns
  `RLSearchResult(graph, edge_actions, failure_actions)` and resets its stashes, because
  `edge.id` is unique only within one `ProofHypergraph`; collect therefore runs theorems
  **sequentially** on one reasoner (`collect_round`, rl_training_driver.py), with
  per-theorem timeout/exception isolation so one dead search cannot kill the round.

## 4. Harvest: turning the finished graph into training targets

`train_step_onpolicy` (pln_rl_training.py) first harvests each result's graph:

**Validity-aware backup** (`search_harvest.compute_backups`) — a memoized AND-OR
recursion that carries both a numeric value and an evidence marker. `KNOWN` means the
search established the outcome. `UNKNOWN` means that it did not. Unknown is not converted
to `0.0`.

```text
SOLVED node                                -> KNOWN(1.0)
locally exhausted node with all edges 0    -> KNOWN(0.0)
OPEN, globally truncated, unelaborated     -> UNKNOWN
AND edge: any known failed child           -> KNOWN(0.0)
AND edge: every child known solved         -> KNOWN(1.0)
AND edge: otherwise                        -> UNKNOWN
OR node: any edge known 1                  -> KNOWN(1.0)
OR node: locally exhausted and all edges 0 -> KNOWN(0.0)
OR node: otherwise                         -> UNKNOWN
```

`CANDIDATES_EXHAUSTED` and `NO_CANDIDATES` are local failure boundaries. `DEPTH_LIMIT`
and `CYCLE` can fail the incoming tactic path, but do not receive standalone critic rows,
because those states were not searched with a fresh root budget. `ELABORATION_ERROR` and
`INFRASTRUCTURE_FAILURE` are unknown for both edge backup and critic supervision.

The critic target is a node property, so `extract_critic_samples` emits at most one row
per node. The actor target is an edge property, so `extract_actor_transitions` emits one
row per selected edge whose outcome is known. A clean `MAX_NODES` episode adds exactly one
root target `0.0` when no elaboration or infrastructure failure occurred.

**Per-edge reward** (`pln_reward.edge_shaped_reward`) — where PLN *does* enter, in the
only provably-safe slot:

```
r = r_term + Σ_j ( γ·Φ(child_j) − Φ(parent) )
r_term = +terminal_success − step_penalty   if the edge is QED (no children, SOLVED)
         terminal_failure − step_penalty    if the edge is DEAD
         −step_penalty                      otherwise
Φ(s)   = PLN strength σ of s, and 0 for SOLVED/DEAD nodes
```

The shaping term is potential-based (Ng–Harada–Russell): because Φ is a function of state
alone and Φ(terminal)=0, the shaping telescopes along every path to
`γ^T·0 − Φ(s_0)` — a constant offset per trajectory — so it can bias *learning speed*
toward states PLN likes but cannot change which policy is optimal. That is the entire
reason a near-constant, unreliable PLN is safe to include.

**Actor transitions** (`extract_actor_transitions`) — one `ActorTransition` per selected
policy-produced edge whose successor evidence is known. Unknown edges are omitted rather
than assigned an invented return:

```
ActorTransition(node_id, goal, tactic, reward=r,
                successor_value = E(e),
                return_ = r + γ·successor_value,
                edge_id)
```

`return_` is a one-step bootstrapped target. The critic uses a separate deduplicated
`CriticSample(node_id, goal, target)` batch, so three accepted tactics from one state do
not duplicate the state label three times.

## 5. The gradient step: recompute, then one update

`compute_onpolicy_loss` (pln_rl_training.py) assembles separate actor and critic batches
from all of the round's results. Actor rows are accepted edges with known outcomes plus
executor-rejected actions. Critic rows are unique states with known node targets. Edge ids
are re-keyed per graph to avoid collisions.

**Log-prob recompute** — `model.evaluate_actions(batch, tactic_ids, arg_indices)` re-runs
the encoder **with gradient** under the current parameters and evaluates the log-probs of
the *stored* actions: `tactic_logp = log_softmax(logits)[τ]`, and the pointer is
teacher-forced through `forced_step` — at step k the stored index `u_k`'s log-prob is
gathered and `u_k`'s embedding (not a fresh sample's) feeds step k+1, so the recomputed
trajectory is exactly the executed one. Invalid/-1 indices contribute log-prob 0 and a
zero embedding. This is the standard A2C evaluate-actions pattern, and it is exactly
on-policy **only because no optimizer step ran between collect and train** — θ is
identical in both phases. Hence the hard invariant: exactly one `optimizer.step()` per
collect round (a second step on the same data would need PPO's ratio clip). The same
argument requires the featurizer identity: collect and train featurize through the *same*
`make_dag_featurizer` instance, so the stored pointer indices land on the same DAG nodes.

**Returns per row type**:

- accepted edge row: `G(e)` from the validity-aware harvest;
- rejected-action row: `G = terminal_failure − step_penalty`, and **no critic target** — the
  *action* failed, but the *state* may be provable by another tactic, so a failed action
  says nothing about `V_theta(s)`.

**Advantage** — over all rows jointly:

```
Â_raw = G(e) − V_theta(s).detach()
Â     = (Â_raw − mean) / (std + 1e-8)        # population std; batch of 1 → 0, not NaN
```

Joint normalization is deliberate: the failure rows' low returns and the accepted-edge rows'
bootstrapped returns supply the *contrast* that the near-constant PLN shaping lacks —
without variance in Â, the actor gradient is zero regardless of how many transitions were
collected. `detach()` on V keeps the critic out of the actor's gradient path (the baseline
must not chase the policy).

**The four loss terms**:

```
L_actor   = − Σ_i m_i · (log π(τ_i|s_i) + w_arg·Σ_k log π(u_k|s_i,τ_i)) · Â_i  /  Σ_i m_i
L_critic  = MSE(V_theta(s), y(s))            # unique valid critic rows only
L_entropy = − mean H(π(·|s))                 # subtracted: rewards exploration
L_bc      = CE(tactic_logits, τ)             # accepted edge rows only, annealed weight
total     = L_actor + 0.5·L_critic − 0.01·L_entropy + w_bc(t)·L_bc
```

Reading `L_actor` mechanically: a positive advantage *increases* the joint log-probability
of the executed tactic **and** its arguments (the same `Â` weights both
levels, `w_arg` balancing their scales); a negative advantage decreases it. The
multiplicity `m` restores the i.i.d. weighting that executor-side dedup removed: an action
drawn 3 times must contribute its gradient term 3 times or the estimator under-weights
high-probability actions. The BC anchor is a supervised cross-entropy toward the
search-executed tactic on accepted edge rows (never failures — that would clone rejections),
annealed from `bc_anneal_start` to `bc_anneal_end` by `bc_weight_at_round`; it is a dense,
low-variance gradient that holds the policy near supervised competence while terminal
rewards are still rare, and it lives in the *loss*, not the reward, so it cannot
contaminate the value target.

Then: `loss.backward()`, `clip_grad_norm_(grad_clip)`, **one** `optimizer.step()`.

## 6. The driver loop around it

`run_rl_training` (rl_training_driver.py) repeats sections 1–5 as rounds:

```
warm start (manifest-driven reconstruction + strict actor-critic state load)
pool ← LeanDojo proof states (parse_state), size-sorted, eval set held out
for round in 0..num_rounds:
    batch    ← sample from the curriculum window (easy prefix of the sorted pool)
    results  ← collect_round(reasoner, batch)          # sequential, fault-isolated
    validate total node/edge counts against the RL update budget
    metrics  ← train_step_onpolicy(..., bc_weight=anneal(round))   # ONE step
    window widens when the recent solve rate ≥ threshold
    last.pt every checkpoint_every rounds (model+optimizer+RNG+round → resumable)
    every eval_every rounds: greedy proof rate on the fixed eval pool → best.pt on improvement
```

Two data-side choices matter for why this learns at all:

- **Rollout roots are interior proof states, not theorem statements.** The terminal reward
  fires only when a rollout reaches QED, and the probability of that scales like `p^k` in
  the distance-to-closure `k`. Dataset proof states sit 1–3 tactics from closure;
  whole statements sit 10–30. Starting near the terminal boundary makes `terminal_success`
  actually occur, and the critic then carries that signal one step outward per curriculum
  widening (value bootstrapping = dynamic programming from the base case inward).
- **Model selection is greedy proof rate on a fixed held-out pool**, never training
  return: return is measured on a sampled policy over a *shifting* curriculum window, so
  it is neither comparable across rounds nor robust to entropy collapse.

## 7. Invariants checklist

| Invariant | Enforced by | Broken consequence |
|---|---|---|
| one optimizer step per collect | loop structure of `train_step_onpolicy` + driver | recomputed log-probs off-policy; biased gradient |
| featurizer identity collect↔train | `reasoner.dag_featurize_data` passed to the trainer | stored arg indices point at wrong DAG nodes |
| vocabs from `prepared_root` only | `_load_vocabs`; never `build_vocab` on rollout states | warm-started embeddings mean the wrong tokens |
| Φ(terminal) = 0 | `potential` returns 0 for SOLVED/DEAD | shaping stops telescoping; terminals get biased by `γ^T Φ(s_T)` |
| critic sees unique valid node rows | `extract_critic_samples` and the separate critic batch | failed actions and duplicate edge labels poison state values |
| dedup with multiplicity | `EdgeAction.multiplicity` weighting | high-probability actions under-weighted |
| value target from backup, not PLN | `compute_backups` uses closure provenance | unreliable PLN and unknown evidence train the critic |
| sequential collect per reasoner | `collect_round` loop | edge-id collisions across concurrent stashes |

## 8. File map

| Step | Files |
|---|---|
| model + sampling + recompute | `atp_lean_gnn/actor_critic.py`, `atp_lean_gnn/argument_selector.py` |
| search + graph | `hybrid_reasoner/joint_inference.py`, `hybrid_reasoner/hypergraph.py` |
| RL search overrides + stash | `atp_lean_gnn/rl_reasoner.py` |
| reward + shaping | `atp_lean_gnn/pln_reward.py` |
| backup + actor/critic harvest | `atp_lean_gnn/search_harvest.py` |
| loss + train step | `atp_lean_gnn/pln_rl_training.py`, `atp_lean_gnn/actor_critic_loss.py` |
| driver, curriculum, eval | `atp_lean_gnn/rl_training_driver.py`, `scripts/rl_train.py`, `configs/rl_actor_critic.json` |
| tests | `tests/test_actor_critic.py`, `tests/test_pln_reward.py`, `tests/test_pln_rl_training.py`, `tests/test_rl_reasoner.py`, `tests/test_rl_training_driver.py` |
