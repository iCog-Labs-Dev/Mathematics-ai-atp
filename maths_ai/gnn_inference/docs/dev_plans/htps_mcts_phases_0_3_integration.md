# HTPS-style MCTS integration, Phases 0–3: implementation plan

## Context

The RL driver currently runs **one best-first pass per theorem**: pop the highest-ranked
OPEN node, expand it once (sample k tactics, execute in Lean, PLN-rank subgoals), mark it
exhausted, repeat until the root resolves or the node budget runs out. Two consequences
limit learning:

1. The critic's regression target (`backup_values`) is effectively binary — leaves are
   1.0 (SOLVED) / 0.0 (DEAD or unexpanded), and products/maxes of those stay in {0, 1}.
   A node the search visited many times without resolving contributes the same 0.0 as a
   node never visited at all.
2. The actor's only dense supervised anchor is the BC term on search-executed tactics.
   Proofs discovered inside a *failed* search's subgraph produce value targets but no
   direct imitation signal.

This plan ports four HTPS mechanisms onto the existing system to fix both: per-edge visit
statistics, repeated PUCT-guided simulation, a soft visit-ratio critic target, and
minimal-hypertree tactic imitation. The design document
`docs/htps_mcts_integration_plan.md` proposed this integration; a code-verification pass
found ten places where it mis-describes the implemented system (Section 8 below lists
them). This plan supersedes it on those points.

**Decisions locked with the user (2026-08-03):**

| Decision | Choice |
|---|---|
| Training objective | **Augment** (design doc Option B): on-policy `L_actor + L_critic + L_entropy + L_bc` untouched, one optimizer step per collect round. New imitation/soft-critic losses train through a separate decoupled step with a separate optimizer. |
| Tactic mining scope | **All solved nodes** by default — minimal hypertrees are mined under every SOLVED node, including in searches whose root failed. `mine_all_solved_nodes: false` restricts to root-only for the ablation. |
| Scope | Phases 0–3 only. Search-hyperparameter sampling (`SearchHParams`), sweep tooling, and async prover infrastructure are **out of scope**. No `temperature` parameter is added to `act()`. |
| Inference | Training-side only. `evaluate_proof_rate` stays greedy single-pass; no `prove_mcts` mode. |

**Guiding property (regression guarantee):** every new config field defaults to a value
that reduces the system to current behavior — `num_simulations=1` runs the legacy prove
loop verbatim, `htps_steps_per_round=0` disables the decoupled step, `visit_threshold`
high enough that no soft target fires at K=1.

## Symbol table

| Symbol | Meaning |
|---|---|
| `g` | a goal node in the proof hypergraph (`ProofNode`: one Lean proof state) |
| `t` | a tactic edge under `g` (`ProofHyperedge`: one accepted tactic application) |
| `N(g,t)` | count of completed simulations that traversed edge `t` |
| `W(g,t)` | sum of backed-up values over those traversals |
| `Q(g,t)` | `W/N` mean action value (first-play-urgency 0.5 when `N=0`) |
| `P(g,t)` | the PUCT prior — the policy probability stored at edge creation (`edge.tactic.probability`) |
| `V(s)` | the critic head's value estimate for state `s` |
| `K` | `num_simulations` — simulations per theorem attempt |
| `B` | `sim_batch_size` — simulations selected before one batched expansion |

---

## 1. Design decisions and alternatives considered

### 1.1 Batched expansion vs. the pending stash

`RLHybridReasoner` stores each sampled action in `self._pending` keyed by
`(goal, tactic, args)` fingerprints, and **flushes the whole stash to failure records at
the start of the next `predict_next_tactic` call** (rl_reasoner.py:118). That contract
assumes strict alternation: one node's proposal, then all its `_link`s, then the next
node's proposal. Batching proposals across several leaves violates it — leaf A's pending
entries would be flushed as failures the moment leaf B proposes.

| Option | Pros | Cons |
|---|---|---|
| **A. Re-key the stash per node (chosen)** — `self._pending: dict[goal_key, dict[fingerprint, EdgeAction]]`; flush happens per node when that node's expansion completes (its `mark_node_exhausted`), not globally on the next proposal. | Fixes the shared component for the expanded scope; GNN proposal can batch across the simulation batch's leaves (one multi-graph forward instead of `k × leaves` single-graph forwards); flush timing becomes explicit instead of positional. | Touches the failure-record path, which is most of the early-training signal — needs its own regression test. |
| B. Keep the stash as-is; expand leaves strictly sequentially within a simulation batch. | No stash change. | Forfeits the batched GNN forward that motivates `sim_batch_size`; the fragile positional contract stays latent and breaks the next time anyone reorders expansion. |

Chosen: **A**. The stash's flush-on-next-proposal design is a defect exposed by the new
scope, so the shared code is fixed rather than worked around. Lean execution stays
sequential per leaf regardless (one Pantograph server), so batching applies to the GNN
proposal step only.

### 1.2 AND-combine rule in `backup_simulation`

Two conventions already coexist: `hypergraph._edge_value` uses `min(child.combined_rank)`
for frontier ranking; `search_harvest._and_combine` defaults to `product` for the critic's
value target.

| Option | Pros | Cons |
|---|---|---|
| **Product (chosen)** | Matches the value-target convention the critic already regresses to (`HarvestConfig.and_combine="product"`); probabilistic reading — the value of a tactic is the joint provability of all its subgoals. | Diverges from `_edge_value`'s `min`. |
| Min | Matches `_edge_value`. | `_edge_value` is a *ranking heuristic* for frontier ordering, not a value estimate; mixing conventions inside the training-signal path is the worse inconsistency. |

Chosen: **product**, reusing `HarvestConfig.and_combine` so the choice stays configurable
in one place. `_edge_value` is left untouched — ranking and value backup are different
jobs, and this plan does not change frontier ranking semantics.

### 1.3 Tactic-only vs. tactic + argument imitation

HTPS mines `(goal, tactic)` pairs; in this system an action is `(τ, u_1…u_K)` — tactic
plus pointer-selected arguments. Every policy-produced edge already carries its full
`EdgeAction` (tactic_id + raw sampled `arg_indices`) in `result.edge_actions`.

| Option | Pros | Cons |
|---|---|---|
| **Tactic + arguments (chosen)** | Arguments are half the action space; a mined `apply` without its premise pointer teaches nothing for arity>0 tactics. The teacher-forced pointer path (`forced_step`, used by `evaluate_actions`) already exists and re-featurization is deterministic (same vocab, same parser), so stored `arg_indices` land on the same DAG nodes. | Slightly larger sample record; edges without an `EdgeAction` (PLN-fallback pseudo-edges) cannot contribute argument targets. |
| Tactic-only | Simplest; matches HTPS literally. | Systematically under-trains the pointer head relative to the tactic head. |

Chosen: **tactic + arguments** where the `EdgeAction` exists; PLN-fallback pseudo-edges
are skipped entirely (they carry no policy action and their "tactic" is synthetic).
Argument positions use the same `-1 = ignore` masking convention `compute_onpolicy_loss`
already uses.

### 1.4 Checkpoint contents

| Option | Pros | Cons |
|---|---|---|
| **Persist `optimizer_htps` state + both replay queues in `last.pt` (chosen)** | Resume reproduces the training trajectory; queue samples are plain tuples of strings/ints/floats and small relative to model weights. | Larger checkpoint; queue schema becomes part of the resume contract. |
| Model/optimizers only, queues rebuilt from scratch | Smaller checkpoint. | A resumed run silently trains the decoupled step on an empty queue for its first rounds — a behavioral discontinuity invisible in the config. |

Chosen: persist both. `save_checkpoint` gains `optimizer_htps_state_dict`,
`tactic_queue`, `critic_queue` keys; the resume path restores them with
`.get(..., default)` so pre-existing checkpoints still load.

### 1.5 Where the simulation loop lives

| Option | Pros | Cons |
|---|---|---|
| **Base `HybridReasoner` in `joint_inference.py` (chosen)** | The multi-simulation loop is search capability, not RL-specific; the RL subclass keeps overriding the same seams (`predict_next_tactic`, `_link`) and adds one new seam (leaf evaluation). Matches the existing inheritance pattern. | Base class grows. |
| RL-subclass-only loop | Base untouched. | Forks the search loop into two implementations; the base best-first pass and the MCTS loop drift apart; hides the capability from non-RL callers. |

Chosen: base class, with the critic-based leaf evaluator supplied through a seam
(Section 3.2), since the base class has no critic.

---

## 2. Phase 0 — `EdgeVisitStats`: tracked, unused

**Files:** `maths_ai/hybrid_reasoner/hypergraph.py`;
new test `maths_ai/gnn_inference/tests/test_visit_stats.py`.

```python
@dataclass
class EdgeVisitStats:
    N: int = 0                 # completed-simulation traversal count
    W: float = 0.0             # total backed-up value across those traversals
    virtual_loss: int = 0      # simulations currently in flight through this edge

    @property
    def Q(self) -> float:
        C = self.N + self.virtual_loss
        if C == 0:
            return 0.5         # first-play-urgency (HTPS Appendix A.2)
        return self.W / max(1, C)
```

`ProofHyperedge` gains `visit_stats: EdgeVisitStats = field(default_factory=EdgeVisitStats)`.
Nothing reads it in this phase.

**Exit criterion:** full existing suite green; new unit tests for `Q` (0.5 at `N=0`,
`W/N` otherwise, virtual-loss denominator effect).

---

## 3. Phase 1 — multi-simulation search with PUCT and virtual loss

**Files:** new `maths_ai/hybrid_reasoner/selection_policy.py`;
`maths_ai/hybrid_reasoner/joint_inference.py`; `maths_ai/hybrid_reasoner/hypergraph.py`;
`maths_ai/gnn_inference/atp_lean_gnn/rl_reasoner.py`;
`maths_ai/gnn_inference/atp_lean_gnn/rl_training_driver.py`;
`maths_ai/gnn_inference/configs/rl_actor_critic.json`;
tests per Section 7.

### 3.1 Selection policy

`selection_policy.py`:

```python
def puct_score(prior: float, stats: EdgeVisitStats, total_node_visits: int, c: float) -> float:
    return stats.Q + c * prior * math.sqrt(total_node_visits) / (1 + stats.N + stats.virtual_loss)
```

`prior` is `edge.tactic.probability` — the policy probability recorded when the edge was
created. PUCT only; the RP variant from the design doc is deferred (config field
`selection_policy` is added now with the single legal value `"puct"` so adding RP later
is a config change, not a schema change).

### 3.2 Search loop in `HybridReasoner`

Constructor gains `num_simulations: int = 1` and `sim_batch_size: int = 4` (constructor
attributes, matching how `max_depth`/`max_nodes` are already threaded). `prove` branches:

- `num_simulations == 1`: the **existing loop, verbatim** — not a re-implementation that
  happens to be equivalent. This is the regression guarantee.
- `num_simulations > 1`: the new loop:

```python
while root not in (SOLVED, DEAD) and simulations_done < K and within_deadline():
    trees = []
    for _ in range(min(B, K - simulations_done)):
        tree = self._select_partial_hypertree(graph)   # PUCT descent; +1 virtual_loss per edge on the path
        if tree is None: break                          # no selectable path (all exhausted)
        trees.append(tree)
    leaves = unexpanded_leaves(trees)                   # deduplicated
    await self._expand_leaves(graph, leaves)            # batched proposal, sequential Lean exec
    for tree in trees:
        self._backup_simulation(graph, tree)            # N/W updates, clears virtual_loss
    simulations_done += len(trees)
```

**`_select_partial_hypertree(graph)`** — descends from the root. At each visited node with
outgoing non-DEAD edges, pick the edge maximizing `puct_score` and descend into **every**
child of that edge (a simulation must reach a full set of leaves consistent with one
candidate proof — the AND semantics), incrementing `virtual_loss` on each traversed edge.
Recursion stops at SOLVED/DEAD nodes and at unexpanded leaves. Returns the tree as a list
of `(node_id, edge_id)` pairs plus the leaf set.

**`_expand_leaves(graph, leaves)`** — the expansion stage of today's `_expand`, factored
into two steps: (1) one **batched** proposal call across all leaves (new seam
`predict_next_tactics_batch`, default implementation loops `predict_next_tactic` so the
base class and non-RL callers are unaffected); (2) per leaf, sequentially, the existing
execute-and-link logic (Lean apply per candidate, PLN ranking, cycle guard,
`mark_node_exhausted`). `_expand` itself remains and is what `num_simulations=1` calls —
untouched.

**`_backup_simulation(graph, tree)`** — walks the simulated tree bottom-up. Node values:

```
value(SOLVED)             = 1.0
value(DEAD)               = 0.0
value(unexpanded leaf)    = self._leaf_value(node)          # new seam
value(interior, edge t)   = AND-combine(children values)    # product, per Decision 1.2
```

Increments `N += 1`, `W += value` and sets `virtual_loss` back down on every edge of the
tree. Status propagation is *not* this walk's job — `add_edge` already ran `_propagate`
during expansion; the two walks stay separate (statistics vs. statuses).

**`_leaf_value` seam** — base implementation returns `0.5` (uninformed). The RL subclass
overrides it with the critic: featurize the goal, `model.encode` under `no_grad`, return
`values.item()`. This mirrors HTPS's `v_T(g) = c_θ(g)` while keeping the base class
critic-free.

### 3.3 Stash re-keying in `RLHybridReasoner`

Per Decision 1.1: `self._pending` becomes `dict[goal_key, dict[fingerprint, EdgeAction]]`.
`predict_next_tactic` (and the new batch variant) writes into its own goal's sub-dict and
no longer flushes globally; the per-node flush to `failure_actions` moves to the point
where that node's expansion completes (end of its sequential execution step in
`_expand_leaves` / end of `_expand`). `_link` pops from the sub-dict for the edge's source
goal. `_flush_pending` takes the goal key. End-of-search flush of all remaining sub-dicts
is unchanged in effect.

### 3.4 Timeout and partial-result survival

Today `collect_round` wraps `prove` in `asyncio.wait_for(timeout_s)`; cancellation
**loses the entire search** — a real defect for multi-simulation runs where a timeout
mid-simulation-500 would discard 499 simulations of experience. Fix in the driver +
reasoner, not by tuning:

- `prove` gains `deadline: float | None` (monotonic timestamp). The simulation loop checks
  it between simulation batches (`within_deadline()` above) and returns the partial graph
  cleanly when exceeded. The K=1 legacy loop also checks it between frontier pops.
- `collect_round` computes `deadline = monotonic() + timeout_s` and passes it down;
  `asyncio.wait_for` stays as a backstop at `timeout_s * 1.25` for hangs inside a single
  Lean call.

### 3.5 Config additions (`RLTrainingConfig` + `rl_actor_critic.json`)

> Amended by `docs/dev_plans/legacy_selection_reconciliation.md`: the mode gate is
> `selection_policy`, not `num_simulations`.

```
selection_policy: str = "legacy"       # "legacy" ⇒ best-first loop; "puct" ⇒ simulation loop
num_simulations: int | None = None     # required under "puct"; forbidden under "legacy"
sim_batch_size: int | None = None      # "puct" default 4; forbidden under "legacy"
puct_c: float | None = None            # "puct" default 1.0; forbidden under "legacy"
```

Threaded to the reasoner at construction (rl_training_driver.py:466-476) and in the
`--eval-only` path — where they are **not** passed (eval stays greedy single-pass per the
locked decision).

**Exit criterion:** a default-constructed reasoner and one with explicit
`selection_policy="legacy"`, same seed and executor script, produce structurally
identical graphs — same node expressions/statuses/creation order, same edge
tactics/child sets — and no edge accumulates visit statistics. At
`selection_policy="puct"` on the multi-level fake executor, visit counts accumulate
(even at `num_simulations=1` — the gate is the policy, not the simulation count) and
virtual loss returns to zero after every batch. Constructing a reasoner or config with
`selection_policy="legacy"` and an explicit simulation budget raises `ValueError`.

---

## 4. Phase 2 — soft critic target and the decoupled training step

**Files:** `maths_ai/gnn_inference/atp_lean_gnn/search_harvest.py`;
`maths_ai/gnn_inference/atp_lean_gnn/pln_rl_training.py`;
`maths_ai/gnn_inference/atp_lean_gnn/rl_training_driver.py`; config; tests.

### 4.1 `extract_critic_samples` (search_harvest.py)

```python
@dataclass(frozen=True)
class CriticSample:
    goal: str
    hypotheses: tuple[str, ...]
    target: float

def extract_critic_samples(graph, *, visit_threshold: int) -> list[CriticSample]:
    for node in graph.nodes.values():                      # graph.nodes is a dict
        if node.status == NodeStatus.SOLVED:  target = 1.0
        elif node.status == NodeStatus.DEAD:  target = 0.0
        else:                                               # OPEN *or* EXPANDED — see note
            t_star = max(edges(node), key=lambda e: e.tactic.probability, default=None)
            if t_star is None or t_star.visit_stats.N < visit_threshold:
                continue                                    # insufficient evidence: no label
            target = t_star.visit_stats.W / t_star.visit_stats.N
        ...
```

Correction to the design doc: unresolved nodes with edges are `EXPANDED`, not `OPEN` —
the doc's `status == OPEN` branch would have skipped exactly the population the soft
target exists for. The extractor treats any node not SOLVED/DEAD as a soft-target
candidate; unexpanded OPEN nodes have no edges and fall out via the `t_star is None`
guard.

This target is pure visit statistics and statuses — **PLN never enters it**, preserving
the "PLN is never the quantity V regresses to" invariant. It is node-level state
evidence, so it does not conflict with the "failed actions give no critic target"
carve-out, which is about action-level failures.

### 4.2 `backup_values` generalization (search_harvest.py)

Signature gains `visit_stats: ... | None = None, visit_threshold: int | None = None`
(defaults preserve current behavior). Where the recursion currently bottoms out at the
hard `0.0` for unresolved nodes (:97), a node whose best-prior edge crossed the threshold
contributes `W/N` instead. At `num_simulations=1` no edge ever crosses the threshold, so
this reduces exactly to current behavior.

### 4.3 Decoupled step (`pln_rl_training.py`)

```python
def train_step_htps_style(model, optimizer_htps, tactic_batch, critic_batch, featurize, *,
                          w_critic_soft: float, arg_loss_weight: float,
                          grad_clip: float, device=None) -> dict[str, float]
```

One `Batch.from_data_list` over the union of the two sample sets, one `model.encode`
forward. Losses:

- `L_critic_soft = MSE(values, targets)` over critic rows (raw scale — the advantage
  normalization convention applies to the actor only).
- (Phase 3 adds `L_tactic_imitation` here; in Phase 2 the tactic term is zero.)

Then `backward`, `clip_grad_norm_`, `optimizer_htps.step()`. This function may run
multiple times per round: its inputs are stored `(input, label)` pairs whose validity
does not depend on which parameters generated them — the one-step-per-collect invariant
is a property of the score-function estimator in `compute_onpolicy_loss` and does not
apply here. Document the exemption in this function's docstring **and** next to the
invariant comment at rl_training_driver.py:513.

`compute_onpolicy_loss` and `train_step_onpolicy` are not modified.

### 4.4 Driver wiring

- `critic_soft_queue: deque[CriticSample]` with `maxlen=critic_queue_size`, extended from
  **every** collected result each round (solved or not — that is the point).
- `optimizer_htps = AdamW(model.parameters(), lr=cfg.htps_learning_rate, ...)` — a
  separate instance so the two gradient signals do not share Adam moment estimates. Guard:
  `assert optimizer_htps is not optimizer` at construction.
- Round loop, between the on-policy step and the curriculum update:

```python
critic_soft_queue.extend(extract_critic_samples(r.graph, visit_threshold=cfg.visit_threshold) for r in results)
for _ in range(cfg.htps_steps_per_round):
    ...sample batches, train_step_htps_style(...)
```

- Checkpoint: per Decision 1.4, `save_checkpoint` gains the optimizer-htps state and
  serialized queues; resume restores with defaults for old checkpoints.
- `metrics.jsonl` rows gain `critic_soft_loss`, `critic_queue_len`.

### 4.5 Config additions

```
visit_threshold: int = 4
critic_queue_size: int = 10000
htps_steps_per_round: int = 0       # 0 ⇒ decoupled step disabled ⇒ current behavior
htps_batch_size: int = 64
htps_learning_rate: float = 1e-4
w_critic_soft: float = 0.5
```

**Exit criterion:** with `htps_steps_per_round=0` everything is bit-identical to Phase 1.
With it enabled on the fake-executor driver test, `critic_soft_loss` decreases over
rounds and the on-policy metrics stay within seed noise of Phase 1 (no interference at
small scale). The Section 10 Ablation-2 comparison (soft vs. hard-only critic) runs on
the real setup after landing.

---

## 5. Phase 3 — minimal-hypertree tactic imitation

**Files:** `search_harvest.py`, `pln_rl_training.py`, `rl_training_driver.py`, config,
tests.

### 5.1 `extract_minimal_hypertree` (search_harvest.py)

```python
@dataclass(frozen=True)
class TacticImitationSample:
    goal: str
    hypotheses: tuple[str, ...]
    tactic_id: int
    arg_indices: tuple[int, ...]     # empty ⇒ tactic-only row

def extract_minimal_hypertree(graph, edge_actions, *, mine_all_solved_nodes: bool = True
                              ) -> list[TacticImitationSample]
```

For each SOLVED node (all of them when `mine_all_solved_nodes`, else the root only, and
only if solved): recursively pick, among its SOLVED edges, the one minimizing total
downstream **step count** (memoized over the shared subgraph), and emit one sample per
edge on that minimal tree. Only edges present in `edge_actions` are emitted — an edge
without a stored policy action is either a PLN-fallback pseudo-edge (no real tactic) or
outside the on-policy filter, and neither is a valid imitation target. Argument indices
come from the stored `EdgeAction.arg_indices` (Decision 1.3); out-of-range/`-1` positions
keep the existing ignore semantics.

The minimality criterion is `step_count` only: `TacticOutcome` carries no timing and
`PantographExecutor.apply` records none, so the design doc's `tactic_cpu_time` option
requires instrumentation that does not exist. Noted as future work, not a config choice
that silently does nothing.

### 5.2 Loss and driver

`train_step_htps_style` gains the tactic term: for imitation rows,
`L_tactic_imitation = CE(tactic_logits, tactic_id)` — reusing `compute_bc_anchor_loss`
(pln_rl_training.py:299), which already implements `-1`-ignored cross-entropy — plus the
teacher-forced argument log-probs through the same `forced_step` path `evaluate_actions`
uses, weighted by `arg_loss_weight`:

```
loss = L_tactic_imitation + w_critic_soft · L_critic_soft
```

Driver: `tactic_imitation_queue: deque[TacticImitationSample]`
(`maxlen=tactic_queue_size`), extended each round from all results' solved subgraphs;
sampled alongside the critic queue in the decoupled step. Config additions:

```
tactic_queue_size: int = 10000
mine_all_solved_nodes: bool = True
```

`metrics.jsonl` gains `tactic_imitation_loss`, `tactic_queue_len`,
`imitation_samples_mined`.

**Exit criterion:** on the fake-executor harness, a search whose root fails but whose
subgraph contains a SOLVED node yields imitation samples (the user's core requirement);
root-only mode yields none for the same graph. Ablation 1 (all-solved vs. root-only)
runs on the real setup after landing; the default stays `mine_all_solved_nodes=True`
unless it loses.

---

## 6. Invariants: preserved, exempted, added

| Invariant | Status |
|---|---|
| one `optimizer.step()` per collect (on-policy) | Preserved for `train_step_onpolicy`. `train_step_htps_style` is exempt (supervised regression on stored pairs; no score-function estimator). Exemption documented at both invariant sites. |
| featurizer identity collect↔train | Preserved for on-policy. Imitation/critic samples are goal-keyed and re-featurized fresh; valid because featurization is deterministic under the fixed prepared vocab. |
| vocabs from `prepared_root` only | Unchanged. |
| Φ(terminal)=0 potential shaping | Untouched — this plan does not modify `pln_reward`. |
| critic sees success rows only (on-policy) | Unchanged. The soft target has its own rule (SOLVED/DEAD/threshold-crossed unresolved) — node-level, distinct mask. |
| PLN never the critic's regression target | Preserved and extended: `W/N` is built from search statistics and statuses only. |
| no autograd tensors stored across search | Preserved: queues hold strings/ints/floats. |
| advantage normalized, critic target raw | Preserved; `L_critic_soft` is raw-scale MSE. |
| no model-architecture change | Preserved; warm-start loader untouched. |
| **new:** soft target only above `visit_threshold` | `extract_critic_samples` / generalized `backup_values`. |
| **new:** `virtual_loss` zero after every backup | `_backup_simulation`; unit-tested. |
| **new:** `num_simulations=1` ≡ legacy search | branch to the verbatim legacy loop + regression test. |
| **new:** decoupled step never touches the on-policy optimizer | separate instance + identity assert in the driver. |

## 7. Test plan

New fake for `maths_ai/gnn_inference/tests/` (added to the existing fake family):
`_SubgoalExecutor(depth_map)` — yields configurable subgoals for the first few
applications, then QED, so graphs have interior structure (the existing `_QEDExecutor`
closes the root on the first expansion, which makes multi-simulation trivially
unobservable).

| Test | Checks |
|---|---|
| `test_visit_stats.py` (new) | `Q` FPU/mean/virtual-loss arithmetic. |
| `test_selection_policy.py` (new) | PUCT ordering: prior dominates at low N, Q at high N; virtual loss suppresses in-flight paths. |
| `test_rl_reasoner.py` (extend) | per-node stash: batched proposal across two leaves does not misattribute failures; per-node flush emits the same failure set as the sequential order did. |
| `test_rl_reasoner.py` (extend) | **Phase-1 regression**: seeded run, `num_simulations=1` vs. pre-change snapshot — identical node/edge/status sets and `RLSearchResult` contents. |
| `test_rl_reasoner.py` (extend) | `num_simulations>1` on `_SubgoalExecutor`: N/W accumulate; `virtual_loss==0` on every edge after `prove` returns; deadline mid-search returns a partial graph with harvestable transitions. |
| `test_search_harvest.py` (**new** — despite the design doc's file map, it does not exist yet) | `extract_critic_samples`: EXPANDED nodes above threshold emit `W/N`; below threshold emit nothing; SOLVED/DEAD emit 1.0/0.0. `backup_values` unchanged when `visit_stats=None`. `extract_minimal_hypertree`: empty for unsolved-root+`mine_all=False`; non-empty for solved interior node with `mine_all=True`; picks the step-minimal edge; skips edges without `EdgeAction`. |
| `test_pln_rl_training.py` (extend) | `train_step_htps_style`: loss decreases on a fixed toy queue; does not step the on-policy optimizer; `-1` arg masking. |
| `test_rl_training_driver.py` (extend) | new config fields default-off ⇒ metrics identical to baseline; enabled ⇒ queue lengths grow, decoupled losses logged; checkpoint round-trips optimizer_htps + queues. |

All existing tests must stay green at every phase (the pre-existing `test_premise_pool`
failure excepted).

## 8. Corrections to `docs/htps_mcts_integration_plan.md` (apply alongside Phase 0)

1. §2.6: unresolved-with-edges nodes are `EXPANDED`, not `OPEN`; `graph.nodes` is a dict;
   `node.edges` does not exist (`outgoing_edge_ids`).
2. §4.3: the RL seams are `predict_next_tactic` and `_link` (plus `_build_gnn_engine`) —
   not "wherever PLN scoring is invoked"; `rank_subgoals` is never overridden. And "no new
   seams needed" is wrong: the pending stash must be re-keyed per node (Section 1.1 here).
3. §2.5: no Lean tactic timing exists; `step_count` is the only implementable minimality
   criterion today.
4. §2.4 / §4.6: `SearchHParams` and per-attempt hyperparameter sampling are out of scope
   (locked decision); remove from the near-term file map.
5. §8 cost table: current cost is k single-graph GNN forwards per node expansion, not ~1.
6. File map: tests live in `maths_ai/gnn_inference/tests/`; `test_search_harvest.py` is
   new, not modified; the config path is `maths_ai/gnn_inference/configs/rl_actor_critic.json`.
7. §5 (`prove_mcts`) deferred entirely (locked decision).

## 9. Verification (end-to-end, per phase)

```bash
uv run python -m pytest maths_ai/gnn_inference/tests/ -q          # every phase
uv run python -m maths_ai.gnn_inference.scripts.rl_smoke           # live chain unchanged
# Phase 1+: short driver runs on the fake harness via test_rl_training_driver.py,
# then a real 10-round run with num_simulations=1 comparing metrics.jsonl to a
# pre-change baseline run (same seed), before enabling num_simulations>1.
```

## 10. Known risks (carried over, updated)

- **Wall-clock envelope**: `theorem_timeout_s=120` was tuned for single-pass search; K>1
  runs will hit the deadline path often. The partial-graph survival fix (3.4) makes this
  degrade to "fewer simulations", not "lost experience", but round wall-clock per solved
  theorem must be re-measured before raising K in real runs.
- **Two optimizers on shared weights**: interference between the on-policy and decoupled
  updates is possible even with separate Adam state; watch the on-policy metrics when
  `htps_steps_per_round` first goes nonzero (Phase 2 exit criterion).
- **PLN cost under K>1**: every newly created subgoal still pays one petta call;
  simulation count multiplies subgoal discovery. Profile before large K.
