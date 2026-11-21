# Integrating PLN reward (Approach 1 + 5) into the Actor-Critic via the existing AND-OR search

## Purpose and the key architectural realization

This plan integrates PLN into the actor-critic reward system using **potential-based reward
shaping (Approach 1)** and **subgoal terminals + curriculum (Approach 5)** from
`pln_reward_integration_approaches.md`, wired into the **Tier B** work (B1 reward source, B2
pointer-as-actor, B3 multi-step credit assignment) from
`actor_critic_warmstart_and_rl_refactor.md`, against the Lean/PLN infrastructure that already
exists in `maths_ai/hybrid_reasoner/joint_inference.py` and `maths_ai/pln_inference/model.py`.
It also fixes the blocking nature of PLN inference.

**The realization that reshapes the whole design:** `HybridReasoner.prove` is not a linear
rollout — it is already a best-first **AND-OR proof search** that (1) applies tactics to real
Lean states through `PantographExecutor.apply`, (2) scores every resulting subgoal with PLN via
`PLNInference.evaluate`, (3) links them into a `ProofHypergraph` whose `_propagate` back-props
solved/rank status to the root, and (4) already enforces `max_depth` and `max_nodes`. So the RL
environment, the PLN evaluation, the terminal detection (empty subgoal list = QED for a branch),
the subgoal structure, and the depth/node budgets **already exist**. The task is therefore not
to build a rollout loop — it is to **harvest RL training signal from the search tree the reasoner
already produces**, add the shaping reward and the critic, sample tactics from the actor policy
so the search is on-policy, and de-block the PLN calls so many searches can run concurrently.

This makes the design an HTPS-style *learn-from-search* system (closer to Approach 4's structure)
even at Stage 1, because the substrate is an AND-OR graph, not a chain. The plan keeps the
Approach-1 reward and Approach-5 terminals exactly as specified, but the credit assignment
(B3) must respect AND-OR backups rather than a single successor.

## Symbol and component table

| Name | Meaning / role | Where |
|------|----------------|-------|
| `Goal(expression, hypotheses)` | a proof state `s`: target string + local hyps | `data_models/proof_components.py` |
| `STV(strength σ, confidence c)`, `.score = σ·c` | PLN truth value of a state | `data_models/proof_components.py` |
| `PLNInference.evaluate(expr, hyps) → PLNResult` | **blocking** PLN query (subprocess.run, 60s) | `pln_inference/model.py:91` |
| `PantographExecutor.apply(server, state, tactic) → TacticOutcome` | async Lean step; `subgoals=[]` on success ⇒ QED for branch | `joint_inference.py:81` |
| `HybridReasoner.prove(goal) → ProofHypergraph` | async AND-OR best-first search | `joint_inference.py:301` |
| `HybridReasoner._expand(graph, node)` | apply GNN top-k tactics, PLN-rank subgoals, link edges | `joint_inference.py:361` |
| `ProofHypergraph` | AND-OR graph; `is_solved`, `is_exhausted`, `frontier`, `add_edge`, `_propagate` | hybrid_reasoner |
| `ActorCriticWithArgsClassifier` | actor + critic + pointer (Tier A, done) | `atp_lean_gnn/actor_critic.py` |
| `Φ(s) = σ(s)` | shaping potential = PLN strength | new reward module |
| `V_φ(s)` | critic value (Tier A `CriticHead`) | `atp_lean_gnn/actor_critic.py` |

Grounding of the reward symbols: `r_term` = trustworthy terminal reward (QED / closed subgoal =
`+1`, failure = `0`/small negative, per-step penalty); `Φ(s) = σ(s)` = PLN strength of state `s`;
`F(s,s') = γΦ(s') − Φ(s)` = potential-based shaping term; `Â` = advantage; `γ` = discount.

---

## Feature 0 — de-block PLN inference (prerequisite for concurrent search)

**Problem.** `PLNInference.evaluate` calls `subprocess.run(..., timeout=60)`
(`pln_inference/model.py:169`) — a synchronous call with no `await`. It is invoked from inside
the async `_expand` (`joint_inference.py:395,411`). On the single event-loop thread, that call
holds the thread hostage for up to 60 s, during which **no other coroutine can progress**. With
one sequential search that is invisible; the moment we run many searches concurrently to collect
training data, every search freezes every other search through each other's `subprocess.run`.

**Fix.** Move the blocking call off the event-loop thread with a bounded thread-pool executor.
`subprocess.run` waits on a separate OS process (`petta`) and **releases the GIL** while waiting,
so a thread executor gives genuine concurrency here.

- Add to `PLNInference`:
  ```python
  async def evaluate_async(self, expression, hypotheses=None, *, loop=None, executor=None):
      loop = loop or asyncio.get_running_loop()
      return await loop.run_in_executor(executor, self.evaluate, expression, list(hypotheses or []))
  ```
- Own a bounded pool so concurrent searches cannot spawn hundreds of `petta` processes:
  `self._pln_pool = ThreadPoolExecutor(max_workers=settings.pln_max_concurrency)` (e.g. 8), or an
  `asyncio.Semaphore(N)` around the `await`.
- Migrate the two blocking call sites in `_expand`/`rank_subgoals` to `await evaluate_async(...)`.
  `rank_subgoals` (`joint_inference.py:234`) becomes `async def rank_subgoals`, and its caller in
  `_expand` awaits it. This is the "migrate callers, no fallback" rule: the sync `evaluate` stays
  for non-async callers/tests, but the search path uses the async wrapper throughout.

**Phase boundary.** Collect trees under one `asyncio.run`, then train synchronously:
```python
trees = asyncio.run(collect_batch(theorems))   # async: searches overlap Lean+PLN waits
targets = extract_targets(trees)                # sync: hypergraph → (state, action, advantage)
train(targets)                                  # sync PyTorch — no event loop
```
The GNN forward inside the search is also synchronous; for Stage 1 leave it inline (Torch ops
release the GIL), and note batched GNN inference across concurrent frontiers as a later
throughput refinement.

---

## Feature 1 — the reward system (Approach 1 potential shaping + Approach 5 terminals)

A new module `atp_lean_gnn/pln_reward.py`, replacing the stub `reward.py` semantics for the
multi-step path.

### Components

1. **`PotentialSource`** — wraps `PLNInference` to return `Φ(s) = σ(s)` for a `Goal`.
   - `Φ(s) = result.stv.strength` (Approach-1 default). Config option `use_score` to use
     `σ·c` instead, folding in confidence.
   - **Convention `Φ(terminal) = 0`** — required for the shaping's optimum-invariance. QED and
     failed states return `Φ = 0`, not a PLN query.
   - Caches per state (`expression, hypotheses`) so a state re-encountered in the graph is
     queried once.

2. **`TerminalReward`** — the trustworthy `r_term` (Approach 5), read from the AND-OR outcome:
   - closed branch / closed subgoal (empty `TacticOutcome.subgoals`, or a node marked solved by
     `_propagate`): `+1`.
   - executor rejected the tactic (`outcome.success == False`): `0` or small negative.
   - depth/node-budget truncation (`mark_node_exhausted`): `0` (truncated ≠ failed — see B3).
   - per-step penalty `−ρ` (small) to prefer short proofs.

3. **`shaped_reward(parent, child, r_term)`** — the per-edge Approach-1 reward:
   ```
   r'(s, s') = r_term  +  ( γ·Φ(s') − Φ(s) )
   ```
   Because `Φ` telescopes over any path, this cannot change the optimal policy for *any* PLN;
   it only densifies the gradient (the guarantee that makes PLN safe to use while unreliable).

4. **`BehavioralCloningAnchor`** (from the approaches doc) — `w_bc(t)·(−log π(label | s))`, an
   annealed supervised term for states that carry a ground-truth tactic label. Dense,
   low-variance signal while PLN is flat and terminals are rare. Implemented in the loss, not the
   reward; `w_bc(t)` decays to 0. **This term is env-independent and can land immediately.**

### Approach-5 curriculum

`max_depth`/`max_nodes` already exist on `HybridReasoner`; truncation is already
`mark_node_exhausted`. Add:
- **Subgoal terminals:** already emitted — every closed subgoal is an AND-OR leaf that
  `_propagate` turns into solved status. `TerminalReward` reads it; no new mechanism.
- **Curriculum ordering:** a `CurriculumSchedule` that feeds the collect phase theorems
  easy→hard (few-step-provable first), widening as the value net improves, so the QED spike is
  reachable from the start.

---

## Feature 2 — on-policy sampling in the search (B2: pointer as part of the actor)

The current search picks tactics from GNN top-k **deterministically** (`predict_next_tactic`,
`joint_inference.py:200,371`). Policy gradient needs **sampled** actions with their log-probs.

- Add a **training-mode executor policy**: in the training variant of `_expand`, instead of
  iterating GNN top-k, call `ActorCriticWithArgsClassifier.act(state_batch, ...)` (the sampling
  method specified in Tier B) to **sample** `(tactic τ, args u)` and record `log π(τ) + Σ_k log
  π(u_k)` and `V_φ(s)`. This is B2 — the pointer's arguments are sampled and contribute to the
  policy gradient under the same advantage as the tactic.
- Featurize the `Goal` for the actor-critic through the **same string path the GNN engine already
  uses** (`Goal.expression` → `proof_state_to_dag` → `dag_to_pyg`), with OOV → `<UNK>` (the
  chosen fallback). This reuses the existing featurization; no new parser.
- Keep exploration governed by the entropy term (Tier A, already wired) and the legal-action mask
  (Tier A `legal_action_mask`; the mask *source* can now be Lean — a tactic the executor rejects
  is an illegal action, so rejections can populate the mask over time).

Design note: run the search **on-policy** so the collected `log π` and `V` match the policy being
updated. Harvesting from a stale deterministic GNN search would require importance weighting; not
worth it at Stage 1.

---

## Feature 3 — AND-OR credit assignment (B3: multi-step, adapted to the hypergraph)

B3 as originally written assumed a single successor `s'`. The real substrate is AND-OR: one
tactic yields a **set** of subgoals, **all** of which must be proved (AND), and a node may be
attacked by several tactics (OR). Credit assignment must respect this.

- **Terminal / solved backup already exists.** `ProofHypergraph._propagate` computes solved
  status bottom-up (a node is solved iff some tactic's subgoals are *all* solved). This is the
  trustworthy terminal signal, propagated for free by the search.
- **Critic target (value backup).** Define `V_φ(s)` to estimate provability. Its regression
  target comes from the AND-OR backup:
  - solved node → target `1`; dead/exhausted node with no hope → target `0`;
  - interior node → bootstrapped from children: `V(s | τ) ≈ ∏_j V(child_j)` (AND: all subgoals
    must close), `V(s) ≈ max_τ V(s | τ)` (OR: best tactic). Use the log-domain sum to avoid
    underflow, and detach the target (semi-gradient), exactly as one-step TD detaches its target.
- **Advantage for the actor.** For the sampled tactic `τ` at state `s`:
  ```
  Â(s, τ) = [ r'_term(s,τ) + γ · V_children(s,τ) ] − V_φ(s)
  ```
  where `V_children(s,τ)` is the AND-combined value of the subgoals `τ` produced (the bootstrap),
  and `r'_term` includes the Approach-1 shaping `γΦ(s') − Φ(s)` summed appropriately over the
  produced subgoals. Normalize `Â` across the collected batch (Tier A A5 normalization), detach
  it for the actor.
- **Reward shaping in AND-OR.** For a tactic producing subgoals `{s'_j}`, apply the shaping per
  child: `Σ_j (γΦ(s'_j) − Φ(s))`, keeping the telescoping property along each root→leaf path.

This is the one genuinely new piece of math relative to the linear B3 sketch; everything else
(sampling, log-probs, entropy, advantage normalization, critic MSE) is the Tier-A/B machinery.

---

## Feature 4 — training loop (collect async → train sync)

New `atp_lean_gnn/pln_rl_training.py`:

```
for round in range(num_rounds):
    theorems = curriculum.sample(round)                     # Approach 5
    graphs   = asyncio.run(collect_batch(reasoner, theorems, policy=model))   # async search
    targets  = extract_targets(graphs, potential_source, reward_cfg)          # AND-OR harvest
    # targets: list of (state_data, tactic_action, tactic_logp, arg_logps, V(s), advantage, label?)
    optimizer.zero_grad()
    loss = compute_pln_ac_loss(targets, w_bc=curriculum.bc_weight(round), ...) # Approach 1 + BC
    loss.backward(); clip; optimizer.step()
    curriculum.update(solved_rate(graphs))                  # widen difficulty
```

- `collect_batch` runs `HybridReasoner.prove` (training variant) for each theorem under
  `asyncio.gather`, with PLN de-blocked via `evaluate_async`.
- `extract_targets` walks each `ProofHypergraph`, emitting one training tuple per sampled
  expansion, with the AND-OR advantage (Feature 3) and Approach-1 shaped reward (Feature 1).
- `compute_pln_ac_loss` = the Tier-A combined loss over these tuples plus the annealed BC anchor.
- The env-dependent code lives entirely inside `collect_batch`; training is plain synchronous
  PyTorch.

---

## File-by-file changes

**New**
- `atp_lean_gnn/pln_reward.py` — `PotentialSource`, `TerminalReward`, `shaped_reward`,
  `CurriculumSchedule`, reward config.
- `atp_lean_gnn/search_harvest.py` — `extract_targets(graph, …)`: AND-OR walk → training tuples;
  AND-OR value backup helpers.
- `atp_lean_gnn/pln_rl_training.py` — collect-async/train-sync loop.
- `tests/test_pln_reward.py`, `tests/test_search_harvest.py` — mock-graph unit tests.

**Modified**
- `pln_inference/model.py` — add `evaluate_async` + bounded pool; keep sync `evaluate`.
- `hybrid_reasoner/joint_inference.py` —
  - `rank_subgoals` → `async`, awaits `evaluate_async`; `_expand` awaits it and the PLN-fallback
    `evaluate` (`:411`) likewise;
  - add a **training-mode expansion** that samples from `model.act` (B2) and records `log π`,
    `V`, chosen action, instead of GNN top-k; guard behind a `policy=` argument so production
    search is unchanged;
  - expose per-node stashing of `Φ(s)`, sampled action, `log π`, `V` for the harvester.
- `atp_lean_gnn/actor_critic_loss.py` — add the annealed BC term `w_bc·(−log π(label|s))` to
  `compute_actor_critic_combined_loss` (env-independent; land first).
- `atp_lean_gnn/actor_critic.py` — the `act()` sampling method (Tier B B2), if not already added.

---

## Data flow

```
theorem ──▶ HybridReasoner.prove (async, on-policy)
              │  _expand:
              │    Goal ─featurize(string→DAG→PyG)─▶ model.act ─▶ sample (τ, u), logπ, V(s)
              │    executor.apply (Pantograph) ─▶ subgoals {s'_j}   (empty ⇒ QED)
              │    PotentialSource.evaluate_async (PLN, thread pool) ─▶ Φ(s'_j)=σ
              │    graph.add_edge ─▶ _propagate solved/rank up the AND-OR graph
              ▼
         ProofHypergraph
              │  extract_targets (sync):
              │    r_term (QED/subgoal/fail/trunc) + shaping (γΦ(s')−Φ(s))
              │    AND-OR value backup ─▶ V-target, advantage Â
              ▼
         training tuples ──▶ compute_pln_ac_loss (+ w_bc·BC anchor) ──▶ backward/step
```

---

## Alternatives considered

1. **Separate linear rollout, ignore the search (original B3 shape).** Drive `executor.apply`
   in a chain, following one subgoal per step. *Pro:* matches the simple linear B3 math. *Con:*
   duplicates the search, discards the AND-OR credit machinery `_propagate` already provides,
   and must arbitrarily pick which subgoal to follow at every AND node. **Rejected** — it throws
   away working infrastructure and the multi-subgoal structure that is the whole point.

2. **Harvest from the existing deterministic GNN search (off-policy).** *Pro:* zero change to
   the search. *Con:* the collected actions aren't sampled from the policy being trained, so
   policy gradient needs importance weighting/PPO clipping. **Rejected for Stage 1** — on-policy
   sampling in the search is simpler and unbiased.

3. **PLN as the value target directly (regress `V_φ → σ`).** **Rejected** — this is exactly the
   "rely on PLN" failure the approaches doc rules out; `V_φ` must regress the AND-OR terminal
   backup, PLN enters only as the shaping potential.

4. **Full MCTS/HTPS guided expansion using `V_φ` (Approach 4).** The end goal, but it needs the
   critic to already be trustworthy to guide search. **Deferred to Stage 3**; Stage 1 harvests
   from the existing best-first (GNN×PLN-ranked) search and only *trains* `V_φ`.

---

## Staging and dependency order

```
Stage 0 (now, env-independent):
  • BC anchor term  w_bc·(−log π(label|s))  in compute_actor_critic_combined_loss
  • model.act() sampling method (B2 core), unit-tested on mock batches

Stage 1a (de-block, no learning change):
  • PLNInference.evaluate_async + bounded pool
  • rank_subgoals/_expand → async awaits; verify prove() still solves the same theorems

Stage 1b (reward + harvest, mock graph):
  • pln_reward.py (Φ, r_term, shaping, Φ(terminal)=0)
  • search_harvest.extract_targets with AND-OR value backup — unit-tested on a hand-built graph

Stage 1c (end-to-end collect→train):
  • training-mode _expand samples from model.act; per-node stash
  • pln_rl_training loop: asyncio.run(collect) → sync train
  • curriculum ordering + subgoal terminals

Stage 2: confidence gate  λ(t)·c(s)  and/or PLN-as-feature (approaches doc Approach 2/3)
Stage 3: V_φ-guided MCTS/HTPS expansion (Approach 4)
```

`Stage 0` is safe to merge before any env work — it only adds a loss term and a sampling method.
Everything from `Stage 1a` on requires the Pantograph server and the `petta` binary.

---

## Open questions / decisions to confirm

1. **`Φ = σ` (strength) vs `Φ = σ·c` (score)?** Strength is the pure Approach-1 default; score
   folds in confidence early. Recommend strength for Stage 1 (cleanest guarantee), revisit at
   Stage 2's confidence gate.
2. **AND value backup: product `∏ V(child)` vs min `min V(child)`?** Product reflects independent
   subgoal provability; min is more pessimistic/robust. Recommend product in log-domain, flag for
   tuning.
3. **Truncation value.** On `max_depth`/`max_nodes` exhaustion, bootstrap `V(s_last)` (proof not
   shown *unprovable*) or force `0`? Recommend bootstrap — truncation ≠ failure.
4. **Failure reward `0` vs small negative?** Small negative discourages illegal tactics faster
   but risks over-penalizing exploration; start at `0`, add the legal-action mask instead.
5. **Parsing.** The existing engine feeds `Goal.expression` (string) to the GNN, so Stage 1 reuses
   the string path (your earlier pending decision). AST→DAG remains the upgrade if OOV/structure
   loss proves limiting.

---

## Test plan

| Unit | Test |
|------|------|
| `evaluate_async` | returns same STV as `evaluate`; N concurrent calls overlap (wall-clock ≪ N×latency) with a fake slow `evaluate` |
| shaping | `Φ(terminal)=0`; telescoping: sum of shaping over a path = `γ^T Φ(s_T) − Φ(s_0)` |
| terminal reward | QED/closed-subgoal → `+1`; failure → `0`; truncation → `0` and `done` set |
| AND-OR backup | hand-built graph: solved leaf → `V=1`; unsolved → `0`; AND node = product; OR node = max |
| harvest | mock `ProofHypergraph` → correct `(state, action, advantage)` tuples; advantage sign matches solved/unsolved |
| BC anchor | `w_bc` term gradient flows to actor; `w_bc=0` reproduces pure RL loss |
| end-to-end (mock env) | `MockLeanEnv` + `MockPLN` → one collect→train round runs, loss finite, one optimizer step |

Run: `uv run python -m pytest maths_ai/gnn_inference/tests/test_pln_reward.py \
  maths_ai/gnn_inference/tests/test_search_harvest.py -q` (plain `python` is permission-denied).

---

## Out of scope

- The `petta`/MeTTa translator internals and Pantograph server setup (already implemented).
- Batched GNN inference across concurrent frontiers (throughput refinement).
- Full MCTS/HTPS guided search (Stage 3).
- Vocabulary extension / subword features (OOV stays `<UNK>` per the confirmed decision).
