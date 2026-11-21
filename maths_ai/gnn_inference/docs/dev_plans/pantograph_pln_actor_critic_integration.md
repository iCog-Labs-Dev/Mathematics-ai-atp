# Integrating the existing Pantograph env + PLN reward into the Actor-Critic

## Headline finding

The two components the Tier-B plan intended to *stub* — a Lean environment that produces
successor states, and a PLN reward that produces a scalar per state — **already exist, already
work, and are already wired together** in `maths_ai/hybrid_reasoner/joint_inference.py`. The
actor-critic integration is therefore not "write stubs for Lean and PLN." It is:

1. **Adapt** two existing objects to the `RewardSource` / environment interfaces the RL loop
   expects, and
2. **Instrument** the existing `HybridReasoner.prove` search — which is *already* the exact
   `state → GNN tactic → Pantograph apply → PLN score → propagate` loop that RL rollouts need —
   to emit training transitions and value targets.

This also resolves the open parsing question (below): the existing pipeline already feeds Lean
output to the GNN as text, so "pretty-print → reuse `proof_state_to_dag`" is not a choice to
make — it is the established, working path.

---

## What already exists (mapping to the RL abstractions)

| RL abstraction (Tier B) | Already implemented as | Location |
|---|---|---|
| `LeanEnv.step(state, tactic, args) → successor(s)` | `PantographExecutor.apply(server, state, tactic) → TacticOutcome` | `joint_inference.py:81` |
| Reconstruct a Lean `GoalState` from a stored goal | `HybridReasoner._start_state(goal)` | `joint_inference.py:338` |
| `StepStatus` (ONGOING / SOLVED / FAILED) | `TacticOutcome(success, subgoals, error)`: `success=False`→FAILED; `subgoals=[]`→QED; else ONGOING | `hypergraph.py:127` |
| `RewardSource` scalar for a state | `PLNInference.evaluate(expr, hypotheses) → PLNResult.stv` | `pln_inference/model.py:91` |
| Reward reliability signal | `PLNResult.is_fallback` (True ⇒ STV is a *random* exploration score, not evidence) + `STV.confidence` | `pln_inference/model.py:52` |
| Reward scalar | `STV.score = strength × confidence` | `proof_components.py:37` |
| Policy (tactic + args) | `GNNModelEngine.inference(goal, top_k) → [TacticCandidate]` | `inference_engine.py:127` |
| Rollout / search loop | `HybridReasoner.prove` (best-first AND/OR expansion) | `joint_inference.py:301` |
| Proof tree + AND/OR value backup | `ProofHypergraph` with `_propagate`, `_edge_value`, `_recompute_node` | `hypergraph.py:179` |
| Depth / size budget | `max_depth`, `max_nodes` | `joint_inference.py:148` |
| Learned exploration over unreliable PLN | `DynamicThompsonSampler` (records observations, samples fallbacks) | used at `joint_inference.py:281` |

The env step, concretely, is a two-call sequence that already runs:
`state = await _start_state(goal)` then `outcome = await executor.apply(server, state, tactic)`.
`outcome.subgoals` is a `List[Goal]` where `Goal.expression` = the target string and
`Goal.hypotheses` = the local context as `"name : type"` strings.

---

## The parsing question is already answered

Earlier we deferred: pretty-print → reuse the string parser, versus AST → DAG directly.
The existing code decides it: `PantographExecutor.apply` converts every Pantograph goal to
`Goal(expression=str(g.target), hypotheses=[str(v) for v in g.variables])`
(`joint_inference.py:111-114`), and the GNN is already fed `goal.expression` as a **string**
through `predict_next_tactic → gnn_engine.inference(goal_expression)`
(`joint_inference.py:213`). So the whole system already runs on **option 1 (text)** with
`<UNK>` fallback for unseen tokens — exactly the two answers previously chosen. Featurizing a
rollout state for the actor-critic is therefore: reconstruct the `"h : T\n⊢ goal"` text from a
`Goal` (the same shape `_start_state` builds) and pass it to `proof_state_to_dag → dag_to_pyg`.
No new parser, no AST adapter, consistent with what inference already does.

One caveat carried over: `_sanitize_inaccessible_names` (`joint_inference.py:21`) rewrites
Lean's pretty-printer tokens like `p✝` before re-parsing. The featurizer must apply the *same*
sanitizer, or the GNN sees tokens the tactic-application path never would.

---

## The reality that reshapes the RL design: this is AND/OR search (HTPS), not a linear MDP

My earlier multi-step answers assumed a **linear** trajectory `s_0 → s_1 → …`. The real system
is an **AND/OR tree**, and `ProofHypergraph` already implements its semantics:

- A tactic applied to a goal produces **several subgoals, all of which must be proved** (AND).
  `_derive_edge_status` marks an edge SOLVED only if *every* child is SOLVED, DEAD if *any*
  child is DEAD (`hypergraph.py:377-385`).
- A goal is proved if **any one** of its tactics leads to a solved edge (OR):
  `_recompute_node` sets a node SOLVED if any outgoing edge is SOLVED (`hypergraph.py:411`).
- Value backup is already the conventional pessimistic AND-aggregate:
  `_edge_value = tactic.probability × min(child.combined_rank)` (`hypergraph.py:387`) — "a
  conjunction is only as strong as its weakest conjunct," which the code comment explicitly
  notes "mirrors value backup in AND-OR search / HTPS."

This matters for credit assignment. "The value of a state" is its **provability**, and the
correct target for `V_φ(s)` is not `r + γV(s')` along one path — it is the AND/OR-backed
outcome: a node is worth 1 if the search closed it (all required subgoals solved through some
tactic), 0 if the search proved it dead. **The hypergraph already computes exactly this backup.**
So the natural RL integration is the **HTPS / AlphaZero pattern** (Approach 4 in
`pln_reward_integration_approaches.md`), and — crucially — the search infrastructure it needs
already exists. We are much closer to Approach 4 than the earlier "start with one-step TD"
framing suggested.

---

## Recommended integration architecture: instrument the search, don't build a parallel rollout loop

Because `HybridReasoner.prove` already performs the rollout, the cleanest design trains the
actor-critic on the **completed search trees** it produces, HTPS-style:

**Phase 1 — search (data generation).** Run `prove(theorem)` as today. It expands nodes,
applies tactics via Pantograph, scores subgoals via PLN, and backs up SOLVED/DEAD through the
hypergraph. Instrument it to record, per expansion, a transition:
`(goal_state_text, sanitized, tactic_candidate, argument_ids, subgoal_texts, stv_per_subgoal,
is_fallback, edge_status)`. No new control flow — just logging what the search already computes.

**Phase 2 — label from the finished tree.** Once `prove` returns, walk the hypergraph and
attach a **value target** to every visited node from its final status:
`V_target(node) = 1.0` if `node.status == SOLVED`, `0.0` if `DEAD`. Nodes left OPEN at budget
exhaustion are *unresolved* (not failures) — mask them out of the value regression, or bootstrap
with `V_φ` (a proof not found in budget is not a disproof; §"Terminal vs truncation" from the
sparsity discussion).

**Phase 3 — train the actor-critic.**
- **Critic** `V_φ(s)`: regress toward the AND/OR-backed `V_target` — real proof outcomes, *not*
  PLN. This is the whole point: the value network learns provability from search results, so the
  unreliable near-constant PLN never becomes its regression target.
- **Actor** (tactic + pointer, per B2): policy-gradient toward tactics that led to SOLVED edges,
  weighted by advantage `A = V_target(node) − V_φ(node)`; or, closer to AlphaZero, a
  cross-entropy toward the search-preferred tactic distribution (which tactic closed the node).
  Both the tactic head and the argument pointer are trained by the same node-level advantage
  (the joint-action credit from B2).
- **Entropy + masking** (A4) and **advantage normalization** (A5) apply unchanged.

**Phase 4 — iterate.** The improved `(π, V_φ)` makes the next search stronger (better tactic
ranking, better frontier ordering via `combined_rank`), which yields better training trees.
This is the AlphaZero/HTPS outer loop.

### Where PLN sits in this design — and why the near-constant problem dissolves

PLN does **not** train the value network here. It keeps its current job: a **search-guidance
prior**, folded into `combined_rank = gnn_prob × strength × confidence`
(`proof_components.py:63`) to order the frontier and pick which subgoals to expand. That is
exactly the "PLN as a leaf evaluator the search can override" role from Approach 4:

- Far from any closed proof, `combined_rank` (PLN-flavored) decides where to look.
- Once the search closes or kills a subtree, the **SOLVED/DEAD** signal — real, trustworthy —
  becomes the value target, overriding whatever PLN guessed.
- The value net is trained on that outcome, so near-constant/​unreliable PLN cannot corrupt it;
  it only affects *search efficiency*, never the *learning target*. If PLN is flat, search is
  less guided (slower) but the value net still learns correct provabilities from outcomes.

The `is_fallback` flag and the `DynamicThompsonSampler` are the existing hedges for "PLN can't
derive this": when PLN falls back, DTS supplies a *learned* exploration score instead of using
the random fallback as if it were a real reward. That machinery stays as search guidance and
should **not** feed the critic target either.

---

## Async boundary (a real engineering constraint, not a design fork)

Pantograph is `async` (`goal_tactic_async`, `goal_start_async`) and `prove` is a coroutine; the
PyTorch training loop is synchronous. Options: (a) run search under `asyncio.run` to collect a
batch of trees, then hand the collected transitions to a synchronous training step — the search
and the gradient update are already naturally separate phases, so this is clean; (b) keep a
persistent event loop and interleave. Recommend (a): search-then-train phases, matching the
HTPS outer loop above. PLN's `evaluate` is a blocking `subprocess.run` (`model.py:169`), so PLN
calls should be run in a thread executor to avoid stalling the event loop during search.

---

## Concrete component plan

1. **`RewardSource` adapter over `PLNInference`.** A `PLNRewardSource` whose `get_reward(state)`
   calls `self.pln.evaluate(expr, hypotheses).stv.score`, and exposes `is_fallback` /
   `confidence` so callers can gate. This replaces the `reward.py` stub; the real object is
   `PLNInference`, already used by `HybridReasoner`. **But note:** per the architecture above,
   this scalar is for *search guidance / optional reward shaping*, not the critic target.

2. **Search instrumentation.** Add an optional `transition_sink` callback to `HybridReasoner`
   (or subclass it) that records the per-expansion transition tuple. Zero change to search
   behavior when the sink is absent.

3. **Tree-to-targets extractor.** A function `hypergraph → [(state_text, tactic, args, V_target,
   advantage_mask)]` reading `node.status` for SOLVED/DEAD/OPEN. This is the HTPS labeling step.

4. **Featurizer reused from inference.** `Goal → text (with `_sanitize_inaccessible_names`) →
   proof_state_to_dag → dag_to_pyg` — the same path `inference.py` and `_start_state` use.

5. **Trainer.** Consumes the extracted targets: critic MSE toward `V_target`, actor
   policy-gradient / cross-entropy toward search-preferred tactics, using the Tier-A
   actor-critic (residual warm-start, single forward, masking, advantage norm) already built.

6. **Outer loop.** `for round: collect trees over a batch of theorems (async) → extract targets
   → train (sync) → checkpoint → repeat`, with an easy-theorem curriculum (Approach 5) so
   SOLVED targets are non-empty early.

---

## Design decisions to confirm before implementing

1. **Policy target: advantage-weighted REINFORCE vs. AlphaZero-style cross-entropy.** The
   hypergraph gives a clean per-node SOLVED/DEAD label, which supports the lower-variance
   AlphaZero cross-entropy-to-search-policy target. Advantage-weighted REINFORCE reuses the
   Tier-A loss as-is but is noisier. Recommend AlphaZero-style given the tree labels are
   available.
2. **Does PLN enter the reward at all, or only search guidance?** Given it is near-constant and
   `is_fallback`-prone, the safe default is **search-guidance only** (frontier ordering), with
   the critic trained purely on SOLVED/DEAD. Optionally add PLN as potential-based shaping
   (Approach 1) *inside* the search value, which is provably optimum-safe. Do **not** regress
   `V_φ` toward STV.
3. **OPEN (budget-exhausted) nodes:** mask out, or bootstrap with `V_φ`? Bootstrapping uses more
   data but injects the model's own bias; masking is safer early. Recommend masking until the
   critic is trustworthy, then bootstrapping.
4. **Search integration point:** subclass `HybridReasoner` with a transition sink, or refactor
   `prove`/`_expand` to accept an optional recorder? Subclassing avoids touching the working
   inference path; recommend that.

---

## Relationship to the other plans

- This **supersedes** the "stub the Lean env and PLN" framing in
  `actor_critic_warmstart_and_rl_refactor.md` §B1/B3: the env and reward are real and wired, so
  B3 becomes "instrument `HybridReasoner`," not "write a rollout collector against a mock env."
- It **realizes** Approach 4 (MCTS/HTPS) from `pln_reward_integration_approaches.md` earlier
  than expected, because the AND/OR search already exists — and it keeps PLN in the Approach-4
  leaf-evaluator role, which is exactly what its near-constant unreliability calls for.
- The Tier-A actor-critic changes (residual warm-start, single forward, masking, advantage
  norm) are all reused unchanged as the network being trained.
