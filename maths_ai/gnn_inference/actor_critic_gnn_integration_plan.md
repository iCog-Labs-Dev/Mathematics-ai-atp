# Integration Plan: Actor-Critic RL into the GNN Architecture

## 0. Context

This plan operationalizes the training pipeline described in *Part II: Training
Pipeline (PLN via Actor-Critic RL)*. The source design specifies a **shared GNN
backbone** processing a proof-state DAG, branching into an **Actor** (policy over
tactics) and a **Critic** (scalar STV predictor), trained with PLN-derived rewards
from a MeTTa interpreter. This document breaks that design into a concrete,
buildable integration plan.

---

## 1. Goals

- Extend the existing (supervised, imitation-trained) GNN into a dual-head
  Actor-Critic network without duplicating the graph encoder.
- Wire a full RL loop: state encoding → tactic sampling → Lean environment step →
  PLN reward → advantage → combined loss → backprop.
- Keep the system modular so the PLN/MeTTa reward source, the Lean environment,
  and the network architecture can be swapped or upgraded independently.

---

## 2. Architecture Integration

### 2.1 Shared GNN Encoder
- Reuse the current GNN exactly as-is for message passing over the proof-state
  DAG (nodes = subgoals/terms, edges = dependency structure).
- Output: a single **global embedding** `h_s` (pooled node representations, e.g.
  attention pooling or mean+max pooling over final-layer node embeddings).
- No architectural change needed here beyond exposing `h_s` as a first-class
  return value (currently it likely feeds straight into a single prediction
  head from the supervised phase).

### 2.2 Actor Head (new)
- Small MLP: `h_s → logits over tactic vocabulary`.
- Output: `π(a|s)`, a categorical distribution over available tactics.
- **Action masking**: at each state only a subset of tactics is legal/applicable.
  Add a mask step (set illegal-tactic logits to `-inf`) before softmax — this is
  not mentioned explicitly in the source doc but is required for correctness
  whenever the tactic vocabulary is fixed-size and state-dependent legality
  exists.

### 2.3 Critic Head (new)
- Small MLP: `h_s → scalar`.
- Output: predicted STV of the current state (expected provability strength),
  matching the PLN target signal used for the reward.

### 2.4 Parameter Sharing
- Encoder parameters are shared and receive gradients from **both** heads
  every step (per the combined loss below). Actor and Critic MLP heads have
  independent parameters.

```
DAG --> [Shared GNN Encoder] --> h_s --+--> [Actor MLP]  --> π(a|s)
                                        +--> [Critic MLP] --> V(s) ≈ STV(s)
```

---

## 3. Training Loop Implementation

Maps directly onto the four steps in the source doc, expanded into concrete
sub-tasks.

### Step 1 — Forward Pass
- Encode current DAG state → `h_s`.
- Actor produces `π(a|s)`; sample (or argmax during eval) a tactic `a`.
- Critic produces `V(s)`.
- Cache `log π(a|s)` and `V(s)` for the loss step (standard on-policy bookkeeping).

### Step 2 — Environment Step
- Apply tactic `a` in Lean.
- Handle three outcomes explicitly:
  1. **Valid step** → new DAG state `s'`.
  2. **Proof complete** → terminal state, episode ends.
  3. **Tactic error / dead end** → terminal or penalized state (needs an explicit
     policy — see Open Questions §6).

### Step 3 — PLN Reward
- Call the MeTTa PLN interpreter with `(s, a, s')`.
- Interpreter returns the strict mathematical STV of `s'` → this is `R`.
- Wrap this call behind a `RewardSource` interface so the PLN/MeTTa backend can
  be mocked, cached, or replaced without touching the training loop.
- Since PLN evaluation is likely the slowest part of the loop, plan for
  batching/caching identical `(s,a,s')` triples across parallel rollouts.

### Step 4 — Advantage Calculation
```
A = R - V(s)
```
- Matches the source doc's one-step advantage. Flag as an extension point:
  a discounted / multi-step or GAE-style advantage (`A = R + γV(s') - V(s)`)
  may reduce variance later, but start with the doc's exact one-step formula
  for parity with the design.

---

## 4. Loss Function Implementation

```
Loss_Total = Loss_Actor + (c1 * Loss_Critic) - (c2 * Entropy)
```

| Term | Formula | Purpose | Notes |
|---|---|---|---|
| `Loss_Actor` | `-log(π(a|s)) * A` | Reinforce tactics that beat the critic's expectation | Detach `A` from the graph — gradients should **not** flow from advantage back through the Critic during the actor update |
| `Loss_Critic` | `MSE(V(s), R)` | Make Critic track true PLN STV | Scaled by `c1 ≈ 0.5` per source doc |
| `Entropy` | `H(π(·|s))` | Encourage exploration, prevent premature collapse onto one tactic (e.g. `simp`) | Subtracted since the loss is minimized but entropy should be maximized |

Implementation notes:
- `A` must be `.detach()`-ed (or `stop_gradient`) before use in `Loss_Actor`
  and before use as the entropy-unrelated term — otherwise the actor loss
  would incorrectly backprop into the Critic head.
- Single optimizer over shared encoder + both heads, single backward pass per
  update — matches "update both networks simultaneously."
- Suggested starting hyperparameters: `c1 = 0.5` (from doc), `c2` small
  (e.g. `0.01`), tuned via ablation once the loop is stable.

---

## 5. Build Phases / Milestones

1. **Head split** — Add Actor and Critic MLPs on top of the existing encoder;
   verify shapes and that both heads run in a single forward pass.
2. **Environment wrapper** — Build a `LeanEnv` step function: `(state, tactic) →
   (next_state, done, info)`. Unit test on a handful of known proof states.
3. **Reward wrapper** — Build `RewardSource.get_stv(s, a, s')` calling the
   MeTTa PLN interpreter; add a mock/stub version for fast local testing before
   the real interpreter is wired in.
4. **Rollout collector** — Loop steps 1–4 above for N steps/episodes, storing
   `(s, a, log π(a|s), V(s), R, done)` tuples.
5. **Loss + optimizer** — Implement the combined loss exactly as in §4; confirm
   gradients reach the shared encoder from both heads.
6. **Stability pass** — Add advantage normalization, gradient clipping, and
   reward scaling if training is unstable (common in actor-critic setups with
   sparse or noisy reward sources like a symbolic interpreter).
7. **Evaluation harness** — Track proof success rate, average STV, entropy
   over time, and Critic MSE as core metrics.

---

## 6. Open Questions / Risks (not fully specified in source doc)

- **Terminal/failure handling**: what reward is assigned on a Lean tactic
  error or dead-end branch? Needs an explicit convention (e.g. fixed penalty
  or STV of 0).
- **Episode structure**: is this single-step bandit-style (one tactic per
  reward) or multi-step with discounting across a full proof? The doc's
  advantage formula (`R - V(s)`) reads as one-step; confirm before adding
  `γV(s')`.
- **PLN/MeTTa latency**: if the interpreter call is expensive, it may bottleneck
  rollout throughput — worth profiling early and considering async/batched
  calls.
- **Action space size and masking source**: where does the list of legal
  tactics per state come from (Lean itself, or precomputed)? This determines
  how the masking step in §2.2 is implemented.

---

## 7. Suggested Module Layout

```
/model
  gnn_encoder.py       # shared backbone (existing, minimally modified)
  actor_head.py
  critic_head.py
  actor_critic_model.py  # wraps encoder + both heads, single forward()
/env
  lean_env.py           # Step 2: apply tactic, return new DAG
/reward
  pln_reward_source.py  # Step 3: MeTTa PLN interpreter call
  mock_reward_source.py # for fast local testing
/training
  rollout_collector.py
  losses.py              # Loss_Actor, Loss_Critic, Entropy, Loss_Total
  train_loop.py
/eval
  metrics.py             # success rate, avg STV, entropy, critic MSE
```
