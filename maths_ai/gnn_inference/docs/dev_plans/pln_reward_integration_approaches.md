# Integrating PLN into the Actor-Critic reward system

## The problem

The actor-critic loop needs a reward signal. The intended source is the PLN (Probabilistic
Logic Networks) truth value of a proof state, obtained from the MeTTa interpreter through the
translator. Two properties of that signal, as it stands today, drive every decision in this
document:

1. **PLN is not yet reliable.** It is incomplete, and empirically it returns the same or a
   very similar value for almost all proof states. We want to *use* it without letting it
   *define* what the value network learns.
2. **Terminal reward is sparse and distant.** Proving a theorem takes many steps, so a QED
   (all goals closed) terminal is reached rarely and far from the initial state — we cannot
   rely on terminal states alone to carry the training signal.

These two are coupled: the safe ways to use an unreliable dense signal all assume the
trustworthy sparse signal is reached *sometimes*, and the trustworthy signal is sparse
precisely because proofs are long. Any workable design must address both at once.

Symbols used throughout: `s` = proof state; `s'` = successor state after applying an action;
`a = (τ, u)` = action (tactic `τ` and its argument list `u`); `V_φ(s)` = critic value with
parameters `φ`; `π` = policy; `γ` = discount; `Â` = advantage; `PLN(s) = (σ(s), c(s))` where
`σ` = strength ("how provable this looks") and `c` = confidence ("how much evidence backs the
estimate"). `1[·]` = indicator.

---

## Why a near-constant reward yields no gradient

This is the governing fact, and it determines whether the current architecture can learn at
all.

The actor gradient is `−∇ log π(a|s) · Â`, with `Â = r − V_φ(s)` (one-step) or
`Â = r + γ·V_φ(s') − V_φ(s)` (multi-step TD). The **level** of the reward is irrelevant — the
critic baseline `V_φ(s)` subtracts it. Only the **variation of `r` across states and actions**
survives into `Â`.

If `PLN(s) ≈ k` for all `s`, then `V_φ(s) → k`, so `Â = k − k = 0` everywhere. Zero advantage
gives **zero actor gradient**: a constant reward and a different constant reward define the
same objective; the policy is told "everything is equally good," which is identical to no
information. So a near-constant PLN reward does not train slowly — it does not train at all,
except wherever the reward genuinely varies.

**Consequence.** The only thing that can teach the policy anything is *reward variance*. Today
the sole source of real variance is the terminal QED spike. Whether the current actor-critic
can learn reduces entirely to: **is a terminal (or subgoal) reward reached with nonzero
frequency?** If yes, the `+1` spike seeds a signal and TD bootstrapping propagates it backward
(`V(s_{T-1}) → γ·1`, `V(s_{T-2}) → γ²·1`, …), turning `V_φ` into a distance-to-QED gradient
whose variation produces nonzero advantages on the approach. If no, there is zero reward
variance anywhere and the loop spins with zero gradient — and no amount of near-constant PLN
changes that. Near-constant PLN is inert: it neither rescues nor blocks learning; terminal
reachability is everything.

This is why "make terminals reachable" (Approach 5) is not optional — it is the precondition
under which every other approach functions.

---

## Design principle

The signal that **defines the objective** — the target the value network is anchored to —
must be the trustworthy one (terminal / subgoal reward). PLN may enter *only* through a
channel that:

- (a) provably cannot change the optimum (potential-based shaping), or
- (b) is bounded/gated by its own confidence and annealed away (confidence-gated reward), or
- (c) shapes the *representation* rather than the *target* (input feature / auxiliary head).

PLN must never be the quantity `V_φ` is regressed *to*. That single sentence is the technical
meaning of "integrate PLN without relying on it."

---

## Approach 1 — PLN as a potential, not a reward (potential-based shaping)

Use PLN as a **potential function** `Φ(s) = σ(s)` and give the shaped reward

```
r'(s, a, s') = r_terminal(s, a, s')  +  ( γ·Φ(s') − Φ(s) )
```

Classical result (Ng, Harada & Russell, 1999): potential-based shaping **leaves the optimal
policy unchanged for any `Φ`, however wrong.** The shaping terms telescope over a trajectory
to `γ^T Φ(s_T) − Φ(s_0)`, depending only on the endpoints, so they cannot bias which policy is
optimal; they only reshape the intermediate gradient to steer toward higher-PLN states sooner.

- **Buys:** exactly the requirement. `V_φ` still learns a terminal-anchored return; PLN only
  densifies the path. Good PLN → faster learning; garbage PLN → optimum provably safe, only
  the speedup is lost. Structurally prevents relying on PLN.
- **Catch:** the guarantee is about the *optimum*, which only bites if terminal reward is
  reached sometimes. If QED is never hit, all gradient comes from shaping and you are relying
  on PLN again. Must be paired with Approach 5.

## Approach 2 — Confidence-gated, annealed reward

Weight PLN by its own confidence and decay it over training:

```
reward = r_terminal  +  λ(t) · c(s') · σ(s')
```

- `c(s')` — PLN's confidence, so low-confidence estimates contribute little. This is the most
  direct use of the `(strength, confidence)` structure: let PLN say when to trust it.
- `λ(t)` — global schedule decaying to 0. Early, when learned value is useless, PLN scaffolds;
  late, when `V_φ` is trustworthy from real returns, PLN fades. "Scaffold then remove."
- **Pros:** simple; the confidence gate is principled.
- **Cons:** no optimum-invariance guarantee — a confidently-wrong PLN can bias the policy while
  `λ` is still large. Relies on the confidence channel being calibrated.

## Approach 3 — Decouple: PLN as feature or auxiliary target, never the value target

Train `V_φ` **only** on bootstrapped terminal-anchored returns. Feed PLN through a channel that
never touches that target:

- **Input feature** — concatenate `(σ(s), c(s))` into the state embedding before the heads.
  The network *learns* how much to trust PLN; if unreliable, the learned weight shrinks. PLN
  informs decisions without defining the reward.
- **Auxiliary prediction head** — add a side task "predict PLN(s)" with a small loss weight.
  Shapes the shared GNN representation (states PLN deems similar are pulled together) without
  contaminating `V_φ`'s regression target.

- **Pros:** the cleanest literal answer to "don't rely on PLN to train my value network" —
  `V_φ` regresses real returns, full stop.
- **Cons:** does nothing for sparsity; if terminal reward never arrives, `V_φ` has nothing to
  learn from. Combine with Approach 1 or 5.

## Approach 4 — MCTS as a policy-value improvement operator (the AlphaZero analogue)

What AlphaZero actually does — the opposite of "use a heuristic": its value network is trained
**purely on self-play terminal outcomes**, with **no external evaluation heuristic**. MCTS is a
**policy-improvement operator**: given the current `(π, V_φ)`, lookahead search produces two
signals *better* than the raw network —

- the **visit-count distribution** over actions → training target for `π` (a search-sharpened
  policy),
- the **search-backed value** (terminal outcomes backed up through the tree, `V_φ` evaluating
  unexpanded leaves) → training target for `V_φ`.

Train toward those; the improved net makes the next search stronger; iterate. The value target
is the search result anchored in real terminals, never a hand-crafted heuristic.

**Where PLN maps in:** PLN is a **leaf evaluator** — one estimate of an unexpanded node's value
that search can *override* once it expands that subtree and finds real terminal signal. PLN
influences *where search looks*; the *training target* is the search-backed value, which
increasingly reflects real proofs as they are discovered. This is precisely "use PLN but let
evidence overrule it": far from any found proof the value is PLN-flavored; near discovered
proofs it is terminal-flavored; the balance shifts automatically toward truth as search
succeeds.

**Theorem-proving-specific caveats (this is not Go):**
- **AND/OR tree, not OR tree.** A tactic can split a goal into several subgoals, *all* of which
  must be proved (AND nodes). A node is proved only if all child subgoals are — the backup rule
  differs from Go's minimax. This is what Meta's **HyperTree Proof Search (HTPS)** and Polu &
  Sutskever's **GPT-f** implement: AlphaZero-style learned policy-value search adapted to the
  AND/OR proof tree, trained on proved/disproved outcomes. These are the closest published
  analogues, more than raw AlphaGo.
- **One-sided outcome.** A strong signal exists when a proof is *found* (`+1`); "not found in
  budget" ≠ "unprovable." The value target is noisier on the negative side than Go's clean
  win/loss, which is exactly why a leaf heuristic (PLN) matters more here than in Go — it fills
  the gap for unresolved states.

- **Pros:** strongest method; search actively manufactures training signal from sparse
  terminals; PLN's role is bounded to leaves and self-corrects. State of the art for neural
  theorem proving.
- **Cons:** by far the most infrastructure — full MCTS/HTPS over the proof tree, many
  Pantograph calls per search, AND/OR backup logic. Not a first step.

## Approach 5 — Attack sparsity directly (complementary to all of the above)

None of 1–4 helps if QED is essentially never reached. Manufacture more terminal-like signal:

- **Curriculum / backward proving** — start from states *near* QED (few steps from a proof) and
  lengthen over training, so terminal reward is reachable from the start.
- **Subgoal terminals** — a proved *subgoal* is a small terminal. In the AND/OR tree every
  closed subgoal is a genuine `+1`, far denser trustworthy signal than whole-theorem QED.
- **Easy-theorem mixing** — keep a fraction of provable-in-few-steps problems in every batch so
  the value net always has some real returns to anchor on.

---

## The label-match question: keep it out of the reward, add it as an annealed auxiliary loss

Because PLN is near-constant today, an obvious temptation is to inject the supervised label as
reward: `r = 1[a == label]` (what `MockRewardSource` does now). This gives dense,
discriminative signal — but it belongs in a different channel, for two reasons:

1. **It conflates two objectives in one scalar.** With `r = 1[a==label] + PLN + terminal`, the
   critic regresses a target mixing "provable-ness" with "matches-the-corpus," and `V_φ` loses
   any clean meaning — you can no longer tell whether PLN is helping.
2. **REINFORCE-toward-label is a high-variance way to use a label you already know.** The label
   reward reaches the policy as `−log π(a_sampled)·(1[a==label] − V)` — sample, then compare.
   But the correct label is known *exactly*, so the low-variance, direct injection is
   **supervised cross-entropy** `−log π(label | s)`: no sampling, no baseline, exact gradient
   toward the right answer.

**Decision.** Keep the label out of the reward; add it as an auxiliary supervised (behavioral-
cloning) term:

```
L_total = L_RL( PLN-shaping + terminal/subgoal reward )  +  w_bc(t) · ( −log π(label | s) )
```

- The RL term keeps its clean, terminal-anchored meaning; PLN enters only as an optimum-safe
  shaping potential (Approach 1).
- The `w_bc` term is a **behavioral-cloning anchor**: dense, low-variance, always-present
  gradient that keeps the policy near supervised competence *while* PLN is flat and terminals
  are rare — the current regime.
- **Anneal `w_bc(t) → 0`** as terminals become reachable and real advantages appear. Early:
  mostly imitation (the only reliable signal). Late: mostly RL. Scaffold, then remove.

This is strictly better than a label reward: same dense signal, lower variance, exact target,
and it does not pollute the value function's regression target. It is nearly free to implement
— the supervised cross-entropy already exists (`backbone.classifier` was trained with it), so
it is one extra weighted term in `compute_actor_critic_combined_loss`, not a reward-source
change.

---

## Recommended staging

Do not start at MCTS. Dependency order that reaches a correct, trustworthy signal soonest:

1. **Stage 1 (once the Lean env is live).** Approach 1 (potential-based shaping, `Φ = σ`) **+**
   Approach 5 (subgoal terminals + easy-theorem curriculum) **+** the annealed BC anchor
   `w_bc`. PLN gives density with a *proven* guarantee it cannot corrupt the optimum; subgoal
   terminals + curriculum make the guarantee actually bite; the BC anchor keeps the policy
   competent while real advantages are still scarce. The critic trains on shaped returns
   anchored to real QED/subgoal terminals — never on PLN as ground truth.

   Stage-1 reward and loss, explicitly:
   ```
   r'(s,a,s') = r_term(s,a,s') + ( γ·σ(s') − σ(s) )          # terminal + PLN shaping
   Â_t        = r'_t + γ·V_φ(s_{t+1})·(1 − done_t) − V_φ(s_t) # TD advantage (detached for actor)
   L_total    = L_actor(Â) + c1·L_critic(returns) − c2·H(π)
              + w_arg·L_arg + w_bc(t)·( −log π(label | s) )
   ```
   where `r_term` = +1 at QED or closed subgoal, 0/small-negative at failure, small step
   penalty otherwise; `w_bc(t)` decays toward 0.

2. **Stage 2 (if PLN proves well-calibrated).** Fold in Approach 2's confidence gate `c(s)` so
   PLN contributes more where it is sure and the anneal removes it as `V_φ` matures. If unsure
   PLN should touch the reward at all, use Approach 3 instead (PLN as input feature / auxiliary
   head; `V_φ` on real returns only).

3. **Stage 3 (for real strength).** Approach 4 (HTPS-style AND/OR search), with everything above
   becoming the leaf evaluator and reward shaping *inside* the search.

**Through-line.** The value network's regression target is always a return grounded in
terminal/subgoal reward. PLN enters through shaping (optimum-safe), confidence-gating
(self-limiting), or representation (non-target) — never as the thing `V_φ` is regressed to. Do
not make PLN carry discrimination it cannot currently provide, and do not smuggle the label
into the reward — hold the label as an explicit, annealed supervised term, and spend effort
making terminal/subgoal reward reachable, because that spike is the one thing in the system that
actually varies.

---

## Comparison table

| Approach | PLN's role | Optimum-safe? | Helps sparsity? | Infra cost | Trust in PLN |
|----------|-----------|---------------|-----------------|-----------|--------------|
| 1 Potential shaping | shaping potential `Φ=σ` | **Yes (proven)** | No (needs A5) | Low | None required |
| 2 Confidence-gated reward | gated additive reward | No | No | Low | Confidence calibration |
| 3 Feature / aux head | input feature or side target | Yes (target untouched) | No | Low–med | None for `V_φ` |
| 4 MCTS / HTPS leaf eval | leaf evaluator, overridable | Yes (search-backed target) | **Yes (manufactures signal)** | **High** | Bounded to leaves |
| 5 Curriculum / subgoals | — (fixes terminal reachability) | — | **Yes (directly)** | Med | — |
| BC anchor (label) | *not PLN* — supervised term | Yes (separate loss) | Provides dense signal | Low | — |

---

## Relationship to the implementation plan

- Stage 1 is the concrete content of **B1 (reward source)** and **B3 (multi-step credit
  assignment)** in `actor_critic_warmstart_and_rl_refactor.md`. `PLNRewardSource` returns
  `σ(s')` (and `c(s')`); the shaping term `γ·σ(s') − σ(s)` and terminal/subgoal reward are
  assembled in the reward/return computation, not inside the loss.
- The BC anchor `w_bc(t)·(−log π(label|s))` is a new weighted term in
  `compute_actor_critic_combined_loss` (Tier A code already computes `tactic_logits` and has the
  labels via `batch.y`), gated by a schedule — implementable now, independent of the env.
- Approaches 4 and 5 (search, curriculum) are new subsystems beyond the current plan's scope and
  would each get their own dev plan.
