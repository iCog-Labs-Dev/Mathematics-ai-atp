# Actor-Critic: warm-start, forward-pass, and RL-integration refactor

## Purpose

The actor-critic scaffolding in `maths_ai/gnn_inference/atp_lean_gnn/` is structurally
complete but semantically incomplete: several of its stated design goals (behavioral
warm-start from the supervised pointer, single-pass rollout, action masking, true
policy-gradient on arguments, multi-step credit assignment) are not actually realized by
the current code. This plan specifies the concrete edits to close that gap.

The work splits into two tiers by dependency:

- **Tier A — environment-independent.** Correct now, without the Lean env or the PLN
  reward being live. These fix defects that exist regardless of reward source.
- **Tier B — environment-dependent.** Require the `LeanEnv` (produces `s'` after applying
  a predicted `(tactic, args)`) and the PLN `STV(s')` reward to be connected. Specified
  here so the Tier-A edits are shaped to accept them, but not to be implemented until the
  env + reward are wired.

Throughout: `H` = GNN hidden width (`hidden_dim`); `T` = number of tactic classes
(`num_tactics`); `B` = batch size; `s` = current proof-state DAG; `s'` = successor state
after applying an action; `V(s)` = critic scalar value; `A` = advantage; `π` = the tactic
policy (softmax over the actor's `T` logits); `STV` = the PLN strength/confidence truth
value used as reward.

---

## Symbol / component table

| Name | Location | Role |
|------|----------|------|
| `GraphSAGEStateClassifier` | `atp_lean_gnn/model.py` | GNN backbone; `encode_nodes`, `readout`, `classifier: Linear(H→T)` |
| `ActorHead` | `atp_lean_gnn/actor_critic.py:15` | Policy head over `T` tactics |
| `CriticHead` | `atp_lean_gnn/actor_critic.py:35` | Value head → scalar `V(s)` |
| `ArgumentSelector` | `atp_lean_gnn/argument_selector.py:26` | Pointer over node embeddings → argument logits |
| `ActorCriticWithArgsClassifier` | `atp_lean_gnn/actor_critic.py:51` | Backbone + actor + critic + tactic_embedding + argument_selector |
| `init_actor_from_supervised` | `atp_lean_gnn/actor_critic.py:150` | Copies `classifier` weights into the actor's final linear |
| `load_from_pointer_checkpoint` | `atp_lean_gnn/actor_critic.py:158` | Warm-start loader (uses `strict=False`) |
| `compute_actor_critic_combined_loss` | `atp_lean_gnn/actor_critic_loss.py:81` | `L = L_actor + c1·L_critic − c2·H(π) + w_arg·L_arg` |
| `train_one_epoch_actor_critic` | `atp_lean_gnn/actor_critic_training.py:61` | Per-epoch RL loop (currently double forward) |
| `RewardSource` / `MockRewardSource` / `PLNRewardSource` | `atp_lean_gnn/reward.py` | Reward provider; mock and PLN are currently identical stubs |

---

## Tier A — environment-independent edits

### A1. Zero-initialized residual actor (exact behavioral warm-start)

**Defect.** The intent is that the supervised tactic classifier `Linear(H→T)` becomes the
starting policy, and RL updates it from there. It does not. `ActorHead.net` is
`Linear(H→H) → ReLU → Dropout → Linear(H→T)` (`actor_critic.py:19`), and
`init_actor_from_supervised` copies the classifier only into the **final** layer `net[3]`
(`actor_critic.py:153`). The first layer `net[0]` stays randomly initialized, so at step 0

```
actor(s) = W₃ · ReLU(W₀ · s)   with W₀ random
        ≠ classifier(s)
```

The random first layer plus the ReLU destroys the supervised mapping before the copied
final layer sees it. The warm-start is nominal only. `test_load_from_pointer_checkpoint`
passes because it asserts tensor-equality of `net[3].weight`, not that the model reproduces
the supervised tactic distribution.

**Fix — residual actor with a zero-initialized correction branch.** Restructure `ActorHead`
so its output is the inherited linear map plus a nonlinear correction that starts at zero:

```
actor(s) = base(s) + residual(s)
  base:     Linear(H → T)                       # inherits classifier weights + bias
  residual: Linear(H → H) → ReLU → Linear(H → T) # final layer weight AND bias init to 0
```

At initialization `residual(s) = 0`, so `actor(s) ≡ base(s) ≡ classifier(s)` exactly,
bit-for-bit. RL then learns only the correction `residual`, growing it from zero. This
gives an exact behavioral warm-start and preserves the extra nonlinear capacity, and it
starts optimization from a known-good point rather than a random perturbation of one.

**Edits.**
- `actor_critic.py` — rewrite `ActorHead`:
  ```python
  class ActorHead(nn.Module):
      def __init__(self, hidden_dim, num_tactics, dropout=0.2):
          super().__init__()
          self.base = nn.Linear(hidden_dim, num_tactics)
          self.residual = nn.Sequential(
              nn.Linear(hidden_dim, hidden_dim),
              nn.ReLU(),
              nn.Dropout(dropout),
              nn.Linear(hidden_dim, num_tactics),
          )
          nn.init.zeros_(self.residual[3].weight)
          nn.init.zeros_(self.residual[3].bias)

      def forward(self, state_emb, mask=None):
          logits = self.base(state_emb) + self.residual(state_emb)
          if mask is not None:
              logits = logits.masked_fill(~mask, float("-inf"))
          return logits
  ```
- `init_actor_from_supervised` — copy classifier into `model.actor.base` (not `net[3]`):
  ```python
  model.actor.base.weight.copy_(model.backbone.classifier.weight)
  if model.backbone.classifier.bias is not None:
      model.actor.base.bias.copy_(model.backbone.classifier.bias)
  ```

**Alternative considered — single-layer actor.** Make `ActorHead` a bare `Linear(H→T)`.
Copying the classifier then gives exact equivalence with no residual machinery.
- Pro: simplest possible; exact by construction.
- Con: discards the extra MLP capacity the two-layer head was added for. The policy is
  then affine in the state embedding, identical in family to the supervised classifier —
  RL can only re-weight, not add nonlinear structure.
- Verdict: rejected in favor of the residual head, which is exact *and* keeps capacity.

**Test.** Replace the tensor-equality check in `test_load_from_pointer_checkpoint` with a
behavioral one: build the pointer model and the warm-started actor-critic on the same
inputs, assert `actor(state) == classifier(state)` (allclose) for a random batch, in
`eval()` mode so dropout is off.

---

### A2. Single forward pass per batch

**Defect.** `train_one_epoch_actor_critic` runs the full model twice per batch:
`model(batch)` at `actor_critic_training.py:105` to sample the tactic, then
`model(batch, teacher_tactic_ids=actions, ...)` at `:113` to obtain argument logits. The
GNN encode+readout is the expensive part and is paid twice; the `tactic_logits`/`values`
from pass 1 are the ones fed to the loss, while pass 2's are recomputed and discarded.

**Fix — split `forward` so the GNN runs once and the argument selector is conditioned on
the sampled action without re-encoding.** Give the model a method that returns the shared
encoder outputs plus the tactic/value heads, and a separate method that runs only the
pointer given a chosen tactic id:

```python
def encode(self, data, *, tactic_mask=None):
    node_embeddings = self.backbone.encode_nodes(data)   # [total_nodes, H]
    state_emb = self.backbone.readout(node_embeddings, data)  # [B, H]
    tactic_logits = self.actor(state_emb, mask=tactic_mask)
    values = self.critic(state_emb)
    return node_embeddings, state_emb, tactic_logits, values

def select_arguments(self, data, node_embeddings, state_emb, tactic_ids, tactic_names=None):
    # existing steps 5-7 of forward(), using the passed-in encoder outputs
    ...
    return arg_logits_list
```

Training loop becomes:

```python
node_emb, state_emb, tactic_logits, values = model.encode(batch, tactic_mask=mask)
actions = torch.distributions.Categorical(logits=tactic_logits).sample()
arg_logits_list = model.select_arguments(batch, node_emb, state_emb, actions, tactic_names)
```

One GNN forward per batch; the sampled action conditions the pointer directly. Keep the
current `forward(...)` as a thin wrapper (`encode` then `select_arguments` with
argmax/teacher ids) so `inference.py` and the tests that call `model(batch)` keep working —
this is a refactor of the internal call graph, not a change to the external signature.

**Edits.** `actor_critic.py` (`encode`, `select_arguments`, `forward` wrapper);
`actor_critic_training.py` (`train_one_epoch_actor_critic` and
`evaluate_model_actor_critic` both use the two-method form).

---

### A3. Checkpoint guard on warm-start load

**Defect.** `load_from_pointer_checkpoint` calls
`model.load_state_dict(new_state_dict, strict=False)` (`actor_critic.py:177`). `strict=False`
silently swallows both missing keys and **shape mismatches**. The actor-critic config
`configs/actor_critic_graphsage_state.json` sets `hidden_dim=512`, while the supervised
pointer default is `128`. If the checkpoint's `H` and the model's `H` disagree, every
backbone and pointer tensor is silently skipped and training proceeds from random
initialization while reporting a successful warm-start. A warm-start that silently did not
happen is worse than a hard error.

**Fix.** Capture the `load_state_dict` return (`missing_keys`, `unexpected_keys`) and,
before that, explicitly diff the shapes of the keys we intend to transfer. Raise on any
intended key whose source and destination shapes differ; log any intended key absent from
the destination.

```python
result = model.load_state_dict(new_state_dict, strict=False)
model_sd = model.state_dict()
mismatched = [
    (k, tuple(v.shape), tuple(model_sd[k].shape))
    for k, v in new_state_dict.items()
    if k in model_sd and v.shape != model_sd[k].shape
]
if mismatched:
    raise ValueError(
        "Warm-start shape mismatch (hidden_dim disagreement between checkpoint and "
        f"model config?): {mismatched}"
    )
console_print(f"Warm-start: loaded {len(new_state_dict)} tensors; "
              f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
```

Note that a `v.shape != model_sd[k].shape` entry is otherwise dropped by `strict=False`
without appearing in `missing_keys`, which is exactly why the explicit shape diff is
required rather than inspecting the return value alone.

**Edits.** `actor_critic.py` (`load_from_pointer_checkpoint`).

**Test.** Save a pointer checkpoint at `H=128`, attempt to warm-start a model built at
`H=256`, assert `ValueError` is raised.

---

### A4. Wire action masking into sampling and entropy

**Defect.** The masking plumbing exists but is never connected. `ActorHead.forward` accepts
`mask` and `compute_entropy_bonus` accepts `mask`, but the training loop calls the model
with no `tactic_mask` (`actor_critic_training.py:105`, `:113`) and
`compute_actor_critic_combined_loss` with no `tactic_mask` (`:128`). Illegal tactics are
never set to `−inf`, so the policy samples tactics that cannot apply at state `s`, and the
entropy bonus `−c2·H(π)` actively rewards spreading probability mass onto illegal actions.

**Fix.** Thread a per-example legal-action mask `M ∈ {0,1}^{B×T}` from the applicability of
each tactic at state `s` through both the actor and the entropy term:
- `model.encode(batch, tactic_mask=M)` → masked logits for sampling.
- `compute_actor_critic_combined_loss(..., tactic_mask=M)` → masked entropy.

The **mask source** is an open question in the integration outline (§6). Until the Lean env
can report applicability, provide the mask via a `legal_action_mask` field on the batch (all-
ones when unknown, so behavior is unchanged) so the wiring lands now and the real signal
drops in later without further code changes. This is the "wire masking" step: no new
logic, only connecting the parameter that already exists to the training and loss calls.

**Edits.** `actor_critic_training.py` (build/pass `M` in train and eval);
`compute_actor_critic_combined_loss` call sites.

---

### A5. Advantage normalization

**Defect / scope.** With one-step 0/1 reward this barely matters, but under multi-step
discounted returns (Tier B) the advantage `A` has a wide dynamic range and raw
policy-gradient variance grows large. `compute_actor_critic_combined_loss` currently forms
`advantages = returns − value_estimates.squeeze(-1).detach()` (`actor_critic_loss.py:95`)
with no normalization.

**Fix.** Normalize the **detached** advantage per batch before it multiplies the log-prob,
leaving the critic's regression target unnormalized (the critic must still regress the true
returns):

```python
adv = returns - value_estimates.squeeze(-1).detach()
adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # actor only
actor_loss = compute_actor_loss(tactic_logits, actions, adv)
critic_loss = compute_critic_loss(value_estimates, returns)  # unnormalized returns
```

Guard the single-example / zero-variance case (`std → 0`) with the `+ 1e-8`. Optionally
maintain a running mean/std of returns to normalize the critic target when reward scale
drifts across training; defer that until reward magnitudes are observed.

**Edits.** `actor_critic_loss.py` (`compute_actor_critic_combined_loss`).

---

## Tier B — environment-dependent edits (specify now, implement once env + reward are live)

### B1. Reward source: retire the mock as a training default

`MockRewardSource` and `PLNRewardSource` both currently return `(actions == targets).float()`
(`reward.py`), i.e. a label-match reward. **As long as the reward is `actions == targets`,
this is REINFORCE-shaped imitation, not RL** — the gradient rewards reproducing the corpus
label, not closing the goal. `train_actor_critic` instantiates `MockRewardSource`
(`training.py:1498`).

- Make the reward source config-selected (`reward_source: "mock" | "pln"`), defaulting to
  `pln` once connected. Keep `MockRewardSource` strictly as a unit-test fixture.
- `PLNRewardSource.get_rewards_batch` must call the MeTTa interpreter on `STV(s')` — the
  successor state produced by the Lean env — not `success_mask`.

### B2. Pointer as part of the actor (joint policy gradient on arguments)

**Current behavior.** `compute_argument_rl_loss` (`actor_critic_loss.py:39`) is cross-entropy
of the argument logits toward the pointer's **own** argmax indices, gated by `success_mask`;
arguments are selected by `argmax` (`actor_critic_training.py:122`). Training a distribution
toward its own argmax merely sharpens current behavior on already-successful examples — the
reward magnitude never enters and there is no exploration, so the pointer can never discover
an argument better than the one it already prefers.

**Fix (once env is live).** Treat the action as the pair `(tactic, args)`. At each argument
step `k`, **sample** the argument from `Categorical(logits=arg_logits_k)` instead of taking
`argmax`, record `log π(arg_k)`, and add

```
L_arg = −( Σ_k log π(arg_k) ) · A.detach()
```

to the actor loss — the same policy-gradient form already used for the tactic in
`compute_actor_loss`. The pointer is then trained by the same advantage `A` (from
`STV(s')`) as the tactic head, learning arguments that raise the successor value rather than
arguments that match the label. Retain the differential learning rate already in
`build_param_groups` (0.1× on `argument_selector` + `tactic_embedding`) so the warm-started
pointer adapts slowly rather than being overwritten early.

**Alternatives.**
- *Freeze the pointer* (RL touches only actor+critic). Pro: stable. Con: the argument policy
  is frozen at the imitation optimum and can never improve from Lean feedback — defeats the
  purpose of adding RL to argument selection.
- *Keep the self-argmax cross-entropy* (current). Con: not policy gradient; no exploration;
  reward-blind. Rejected.
- Verdict: joint advantage-weighted log-prob of the *sampled* argument.

### B3. Multi-step credit assignment (replace one-step bandit)

A proof is a sequence, so the current one-step model (`returns = reward`, no bootstrap) is
the wrong episode structure. Three additions:

1. **Rollout collector** (`rollout_collector.py`, specified in the integration outline and
   dropped from the implementation plan — it must return). From `s₀`, predict `(tactic,
   args)`, apply via the Lean env → `s₁`, obtain `r₁ = STV(s₁)`, repeat to QED / failure /
   step budget. Collect `(sₜ, aₜ, rₜ, sₜ₊₁, doneₜ)`. Multi-step credit cannot be expressed
   inside the current `for batch in loader` supervised loop, because the states are no
   longer independent draws — they are produced by the policy itself.
2. **Bootstrapped critic target.** Replace `returns = r` with the TD target
   `yₜ = rₜ + γ·V(sₜ₊₁)·(1 − doneₜ)` and advantage `Aₜ = yₜ − V(sₜ)`, with `γ ≈ 0.99`.
   The critic and `V(s)` already exist; the change is to stop treating `returns` as the raw
   reward and start bootstrapping the successor value. Start with one-step TD; add GAE(λ)
   only if variance warrants it.
3. **Terminal / failure reward** (open question, outline §6). Set QED = +1, illegal/failed
   tactic = 0 or small negative, and a small per-step penalty to prefer shorter proofs.
   Per-step `STV(s')` is the dense shaping signal between terminals.

Because the env step carries the cost (PLN latency, Lean invocation), the collector should
batch/parallelize env steps, and rollout and update will run in alternating phases rather
than per-minibatch.

---

## Dependency ordering

```
A3 (guard) ─┐
A1 (residual warm-start) ─┼─ independent, do first, any order
A5 (adv norm) ─┘
A2 (single forward) ──────── refactors the loop A4/B2 build on
A4 (wire masking) ────────── needs A2's encode(tactic_mask=…) path
                                   │
                       (Lean env + PLN reward land)
                                   │
B1 (reward source) ──── B3 (multi-step rollout) ──── B2 (pointer-as-actor)
```

Tier A is safe to implement and merge before the env exists; none of it changes results
under the current mock reward except A1 (which makes warm-start real) and A3 (which turns a
silent no-op into an error). Tier B waits for `LeanEnv` + `PLNRewardSource`.

---

## Test plan

| Edit | Test |
|------|------|
| A1 | `actor(state) ≈ classifier(state)` (allclose, eval mode) right after warm-start |
| A2 | GNN `encode_nodes` called once per batch (spy/counter); loss unchanged vs. old two-pass on a fixed seed within tolerance |
| A3 | warm-start across mismatched `hidden_dim` raises `ValueError` |
| A4 | masked tactic positions have `−inf` logits and zero sampling probability; entropy ignores masked classes |
| A5 | normalized advantage has ≈0 mean, ≈1 std per batch; zero-variance batch does not divide by zero |
| B1 | `pln` reward source selected by config; mock only in fixtures |
| B2 | argument log-prob gradient flows into `argument_selector` under a nonzero advantage |
| B3 | one-step trajectory reproduces the bandit target; `done` masks the bootstrap term |

Run: `uv run python -m pytest maths_ai/gnn_inference/tests/test_actor_critic.py -q`
(plain `python` is permission-denied in this repo; use `uv run`).

---

## Out of scope

- The `LeanEnv` implementation itself (successor-state construction, Lean invocation).
- The PLN/MeTTa `STV` computation.
- Argument-space masking beyond premise-node validity (the pointer already masks non-premise
  nodes via `premise_mask`).
