# Implementation Plan: Actor-Critic RL Integration into GNN Architecture

## 0. Context & Resolved Design Decisions

This plan integrates Actor-Critic RL into the existing GNN-based tactic
prediction system. It is based on the design in
[actor_critic_gnn_integration_plan.md](file:///home/nolawi/main-maths/maths_ai/gnn_inference/actor_critic_gnn_integration_plan.md)
and incorporates the following resolved decisions:

| Decision | Resolution |
|----------|------------|
| **Checkpoint loading** | The actor-critic model supports loading weights from an existing pointer (`TacticWithArgsClassifier`) checkpoint. The backbone, tactic embeddings, and argument selector are initialized from the pretrained weights; the actor head is warm-started from `backbone.classifier`. |
| **Argument selector in RL rollouts** | The argument selector runs during RL rollouts. Both tactic and arguments are required to interact with Lean — the full `(tactic, arguments)` pair is applied in Lean, the resulting proof state is sent to PLN for STV computation. |
| **Argument head training strategy** | **Hybrid**: pretrained initialization (from pointer checkpoint) + joint RL training with a differential learning rate (0.1× multiplier for argument selector parameters). The argument loss term remains in `L_total` using the Lean environment outcome as signal. |

---

## 1. Current Architecture Summary

The codebase has two model variants, both using supervised imitation learning:

```
GraphSAGEStateClassifier (model.py)
├── label_embedding          — nn.Embedding(num_node_labels, H)
├── node_type_embedding      — nn.Embedding(num_node_types, H)
├── convs                    — ModuleList[SAGEConv] × num_layers
├── dropout                  — nn.Dropout
├── classifier               — nn.Linear(H, num_tactics)       ← tactic head
├── encode_nodes(data)       → [total_nodes, H]
├── readout(node_embs, data) → [B, H]   (state_node_index select)
└── forward(data)            → [B, num_tactics]

TacticWithArgsClassifier (argument_selector.py)
├── backbone                 — GraphSAGEStateClassifier (shared encoder)
├── tactic_embedding         — nn.Embedding(num_tactics, H)
├── argument_selector        — ArgumentSelector (pointer head)
├── forward(data, ...)       → (tactic_logits [B, T], arg_logits_list [list of [B, N_max]])
```

The existing loss ([compute_combined_loss](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/argument_selector.py#L303-L368)):
```
L_supervised = L_tactic_CE + w_arg × L_arg_CE
```

Training pipelines:
- [train_baseline](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/training.py#L835-L1039) — trains `GraphSAGEStateClassifier`
- [train_pointer](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/training.py#L1042-L1254) — trains `TacticWithArgsClassifier`

**None of these are modified by this implementation.**

---

## 2. Target Architecture

```
ActorCriticWithArgsClassifier (actor_critic.py)  [NEW]
├── backbone                 — GraphSAGEStateClassifier (shared encoder, unchanged)
├── actor                    — ActorHead MLP (replaces backbone.classifier for tactic selection)
├── critic                   — CriticHead MLP (new — predicts V(s) ≈ STV)
├── tactic_embedding         — nn.Embedding(num_tactics, H) (same role as in TacticWithArgsClassifier)
├── argument_selector        — ArgumentSelector (same pointer head, jointly trained)
├── forward(data, ...)       → (tactic_logits [B, T], value [B, 1], arg_logits_list [...])
```

```
DAG ──→ [Shared GNN Encoder] ──→ node_embeddings
                                       │
                                       ▼
                               readout(state_node_index)
                                       │
                                       ▼
                                   state_emb (h_s)
                                    ┌───┴───┐
                                    │       │
                                    ▼       ▼
                              [ActorHead] [CriticHead]
                                    │       │
                                    ▼       ▼
                              π(a|s)      V(s)
                                    │
                                    ▼
                          tactic_embedding(a)
                                    │
                                    ▼
                          [ArgumentSelector]  ← also receives node_embeddings
                                    │
                                    ▼
                            arg_logits_list
```

---

## 3. Combined Loss Function

```
L_total = L_actor + c1 · L_critic − c2 · H(π) + w_arg · L_arg
```

| Term | Formula | Default Weight | Gradient Targets |
|------|---------|----------------|-----------------|
| `L_actor` | `−log π(a\|s) · A.detach()` | 1.0 | Actor MLP → shared encoder |
| `L_critic` | `MSE(V(s), R)` | c1 = 0.5 | Critic MLP → shared encoder |
| `H(π)` | Entropy of `π(·\|s)` | c2 = 0.01 | Actor MLP → shared encoder |
| `L_arg` | Cross-entropy on argument positions | w_arg = 0.5 | Argument selector → shared encoder |

Where:
- `A = R − V(s)` — one-step advantage (detached before entering actor loss)
- `R` — PLN STV reward from the MeTTa interpreter for the resulting proof state

**Gradient isolation rules:**
- `A` is `.detach()`-ed — actor loss does NOT backprop into the critic
- `V(s)` is only used in `L_critic` — critic gradients are independent of actor
- `L_arg` gradients flow through the argument selector AND the shared encoder
- All four loss terms contribute gradients to the shared GNN encoder via a single `backward()` call

**Differential learning rates:**
- Actor head, Critic head: base learning rate (e.g. `1e-3`)
- Shared encoder (backbone): base learning rate
- Argument selector + tactic_embedding: `0.1 × base_lr` (slower adaptation to preserve pretrained quality)

---

## 4. RL Rollout Flow

Each training step during RL follows this sequence:

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Forward Pass                                            │
│   data (proof-state DAG) → model.forward(data)                  │
│   → tactic_logits π(a|s), value V(s), arg_logits_list           │
│                                                                 │
│ Step 2: Action Sampling                                         │
│   Sample tactic a ~ Categorical(softmax(tactic_logits))         │
│   Select arguments via argmax on arg_logits (or sample)         │
│   Cache: log π(a|s), V(s), selected args                       │
│                                                                 │
│ Step 3: Environment Step                                        │
│   Apply (tactic_name, [arg1, arg2, ...]) in Lean                │
│   Outcomes:                                                     │
│     ✓ Valid step      → new DAG state s'                        │
│     ✓ Proof complete  → terminal, episode ends                  │
│     ✗ Tactic error    → terminal, penalty reward                │
│                                                                 │
│ Step 4: PLN Reward                                              │
│   Send resulting proof state s' to PLN (MeTTa interpreter)      │
│   R = STV(s')                                                   │
│   (Terminal failure: R = 0.0)                                   │
│                                                                 │
│ Step 5: Advantage & Loss                                        │
│   A = R − V(s).detach()                                         │
│   L_total = −log π(a|s) · A                                     │
│           + c1 · MSE(V(s), R)                                   │
│           − c2 · H(π)                                           │
│           + w_arg · L_arg                                        │
│                                                                 │
│ Step 6: Backprop                                                │
│   L_total.backward()                                            │
│   clip_grad_norm_(all params)                                   │
│   optimizer.step() (param groups with differential LR)          │
└─────────────────────────────────────────────────────────────────┘
```

**Argument loss signal during RL rollouts:**
Since there are no ground-truth argument labels during RL, the argument loss
uses the Lean environment outcome as signal:
- If the `(tactic, arguments)` pair succeeds in Lean → the argument cross-entropy
  loss is computed with the selected argument indices as pseudo-targets (REINFORCE
  on the argument selection)
- If it fails → no argument loss for this step (the actor loss already penalizes
  the failed tactic choice)

---

## 5. New Files

### 5.1 `atp_lean_gnn/actor_critic.py`

The core model module. Contains:

**`ActorHead(nn.Module)`**
```python
class ActorHead(nn.Module):
    """Policy head: state embedding → tactic logits."""
    def __init__(self, hidden_dim: int, num_tactics: int, dropout: float = 0.2):
        # Linear(H, H) → ReLU → Dropout → Linear(H, num_tactics)

    def forward(self, state_emb: Tensor, mask: Tensor | None = None) -> Tensor:
        # Returns [B, num_tactics] logits
        # If mask is provided, sets masked positions to -inf before returning
```

**`CriticHead(nn.Module)`**
```python
class CriticHead(nn.Module):
    """Value head: state embedding → scalar V(s)."""
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        # Linear(H, H) → ReLU → Dropout → Linear(H, 1)

    def forward(self, state_emb: Tensor) -> Tensor:
        # Returns [B, 1] value estimates
```

**`ActorCriticWithArgsClassifier(nn.Module)`**
```python
class ActorCriticWithArgsClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_node_labels: int,
        num_tactics: int,
        num_node_types: int = len(NODE_TYPE_TO_ID),
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_node_type: bool = True,
        max_args: int = 3,
    ):
        self.backbone = GraphSAGEStateClassifier(...)   # shared encoder
        self.actor = ActorHead(hidden_dim, num_tactics, dropout)
        self.critic = CriticHead(hidden_dim, dropout)
        self.tactic_embedding = nn.Embedding(num_tactics, hidden_dim)
        self.argument_selector = ArgumentSelector(hidden_dim)
        self.max_args = max_args
        self.hidden_dim = hidden_dim

    def forward(
        self,
        data,
        *,
        tactic_mask: Tensor | None = None,
        teacher_tactic_ids: Tensor | None = None,
        tactic_names: list[str] | None = None,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Return (tactic_logits, value_estimates, arg_logits_list)."""
        # 1. Encode: node_embeddings = backbone.encode_nodes(data)
        # 2. Readout: state_emb = backbone.readout(node_embeddings, data)
        # 3. Actor:   tactic_logits = self.actor(state_emb, mask=tactic_mask)
        # 4. Critic:  values = self.critic(state_emb)
        # 5. Tactic embedding for argument selection (teacher-forced or argmax)
        # 6. Autoregressive argument selection (same logic as TacticWithArgsClassifier)
        return tactic_logits, values, arg_logits_list
```

**`init_actor_from_supervised(model: ActorCriticWithArgsClassifier)`**
- Copies `model.backbone.classifier.weight` → `model.actor` final linear layer
- Enables warm-start from supervised pretraining

**`load_from_pointer_checkpoint(model, checkpoint_path, device)`**
- Loads a `TacticWithArgsClassifier` checkpoint
- Maps weights to the actor-critic model:
  - `backbone.*` → `backbone.*` (direct copy)
  - `backbone.classifier` → `actor` final layer (warm-start)
  - `tactic_embedding` → `tactic_embedding` (direct copy)
  - `argument_selector.*` → `argument_selector.*` (direct copy)
- Critic head is randomly initialized (no supervised equivalent)

---

### 5.2 `atp_lean_gnn/actor_critic_loss.py`

Dedicated loss module — does NOT modify `argument_selector.compute_combined_loss`.

**`compute_actor_loss(tactic_logits, actions, advantages)`**
```python
def compute_actor_loss(
    tactic_logits: Tensor,  # [B, num_tactics]
    actions: Tensor,        # [B] sampled tactic ids
    advantages: Tensor,     # [B] detached advantages
) -> Tensor:
    log_probs = F.log_softmax(tactic_logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    return -(selected_log_probs * advantages.detach()).mean()
```

**`compute_critic_loss(predicted_values, returns)`**
```python
def compute_critic_loss(
    predicted_values: Tensor,  # [B, 1]
    returns: Tensor,           # [B]
) -> Tensor:
    return F.mse_loss(predicted_values.squeeze(-1), returns)
```

**`compute_entropy_bonus(tactic_logits, mask=None)`**
```python
def compute_entropy_bonus(
    tactic_logits: Tensor,           # [B, num_tactics]
    mask: Tensor | None = None,      # [B, num_tactics] bool
) -> Tensor:
    # Apply mask if provided (set masked logits to -inf)
    # Compute: H = -sum(p * log(p)) over valid actions
    # Return mean entropy across batch
```

**`compute_argument_rl_loss(arg_logits_list, selected_arg_indices, success_mask)`**
```python
def compute_argument_rl_loss(
    arg_logits_list: list[Tensor],       # per-step arg logits
    selected_arg_indices: list[Tensor],  # per-step selected arg positions
    success_mask: Tensor,                # [B] bool — did the (tactic, args) succeed in Lean?
) -> Tensor:
    # Only compute loss for samples where the tactic+args succeeded
    # Cross-entropy with selected indices as pseudo-targets
    # Returns scalar loss
```

**`compute_actor_critic_combined_loss(...)`** — the main entry point:
```python
def compute_actor_critic_combined_loss(
    tactic_logits: Tensor,
    value_estimates: Tensor,
    arg_logits_list: list[Tensor],
    actions: Tensor,
    returns: Tensor,
    selected_arg_indices: list[Tensor],
    success_mask: Tensor,
    *,
    tactic_mask: Tensor | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    arg_loss_weight: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    advantages = returns - value_estimates.squeeze(-1).detach()

    actor_loss = compute_actor_loss(tactic_logits, actions, advantages)
    critic_loss = compute_critic_loss(value_estimates, returns)
    entropy = compute_entropy_bonus(tactic_logits, mask=tactic_mask)
    arg_loss = compute_argument_rl_loss(arg_logits_list, selected_arg_indices, success_mask)

    total = actor_loss + critic_weight * critic_loss - entropy_weight * entropy + arg_loss_weight * arg_loss

    metrics = {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(entropy.item()),
        "arg_loss": float(arg_loss.item()),
        "total_loss": float(total.item()),
        "mean_advantage": float(advantages.mean().item()),
        "mean_value": float(value_estimates.mean().item()),
        "mean_return": float(returns.mean().item()),
    }
    return total, metrics
```

---

### 5.3 `atp_lean_gnn/actor_critic_training.py`

Training and evaluation loops for actor-critic. Mirrors the structure of
[argument_training.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/argument_training.py)
but uses the actor-critic loss. The existing `argument_training.py` is **not modified**.

**`build_param_groups(model, base_lr, arg_lr_multiplier=0.1)`**
```python
def build_param_groups(
    model: ActorCriticWithArgsClassifier,
    base_lr: float,
    arg_lr_multiplier: float = 0.1,
) -> list[dict]:
    """Create optimizer parameter groups with differential learning rates.

    - Actor head, Critic head, shared encoder: base_lr
    - Argument selector + tactic_embedding: base_lr * arg_lr_multiplier
    """
    arg_param_ids = set(
        id(p) for p in list(model.argument_selector.parameters())
                      + list(model.tactic_embedding.parameters())
    )
    return [
        {"params": [p for p in model.parameters() if id(p) not in arg_param_ids],
         "lr": base_lr},
        {"params": [p for p in model.parameters() if id(p) in arg_param_ids],
         "lr": base_lr * arg_lr_multiplier},
    ]
```

**`train_one_epoch_actor_critic(model, loader, *, reward_source, optimizer, ...)`**
- Iterates over batches from the DataLoader
- Runs `model.forward(batch)` → `(tactic_logits, values, arg_logits_list)`
- Samples tactic actions from `Categorical(softmax(tactic_logits))`
- Selects arguments via argmax on `arg_logits_list`
- Queries `reward_source.get_rewards_batch(...)` for PLN STV rewards
- Computes `compute_actor_critic_combined_loss(...)`
- Backward pass with AMP and gradient clipping
- Logs per-batch: actor_loss, critic_loss, entropy, arg_loss, combined_loss
- Returns epoch-level aggregated metrics

**`evaluate_model_actor_critic(model, loader, *, reward_source, ...)`**
- `torch.no_grad()` evaluation loop
- Tracks: combined loss, tactic top-1 accuracy, critic MSE, mean policy entropy, 
  mean advantage, mean value, mean return

---

### 5.4 `atp_lean_gnn/reward.py`

Reward source abstraction layer.

**`RewardSource(ABC)`**
```python
class RewardSource(ABC):
    @abstractmethod
    def get_reward(self, state: Any, action: int, next_state: Any) -> float:
        """Return the STV reward for a single (state, action, next_state) transition."""
        ...

    @abstractmethod
    def get_rewards_batch(
        self,
        actions: Tensor,
        tactic_targets: Tensor,
        success_mask: Tensor,
    ) -> Tensor:
        """Compute rewards for an entire batch. Returns [B] tensor of rewards."""
        ...
```

**`MockRewardSource(RewardSource)`**
- For offline supervised RL training / testing the loop before Lean+PLN are wired in
- `get_rewards_batch(actions, tactic_targets, success_mask)`:
  - Returns `1.0` where `actions == tactic_targets`
  - Returns `0.0` elsewhere
- This allows end-to-end testing of the training loop against the existing
  supervised dataset

**`PLNRewardSource(RewardSource)`** (stub / interface)
- Calls the MeTTa PLN interpreter with the resulting proof state
- Returns `STV(s')` as the reward
- Includes a caching layer for repeated `(state, action, next_state)` triples
- Actual implementation depends on the PLN/MeTTa integration (out of scope for
  this plan, but the interface is defined here)

---

### 5.5 `configs/actor_critic_graphsage_state.json`

```json
{
  "prepared_root": "artifacts/prepared/v1",
  "run_root": "runs/actor_critic_gnn",
  "seed": 42,
  "device": "auto",
  "edge_mode": "bidirectional",
  "use_node_type": true,
  "max_args": 3,
  "arg_loss_weight": 0.5,
  "critic_weight": 0.5,
  "entropy_weight": 0.01,
  "arg_lr_multiplier": 0.1,
  "pretrained_pointer_checkpoint": null,
  "model": {
    "hidden_dim": 512,
    "num_layers": 4,
    "dropout": 0.2,
    "max_args": 3,
    "arg_loss_weight": 0.5
  },
  "training": {
    "batch_size": 256,
    "epochs": 20,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "log_every_batches": 100,
    "num_workers": 12,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "use_amp": true
  }
}
```

---

### 5.6 `tests/test_actor_critic.py`

| Test | What it verifies |
|------|-----------------|
| `test_actor_head_output_shape` | `ActorHead` produces `[B, num_tactics]` |
| `test_critic_head_output_shape` | `CriticHead` produces `[B, 1]` |
| `test_action_masking` | Masked logit positions are `−inf`, unmasked are finite |
| `test_full_model_forward` | `ActorCriticWithArgsClassifier.forward()` returns `(tactic_logits, values, arg_logits_list)` with correct shapes |
| `test_argument_selector_unchanged` | The argument selector inside actor-critic produces identical output to a standalone `TacticWithArgsClassifier` given the same inputs |
| `test_combined_loss_components` | All four loss terms are finite scalars; `total_loss` is their weighted sum |
| `test_gradient_flow_actor` | Actor head params have non-zero `.grad` after `total_loss.backward()` |
| `test_gradient_flow_critic` | Critic head params have non-zero `.grad` |
| `test_gradient_flow_encoder` | Backbone encoder params have non-zero `.grad` |
| `test_gradient_flow_arg_selector` | Argument selector params have non-zero `.grad` |
| `test_advantage_detached` | Critic params receive zero grad from actor loss alone (advantage is detached) |
| `test_differential_lr_param_groups` | `build_param_groups` assigns correct LR to each group |
| `test_load_from_pointer_checkpoint` | Loading a `TacticWithArgsClassifier` checkpoint into `ActorCriticWithArgsClassifier` produces matching backbone weights |
| `test_init_actor_from_supervised` | `backbone.classifier.weight` matches `actor` final layer weight after `init_actor_from_supervised` |
| `test_mock_reward_source` | `MockRewardSource` returns `1.0` for correct actions, `0.0` for incorrect |
| `test_training_step_e2e` | One full training step (forward → loss → backward → optimizer.step) completes without error |

---

## 6. Modified Files

### 6.1 `atp_lean_gnn/training.py` — additive changes only

**New dataclass: `ActorCriticConfig`** (after `PointerConfig`, ~line 270)
```python
@dataclass(frozen=True)
class ActorCriticConfig:
    prepared_root: Path
    run_root: Path
    seed: int = 42
    device: str = "auto"
    edge_mode: str = "bidirectional"
    use_node_type: bool = True
    max_args: int = 3
    arg_loss_weight: float = 0.5
    critic_weight: float = 0.5
    entropy_weight: float = 0.01
    arg_lr_multiplier: float = 0.1
    pretrained_pointer_checkpoint: str | None = None
    model: TacticWithArgsConfig = field(default_factory=TacticWithArgsConfig)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)

    @classmethod
    def from_dict(cls, payload: dict) -> "ActorCriticConfig": ...
    def normalized(self) -> "ActorCriticConfig": ...
    def to_dict(self) -> dict: ...
```

**New functions:**
- `load_actor_critic_config(config_path, *, overrides...)` — same pattern as `load_pointer_config`
- `build_actor_critic_model(metadata, config)` → `ActorCriticWithArgsClassifier`
- `train_actor_critic(config, *, resume_run_dir=None)` — top-level training entrypoint
  - Builds model, optimizer (with `build_param_groups` for differential LR), grad scaler
  - If `config.pretrained_pointer_checkpoint` is set, calls `load_from_pointer_checkpoint`
  - Delegates epoch training to `actor_critic_training.train_one_epoch_actor_critic`
  - Saves checkpoints, writes metrics.jsonl, eval files, summary.json

**Modified function: `train_main()`** (~line 1357)
- Add `"actor_critic"` to `--model-type` choices:
  ```python
  parser.add_argument(
      "--model-type",
      choices=["baseline", "pointer", "actor_critic"],
      ...
  )
  ```
- Add `elif model_type == "actor_critic":` branch that calls `train_actor_critic`

**Modified function: `_save_checkpoint()`** (~line 803)
- Update type annotation to include `ActorCriticWithArgsClassifier`:
  ```python
  model: GraphSAGEStateClassifier | TacticWithArgsClassifier | ActorCriticWithArgsClassifier,
  ```

No existing functions are deleted or have their behavior changed.

---

### 6.2 `inference_engine.py` — backward-compatible extension

**`GNNModelEngine.__init__()`** — add optional `model_type` parameter:
```python
def __init__(
    self,
    config_path: Path,
    tactic_predictor_model_path: Path,
    argument_predictor_model_path: Path,
    *,
    model_type: str = "pointer",   # NEW — "pointer" or "actor_critic"
    ...
):
```

When `model_type="actor_critic"`:
- Instantiate `ActorCriticWithArgsClassifier` instead of `TacticWithArgsClassifier`
- Load checkpoint using `load_from_pointer_checkpoint` or direct `load_state_dict`

The `inference()` method needs no change — it accesses `self.gnn_inference` which
calls `predict_tactics_with_arguments()`, and the `GNNPredictor` interface is
compatible with both model types.

---

### 6.3 `model.py` (outer `gnn_inference/model.py`) — minor update

**`GNNPredictor`** — update to handle the extra return value:
```python
def predict_tactics_with_arguments(self, goal_expression: str, top_k: int = 3):
    result = self.pipeline.predict_tactic_result(goal_expression, top_k=top_k)
    return result.top_tactic_predictions
```

The `InferencePipeline` in [inference.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/inference.py)
accesses `model.backbone.encode_nodes()`, `model.backbone.readout()`, and
`model.backbone.classifier()` directly. For actor-critic, the pipeline should
use `model.actor()` instead of `model.backbone.classifier()` for tactic logits.

Add a conditional in `InferencePipeline.predict_tactic_result()`:
```python
# Line ~159
if hasattr(self.model, 'actor'):
    tactic_logits = self.model.actor(state_emb)
else:
    tactic_logits = self.model.backbone.classifier(state_emb)
```

---

## 7. Files NOT Modified

| File | Reason |
|------|--------|
| [argument_selector.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/argument_selector.py) | `ArgumentSelector`, `TacticWithArgsClassifier`, `compute_combined_loss` — all untouched. The actor-critic model imports `ArgumentSelector` but doesn't change it. |
| [argument_training.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/argument_training.py) | Supervised argument training loops remain as-is |
| [model.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/model.py) (inner) | `GraphSAGEStateClassifier` backbone is used unchanged |
| [premise_scoring.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/premise_scoring.py) | Premise scorer is orthogonal |
| [premise_training.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/premise_training.py) | Premise training pipeline unchanged |
| [graph.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/graph.py) | DAG construction unchanged |
| [pyg.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/pyg.py) | PyG conversion unchanged |
| [labels.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/labels.py) | Tactic vocab/arity unchanged |
| [dataset.py](file:///home/nolawi/main-maths/maths_ai/gnn_inference/atp_lean_gnn/dataset.py) | Dataset logic unchanged |
| All existing test files | No modifications to existing tests |

---

## 8. Implementation Order

| Phase | Files | Depends On | Validates |
|-------|-------|------------|-----------|
| **1. Model heads** | `actor_critic.py` | — | Output shapes, action masking, checkpoint loading |
| **2. Loss functions** | `actor_critic_loss.py` | Phase 1 | Loss components, gradient flow, advantage detachment |
| **3. Reward abstraction** | `reward.py` | — | Mock rewards for testing |
| **4. Training loop** | `actor_critic_training.py` | Phases 1–3 | End-to-end training step, differential LR |
| **5. Config & integration** | `training.py` additions, config JSON | Phases 1–4 | Full training run with `--model-type actor_critic` |
| **6. Inference support** | `inference_engine.py`, `model.py` (outer), `inference.py` | Phase 1 | Actor-critic model produces same inference output format |
| **7. Tests** | `tests/test_actor_critic.py` | All phases | Full test suite passes |

---

## 9. Verification Plan

### Automated Tests
```bash
# Phase 1-3: Unit tests for model, loss, reward
python -m pytest maths_ai/gnn_inference/tests/test_actor_critic.py -v -k "head or loss or reward"

# Phase 4-5: Full training loop smoke test
python -m pytest maths_ai/gnn_inference/tests/test_actor_critic.py -v -k "training_step"

# Phase 7: Full suite + regression check
python -m pytest maths_ai/gnn_inference/tests/ -v
```

### Manual Verification
1. **Gradient sanity**: Run 5 training steps with `torch.autograd.set_detect_anomaly(True)`.
   Confirm no anomalous gradients, both heads receive non-zero gradients, and the
   argument selector receives attenuated gradients (proportional to `arg_lr_multiplier`).

2. **Checkpoint round-trip**: Save and reload an actor-critic checkpoint. Verify that
   `model.eval()` inference produces identical outputs before and after reload.

3. **Pointer checkpoint loading**: Load a pretrained pointer checkpoint into the
   actor-critic model. Run inference and confirm tactic predictions are similar
   to the original pointer model (they won't be identical because the actor head
   architecture differs from `nn.Linear`, but top-k tactics should overlap).

4. **Training stability**: Run 3 epochs with `MockRewardSource` on the existing
   dataset. Confirm that loss decreases monotonically and entropy doesn't collapse
   to zero.
