"""End-to-end PLN actor-critic RL: collect search trees (async) → train (sync).

Two phases, cleanly separated by the ``asyncio.run`` boundary:

  * **collect** — run the hybrid reasoner's AND-OR search (async, PLN de-blocked via
    ``evaluate_async``) to produce ``ProofHypergraph`` s for a batch of theorems.
  * **train** — harvest per-transition targets from those graphs (``search_harvest``) and take
    a synchronous actor-critic gradient step: advantage = ``return − V_pred(s)`` with the
    AND-OR-backed return, critic regressing the AND-OR backup value, plus the annealed BC
    anchor. PLN enters the reward only as the potential-based shaping term (Approach 1).

The trainer takes an injected ``featurize`` callable (``Goal -> PyG Data``) so it is testable
without a Lean/Pantograph backend; ``make_featurizer`` builds the default one reusing the same
string path the GNN engine already uses (OOV → ``<UNK>``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

import torch
from torch_geometric.data import Batch, Data

from .actor_critic import ActorCriticWithArgsClassifier
from .actor_critic_loss import compute_bc_anchor_loss, compute_critic_loss, compute_entropy_bonus
from .graph import proof_state_to_dag
from .pyg import build_premise_mask, dag_to_pyg
from .pln_reward import RewardConfig
from .search_harvest import HarvestConfig, HarvestedTransition, extract_transitions

from maths_ai.data_models.proof_components import Goal


def _validate_update_size(
    datas: list[Data],
    *,
    max_update_nodes: int,
    max_update_edges: int,
) -> tuple[int, int]:
    if max_update_nodes < 0 or max_update_edges < 0:
        raise ValueError("RL update node and edge budgets cannot be negative.")
    node_count = sum(int(data.num_nodes) for data in datas)
    edge_count = sum(int(data.edge_index.size(1)) for data in datas)
    if (max_update_nodes and node_count > max_update_nodes) or (
        max_update_edges and edge_count > max_update_edges
    ):
        raise ValueError(
            "Collected RL update exceeds its graph budget: "
            f"graphs={len(datas)}, nodes={node_count}, edges={edge_count}, "
            f"max_nodes={max_update_nodes}, max_edges={max_update_edges}."
        )
    return node_count, edge_count


@dataclass(frozen=True)
class EdgeAction:
    """The integer record of one sampled action — everything the train phase needs to
    recompute its log-prob under the current parameters (no autograd graph is stashed).

    ``multiplicity`` is how many of the k i.i.d. draws produced this same action: the
    action was applied to Lean once (dedup for the executor) but its policy-gradient term
    must be weighted by ``m`` or high-probability actions are under-weighted.
    """
    tactic_id: int
    arg_indices: tuple[int, ...] = ()
    multiplicity: int = 1


@dataclass(frozen=True)
class FailureRecord:
    """A sampled action the executor rejected (or that never produced an edge).

    There is no hyperedge and no successor state, so it enters the loss actor-only:
    ``return = terminal_failure − step_penalty`` and NO critic regression target — the
    *state* may still be provable via another tactic; only this *action* failed.
    """
    goal: Goal
    action: EdgeAction


def goal_to_state(goal: Goal) -> str:
    """Render a ``Goal`` as the proof-state string the parser expects.

    ``Goal.hypotheses`` are Pantograph's ``name : type`` lines; joining them above the
    turnstile reproduces the state the executor observed, so the DAG contains the
    hypothesis nodes the pointer needs as argument candidates. A goal with no
    hypotheses passes its expression through unchanged (it may already embed a full
    state string, as the harvest tests do).
    """
    if goal.hypotheses:
        return "\n".join(goal.hypotheses) + "\n⊢ " + goal.expression
    return goal.expression


def make_featurizer(node_vocab: dict[str, int]) -> Callable[[Goal], Data]:
    """Build a ``Goal -> PyG Data`` featurizer reusing the string parsing path.

    ``goal_to_state`` → ``proof_state_to_dag`` → ``dag_to_pyg`` with the fixed ``node_vocab``
    (unseen tokens map to ``<UNK>``). Adds the ``premise_mask`` and ``state_node_index`` the
    encoder's readout needs.
    """
    dag_featurize = make_dag_featurizer(node_vocab)

    def featurize(goal: Goal) -> Data:
        _dag, data = dag_featurize(goal)
        return data

    return featurize


def make_dag_featurizer(node_vocab: dict[str, int]):
    """Like ``make_featurizer`` but return ``(dag, data)``.

    The RL reasoner needs the ``DAGBuilder`` alongside the tensors: the pointer's sampled
    argument indices are positions in this DAG's node list, and decoding them to Lean
    argument strings (``_resolve_local_node_name``) requires the nodes themselves. Collect
    and train MUST featurize through the same path so the stored indices stay aligned.
    """
    state_label_id = node_vocab.get("State", 0)

    def featurize(goal: Goal):
        dag = proof_state_to_dag(goal_to_state(goal))
        data = dag_to_pyg(dag, node_vocab, add_reverse_edges=True)
        data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
        matches = (data.x == state_label_id).nonzero(as_tuple=False).view(-1)
        data.state_node_index = matches[-1:] if matches.numel() else torch.tensor([0], dtype=torch.long)
        return dag, data

    return featurize


def compute_transition_loss(
    model: ActorCriticWithArgsClassifier,
    transitions: list[HarvestedTransition],
    featurize: Callable[[Goal], Data],
    tactic_to_id: dict[str, int],
    *,
    device: torch.device | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    bc_weight: float = 0.0,
    max_update_nodes: int = 0,
    max_update_edges: int = 0,
) -> tuple[torch.Tensor, dict[str, float]] | None:
    """One actor-critic loss over a batch of harvested transitions (tactic-level policy grad).

    Advantage ``Â(s,τ) = return − V_pred(s)`` with the AND-OR-backed ``return`` from the
    harvest; the critic regresses the AND-OR backup ``value_target``. Transitions whose tactic
    is outside ``tactic_to_id`` (e.g. the ``PLN_fallback`` pseudo-tactic) are dropped. Returns
    ``None`` if nothing survives. The BC anchor uses each transition's own tactic id as the
    supervised label (the search-chosen action), annealed by ``bc_weight``.
    """
    device = device or torch.device("cpu")
    datas: list[Data] = []
    action_ids: list[int] = []
    returns: list[float] = []
    value_targets: list[float] = []
    critic_rows: list[int] = []
    for t in transitions:
        tid = tactic_to_id.get(t.tactic.tactic_name)
        if tid is None:
            continue
        datas.append(featurize(t.goal))
        action_ids.append(tid)
        returns.append(t.return_)
        if t.value_target is not None:
            critic_rows.append(len(datas) - 1)
            value_targets.append(t.value_target)

    if not datas:
        return None

    node_count, edge_count = _validate_update_size(
        datas,
        max_update_nodes=max_update_nodes,
        max_update_edges=max_update_edges,
    )
    batch = Batch.from_data_list(datas).to(device)
    _node_emb, _state_emb, tactic_logits, values = model.encode(batch)

    actions = torch.tensor(action_ids, dtype=torch.long, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    value_pred = values.squeeze(-1)

    # Advantage: AND-OR-backed return minus the critic baseline (detached), normalized.
    raw_adv = returns_t - value_pred.detach()
    advantages = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)

    log_probs = torch.log_softmax(tactic_logits, dim=-1)
    selected_logp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    actor_loss = -(selected_logp * advantages).mean()

    if critic_rows:
        value_targets_t = torch.tensor(value_targets, dtype=torch.float32, device=device)
        critic_loss = compute_critic_loss(values[critic_rows], value_targets_t)
        mean_value_target = float(value_targets_t.mean().item())
    else:
        critic_loss = values.sum() * 0.0
        mean_value_target = 0.0
    entropy = compute_entropy_bonus(tactic_logits)

    total = actor_loss + critic_weight * critic_loss - entropy_weight * entropy

    if bc_weight != 0.0:
        bc_loss = compute_bc_anchor_loss(tactic_logits, actions)
        total = total + bc_weight * bc_loss
        bc_val = float(bc_loss.item())
    else:
        bc_val = 0.0

    metrics = {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(entropy.item()),
        "bc_loss": bc_val,
        "total_loss": float(total.item()),
        "mean_return": float(returns_t.mean().item()),
        "mean_value_target": mean_value_target,
        "num_transitions": float(len(datas)),
        "update_node_count": float(node_count),
        "update_edge_count": float(edge_count),
    }
    return total, metrics


def compute_onpolicy_loss(
    model: ActorCriticWithArgsClassifier,
    transitions: list[HarvestedTransition],
    edge_actions: dict[int, EdgeAction],
    failures: list[FailureRecord],
    featurize: Callable[[Goal], Data],
    *,
    device: torch.device | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    arg_loss_weight: float = 0.5,
    bc_weight: float = 0.0,
    reward_cfg: RewardConfig | None = None,
    max_update_nodes: int = 0,
    max_update_edges: int = 0,
) -> tuple[torch.Tensor, dict[str, float]] | None:
    """On-policy actor-critic loss with the argument-level policy gradient (B2).

    Recompute pattern (refinement 2): the collect phase stored only the integer actions
    (``EdgeAction``); here ``model.evaluate_actions`` rebuilds ``log π(τ|s)`` and
    ``Σ_k log π(u_k|s,τ)`` with gradient under the current parameters. Exactly on-policy only
    when no optimizer step ran between collect and this call.

    Per successful edge (a ``HarvestedTransition`` whose ``edge_id`` is in ``edge_actions``):
      ``L_actor = −m · (log π(τ) + w_arg · Σ_k log π(u_k)) · Â.detach()``
    with ``Â = return − V_pred(s)`` (normalized jointly with the failure rows) and
    ``m`` the sample multiplicity (refinement 1). The critic regresses the AND-OR
    ``value_target`` on these rows only.

    Per ``FailureRecord`` (executor-rejected sample, refinement 4): the same actor term with
    ``return = terminal_failure − step_penalty`` and NO critic target.
    """
    device = device or torch.device("cpu")
    reward_cfg = reward_cfg or RewardConfig()

    datas: list[Data] = []
    tactic_ids: list[int] = []
    arg_rows: list[tuple[int, ...]] = []
    multiplicities: list[float] = []
    returns: list[float] = []
    is_success: list[bool] = []
    value_targets: list[float] = []
    critic_rows: list[int] = []

    for t in transitions:
        action = edge_actions.get(t.edge_id)
        if action is None:
            continue
        datas.append(featurize(t.goal))
        tactic_ids.append(action.tactic_id)
        arg_rows.append(action.arg_indices)
        multiplicities.append(float(action.multiplicity))
        returns.append(t.return_)
        is_success.append(True)
        if t.value_target is not None:
            critic_rows.append(len(datas) - 1)
            value_targets.append(t.value_target)

    failure_return = reward_cfg.terminal_failure - reward_cfg.step_penalty
    for f in failures:
        datas.append(featurize(f.goal))
        tactic_ids.append(f.action.tactic_id)
        arg_rows.append(f.action.arg_indices)
        multiplicities.append(float(f.action.multiplicity))
        returns.append(failure_return)
        is_success.append(False)

    if not datas:
        return None

    node_count, edge_count = _validate_update_size(
        datas,
        max_update_nodes=max_update_nodes,
        max_update_edges=max_update_edges,
    )
    batch = Batch.from_data_list(datas).to(device)
    max_args = max((len(r) for r in arg_rows), default=0)
    if max_args > 0:
        arg_indices = torch.full((len(arg_rows), max_args), -1, dtype=torch.long, device=device)
        for i, row in enumerate(arg_rows):
            for j, idx in enumerate(row):
                arg_indices[i, j] = idx
    else:
        arg_indices = None

    tactic_ids_t = torch.tensor(tactic_ids, dtype=torch.long, device=device)
    tactic_logp, arg_logp, entropy, values, tactic_logits = model.evaluate_actions(
        batch, tactic_ids_t, arg_indices
    )

    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    mult_t = torch.tensor(multiplicities, dtype=torch.float32, device=device)
    success_t = torch.tensor(is_success, dtype=torch.bool, device=device)

    # Advantage over ALL rows (success + failure), normalized jointly so the failure
    # rows' negative returns supply the contrast the near-constant PLN reward lacks.
    # Population std (unbiased=False): a single-row batch normalizes to 0, not NaN.
    raw_adv = returns_t - values.detach()
    advantages = (raw_adv - raw_adv.mean()) / (raw_adv.std(unbiased=False) + 1e-8)

    joint_logp = tactic_logp + arg_loss_weight * arg_logp
    actor_terms = -(mult_t * joint_logp * advantages)
    actor_loss = actor_terms.sum() / mult_t.sum()

    # Critic: regress the AND-OR backup on success rows only (a failed ACTION says
    # nothing about the STATE's value).
    if critic_rows:
        value_targets_t = torch.tensor(value_targets, dtype=torch.float32, device=device)
        critic_loss = compute_critic_loss(values[critic_rows], value_targets_t)
    else:
        critic_loss = values.sum() * 0.0

    entropy_bonus = entropy.mean()
    total = actor_loss + critic_weight * critic_loss - entropy_weight * entropy_bonus

    if bc_weight != 0.0:
        # BC anchor toward the search-executed tactic on success rows only — anchoring
        # toward rejected tactics would clone the failures.
        labels = torch.where(success_t, tactic_ids_t, torch.full_like(tactic_ids_t, -1))
        bc_loss = compute_bc_anchor_loss(tactic_logits, labels)
        total = total + bc_weight * bc_loss
        bc_val = float(bc_loss.item())
    else:
        bc_val = 0.0

    metrics = {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(entropy_bonus.item()),
        "bc_loss": bc_val,
        "total_loss": float(total.item()),
        "mean_return": float(returns_t.mean().item()),
        "mean_arg_logp": float(arg_logp.mean().item()),
        "num_transitions": float(int(success_t.sum().item())),
        "num_failures": float(len(failures)),
        "update_node_count": float(node_count),
        "update_edge_count": float(edge_count),
    }
    return total, metrics


def train_step(
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    graphs: list,
    featurize: Callable[[Goal], Data],
    tactic_to_id: dict[str, int],
    *,
    reward_cfg: RewardConfig | None = None,
    harvest_cfg: HarvestConfig | None = None,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    bc_weight: float = 0.0,
) -> dict[str, float]:
    """Harvest transitions from a batch of search graphs and take one gradient step."""
    reward_cfg = reward_cfg or RewardConfig()
    harvest_cfg = harvest_cfg or HarvestConfig()
    transitions: list[HarvestedTransition] = []
    for graph in graphs:
        transitions.extend(extract_transitions(graph, reward_cfg, harvest_cfg))

    if not transitions:
        return {"num_transitions": 0.0}

    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = compute_transition_loss(
        model, transitions, featurize, tactic_to_id,
        device=device, critic_weight=critic_weight,
        entropy_weight=entropy_weight, bc_weight=bc_weight,
    )
    if result is None:
        return {"num_transitions": 0.0}
    loss, metrics = result
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return metrics


async def collect_batch(reasoner, theorems: list[dict]) -> list:
    """Run the hybrid reasoner's AND-OR search concurrently over a batch of theorems.

    Each item is ``{"goal": str, "hypotheses": list[str]}``. Requires a live Pantograph
    server + ``petta`` binary (env-gated); PLN is de-blocked via ``evaluate_async`` so the
    searches overlap their Lean/PLN waits. Returns the finished ``ProofHypergraph`` s.
    """
    return await asyncio.gather(
        *(
            reasoner.prove(item["goal"], hypotheses=item.get("hypotheses"))
            for item in theorems
        )
    )


def collect_and_train(
    reasoner,
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    theorems: list[dict],
    featurize: Callable[[Goal], Data],
    tactic_to_id: dict[str, int],
    **train_kwargs,
) -> dict[str, float]:
    """One round: collect search graphs under ``asyncio.run`` (async), then train (sync)."""
    graphs = asyncio.run(collect_batch(reasoner, theorems))
    return train_step(model, optimizer, graphs, featurize, tactic_to_id, **train_kwargs)


# ---------------------------------------------------------------------------
# On-policy path (training-mode expand): RLHybridReasoner results → gradient step
# ---------------------------------------------------------------------------


def train_step_onpolicy(
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    results: list,
    featurize: Callable[[Goal], Data],
    *,
    reward_cfg: RewardConfig | None = None,
    harvest_cfg: HarvestConfig | None = None,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    arg_loss_weight: float = 0.5,
    bc_weight: float = 0.0,
    max_update_nodes: int = 0,
    max_update_edges: int = 0,
) -> dict[str, float]:
    """One gradient step over a batch of ``RLSearchResult`` s (rl_reasoner.py).

    Per result: harvest only the edges the policy produced
    (``extract_transitions(edge_ids=…)``), then join each transition to its ``EdgeAction``
    via ``edge_id`` and fold in the failure records. Exactly ONE ``optimizer.step()`` per
    collect round — the recomputed log-probs are on-policy only while the parameters match
    those that sampled the actions.
    """
    reward_cfg = reward_cfg or RewardConfig()
    harvest_cfg = harvest_cfg or HarvestConfig()

    transitions: list[HarvestedTransition] = []
    edge_actions: dict[int, EdgeAction] = {}
    failures: list[FailureRecord] = []
    # edge_id is unique only within one graph (refinement 6): re-key each result's
    # actions by a per-batch offset before merging.
    offset = 0
    for result in results:
        ts = extract_transitions(
            result.graph, reward_cfg, harvest_cfg,
            edge_ids=list(result.edge_actions.keys()),
        )
        for t in ts:
            t.edge_id += offset
        transitions.extend(ts)
        edge_actions.update({eid + offset: a for eid, a in result.edge_actions.items()})
        failures.extend(result.failure_actions)
        if result.edge_actions:
            offset += max(result.edge_actions.keys()) + 1

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_result = compute_onpolicy_loss(
        model, transitions, edge_actions, failures, featurize,
        device=device, critic_weight=critic_weight, entropy_weight=entropy_weight,
        arg_loss_weight=arg_loss_weight, bc_weight=bc_weight, reward_cfg=reward_cfg,
        max_update_nodes=max_update_nodes,
        max_update_edges=max_update_edges,
    )
    if loss_result is None:
        return {"num_transitions": 0.0, "num_failures": 0.0}
    loss, metrics = loss_result
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return metrics


async def collect_batch_onpolicy(reasoner, theorems: list[dict]) -> list:
    """Run the RL reasoner SEQUENTIALLY over the batch and return its ``RLSearchResult`` s.

    Sequential because one reasoner instance holds one action stash at a time
    (refinement 6); PLN concurrency *within* each search still applies via
    ``evaluate_async``. Cross-theorem concurrency needs per-search reasoner instances
    and is deferred.
    """
    results = []
    for item in theorems:
        results.append(await reasoner.prove(item["goal"], hypotheses=item.get("hypotheses")))
    return results


def collect_and_train_onpolicy(
    reasoner,
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    theorems: list[dict],
    featurize: Callable[[Goal], Data],
    **train_kwargs,
) -> dict[str, float]:
    """One on-policy round: sequential collect (async) → one gradient step (sync)."""
    results = asyncio.run(collect_batch_onpolicy(reasoner, theorems))
    return train_step_onpolicy(model, optimizer, results, featurize, **train_kwargs)
