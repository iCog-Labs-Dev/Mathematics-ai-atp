"""End-to-end PLN actor-critic RL: collect search trees (async) → train (sync).

Two phases share the caller's event loop; model updates begin after collection:

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

import re
from dataclasses import dataclass, replace
from typing import Callable

import torch
from torch_geometric.data import Batch, Data

from .actor_critic import ActorCriticWithArgsClassifier
from .policy_actions import ArgumentReference, CandidatePoolSnapshot, fingerprint
from .retrieval_artifacts import goal_query
from .actor_critic_loss import (
    compute_bc_anchor_loss,
    compute_critic_loss,
    compute_entropy_bonus,
)
from .graph import dag_fingerprint, model_goal_to_dag
from .pyg import build_premise_mask, dag_to_pyg
from .pln_reward import RewardConfig
from .search_harvest import (
    ActorTransition,
    CriticSample,
    HarvestConfig,
    extract_actor_transitions,
    extract_critic_samples,
)

from maths_ai.data_models.proof_components import Goal
from maths_ai.hybrid_reasoner.hypergraph import UnsupportedObservation
from maths_ai.lean_replay import LeanGoalSeed


_STRUCTURAL_LABELS = {
    "State",
    "Goal",
    "Hyp",
    ":app",
    ":arg",
    ":forall",
    ":instance-of",
    ":lambda",
    ":let",
    ":lit",
    ":mdata",
    ":metavar",
    ":proj",
    ":proof-of",
}
_FV_RE = re.compile(r"^FV\d+$")


def _is_structural_label(label: str) -> bool:
    return (
        label in _STRUCTURAL_LABELS
        or label.startswith(":")
        or label.startswith("HypRole:")
        or label.startswith("BinderRole:")
        or label.startswith("ArgRole:")
        or label.startswith("ArgPosition:")
        or _FV_RE.fullmatch(label) is not None
    )


# Roles the model serializer emits for binders, application arguments, and
# context locals (``instance``/``let`` refine the local's binder role).
_BINDER_ROLES = (":explicit", ":implicit", ":strict-implicit", ":instance-implicit")
_ARG_ROLES = (":explicit", ":implicit", ":implicit-type", ":instance", ":proof")
_HYP_ROLES = (*(role.removeprefix(":") for role in _BINDER_ROLES), "instance", "let")


def supported_structural_labels(*, max_locals: int, max_arity: int) -> list[str]:
    """Every structural label a state within these bounds can contain.

    A vocabulary built from a seed goal alone lacks labels its successors
    introduce, such as ``FV1`` after ``intro``. Vocabularies for synthetic runs
    include this set; states beyond the bounds still fail the missing-label check.
    """
    if max_locals < 0 or max_arity < 0:
        raise ValueError("Structural label bounds must be nonnegative.")
    labels = [
        *sorted(_STRUCTURAL_LABELS),
        *(f"BinderRole:{role}" for role in _BINDER_ROLES),
        *(f"ArgRole:{role}" for role in _ARG_ROLES),
        *(f"ArgPosition:{position}" for position in range(max_arity)),
        *(f"HypRole:{role}" for role in _HYP_ROLES),
        *(f"FV{index}" for index in range(max_locals)),
    ]
    assert all(_is_structural_label(label) for label in labels)
    return labels


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
    candidate_pool: CandidatePoolSnapshot
    graph_fingerprint: str
    arguments: tuple[ArgumentReference, ...] = ()
    multiplicity: int = 1


@dataclass(frozen=True)
class FailureRecord:
    """A sampled action that the executor rejected.

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

    def featurize(goal: Goal):
        dag = model_goal_to_dag(goal)
        data = dag_to_pyg(dag, node_vocab, add_reverse_edges=True)
        data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
        if dag.state_root_id is None:
            raise ValueError("RL DAG has no State root.")
        data.state_node_index = torch.tensor([dag.state_root_id], dtype=torch.long)
        data.graph_fingerprint = dag_fingerprint(dag)
        unknown_labels = [
            node.label for node in dag.nodes if node.label not in node_vocab
        ]
        structural_unknown = sorted(
            {label for label in unknown_labels if _is_structural_label(label)}
        )
        if structural_unknown:
            raise UnsupportedObservation(
                "Prepared node vocabulary is missing required structural labels: "
                + ", ".join(structural_unknown)
            )
        semantic_unknown_count = sum(
            not _is_structural_label(label) for label in unknown_labels
        )
        data.unknown_label_count = len(unknown_labels)
        data.structural_unknown_label_count = 0
        data.semantic_unknown_label_count = semantic_unknown_count
        data.structural_label_count = sum(
            _is_structural_label(node.label) for node in dag.nodes
        )
        data.semantic_label_count = int(data.num_nodes) - int(
            data.structural_label_count
        )
        data.total_label_count = int(data.num_nodes)
        return dag, data

    return featurize


def compute_transition_loss(
    model,
    actor_transitions,
    critic_samples,
    featurize,
    *,
    edge_actions,
    device=None,
    critic_weight=0.5,
    entropy_weight=0.01,
    bc_weight=0.0,
    max_update_nodes=0,
    max_update_edges=0,
):
    """Evaluate harvested targets only with the complete recorded actions."""
    result = compute_onpolicy_loss(
        model,
        actor_transitions,
        critic_samples,
        edge_actions,
        [],
        featurize,
        device=device,
        critic_weight=critic_weight,
        entropy_weight=entropy_weight,
        bc_weight=bc_weight,
        max_update_nodes=max_update_nodes,
        max_update_edges=max_update_edges,
    )
    if result is not None:
        result[1]["mean_value_target"] = sum(x.target for x in critic_samples) / max(
            len(critic_samples), 1
        )
    return result


def compute_onpolicy_loss(
    model: ActorCriticWithArgsClassifier,
    actor_transitions: list[ActorTransition],
    critic_samples: list[CriticSample],
    edge_actions: dict[int, EdgeAction],
    failures: list[FailureRecord],
    featurize: Callable[[Goal], Data],
    *,
    device: torch.device | None = None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    bc_weight: float = 0.0,
    reward_cfg: RewardConfig | None = None,
    max_update_nodes: int = 0,
    max_update_edges: int = 0,
    max_update_candidates: int = 100000,
    max_update_candidate_bytes: int = 268435456,
) -> tuple[torch.Tensor, dict[str, float]] | None:
    """Build separate actor and critic forwards, then one combined backward loss."""
    device = device or torch.device("cpu")
    reward_cfg = reward_cfg or RewardConfig()

    actor_datas: list[Data] = []
    tactic_ids: list[int] = []
    argument_rows = []
    candidate_pools = []
    multiplicities: list[float] = []
    returns: list[float] = []
    is_accepted: list[bool] = []

    for t in actor_transitions:
        action = edge_actions.get(t.edge_id)
        if action is None:
            raise ValueError(
                "Accepted transition is missing its complete collection action."
            )
        data = featurize(t.goal)
        if data.graph_fingerprint != action.graph_fingerprint:
            raise ValueError(
                "Accepted action graph changed between collection and training."
            )
        if action.candidate_pool.query_fingerprint != fingerprint(goal_query(t.goal)):
            raise ValueError(
                "Collection query text changed between collection and update."
            )
        actor_datas.append(data)
        tactic_ids.append(action.tactic_id)
        argument_rows.append(action.arguments)
        candidate_pools.append(action.candidate_pool)
        multiplicities.append(float(action.multiplicity))
        returns.append(t.return_)
        is_accepted.append(True)

    failure_return = reward_cfg.terminal_failure - reward_cfg.step_penalty
    for f in failures:
        data = featurize(f.goal)
        if data.graph_fingerprint != f.action.graph_fingerprint:
            raise ValueError(
                "Rejected action graph changed between collection and training."
            )
        if f.action.candidate_pool.query_fingerprint != fingerprint(goal_query(f.goal)):
            raise ValueError(
                "Rejected action query text changed between collection and update."
            )
        actor_datas.append(data)
        tactic_ids.append(f.action.tactic_id)
        argument_rows.append(f.action.arguments)
        candidate_pools.append(f.action.candidate_pool)
        multiplicities.append(float(f.action.multiplicity))
        returns.append(failure_return)
        is_accepted.append(False)

    critic_datas = [featurize(sample.goal) for sample in critic_samples]
    if not actor_datas and not critic_datas:
        return None

    node_count, edge_count = _validate_update_size(
        actor_datas + critic_datas,
        max_update_nodes=max_update_nodes,
        max_update_edges=max_update_edges,
    )
    zero = next(model.parameters()).sum() * 0.0
    actor_loss = zero
    entropy_bonus = zero
    bc_loss = zero
    mean_return = 0.0
    mean_arg_logp = 0.0
    sampled_entropy = 0.0
    policy_diagnostics = {}
    if actor_datas:
        batch = Batch.from_data_list(actor_datas).to(device)
        if any(pool is None for pool in candidate_pools):
            raise ValueError(
                "Collected actions require their immutable candidate pools."
            )
        for data, pool in zip(actor_datas, candidate_pools):
            if pool.graph_fingerprint != data.graph_fingerprint:
                raise ValueError("Collection candidate pool graph changed.")
        candidate_count = sum(
            len(p.local_ids) + len(p.lemma_ids) for p in candidate_pools
        )
        candidate_bytes = sum(
            (
                len(p.local_ids) * model.hidden_dim
                + len(p.lemma_ids) * model.decoder.text_dim
            )
            * 4
            for p in candidate_pools
        )
        if (
            candidate_count > max_update_candidates
            or candidate_bytes > max_update_candidate_bytes
        ):
            raise ValueError("Collected update exceeds its candidate tensor budget.")
        tactic_ids_t = torch.tensor(tactic_ids, dtype=torch.long, device=device)
        evaluated = model.evaluate_actions(
            batch, tactic_ids_t, argument_rows, candidate_pools=candidate_pools
        )
        tactic_logp, arg_logp = evaluated.tactic_logp, evaluated.arg_logp
        values, tactic_logits = evaluated.value, evaluated.tactic_logits
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        mult_t = torch.tensor(multiplicities, dtype=torch.float32, device=device)
        accepted_t = torch.tensor(is_accepted, dtype=torch.bool, device=device)
        raw_adv = returns_t - values.detach()
        mean_adv = (raw_adv * mult_t).sum() / mult_t.sum()
        variance = ((raw_adv - mean_adv).square() * mult_t).sum() / mult_t.sum()
        advantages = (raw_adv - mean_adv) / (variance.sqrt() + 1e-8)
        joint_logp = evaluated.logp
        actor_loss = (-(mult_t * joint_logp * advantages)).sum() / mult_t.sum()
        entropy_bonus = (
            -((joint_logp.detach() + 1) * joint_logp * mult_t).sum() / mult_t.sum()
        )
        if bc_weight != 0.0:
            labels = torch.where(
                accepted_t, tactic_ids_t, torch.full_like(tactic_ids_t, -1)
            )
            bc_loss = compute_bc_anchor_loss(tactic_logits, labels)
        mean_return = float(returns_t.mean().item())
        mean_arg_logp = float(arg_logp.mean().item())
        sampled_entropy = float(
            (-(joint_logp.detach() * mult_t).sum() / mult_t.sum()).item()
        )
        policy_diagnostics = {
            "mean_count_logp": float(evaluated.count_logp.detach().mean()),
            "mean_source_logp": float(evaluated.source_logp.detach().mean()),
            "mean_identity_logp": float(evaluated.identity_logp.detach().mean()),
            "update_candidate_count": float(candidate_count),
            "update_candidate_bytes": float(candidate_bytes),
        }

    critic_loss = zero
    if critic_datas:
        critic_batch = Batch.from_data_list(critic_datas).to(device)
        _nodes, _states, _logits, critic_values = model.encode(critic_batch)
        critic_targets = torch.tensor(
            [sample.target for sample in critic_samples],
            dtype=torch.float32,
            device=device,
        )
        critic_loss = compute_critic_loss(critic_values, critic_targets)
    total = actor_loss + critic_weight * critic_loss - entropy_weight * entropy_bonus
    if actor_datas and bc_weight != 0.0:
        total = total + bc_weight * bc_loss
        bc_val = float(bc_loss.item())
    else:
        bc_val = 0.0

    metrics = {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": sampled_entropy,
        "entropy_gradient_surrogate": float(-entropy_bonus.detach()),
        **policy_diagnostics,
        "bc_loss": bc_val,
        "total_loss": float(total.item()),
        "mean_return": mean_return,
        "mean_arg_logp": mean_arg_logp,
        "num_transitions": float(sum(is_accepted)),
        "num_critic_samples": float(len(critic_datas)),
        "num_failures": float(len(failures)),
        "update_node_count": float(node_count),
        "update_edge_count": float(edge_count),
    }
    return total, metrics


def train_step(model, optimizer, results, featurize, **kwargs):
    """Train from complete result/action records rather than bare proof graphs."""
    return train_step_onpolicy(model, optimizer, results, featurize, **kwargs)


async def collect_batch(reasoner, theorems: list[LeanGoalSeed]) -> list:
    """Run one shared reasoner sequentially over a batch of theorem seeds.

    Requires a live Pantograph
    server + ``petta`` binary (env-gated); PLN is de-blocked via ``evaluate_async`` so the
    independent pool workers provide cross-theorem concurrency. Returns finished graphs.
    """
    return [await reasoner.prove(seed) for seed in theorems]


async def collect_and_train(reasoner, model, optimizer, theorems, featurize, **kwargs):
    results = await collect_batch(reasoner, theorems)
    return train_step_onpolicy(model, optimizer, results, featurize, **kwargs)


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
    bc_weight: float = 0.0,
    max_update_nodes: int = 0,
    max_update_edges: int = 0,
    max_update_candidates: int = 100000,
    max_update_candidate_bytes: int = 268435456,
) -> dict[str, float]:
    """One gradient step over a batch of ``RLSearchResult`` s (rl_reasoner.py).

    Per result: harvest only the edges the policy produced
    (``extract_actor_transitions(edge_ids=...)``), then join each transition to its ``EdgeAction``
    via ``edge_id`` and fold in the failure records. Exactly ONE ``optimizer.step()`` per
    collect round — the recomputed log-probs are on-policy only while the parameters match
    those that sampled the actions.
    """
    reward_cfg = reward_cfg or RewardConfig()
    harvest_cfg = harvest_cfg or HarvestConfig()

    actor_transitions: list[ActorTransition] = []
    critic_samples: list[CriticSample] = []
    edge_actions: dict[int, EdgeAction] = {}
    failures: list[FailureRecord] = []
    unknown_edges_skipped = 0
    unknown_nodes_skipped = 0
    # edge_id is unique only within one graph (refinement 6): re-key each result's
    # actions by a per-batch offset before merging.
    offset = 0
    for result in results:
        ts = extract_actor_transitions(
            result.graph,
            reward_cfg,
            harvest_cfg,
            edge_ids=list(result.edge_actions.keys()),
        )
        actor_transitions.extend(replace(t, edge_id=t.edge_id + offset) for t in ts)
        result_critic_samples = extract_critic_samples(result.graph, harvest_cfg)
        critic_samples.extend(result_critic_samples)
        unknown_edges_skipped += len(result.edge_actions) - len(ts)
        unknown_nodes_skipped += len(result.graph.nodes) - len(result_critic_samples)
        edge_actions.update({eid + offset: a for eid, a in result.edge_actions.items()})
        failures.extend(result.failure_actions)
        if result.edge_actions:
            offset += max(result.edge_actions.keys()) + 1

    model.eval()
    optimizer.zero_grad(set_to_none=True)
    loss_result = compute_onpolicy_loss(
        model,
        actor_transitions,
        critic_samples,
        edge_actions,
        failures,
        featurize,
        device=device,
        critic_weight=critic_weight,
        entropy_weight=entropy_weight,
        bc_weight=bc_weight,
        reward_cfg=reward_cfg,
        max_update_nodes=max_update_nodes,
        max_update_edges=max_update_edges,
        max_update_candidates=max_update_candidates,
        max_update_candidate_bytes=max_update_candidate_bytes,
    )
    if loss_result is None:
        return {
            "num_transitions": 0.0,
            "num_critic_samples": 0.0,
            "num_failures": 0.0,
            "unknown_edges_skipped": float(unknown_edges_skipped),
            "unknown_nodes_skipped": float(unknown_nodes_skipped),
            "unknown_label_count": 0.0,
            "structural_unknown_label_count": 0.0,
            "semantic_unknown_label_count": 0.0,
            "total_label_count": 0.0,
            "optimizer_step": 0.0,
        }
    loss, metrics = loss_result
    loss.backward()
    for name, component in {
        "encoder": model.encoder,
        "tactic_actor": model.actor,
        "critic": model.critic,
        "count": model.decoder.arg_count_head,
        "source": model.decoder.source_head,
        "recurrent_pointer": model.decoder.argument_selector,
        "lemma_adapter": model.decoder.lemma_adapter,
    }.items():
        squares = [
            parameter.grad.detach().float().square().sum()
            for parameter in component.parameters()
            if parameter.grad is not None
        ]
        metrics["gradient_norm_" + name] = (
            float(torch.stack(squares).sum().sqrt()) if squares else 0.0
        )
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    metrics["optimizer_step"] = 1.0
    metrics["unknown_edges_skipped"] = float(unknown_edges_skipped)
    metrics["unknown_nodes_skipped"] = float(unknown_nodes_skipped)
    return metrics


async def collect_batch_onpolicy(reasoner, theorems: list[LeanGoalSeed]) -> list:
    """Run the RL reasoner SEQUENTIALLY over the batch and return its ``RLSearchResult`` s.

    Sequential because one reasoner instance holds one action stash at a time
    (refinement 6); PLN concurrency *within* each search still applies via
    ``evaluate_async``. Cross-theorem concurrency needs per-search reasoner instances
    and is deferred.
    """
    results = []
    for seed in theorems:
        results.append(await reasoner.prove(seed))
    return results


async def collect_and_train_onpolicy(
    reasoner,
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    theorems: list[LeanGoalSeed],
    featurize: Callable[[Goal], Data],
    **train_kwargs,
) -> dict[str, float]:
    """One on-policy round: sequential collect (async) → one gradient step (sync)."""
    results = await collect_batch_onpolicy(reasoner, theorems)
    return train_step_onpolicy(model, optimizer, results, featurize, **train_kwargs)
