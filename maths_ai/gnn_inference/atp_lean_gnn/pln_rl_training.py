"""End-to-end PLN actor-critic RL: collect search trees (async) → train (sync).

Two phases share the caller's event loop; model updates begin after collection:

  * **collect** — run the hybrid reasoner's AND-OR search (async, PLN de-blocked via
    ``evaluate_async``) to produce ``ProofHypergraph`` s for a batch of theorems.
  * **train** — harvest per-transition targets from those graphs (``search_harvest``) and take
    a synchronous actor-critic gradient step: advantage = ``return − V_pred(s)`` with the
    AND-OR-backed return, critic regressing the AND-OR backup value, plus the annealed BC
    anchor. PLN enters the reward only as the potential-based shaping term (Approach 1).

The trainer takes an injected ``featurize`` callable (``Goal -> PyG Data``) so it is testable
without a Lean/Pantograph backend. ``make_featurizer`` builds the strict normalized model
S-expression path used by live collection and both optimizer paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

import torch
from torch_geometric.data import Batch, Data

from .actor_critic import ActorCriticWithArgsClassifier
from .actor_critic_loss import compute_bc_anchor_loss, compute_entropy_bonus
from .update_batching import (
    UpdateLimits,
    UpdateRow,
    UpdatePlan,
    prepare_plan,
    materialize,
    normalized_advantages,
)
from .policy_actions import ArgumentReference, CandidatePoolSnapshot, fingerprint
from .retrieval_artifacts import goal_query
from .graph import dag_fingerprint, model_goal_to_dag
from .pyg import build_premise_mask, dag_to_pyg
from .pln_reward import RewardConfig
from .search_harvest import (
    ActorTransition,
    CriticSample,
    HarvestConfig,
    TacticImitationSample,
    extract_critic_samples,
    extract_transitions,
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


@dataclass(frozen=True)
class EdgeAction:
    """The integer record of one sampled action — everything the train phase needs to
    recompute its log-prob under the current parameters (no autograd graph is stashed).

    ``arguments`` are the decoder's typed references (a local node or a library
    lemma), resolved against ``candidate_pool``, the immutable support the action
    was drawn from; the update re-evaluates them against that same pool.

    ``multiplicity`` is how many of the k i.i.d. draws produced this same action: the
    action was applied to Lean once (dedup for the executor) but its policy-gradient term
    must be weighted by ``m`` or high-probability actions are under-weighted.
    """

    tactic_id: int
    candidate_pool: CandidatePoolSnapshot
    graph_fingerprint: str
    arguments: tuple[ArgumentReference, ...] = ()
    multiplicity: int = 1

    def __post_init__(self) -> None:
        if not self.graph_fingerprint:
            raise ValueError("EdgeAction requires a graph fingerprint.")

    @property
    def raw_action_id(self) -> tuple:
        """The concrete choice: tactic id and each argument's source and identity."""
        return (
            self.tactic_id,
            tuple((ref.source, ref.identity) for ref in self.arguments),
        )


class ObservationOutcome(str, Enum):
    ACCEPTED = "accepted"  # Lean accepted the command: it has an edge
    REJECTED = "rejected"  # Lean checked the command and rejected it
    UNRENDERABLE = "unrenderable"  # the draw could not become a command
    RESOURCE_LIMIT = "resource_limit"  # no negative action evidence


@dataclass(frozen=True)
class PolicyObservation:
    """Every draw of one raw action at one search node, with its checked outcome.

    ``action`` holds the raw tactic id and argument references, and its
    ``multiplicity`` is how many settled draws produced them, across every
    proposal round at the node. Two raw actions that render the same command
    share its one execution outcome but stay separate observations, so each
    keeps its own argument references and gradient. An accepted observation names
    its ``edge_id``; its return comes from that edge. A rejected or
    unrenderable one enters the loss actor-only, with
    ``return = terminal_failure − step_penalty`` and no critic target: the
    state may still be provable; only this action failed. ``log_prior`` is the
    proposal-time prior (None when no complete action existed); it is a record,
    and the loss recomputes log-probabilities instead of using it.
    """

    node_id: int
    goal: Goal
    action: EdgeAction
    outcome: ObservationOutcome
    first_round: int
    edge_id: int | None = None
    log_prior: float | None = None
    provider: str = ""

    def __post_init__(self) -> None:
        if (self.outcome == ObservationOutcome.ACCEPTED) != (self.edge_id is not None):
            raise ValueError("Only an accepted observation has an edge.")
        if self.outcome == ObservationOutcome.UNRENDERABLE:
            if self.log_prior is not None:
                raise ValueError("An unrenderable draw has no complete-action prior.")
        elif self.log_prior is None:
            raise ValueError("An executable observation needs its action prior.")


def make_featurizer(node_vocab: dict[str, int]) -> Callable[[Goal], Data]:
    """Build the canonical ``Goal -> PyG Data`` model-S-expression featurizer."""
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
        dag = model_goal_to_dag(goal.require_model_state())
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
        data.unknown_label_count = len(unknown_labels)
        data.structural_unknown_label_count = 0
        data.semantic_unknown_label_count = sum(
            not _is_structural_label(label) for label in unknown_labels
        )
        data.structural_label_count = sum(
            _is_structural_label(node.label) for node in dag.nodes
        )
        data.semantic_label_count = int(data.num_nodes) - int(
            data.structural_label_count
        )
        data.total_label_count = int(data.num_nodes)
        return dag, data

    return featurize


def prepare_transition_update(
    model,
    transitions,
    critic_samples,
    featurize,
    tactic_to_id,
    *,
    max_update_nodes=50000,
    max_update_edges=250000,
):
    """Preflight all tactic-level RL rows before any model or optimizer work."""
    actors = []
    for t in transitions:
        tid = tactic_to_id.get(t.tactic.tactic_name)
        if tid is not None:
            data = featurize(t.goal)
            actors.append(
                UpdateRow(
                    t.goal, str(data.graph_fingerprint), t.return_, tid, accepted=True
                )
            )
    return _prepare(
        model,
        actors,
        critic_samples,
        featurize,
        "transition",
        max_update_nodes,
        max_update_edges,
    )


def _prepare(
    model,
    actors,
    critics,
    featurize,
    phase,
    max_update_nodes,
    max_update_edges,
    max_update_candidates=100000,
    max_update_candidate_bytes=268435456,
):
    critic_rows = [
        UpdateRow(s.goal, s.graph_fingerprint, s.target, source=s.source.value)
        for s in critics
    ]
    limits = UpdateLimits(
        max_update_nodes,
        max_update_edges,
        max_update_candidates,
        max_update_candidate_bytes,
    )
    return prepare_plan(
        model, actors, critic_rows, featurize, _check_action, phase=phase, limits=limits
    )


def _check_action(
    model: ActorCriticWithArgsClassifier,
    goal: Goal,
    data: Data,
    action: EdgeAction | TacticImitationSample,
    owner: str,
) -> None:
    """Validate one stored complete action before either optimizer allocates inputs."""
    if str(data.graph_fingerprint) != action.graph_fingerprint:
        raise ValueError(f"{owner} graph changed between collection and training.")
    if action.candidate_pool is None:
        raise ValueError(f"{owner} needs its immutable candidate pool.")
    if action.candidate_pool.graph_fingerprint != data.graph_fingerprint:
        raise ValueError(f"{owner}: the candidate pool's graph changed.")
    if action.candidate_pool.query_fingerprint != fingerprint(goal_query(goal)):
        raise ValueError(
            f"{owner}: the query text changed between collection and update."
        )
    model.validate_candidate_pool(action.candidate_pool, num_nodes=data.num_nodes)
    action.candidate_pool.validate(action.arguments)
    if len(action.arguments) > model.max_args:
        raise ValueError(f"{owner}: argument count exceeds the policy budget.")


def prepare_onpolicy_update(
    model,
    transitions,
    critic_samples,
    observations,
    featurize,
    *,
    reward_cfg=None,
    max_update_nodes=50000,
    max_update_edges=250000,
    max_update_candidates=100000,
    max_update_candidate_bytes=268435456,
):
    """Validate and plan all eligible complete-action observations on the CPU."""
    reward_cfg = reward_cfg or RewardConfig()
    by_edge = {t.edge_id: t for t in transitions}
    actors = []
    for o in observations:
        if o.outcome is ObservationOutcome.RESOURCE_LIMIT:
            continue
        accepted = o.outcome is ObservationOutcome.ACCEPTED
        transition = by_edge.get(o.edge_id) if accepted else None
        if accepted and transition is None:
            continue
        goal = transition.goal if accepted else o.goal
        target = (
            transition.return_
            if accepted
            else reward_cfg.terminal_failure - reward_cfg.step_penalty
        )
        actors.append(
            UpdateRow(
                goal,
                o.action.graph_fingerprint,
                target,
                o.action.tactic_id,
                o.action,
                float(o.action.multiplicity),
                accepted,
            )
        )
    return _prepare(
        model,
        actors,
        critic_samples,
        featurize,
        "onpolicy",
        max_update_nodes,
        max_update_edges,
        max_update_candidates,
        max_update_candidate_bytes,
    )


def prepare_htps_update(
    model,
    tactic_batch,
    critic_batch,
    featurize,
    *,
    max_update_nodes=50000,
    max_update_edges=250000,
    max_update_candidates=100000,
    max_update_candidate_bytes=268435456,
):
    actors = [
        UpdateRow(s.goal, s.graph_fingerprint, 0.0, s.tactic_id, s)
        for s in tactic_batch
    ]
    return _prepare(
        model,
        actors,
        critic_batch,
        featurize,
        "htps",
        max_update_nodes,
        max_update_edges,
        max_update_candidates,
        max_update_candidate_bytes,
    )


def backward_update(
    model,
    plan: UpdatePlan,
    featurize,
    *,
    device=None,
    critic_weight=0.5,
    entropy_weight=0.01,
    bc_weight=0.0,
    arg_loss_weight=0.5,
):
    """Accumulate globally reduced gradients, freeing each batch before the next.

    The caller owns zeroing, clipping, stepping, and cleanup. No returned tensor
    retains an autograd graph. The value prepass has no argument tensors.
    """
    device = device or next(model.parameters()).device
    model.eval()
    metrics = dict(plan.metrics)
    n_actor, n_critic = len(plan.actors), len(plan.critics)
    accepted = sum(r.record.accepted for r in plan.actors)
    weight = sum(r.record.weight for r in plan.actors)
    values = []
    if plan.phase != "htps":
        with torch.no_grad():
            for indexes in plan.actor_batches:
                batch = Batch.from_data_list(
                    materialize(model, plan.actors, indexes, featurize)
                ).to(device)
                encoded = model.encode_graph(batch)
                predicted = model.critic(encoded.state_embeddings).reshape(-1)
                if not torch.isfinite(predicted).all():
                    raise ValueError("Nonfinite values in RL advantage prepass.")
                values.extend(predicted.cpu().tolist())
                del batch, encoded, predicted
        advantages = (
            normalized_advantages(
                [r.record.target for r in plan.actors],
                values,
                [r.record.weight for r in plan.actors],
            )
            if n_actor
            else []
        )
        metrics["update_value_batches"] = float(len(plan.actor_batches))
    else:
        advantages = []
    sums = {
        k: 0.0
        for k in (
            "actor_loss",
            "critic_loss",
            "entropy",
            "entropy_gradient_surrogate",
            "bc_loss",
            "mean_arg_logp",
            "mean_count_logp",
            "mean_source_logp",
            "mean_identity_logp",
            "tactic_imitation_loss",
        )
    }
    with torch.enable_grad():
        for indexes in plan.actor_batches:
            rows = [plan.actors[i].record for i in indexes]
            batch = Batch.from_data_list(
                materialize(model, plan.actors, indexes, featurize)
            ).to(device)
            labels = torch.tensor(
                [r.tactic_id for r in rows], dtype=torch.long, device=device
            )
            if plan.phase == "transition":
                _nodes, _states, logits, _values = model.encode(batch)
                logp = (
                    torch.log_softmax(logits, -1).gather(1, labels[:, None]).reshape(-1)
                )
                # encode() also returns the critic branch, which this actor loss does not use.
                del _nodes, _states, _values
            else:
                evaluated = model.evaluate_actions(
                    batch,
                    labels,
                    [r.action.arguments for r in rows],
                    candidate_pools=[r.action.candidate_pool for r in rows],
                )
                logits, logp = evaluated.tactic_logits, evaluated.logp
            if plan.phase == "htps":
                imitation = (
                    -evaluated.tactic_logp.sum()
                    - arg_loss_weight * evaluated.arg_logp.sum()
                ) / n_actor
                loss = imitation
                sums["tactic_imitation_loss"] += float(imitation.detach())
            else:
                mult = torch.tensor(
                    [r.weight for r in rows], device=device, dtype=logp.dtype
                )
                adv = torch.tensor(
                    [advantages[i] for i in indexes], device=device, dtype=logp.dtype
                )
                actor = -(mult * logp * adv).sum() / weight
                if plan.phase == "transition":
                    bonus = compute_entropy_bonus(logits) * len(rows) / n_actor
                    sums["entropy"] += float(bonus.detach())
                else:
                    bonus = -((logp.detach() + 1) * logp * mult).sum() / weight
                    sums["entropy"] += float(-(logp.detach() * mult).sum() / weight)
                    sums["entropy_gradient_surrogate"] += float(-bonus.detach())
                loss = actor - entropy_weight * bonus
                sums["actor_loss"] += float(actor.detach())
                if bc_weight and accepted:
                    anchor_labels = torch.tensor(
                        [r.tactic_id if r.accepted else -1 for r in rows], device=device
                    )
                    local_accepted = sum(r.accepted for r in rows)
                    if local_accepted:
                        anchor = (
                            compute_bc_anchor_loss(logits, anchor_labels)
                            * local_accepted
                            / accepted
                        )
                        loss = loss + bc_weight * anchor
                        sums["bc_loss"] += float(anchor.detach())
                    del anchor_labels
                if plan.phase == "onpolicy":
                    for key, output in (
                        ("mean_arg_logp", evaluated.arg_logp),
                        ("mean_count_logp", evaluated.count_logp),
                        ("mean_source_logp", evaluated.source_logp),
                        ("mean_identity_logp", evaluated.identity_logp),
                    ):
                        sums[key] += float(output.detach().sum()) / n_actor
                    del output
                del mult, adv, actor, bonus
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {plan.phase} actor loss.")
            loss.backward()
            # Delete every graph-bearing reference, including local component tensors.
            del loss, batch, labels, logits, logp
            if plan.phase != "transition":
                del evaluated
            if plan.phase == "htps":
                del imitation
            elif bc_weight and accepted and local_accepted:
                del anchor
        for indexes in plan.critic_batches:
            batch = Batch.from_data_list(
                materialize(model, plan.critics, indexes, featurize)
            ).to(device)
            encoded = model.encode_graph(batch)
            predictions = model.critic(encoded.state_embeddings).reshape(-1)
            targets = predictions.new_tensor(
                [plan.critics[i].record.target for i in indexes]
            )
            critic = (predictions - targets).square().sum() / n_critic
            if not torch.isfinite(critic):
                raise ValueError(f"Nonfinite {plan.phase} critic loss.")
            sums["critic_loss"] += float(critic.detach())
            (critic_weight * critic).backward()
            del critic, batch, encoded, predictions, targets
    if plan.phase == "htps":
        metrics.update(
            tactic_imitation_loss=sums["tactic_imitation_loss"],
            critic_soft_loss=sums["critic_loss"],
            htps_total_loss=sums["tactic_imitation_loss"]
            + critic_weight * sums["critic_loss"],
            num_imitation_rows=float(n_actor),
            num_critic_rows=float(n_critic),
        )
    else:
        metrics.update(
            {
                key: value
                for key, value in sums.items()
                if key != "tactic_imitation_loss"
            }
        )
        metrics["total_loss"] = (
            sums["actor_loss"]
            + critic_weight * sums["critic_loss"]
            + bc_weight * sums["bc_loss"]
        )
        metrics["total_loss"] -= entropy_weight * (
            sums["entropy"]
            if plan.phase == "transition"
            else -sums["entropy_gradient_surrogate"]
        )
        metrics.update(
            num_transitions=float(accepted),
            num_failures=float(n_actor - accepted),
            num_critic_samples=float(n_critic),
            mean_return=(
                sum(r.record.target for r in plan.actors) / n_actor if n_actor else 0.0
            ),
            mean_value_target=(
                sum(r.record.target for r in plan.critics) / n_critic
                if n_critic
                else 0.0
            ),
        )
        for row in plan.critics:
            key = f"critic_source_{row.record.source}"
            metrics[key] = metrics.get(key, 0.0) + 1.0
    return metrics


def _execute_update(model, optimizer, prepare, featurize, *, grad_clip, **loss_options):
    """Own the gradient lifetime and one optimizer mutation for a logical update."""
    plan = None
    try:
        plan = prepare()
        optimizer.zero_grad(set_to_none=True)
        metrics = backward_update(model, plan, featurize, **loss_options)
        step_key = (
            "htps_optimizer_step" if plan.phase == "htps" else "onpolicy_optimizer_step"
        )
        metrics[step_key] = 0.0
        if not plan.empty:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            metrics[step_key] = 1.0
        return metrics
    except BaseException as exc:
        optimizer.zero_grad(set_to_none=True)
        phase = "preflight" if plan is None else plan.phase
        exc.add_note(
            f"RL {phase} computational update failed; partial gradients were cleared."
        )
        raise


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
    max_update_nodes: int = 50000,
    max_update_edges: int = 250000,
) -> dict[str, float]:
    """Harvest transitions from a batch of search graphs and take one gradient step."""
    reward_cfg = reward_cfg or RewardConfig()
    harvest_cfg = harvest_cfg or HarvestConfig()
    transitions: list[ActorTransition] = []
    critic_samples: list[CriticSample] = []
    for graph in graphs:
        transitions.extend(extract_transitions(graph, reward_cfg, harvest_cfg))
        critic_samples.extend(extract_critic_samples(graph, harvest_cfg=harvest_cfg))

    return _execute_update(
        model,
        optimizer,
        lambda: prepare_transition_update(
            model,
            transitions,
            critic_samples,
            featurize,
            tactic_to_id,
            max_update_nodes=max_update_nodes,
            max_update_edges=max_update_edges,
        ),
        featurize,
        grad_clip=grad_clip,
        device=device,
        critic_weight=critic_weight,
        entropy_weight=entropy_weight,
        bc_weight=bc_weight,
    )


async def collect_batch(reasoner, theorems: list[LeanGoalSeed]) -> list:
    """Run one shared reasoner sequentially over a batch of theorem seeds.

    Requires a live Pantograph client + ``petta`` binary (env-gated). A reasoner runs one search at a time;
    independent pool workers provide cross-theorem concurrency. Returns the finished
    ``ProofHypergraph`` s.
    """
    return [await reasoner.prove(seed) for seed in theorems]


async def collect_and_train(
    reasoner,
    model: ActorCriticWithArgsClassifier,
    optimizer: torch.optim.Optimizer,
    theorems: list[LeanGoalSeed],
    featurize: Callable[[Goal], Data],
    tactic_to_id: dict[str, int],
    **train_kwargs,
) -> dict[str, float]:
    """Await collection on the owning loop, then perform one synchronous update."""
    graphs = await collect_batch(reasoner, theorems)
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
    bc_weight: float = 0.0,
    max_update_nodes: int = 50000,
    max_update_edges: int = 250000,
    max_update_candidates: int = 100000,
    max_update_candidate_bytes: int = 268435456,
) -> dict[str, float]:
    """One gradient step over a batch of ``RLSearchResult`` s (rl_reasoner.py).

    Per result: harvest only the edges the policy produced
    (``extract_transitions(edge_ids=…)``), once each, and join every accepted
    observation to its edge's transition; rejected and unrenderable observations
    enter as failure rows. Exactly ONE ``optimizer.step()`` per
    collect round — the recomputed log-probs are on-policy only while the parameters match
    those that sampled the actions.
    """
    reward_cfg = reward_cfg or RewardConfig()
    harvest_cfg = harvest_cfg or HarvestConfig()

    transitions: list[ActorTransition] = []
    critic_samples: list[CriticSample] = []
    observations: list[PolicyObservation] = []
    accepted_edges = 0
    # edge_id is unique only within one graph (refinement 6): re-key each result's
    # edges by a per-batch offset before merging.
    offset = 0
    for result in results:
        edge_ids = sorted(result.accepted_edge_ids())
        accepted_edges += len(edge_ids)
        ts = extract_transitions(
            result.graph, reward_cfg, harvest_cfg, edge_ids=edge_ids
        )
        for t in ts:
            t.edge_id += offset
        transitions.extend(ts)
        critic_samples.extend(
            extract_critic_samples(result.graph, harvest_cfg=harvest_cfg)
        )
        observations.extend(
            replace(o, edge_id=o.edge_id + offset) if o.edge_id is not None else o
            for o in result.observations
        )
        if edge_ids:
            offset += edge_ids[-1] + 1

    unknown_edges_skipped = accepted_edges - len(transitions)
    unknown_nodes_skipped = sum(len(result.graph.nodes) for result in results) - len(
        critic_samples
    )

    metrics = _execute_update(
        model,
        optimizer,
        lambda: prepare_onpolicy_update(
            model,
            transitions,
            critic_samples,
            observations,
            featurize,
            reward_cfg=reward_cfg,
            max_update_nodes=max_update_nodes,
            max_update_edges=max_update_edges,
            max_update_candidates=max_update_candidates,
            max_update_candidate_bytes=max_update_candidate_bytes,
        ),
        featurize,
        grad_clip=grad_clip,
        device=device,
        critic_weight=critic_weight,
        entropy_weight=entropy_weight,
        bc_weight=bc_weight,
    )
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


# ---------------------------------------------------------------------------
# Decoupled HTPS-style step (Phases 2–3): supervised regression on stored pairs
# ---------------------------------------------------------------------------


def train_step_htps_style(
    model: ActorCriticWithArgsClassifier,
    optimizer_htps: torch.optim.Optimizer,
    tactic_batch: list[TacticImitationSample],
    critic_batch: list[CriticSample],
    featurize: Callable[[Goal], Data],
    *,
    w_critic_soft: float = 0.5,
    arg_loss_weight: float = 0.5,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    max_update_nodes: int = 50000,
    max_update_edges: int = 250000,
    max_update_candidates: int = 100000,
    max_update_candidate_bytes: int = 268435456,
) -> dict[str, float]:
    """One decoupled supervised step: tactic imitation + soft critic regression.

    ``tactic_batch`` rows are complete actions (tactic id, argument references and
    their candidate pool) mined from the certified proof hypertree
    (``extract_minimal_hypertree``); ``critic_batch`` rows are (goal, Q-or-status)
    targets (``extract_critic_samples``). Losses:

      * ``L_tactic_imitation`` — the negative log-probability of the tactic plus,
        weighted by ``arg_loss_weight``, that of its arguments (count, source and
        identity), teacher-forced through ``evaluate_actions`` against the pool the
        action was drawn from.
      * ``L_critic_soft`` — MSE of the value head against the stored targets on
        critic rows, raw scale (the advantage-normalization convention applies to the
        actor's score-function estimator only).

      ``loss = L_tactic_imitation + w_critic_soft · L_critic_soft``

    **One-step-per-collect exemption:** this function may run many times per round.
    That invariant belongs to ``prepare_onpolicy_update``'s score-function estimator,
    whose recomputed log-probs are on-policy only under unchanged parameters. Here
    every input is a stored ``(input, label)`` pair whose validity does not depend on
    which parameters generated it — supervised regression, no importance ratio to
    invalidate. ``optimizer_htps`` must be a separate optimizer instance from the
    on-policy one (separate Adam moments; the driver asserts identity inequality).
    """
    return _execute_update(
        model,
        optimizer_htps,
        lambda: prepare_htps_update(
            model,
            tactic_batch,
            critic_batch,
            featurize,
            max_update_nodes=max_update_nodes,
            max_update_edges=max_update_edges,
            max_update_candidates=max_update_candidates,
            max_update_candidate_bytes=max_update_candidate_bytes,
        ),
        featurize,
        grad_clip=grad_clip,
        device=device,
        critic_weight=w_critic_soft,
        arg_loss_weight=arg_loss_weight,
    )
