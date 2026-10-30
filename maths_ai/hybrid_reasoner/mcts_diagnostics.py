"""CPU-only live verification of MCTS and best-first search over the installed client.

The application gate (``verify_live_application``) runs the client's protocol
checks, admission rejection, and at least 100 scoped proofs, then this
repository's canonical graphs and replay. This module runs
the search itself against the actual executable: a genuine multi-leaf PUCT
batch, graph equivalence of search nodes, checked rejection of alternatives,
scope closure after a cooperative cutoff, repeated searches on one session,
two independent pool workers sharing one model device, and bounded on-policy
updates with HTPS disabled and enabled. Proposals are scripted, or drawn from a fresh small
model, so the evidence does not depend on a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

from pantograph_async import PantographClient, PantographEnv, PantographProtocolError
from pantograph_async.diagnostics import distribution_source
from pantograph_async.probes import require

from maths_ai.core.git_worktree import dirty_provenance, read_status
from maths_ai.data_models.proof_components import ProposedAction, TacticCandidate
from maths_ai.lean_replay import seed_from_text

from .hypergraph import EdgeStatus, NodeStatus, SearchEndReason
from .joint_inference import HybridReasoner, PantographExecutor
from .pantograph_diagnostics import verify_live_application
from .pantograph_model_sexpr import create_model_sexpr_client, materialize_seed
from .proposal import ProposalSpec


# The theorems the live checks search, as the typed seeds search accepts.
CONJUNCTION = seed_from_text(goal="True ∧ True")
IDENTITY = seed_from_text(goal="∀ (p : Prop), p → p")

TARGET_BRANCH = "MCTS-AC-branch"
TARGET_BASE_COMMIT = "61f6d384730fca3dd7a56918377af4efb2c912cc"
APPLICATION_SOURCE_COMMIT = "54d7cefe04aed980734798f06528ed80c71b81ac"
CLIENT_REPOSITORY = "https://github.com/123nol/Pantograph-Client"
CLIENT_COMMIT = "955a4921b4cacaa49f4b3087f3982d7ca3629566"
GOLDEN_PATH = (
    Path(__file__).parents[1]
    / "gnn_inference/tests/fixtures/pantograph/canonical_graph_expectations.json"
)
SEARCH_SOURCES = (
    "maths_ai/hybrid_reasoner/joint_inference.py",
    "maths_ai/hybrid_reasoner/hypergraph.py",
    "maths_ai/hybrid_reasoner/mcts_diagnostics.py",
    "maths_ai/hybrid_reasoner/selection_policy.py",
    "maths_ai/hybrid_reasoner/execution_types.py",
    "maths_ai/hybrid_reasoner/execution_registry.py",
    "maths_ai/hybrid_reasoner/execution_limits.py",
    "maths_ai/hybrid_reasoner/retained_executor.py",
    "maths_ai/hybrid_reasoner/proof_recipes.py",
    "maths_ai/hybrid_reasoner/proof_replay.py",
    "maths_ai/gnn_inference/atp_lean_gnn/rl_reasoner.py",
    "maths_ai/gnn_inference/atp_lean_gnn/rl_live_collection.py",
    "maths_ai/gnn_inference/atp_lean_gnn/rl_training_driver.py",
    "maths_ai/gnn_inference/atp_lean_gnn/pln_rl_training.py",
    "maths_ai/hybrid_reasoner/pantograph_adapter.py",
    "maths_ai/hybrid_reasoner/pantograph_model_sexpr.py",
    "maths_ai/hybrid_reasoner/pantograph_diagnostics.py",
    "maths_ai/data_models/model_identity.py",
    "maths_ai/data_models/proof_components.py",
    "maths_ai/gnn_inference/atp_lean_gnn/inference.py",
    "maths_ai/gnn_inference/atp_lean_gnn/seed_admission.py",
    "maths_ai/lean_replay/columns.py",
    "maths_ai/lean_replay/errors.py",
    "maths_ai/lean_replay/lexer.py",
    "maths_ai/lean_replay/raw.py",
    "maths_ai/lean_replay/seed.py",
    "maths_ai/lean_replay/sexp.py",
    "maths_ai/lean_replay/text.py",
    "maths_ai/lean_replay/universe.py",
    "maths_ai/lean_replay/validation.py",
)
# Each leaf tries an elaboration failure and an admission before the tactic
# that closes it, so every leaf runs several alternatives on one state.
CONJUNCTION_SCRIPT = {
    "True ∧ True": (("constructor", 1.0),),
    "True": (("exact False.elim", 0.5), ("exact (by sorry)", 0.3), ("trivial", 0.2)),
}
PUCT_SETTINGS = {
    "selection_policy": "puct",
    "num_simulations": 8,
    "sim_batch_size": 2,
    "max_depth": 4,
    "max_nodes": 20,
}
LEARNING_TACTICS = {"constructor": 0, "trivial": 1, "intro": 2, "exact": 3}
LEARNING_MAX_LOCALS = 8
LEARNING_MAX_ARITY = 8
IDENTITY_SCRIPT = {
    "∀ (p : Prop), p → p": (("intro p h", 1.0),),
    "p": (("exact h", 1.0),),
}


class ScriptedSearchReasoner(HybridReasoner):
    """Deterministic proposals keyed by goal text; no checkpoint or PLN."""

    def __init__(self, server: PantographClient, script, **search) -> None:
        self.script = script
        self.proposal_batches: list[list[int]] = []
        super().__init__(
            None,
            None,
            None,
            executor=PantographExecutor(server),
            use_pln=False,
            **search,
        )

    def _build_gnn_engine(self, **_kwargs):
        return None

    def predict_next_tactics_for_nodes(self, nodes, draws=None):
        self.proposal_batches.append([node.id for node in nodes])
        return [
            [scripted_action(tactic, probability) for tactic, probability in self.script.get(node.goal.expression, ())]
            for node in nodes
        ]


def scripted_action(tactic: str, probability: float) -> ProposedAction:
    """A scripted zero-argument action whose probability is the complete action's."""
    return ProposedAction(
        candidate=TacticCandidate(tactic_name=tactic, arguments=[], probability=probability),
        raw_action_id=("scripted", tactic),
        log_prior=math.log(probability),
        provider="scripted-v1",
    )


# Widening on the real REPL: the root's first round has only a rejected
# command; its second round draws that command again (served from the node's
# cache) and the conjunction's constructor, whose two subgoals then close.
WIDENING_ROUNDS = {
    "True ∧ True": (
        (("exact False.elim", 0.6),),
        (("exact False.elim", 0.6), ("constructor", 0.4)),
    ),
    "True": ((("trivial", 1.0),),),
}
WIDENING_SETTINGS = {
    "selection_policy": "puct",
    "num_simulations": 12,
    "sim_batch_size": 1,
    "max_depth": 4,
    "max_nodes": 20,
    "top_k_tactics": 2,
    "widening_initial_actions": 1,
    "widening_coefficient": 2.0,
}


class WideningScriptedReasoner(ScriptedSearchReasoner):
    """Scripted rounds per goal text, served the way a sampling provider widens."""

    def __init__(self, server: PantographClient, rounds, **search) -> None:
        self.rounds = rounds
        self.served: dict[int, int] = {}
        super().__init__(server, {}, **search)

    def _proposal_source_is_finite(self) -> bool:
        return False

    def predict_next_tactics_for_nodes(self, nodes, draws=None):
        self.proposal_batches.append([node.id for node in nodes])
        batches = []
        for node, count in zip(nodes, draws or [self.top_k_tactics] * len(nodes)):
            script = self.rounds.get(node.goal.expression, ())
            index = self.served.get(node.id, 0)
            self.served[node.id] = index + 1
            entries = script[min(index, len(script) - 1)] if script else ()
            batches.append([scripted_action(tactic, p) for tactic, p in entries[:count]])
        return batches


class _CountingExecutor:
    """The real executor, recording every command it sends to Lean."""

    def __init__(self, inner: PantographExecutor) -> None:
        self.inner = inner
        self.server = inner.server
        self.commands: list[str] = []

    async def apply(self, registry, occurrence, command, *, holder="operation"):
        self.commands.append(command)
        return await self.inner.apply(registry, occurrence, command, holder=holder)


async def verify_widening(client: PantographClient) -> list[str]:
    """A later round finds the action the first lacked, on the real REPL."""
    baseline = (await client.stat()).n_goals
    reasoner = WideningScriptedReasoner(client, WIDENING_ROUNDS, **WIDENING_SETTINGS)
    executor = _CountingExecutor(reasoner.executor)
    reasoner.executor = executor
    graph = await reasoner.prove(CONJUNCTION)
    require((await client.stat()).n_goals == baseline, "Widening search leaked states.")
    require(graph.is_solved(), "Widening did not reach the conjunction's proof.")
    root = graph.root
    require(root.proposal_rounds >= 2, "The root was not widened.")
    require(
        root.command_outcomes.get("exact False.elim") is None
        and "exact False.elim" in root.command_outcomes,
        "The rejected command has no checked rejection on the root.",
    )
    require(
        executor.commands.count("exact False.elim") == 1,
        "A rejected command ran in Lean twice at one node.",
    )
    (edge,) = (graph.edges[eid] for eid in root.outgoing_edge_ids)
    require(edge.tactic.tactic_name == "constructor", "The widened edge is not the constructor.")
    require(len(edge.child_ids) == 2, "The constructor's AND obligations were not both kept.")
    require(
        abs(edge.log_mass - math.log(0.4)) < 1e-9,
        "The constructor's prior is not its scripted complete-action probability.",
    )
    priors = graph.puct_priors(root.id)
    require(abs(sum(priors.values()) - 1.0) < 1e-9, "Root priors do not sum to one.")
    return [
        "widening-later-action-after-checked-rejection",
        "widening-rejection-served-from-cache",
        "widening-and-obligations-kept",
        "widening-priors-normalized",
    ]


class _FirstBatchCutoff(ScriptedSearchReasoner):
    """A cooperative deadline that expires after the first simulation batch."""

    def __init__(self, server: PantographClient, script, **search) -> None:
        super().__init__(server, script, **search)
        self._deadline_checks = 0

    def _within_deadline(self, deadline) -> bool:
        self._deadline_checks += 1
        return self._deadline_checks == 1


def _puct_reasoner(client: PantographClient) -> ScriptedSearchReasoner:
    return ScriptedSearchReasoner(client, CONJUNCTION_SCRIPT, **PUCT_SETTINGS)


def _require_graph_equivalence(graph, edge, expected, label: str) -> None:
    from maths_ai.gnn_inference.atp_lean_gnn.graph import (
        dag_fingerprint,
        model_goal_to_dag,
    )

    require(len(edge.child_ids) == len(expected), f"Child count changed: {label}")
    for child_id, golden in zip(edge.child_ids, expected):
        goal = graph.nodes[child_id].goal
        require(
            goal.state_fingerprint() == golden["state_fingerprint"],
            f"Search node state differs from the canonical graph: {label}",
        )
        require(
            dag_fingerprint(model_goal_to_dag(goal)) == golden["dag_fingerprint"],
            f"Search node DAG differs from the canonical graph: {label}",
        )


async def verify_mcts_search(client: PantographClient, *, searches: int = 20) -> list[str]:
    """Run both search modes on one live session and check ownership afterwards."""
    if type(searches) is not int or searches < 2:
        raise ValueError("Repeated-search verification requires at least two searches.")
    goldens = json.loads(GOLDEN_PATH.read_text())["goals"]
    baseline = (await client.stat()).n_goals
    checks = []

    reasoner = _puct_reasoner(client)
    graph = await reasoner.prove(CONJUNCTION)
    require(graph.is_solved(), "PUCT search did not prove the conjunction.")
    require(
        any(len(batch) > 1 for batch in reasoner.proposal_batches),
        "PUCT search never proposed a multi-leaf batch.",
    )
    root_edges = [graph.edges[eid] for eid in graph.nodes[graph.root_id].outgoing_edge_ids]
    require(len(root_edges) == 1, "Conjunction split produced extra root edges.")
    for child_id in root_edges[0].child_ids:
        child = graph.nodes[child_id]
        require(child.status == NodeStatus.READY, "A conjunction leaf stayed open.")
        tactics = [graph.edges[eid].tactic.tactic_name for eid in child.outgoing_edge_ids]
        # The elaboration failure and the admission were rejected by Lean.
        require(tactics == ["trivial"], f"Rejected alternatives became edges: {tactics}")
    require(
        all(edge.visit_stats.virtual_loss == 0 for edge in graph.edges.values()),
        "PUCT search left virtual losses.",
    )
    require(
        root_edges[0].visit_stats.N >= 1
        and root_edges[0].visit_stats.N_v == root_edges[0].visit_stats.N,
        "Completed PUCT simulations did not back up valid evidence.",
    )
    _require_graph_equivalence(
        graph, root_edges[0], goldens["conjunction_split"], "conjunction_split"
    )
    require((await client.stat()).n_goals == baseline, "PUCT search leaked states.")
    checks += [
        "puct-multi-leaf-batch",
        "checked-rejection-of-alternatives",
        "puct-graph-equivalence",
    ]

    best_first = ScriptedSearchReasoner(client, IDENTITY_SCRIPT, max_depth=4)
    graph = await best_first.prove(IDENTITY)
    require(graph.is_solved(), "Best-first search did not prove the identity.")
    root_edge = graph.edges[graph.nodes[graph.root_id].outgoing_edge_ids[0]]
    require(root_edge.status == EdgeStatus.READY, "Identity edge stayed unsolved.")
    _require_graph_equivalence(
        graph, root_edge, goldens["identity_intro"], "identity_intro"
    )
    require((await client.stat()).n_goals == baseline, "Best-first search leaked states.")
    checks.append("best-first-graph-equivalence")

    cutoff = _FirstBatchCutoff(client, CONJUNCTION_SCRIPT, **PUCT_SETTINGS)
    graph = await cutoff.prove(CONJUNCTION, deadline=0.0)
    require(
        graph.end_reason == SearchEndReason.DEADLINE and not graph.is_solved(),
        "Cooperative cutoff did not end the search with a partial graph.",
    )
    require(
        bool(graph.nodes[graph.root_id].outgoing_edge_ids),
        "Cooperative cutoff discarded the completed root expansion.",
    )
    require(
        (await client.stat()).n_goals == baseline,
        "Theorem scope did not close after a cooperative cutoff.",
    )
    checks.append("cooperative-deadline-scope-closure")

    for _ in range(searches):
        graph = await _puct_reasoner(client).prove(CONJUNCTION)
        require(graph.is_solved(), "A repeated PUCT search failed.")
        require(
            (await client.stat()).n_goals == baseline,
            "Repeated searches did not return allocations to baseline.",
        )
    checks.append(f"repeated-searches-{searches}")
    return checks


async def verify_shared_device_workers(env: PantographEnv) -> list[str]:
    """Two pool workers with independent clients share one model device."""
    import torch

    from maths_ai.gnn_inference.atp_lean_gnn.rl_live_collection import LiveReasonerPool
    from maths_ai.gnn_inference.atp_lean_gnn.rl_resources import ResourcePlan

    plan = ResourcePlan(
        reported_cpu_count=2,
        affinity_cpu_count=2,
        quota_cpu_count=None,
        usable_cpu_count=2,
        requested_worker_count=2,
        worker_count=2,
        collection_devices=(torch.device("cpu"),),
        update_device=torch.device("cpu"),
        gpu_resources=(),
    )

    async def server_factory(_worker_id):
        return await create_model_sexpr_client(env)

    def reasoner_factory(_worker_id, _device, _model, server):
        return _puct_reasoner(server)

    async def search_factory(reasoner, theorem, _proposal, deadline):
        return await reasoner.prove(theorem, deadline=deadline)

    pool = await LiveReasonerPool.create(
        model=torch.nn.Linear(1, 1),
        plan=plan,
        server_factory=server_factory,
        reasoner_factory=reasoner_factory,
        search_factory=search_factory,
        is_solved=lambda graph: graph.is_solved(),
        is_cooperative_deadline=lambda graph: False,
        startup_timeout_s=env.server_start_timeout_s,
        shutdown_timeout_s=env.worker_shutdown_timeout_s,
    )
    clients = [worker.server for worker in pool.workers]
    async with pool:
        require(
            len({client.session_id for client in clients}) == 2,
            "Pool workers share one client session.",
        )
        require(
            pool.workers[0].model is pool.workers[1].model,
            "Workers on one device did not share the model replica.",
        )
        pool.assert_generation(0)
        # Scripted reasoners propose no policy actions, so their proposals are unused.
        result = await pool.collect_round(
            [CONJUNCTION] * 4, timeout_s=120, model_generation=0, proposals=[None] * 4
        )
        require(result.stats["solved"] == 4, "Pool workers did not prove every theorem.")
        for client in clients:
            require((await client.stat()).n_goals == 0, "A pool worker leaked states.")
    require(
        all(client.phase.value == "CLOSED" for client in clients),
        "A pool client was not closed and reaped.",
    )
    return ["two-workers-one-model-device"]


async def verify_learning_round(client: PantographClient) -> dict:
    """Bounded RL rounds with HTPS disabled and enabled.

    Each round runs PUCT collection, one on-policy step, mining, and (when
    enabled) the HTPS step, on a fresh small actor-critic, so this checks the
    update path and its accounting, not model quality.
    """
    import math

    import torch
    from torch.optim import AdamW

    from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
    from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
    from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import LocalCandidateProvider
    from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
    from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
    from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
        supported_structural_labels,
        train_step_htps_style,
        train_step_onpolicy,
    )
    from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab_from_labels
    from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
    from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
        extract_critic_samples,
        extract_minimal_hypertree,
    )

    async with client.state_scope() as scope:
        _, root, _ = await materialize_seed(scope, CONJUNCTION)
    # Successor states add structural labels the seed goal lacks.
    node_vocab = build_vocab_from_labels(
        [
            *(node.label for node in model_goal_to_dag(root).nodes),
            *supported_structural_labels(
                max_locals=LEARNING_MAX_LOCALS, max_arity=LEARNING_MAX_ARITY
            ),
        ]
    )

    async def learning_round(htps_enabled: bool) -> dict:
        torch.manual_seed(0)
        model = build_actor_critic_model(
            model_spec=ModelSpec.from_dict(
                {
                    "architecture": "graphsage",
                    "hidden_dim": 32,
                    "dropout": 0.1,
                    "encoder": {"num_layers": 2},
                    "use_node_type": True,
                    "max_args": 1,
                }
            ),
            num_node_labels=len(node_vocab),
            num_tactics=len(LEARNING_TACTICS),
        )
        with torch.no_grad():
            # Favor the two tactics that close the seed so the round has a proof.
            model.actor.base.bias[LEARNING_TACTICS["constructor"]] += 4.0
            model.actor.base.bias[LEARNING_TACTICS["trivial"]] += 4.0
        reasoner = RLHybridReasoner(
            model,
            node_vocab,
            LEARNING_TACTICS,
            executor=PantographExecutor(client),
            candidate_provider=LocalCandidateProvider(),
            top_k_tactics=3,
            use_pln=False,
            **PUCT_SETTINGS,
        )
        baseline = (await client.stat()).n_goals
        result = await reasoner.prove(CONJUNCTION, proposal=ProposalSpec(mode="sample", seed=0))
        require((await client.stat()).n_goals == baseline, "Learning search leaked states.")
        require(
            bool(result.observations),
            "The learning search produced no on-policy actions.",
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]

        # The driver's order: one on-policy step, mining, then HTPS steps.
        metrics = train_step_onpolicy(
            model,
            AdamW(model.parameters(), lr=1e-3),
            [result],
            reasoner.dag_featurize_data,
            reward_cfg=RewardConfig(step_penalty=0.01),
            bc_weight=0.0,
        )
        onpolicy_steps = int(metrics["onpolicy_optimizer_step"])
        critic_samples = extract_critic_samples(result.graph, visit_threshold=1)
        imitation_samples = extract_minimal_hypertree(result.graph, result.observations)
        htps_steps = 0
        if htps_enabled:
            htps = train_step_htps_style(
                model,
                AdamW(model.parameters(), lr=1e-4),
                imitation_samples,
                critic_samples,
                reasoner.dag_featurize_data,
            )
            metrics = {**metrics, **htps}
            htps_steps = int(htps["htps_optimizer_step"])
            require(
                htps_steps == int(bool(imitation_samples or critic_samples)),
                "The HTPS update step count does not match the mined replay.",
            )
        require(
            all(math.isfinite(float(value)) for value in metrics.values()),
            "A bounded learning round produced a non-finite metric.",
        )
        require(onpolicy_steps == 1, "The on-policy update did not take its single step.")
        require(
            int(metrics["structural_unknown_label_count"]) == 0,
            "A learning state had structural labels outside the vocabulary.",
        )
        require(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters())
            ),
            "The optimizer steps did not change any parameter.",
        )
        return {
            "solved": result.graph.is_solved(),
            "accepted_edges": len(result.accepted_edge_ids()),
            "observations": len(result.observations),
            "draws": result.draws(),
            "imitation_samples": len(imitation_samples),
            "critic_samples": len(critic_samples),
            "onpolicy_optimizer_steps": onpolicy_steps,
            "htps_optimizer_steps": htps_steps,
            "model_generation": onpolicy_steps + htps_steps,
        }

    disabled = await learning_round(htps_enabled=False)
    enabled = await learning_round(htps_enabled=True)
    require(
        disabled["model_generation"] == 1 and enabled["model_generation"] == 2,
        "Generation did not advance by the actual optimizer steps.",
    )
    return {"htps_disabled": disabled, "htps_enabled": enabled}


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _installed_client() -> dict:
    """The installed distribution must be the pinned maintained-repository commit."""
    source = distribution_source() or {}
    vcs = source.get("vcs_info", {})
    require(
        source.get("url") == CLIENT_REPOSITORY and vcs.get("commit_id") == CLIENT_COMMIT,
        f"Installed client is not {CLIENT_REPOSITORY}@{CLIENT_COMMIT}: {source}",
    )
    return {"repository": CLIENT_REPOSITORY, "commit": CLIENT_COMMIT}


def _target_identity() -> dict:
    root = Path(__file__).parents[2]

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, timeout=5
        ).strip()

    dirty, entries = dirty_provenance(root, read_status(root))
    return {
        "target_branch": TARGET_BRANCH,
        "target_base_commit": TARGET_BASE_COMMIT,
        "application_source_commit": APPLICATION_SOURCE_COMMIT,
        "client": _installed_client(),
        "repository_revision": git("rev-parse", "HEAD"),
        "worktree_clean": not dirty,
        "dirty_sha256": dirty,
        "dirty_entries": entries,
        "search_sources_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SEARCH_SOURCES
        },
    }


async def verify_live_mcts(
    env: PantographEnv, *, proofs: int = 100, searches: int = 20
) -> dict:
    """The application gate, then the search gate, with the target's own identity."""
    report = {
        "target": _target_identity(),
        "configuration": {
            "imports": list(env.imports),
            "request_timeout_s": env.timeout,
            "server_start_timeout_s": env.server_start_timeout_s,
            "worker_shutdown_timeout_s": env.worker_shutdown_timeout_s,
            "source_root_revision": _git_revision(env.source_root),
            "scoped_proofs": proofs,
            "repeated_searches": searches,
            "search": PUCT_SETTINGS,
            "device": "cpu",
        },
        "unavailable_resources": {
            "gpu": "not used; this is a CPU ownership gate, not GPU deployment evidence",
            "checkpoint": "not used; proposals are scripted or from a fresh small model",
            "petta": "not used; searches run with use_pln=False",
            "complete_theorem_certification": False,
        },
    }
    report["client"] = await verify_live_application(env, proofs=proofs)
    client = await create_model_sexpr_client(env)
    try:
        initial = (await client.stat()).n_goals
        checks = await verify_mcts_search(client, searches=searches)
        checks += await verify_widening(client)
        learning = await verify_learning_round(client)
        final = (await client.stat()).n_goals
        if final != initial:
            raise PantographProtocolError("MCTS search checks leaked states.")
    finally:
        await client.aclose()
    report["search"] = {
        "checks": checks,
        "initial_states": initial,
        "final_states": final,
        "phase": client.phase.value,
        "subprocess_reaped": client.phase.value == "CLOSED",
    }
    report["search"]["checks"] += await verify_shared_device_workers(env)
    report["learning"] = learning
    return report
