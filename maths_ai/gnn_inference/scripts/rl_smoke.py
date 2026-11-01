"""Live smoke harness for the on-policy RL loop (manual — needs Lean + petta).

Run:  uv run python -m maths_ai.gnn_inference.scripts.rl_smoke

Pass ``--source-root <lake project>`` to run the same check against a compiled
Mathlib project instead of core Lean. The seed goal is provable in either, so a
failure under ``--source-root`` that does not reproduce without it points at the
Lean environment rather than the RL loop.

The search runs with the branch's PUCT simulation settings by default; pass
``--selection-policy legacy`` for the best-first loop, and ``--no-pln`` to run
without the petta binary.

What it validates end-to-end with a REAL Pantograph server and a REAL petta binary:
  collect: RLHybridReasoner samples from a fresh (untrained) actor-critic, decodes each
  draw into a Lean tactic, the PantographExecutor applies it, PLN scores the survivors,
  and the search returns an RLSearchResult.
  train:   train_step_onpolicy harvests the on-policy edges + failure records and takes
  exactly one gradient step.

The policy is untrained, so most draws are rejected by Lean — that IS the expected
signal path (failure records dominate). To make the QED branch reachable within a few
draws, the actor's base head is biased toward "intro"/"exact"/"assumption" on the seed
goal ``∀ (p : Prop), p → p``. This validates control flow and reward plumbing only;
proving quality waits on trained checkpoints.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import torch
from torch.optim import AdamW

from maths_ai.lean_replay import seed_from_text
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor
from maths_ai.hybrid_reasoner.proposal import PROPOSAL_MODES, ProposalSpec
from pantograph_async import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client,
    materialize_seed,
)

from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import (
    ActorCriticWithArgsClassifier,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import LocalCandidateProvider
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    supported_structural_labels,
    train_step_onpolicy,
)
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab_from_labels
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner


TACTIC_VOCAB = {
    "intro": 0,
    "exact": 1,
    "assumption": 2,
    "apply": 3,
    "constructor": 4,
    "trivial": 5,
}

# Seed theorem: identity. `intro h` then `exact h` (or `assumption`) closes it.
SEED_GOAL = "p → p"
SEED_HYPS = ["p : Prop"]
# Bounds for successor states within the search depth; a state beyond them
# still fails the training step's missing structural-label check.
SMOKE_MAX_LOCALS = 8
SMOKE_MAX_ARITY = 8


def build_model(node_vocab: dict[str, int]) -> ActorCriticWithArgsClassifier:
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
        num_tactics=len(TACTIC_VOCAB),
    )
    # Bias the fresh policy toward the tactics that can actually close the seed goal,
    # so the QED branch fires within a handful of i.i.d. draws. The base head's random
    # logits are O(1), so +6/+4 makes intro/assumption dominate the softmax.
    with torch.no_grad():
        model.actor.base.bias[TACTIC_VOCAB["intro"]] += 6.0
        model.actor.base.bias[TACTIC_VOCAB["assumption"]] += 4.0
    return model


async def run_smoke(
    env: PantographEnv,
    *,
    use_pln: bool = True,
    proposal: ProposalSpec = ProposalSpec(mode="sample", seed=0),
    **search,
) -> None:
    torch.manual_seed(0)

    env.verify()
    print(f"[rl_smoke] starting Pantograph server: {env.describe()}")
    async with await create_model_sexpr_client(env) as server:
        seed = seed_from_text(goal=SEED_GOAL, hypotheses=SEED_HYPS)
        async with server.state_scope() as scope:
            _, seed_goal, _ = await materialize_seed(scope, seed)
        # Successor states add structural labels the seed goal lacks.
        node_vocab = build_vocab_from_labels(
            [
                *(node.label for node in model_goal_to_dag(seed_goal).nodes),
                *supported_structural_labels(
                    max_locals=SMOKE_MAX_LOCALS, max_arity=SMOKE_MAX_ARITY
                ),
            ]
        )
        model = build_model(node_vocab)
        executor = PantographExecutor(server)

        reasoner = RLHybridReasoner(
            model,
            node_vocab,
            TACTIC_VOCAB,
            executor=executor,
            candidate_provider=LocalCandidateProvider(),
            top_k_tactics=4,
            max_depth=4,
            max_nodes=30,
            use_pln=use_pln,
            **search,
        )

        print(
            f"[rl_smoke] collect ({reasoner.selection_policy}, {proposal.mode} proposals): proving "
            f"`{SEED_GOAL}` with hyps {SEED_HYPS}"
        )
        result = await reasoner.prove(seed, proposal=proposal)

        graph = result.graph
        print(
            f"[rl_smoke] search done: certified={graph.is_solved()} "
            f"candidate={graph.has_candidate()} end={graph.end_reason.value} "
            f"nodes={len(graph.nodes)} edges={len(graph.edges)} "
            f"on-policy edges={len(result.accepted_edge_ids())} "
            f"observations={len(result.observations)} draws={result.draws()}"
        )
        print(f"[rl_smoke] execution: {graph.execution}")
        assert graph.execution["intermediate_reconstructions"] == 0
        assert result.observations, "no on-policy signal collected — no settled policy draw"

        optimizer = AdamW(model.parameters(), lr=1e-3)
        metrics = train_step_onpolicy(
            model,
            optimizer,
            [result],
            reasoner.dag_featurize_data,
            reward_cfg=RewardConfig(step_penalty=0.01),
            bc_weight=0.0,
        )
        print(f"[rl_smoke] train step metrics: {metrics}")
        assert (
            metrics.get("num_transitions", 0.0) + metrics.get("num_failures", 0.0) > 0
        )
        assert all(v == v for v in metrics.values()), "NaN in training metrics"
        print(
            "[rl_smoke] OK — collect → harvest → one on-policy gradient step completed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live smoke test for the on-policy RL loop"
    )
    parser.add_argument(
        "--source-root",
        type=str,
        required=True,
        help="Pinned Mathlib Lake project root",
    )
    parser.add_argument(
        "--pantograph-repl",
        type=str,
        required=True,
        help="Pinned model-S-expression Pantograph REPL binary",
    )
    parser.add_argument(
        "--selection-policy", choices=("legacy", "puct"), default="puct"
    )
    parser.add_argument("--num-simulations", type=int, default=8)
    parser.add_argument("--sim-batch-size", type=int, default=2)
    parser.add_argument(
        "--no-pln", action="store_true", help="Run without the petta binary"
    )
    parser.add_argument(
        "--proposal-mode", choices=PROPOSAL_MODES, default="sample",
        help="'sample' draws tactics and arguments; 'greedy' takes the step-wise argmax of both",
    )
    parser.add_argument("--proposal-seed", type=int, default=0, help="Seed of the search's draws")
    args = parser.parse_args()
    search = (
        {
            "selection_policy": "puct",
            "num_simulations": args.num_simulations,
            "sim_batch_size": args.sim_batch_size,
        }
        if args.selection_policy == "puct"
        else {}
    )

    source_root = Path(args.source_root)
    env = PantographEnv(
        source_root=source_root,
        pantograph_repl=Path(args.pantograph_repl),
        imports=("Init", "Mathlib"),
    )
    proposal = ProposalSpec(mode=args.proposal_mode, seed=args.proposal_seed)
    asyncio.run(run_smoke(env, use_pln=not args.no_pln, proposal=proposal, **search))


if __name__ == "__main__":
    main()
