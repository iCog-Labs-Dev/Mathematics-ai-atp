import asyncio
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from pantograph_async import PantographEnv
from maths_ai.lean_replay import seed_from_text
from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor, HybridReasoner
from maths_ai.hybrid_reasoner.execution_registry import SnapshotRegistry
from maths_ai.hybrid_reasoner.execution_types import (
    ExecutionOccurrence,
    ObligationKey,
    root_declarations,
    bind_arguments,
)
from maths_ai.data_models.tactic_commands import render_tactic_command
from maths_ai.data_models.proof_components import ProposedAction
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client,
    materialize_seed,
)
from maths_ai.gnn_inference.atp_lean_gnn.inference import (
    _resolve_local_node_name,
    local_declarations,
)
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import make_dag_featurizer
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import (
    ArgumentReference,
    CandidatePoolSnapshot,
)
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import LibraryMemory
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model


@pytest.mark.pantograph_live
def test_controlled_local_library_and_mixed_actions_against_custom_repl():
    root = os.environ.get("PANTOGRAPH_TEST_SOURCE_ROOT")
    repl = os.environ.get("PANTOGRAPH_TEST_REPL")
    if not root or not repl:
        pytest.skip("Explicit live Lean paths not configured.")

    async def run():
        env = PantographEnv(
            source_root=Path(root),
            pantograph_repl=Path(repl),
            imports=("Init", "Mathlib"),
            server_start_timeout_s=120,
        )
        async with await create_model_sexpr_client(env) as client:
            cases = [
                ("p", ["p : Prop", "h : p"], "exact", None),
                ("True", [], "exact", "True.intro"),
                (
                    "n + 0 = m",
                    ["n : Nat", "m : Nat", "h : n = m"],
                    "rw",
                    "Nat.add_zero",
                ),
            ]
            for target, hyps, tactic, lemma in cases:
                async with client.state_scope() as scope:
                    state, goal, _ = await materialize_seed(
                        scope, seed_from_text(goal=target, hypotheses=hyps)
                    )
                    registry = SnapshotRegistry(scope, max_snapshots=3)
                    anchor = registry.adopt_root(state, "source-aware test root")
                    occurrence = ExecutionOccurrence(
                        0,
                        ObligationKey.root(),
                        state.goals[0].goal_name,
                        goal,
                        root_declarations(goal),
                        anchor,
                    )
                    try:
                        dag = model_goal_to_dag(goal)
                        vocab = build_vocab([dag])
                        _, data = make_dag_featurizer(vocab)(goal)
                        local_ids = tuple(
                            data.premise_mask.nonzero().flatten().tolist()
                        )
                        refs = []
                        names = []
                        if hyps:
                            index = next(
                                i
                                for i, n in enumerate(dag.nodes)
                                if n.label == "Hyp"
                                and _resolve_local_node_name(
                                    n, dag, local_declarations(goal)
                                )
                                == "h"
                            )
                            refs.append(ArgumentReference("local", index))
                            names.append("h")
                        library = None
                        if lemma:
                            library = LibraryMemory(
                                [100],
                                [lemma],
                                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                                "live",
                            )
                            refs.append(ArgumentReference("library", 100))
                            names.append(lemma)
                        pool = CandidatePoolSnapshot(
                            local_ids,
                            (100,) if lemma else (),
                            (0.5,) if lemma else (),
                            artifact_fingerprint="live" if lemma else "",
                            library=library,
                        )
                        spec = ModelSpec.from_dict(
                            dict(
                                architecture="graphsage",
                                hidden_dim=8,
                                encoder={"num_layers": 1},
                                text_dim=4,
                                retrieval_binding="live" if lemma else "",
                            )
                        )
                        model = build_actor_critic_model(
                            model_spec=spec, num_node_labels=len(vocab), num_tactics=1
                        )
                        evaluated = model.evaluate_actions(
                            Batch.from_data_list([data]),
                            torch.tensor([0]),
                            (tuple(refs),),
                            candidate_pools=(pool,),
                        )
                        assert torch.isfinite(evaluated.logp).all()
                        command = render_tactic_command(
                            TacticCandidate(
                                tactic_name=tactic,
                                arguments=names,
                                probability=1.0,
                                argument_refs=bind_arguments(goal, names),
                            )
                        )
                        outcome = await PantographExecutor(client).apply(
                            registry, occurrence, command
                        )
                        assert outcome.success and not outcome.children, outcome.error
                        assert outcome.selected_complete
                        certificate = await scope.certify(registry.state(outcome.lease))
                        assert certificate.status == "certified"
                        await registry.release(outcome.lease)
                        await registry.release(anchor)
                    finally:
                        await registry.close()
                assert (await client.stat()).n_goals == 0

    asyncio.run(run())


@pytest.mark.pantograph_live
@pytest.mark.parametrize("mixed", [False, True])
def test_global_and_mixed_arguments_certify_through_production_search(live_env, mixed):
    class ControlledReasoner(HybridReasoner):
        def _build_gnn_engine(self, **kwargs):
            return None

        def predict_next_tactics_for_nodes(self, nodes, draws=None):
            result = []
            for node in nodes:
                goal = node.goal
                if not mixed:
                    name, arguments = "exact", ["True.intro"]
                elif node.depth == 0:
                    local = next(
                        x.execution_or_live_name
                        for x in goal.locals
                        if x.execution_or_live_name == "h"
                    )
                    name, arguments = "rw", [local, "Nat.add_zero"]
                else:
                    name, arguments = "rfl", []
                tactic = TacticCandidate(
                    tactic_name=name,
                    arguments=arguments,
                    probability=1.0,
                    argument_refs=bind_arguments(goal, arguments),
                )
                result.append(
                    [
                        ProposedAction(
                            tactic, (name, tuple(arguments)), 0.0, "controlled"
                        )
                    ]
                )
            return result

    async def run():
        async with await create_model_sexpr_client(live_env) as client:
            reasoner = ControlledReasoner(
                None,
                None,
                None,
                executor=PantographExecutor(client),
                use_pln=False,
                selection_policy="puct",
                num_simulations=8,
                top_k_tactics=1,
            )
            seed = seed_from_text(
                goal="n + 0 = m" if mixed else "True",
                hypotheses=["n : Nat", "m : Nat", "h : n = m"] if mixed else [],
            )
            graph = await reasoner.prove(seed)
            assert graph.is_solved(), graph.end_reason
            assert graph.execution["replay_attempts"] >= 1
            assert graph.execution["intermediate_reconstructions"] == 0
            assert (await client.stat()).n_goals == 0

    asyncio.run(run())
