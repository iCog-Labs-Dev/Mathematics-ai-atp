"""Live graph parity, runtime reference identity, and native branch isolation."""

import asyncio
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import _build_policy
from maths_ai.gnn_inference.atp_lean_gnn.live_policy import LiveJointPolicy
from maths_ai.gnn_inference.atp_lean_gnn.live_state import (
    LiveLeanSession, LiveStateError, TacticFailure, normalized_goal_record, prepare_live_goal,
)
from maths_ai.gnn_inference.atp_lean_gnn.preparation import prepare_example
from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import _make_model_record
from maths_ai.gnn_inference.scripts.run_gnn_live import run_live_smoke


def expression(sexp, pp):
    return {"modelSexpVersion": 1, "modelSexp": sexp, "pp": pp}


def local_goal(name="h_fresh_987"):
    return {"name": "goal0", "isConversion": False,
            "target": expression("(:fv FV0)", "P"), "vars": [
                {"name": "p_internal", "userName": "P", "contextIndex": 0,
                 "binderRole": ":explicit", "type": expression("(:sort Prop)", "Prop")},
                {"name": "h_internal", "userName": name, "contextIndex": 1,
                 "binderRole": ":explicit", "type": expression("(:fv FV0)", "P")},
            ]}


def fixture_policy(goal, *, library=False):
    """Actual GNN/GRU with controlled heads; not the accepted trained weights."""
    dag = proof_state_to_dag("⊢ P", **normalized_goal_record(goal))
    # Intentionally omit display names and FV labels: runtime mapping must not
    # infer identity from <UNK> vocabulary entries.
    labels = {node.label for node in dag.nodes if not node.label.startswith("FV")}
    labels -= {variable["userName"] for variable in goal["vars"]}
    vocab = {name: i for i, name in enumerate(["<UNK>", *sorted(labels)])}
    config = {"pointer_config": {"prepared_root": ".", "run_root": ".", "gnn_type": "sage", "use_node_type": True,
        "edge_mode": "bidirectional", "max_args": 3,
        "model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0, "max_args": 3}}}
    tactics = {"<UNK_TACTIC>": 0, "exact": 1, "sorry": 2}
    policy = _build_policy(config, vocab, tactics, 8).eval()
    with torch.no_grad():
        policy.pointer.backbone.classifier.weight.zero_()
        policy.pointer.backbone.classifier.bias.copy_(torch.tensor([-10., 10., 0.]))
        policy.arg_count_head.weight.zero_()
        policy.arg_count_head.bias.copy_(torch.tensor([-20., 20., -20., -20.]))
        policy.source_head.weight.zero_()
        policy.source_head.bias.copy_(torch.tensor([0., 20.]) if library else torch.tensor([20., 0.]))
        policy.pointer.argument_selector.out_proj.weight.zero_()
        policy.pointer.argument_selector.out_proj.bias.zero_()
    # local_type_signals intentionally uses the real training vocabulary. For
    # this OOV test, select the desired runtime FV node through controlled scores.
    original_score = policy.score_step

    def score_step(hidden, pool, **kwargs):
        source, local, lemmas = original_score(hidden, pool, **kwargs)
        wanted = next(node.id for node in dag.nodes if node.label == "FV1") if goal["vars"] else None
        if wanted in pool.local_ids:
            local = local.clone()
            local[pool.local_ids.index(wanted)] += 30
        return source, local, lemmas

    policy.score_step = score_step
    index = SimpleNamespace(lemma_ids=[7], lemma_names=["True.intro"])
    system = SimpleNamespace(policy=policy, node_vocab=vocab, tactic_vocab=tactics, profile="base",
        lemma_index=index, retrieve=lambda states, **kwargs: ([[7]], np.zeros((1, 1, 8), dtype=np.float32),
                                                  np.zeros((1, 1), dtype=np.float32)))
    return LiveJointPolicy(system, edge_mode="bidirectional")


def test_live_graph_matches_offline_preparation_and_resolves_oov_names():
    goal = local_goal()
    raw = {"invocation": {"goalsBefore": [goal]}}
    row = DatasetRow("P : Prop\nh_fresh_987 : P\n⊢ P", "Fixture", "exact h_fresh_987", "val", 0)
    record = _make_model_record(row, raw, {})
    offline = prepare_example(row, sexpr_data=record, use_sexpr=True)
    policy = fixture_policy(goal)
    prediction = policy.predict(goal, top_k=3)
    assert [node.as_dict() for node in prediction.graph.dag.nodes] == [node.as_dict() for node in offline.dag.nodes]
    assert prediction.graph.dag.edges == offline.dag.edges
    candidate, = prediction.candidates
    assert candidate.code == "exact «h_fresh_987»"
    assert candidate.references[0].identity == 1  # Context index, not graph node ID.
    assert candidate.log_probability <= 0
    assert {item["tactic"] for item in prediction.rejected} == {"sorry", "<UNK_TACTIC>"}


@pytest.mark.parametrize("damage", ["missing_model", "old_version", "missing_index", "duplicate_index", "unknown_local", "conversion"])
def test_live_graph_rejects_lossy_inputs(damage):
    goal = local_goal()
    if damage == "missing_model":
        del goal["target"]["modelSexp"]
    elif damage == "old_version":
        goal["target"]["modelSexpVersion"] = 0
    elif damage == "missing_index":
        del goal["vars"][0]["contextIndex"]
    elif damage == "duplicate_index":
        goal["vars"][1]["contextIndex"] = 0
    elif damage == "unknown_local":
        goal["target"]["modelSexp"] = "(:fv FV99)"
    else:
        goal["isConversion"] = True
    with pytest.raises(LiveStateError):
        prepare_live_goal(goal, {"<UNK>": 0}, edge_mode="bidirectional")


def test_library_arguments_use_bound_corpus_names():
    goal = {"name": "g", "isConversion": False, "vars": [],
            "target": expression("(:c True)", "True")}
    prediction = fixture_policy(goal, library=True).predict(goal, top_k=1)
    assert prediction.candidates[0].code == "exact _root_.«True».«intro»"
    assert prediction.candidates[0].references[0].identity == "True.intro"


def test_ambiguous_local_name_is_rejected_without_dropping_argument():
    goal = local_goal("P")
    prediction = fixture_policy(goal).predict(goal, top_k=1)
    assert not prediction.candidates
    assert "ambiguous local name" in prediction.rejected[0]["reason"]


def test_accepted_cues_and_action_scores_match_bundle_greedy_decode():
    goal = local_goal()
    adapter = fixture_policy(goal)
    # Retain FV labels in the model vocabulary so the accepted soft cues can
    # recognize them, exactly as in ordinary prepared inputs.
    for label in ("FV0", "FV1"):
        adapter.system.node_vocab[label] = len(adapter.system.node_vocab)
    # Rebuild the fixture backbone's label table for the expanded vocabulary.
    old = adapter.system.policy.pointer.backbone.label_embedding
    embedding = torch.nn.Embedding(len(adapter.system.node_vocab), old.embedding_dim)
    with torch.no_grad():
        embedding.weight[:old.num_embeddings].copy_(old.weight)
    adapter.system.policy.pointer.backbone.label_embedding = embedding
    adapter.system.profile = "accepted_local_cues"
    adapter.system.policy.local_type_bonus_strengths = {"exact": 8.0}
    with torch.no_grad():
        adapter.system.policy.arg_count_head.bias.copy_(torch.tensor([-1., 2., -1., -1.]))
        adapter.system.policy.source_head.bias.copy_(torch.tensor([2., 0.]))
    adapter.id_to_node = {identity: name for name, identity in adapter.system.node_vocab.items()}
    predicted = adapter.predict(goal, top_k=1)
    # The live wrapper must use the accepted bundle's own top-one result and
    # predicted-count semantics, rather than introducing an inference policy.
    from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import LoadedJointSourceBundle, GRAPH_FORMAT
    reference = LoadedJointSourceBundle.predict(adapter.system, predicted.graph.batch,
        [predicted.graph.text], graph_format=GRAPH_FORMAT, normalized_records=[predicted.graph.normalized])
    selected = reference[0]["arguments"][0]
    assert selected["source"] == "local"
    assert predicted.graph.local_node_context[selected["identity"]] == predicted.candidates[0].references[0].identity
    expected = (torch.log_softmax(torch.tensor([-10., 10., 0.]), dim=0)[1]
                + torch.log_softmax(torch.tensor([-1., 2., -1., -1.]), dim=0)[1]
                + torch.log_softmax(torch.tensor([2., 0.]), dim=0)[0])
    assert predicted.candidates[0].log_probability == pytest.approx(float(expected), abs=1e-5)


def native_root():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for pinned native integration")
    root = Path(configured).resolve()
    if not (root / ".lake/build/bin/repl").is_file():
        pytest.fail("MATHS_AI_PANTOGRAPH_ROOT has no built REPL")
    return root


def test_native_predictions_preserve_siblings_and_parent_after_failure():
    root = native_root()

    async def run():
        async with LiveLeanSession(source_root=root, pantograph_repl=root / ".lake/build/bin/repl",
                                   imports=("Init",)) as session:
            initial = await session.start("∀ (P : Prop), P → P ∧ P")
            assert len(initial.goals) == 1
            noop = await session.step(initial, "skip")
            assert len(noop.goals) == 1  # Automatic mode used to duplicate it.
            introduced = await session.step(initial, "intro P h_fresh_987")
            assert len(introduced.goals[0]["vars"]) == 2
            split = await session.step(introduced, "constructor")
            assert len(split.goals) == 2
            before = copy.deepcopy(split.goals)
            with pytest.raises(TacticFailure):
                await session.step(split, "exact DoesNotExist")
            with pytest.raises(TacticFailure):
                await session.step(split, "exact (")
            policy = fixture_policy(split.goals[0])
            first = policy.predict(split.goals[0], top_k=1).candidates[0]
            left_done = await session.step(split, first.code)
            assert len(left_done.goals) == 1
            assert left_done.goals[0]["name"] == split.goals[1]["name"]
            # Same saved parent can independently produce another child.
            other_child = await session.step(split, first.code)
            assert len(other_child.goals) == 1 and split.goals == before
            second = policy.predict(left_done.goals[0], top_k=1).candidates[0]
            done = await session.step(left_done, second.code)
            assert done.closed
            root_expr = await session.client.call("goal.print", {"stateId": done.state_id}, timeout=30)
            assert root_expr.get("root", {}).get("pp")

            let_state = await session.step(introduced, "let local_copy := h_fresh_987")
            graph = prepare_live_goal(let_state.goals[0], {"<UNK>": 0}, edge_mode="bidirectional")
            assert graph.normalized["hyp_sexps"][-1]["is_let"]
            assert "local_copy : P := h_fresh_987" in graph.text

            instance_root = await session.start("∀ {α : Type} [Inhabited α], True")
            instance_state = await session.step(instance_root, "intro α inst")
            instance_graph = prepare_live_goal(instance_state.goals[0], {"<UNK>": 0}, edge_mode="bidirectional")
            assert instance_graph.normalized["hyp_sexps"][0]["binder_role"] == ":implicit"
            assert instance_graph.normalized["hyp_sexps"][1]["is_instance"]

    asyncio.run(run())


def test_native_live_graph_matches_captured_frontend_graph():
    root = native_root()

    async def run():
        async with LiveLeanSession(source_root=root, pantograph_repl=root / ".lake/build/bin/repl",
                                   imports=("Init",)) as session:
            source = "theorem liveParity (P : Prop) (h_fresh_987 : P) : P := by\n  exact h_fresh_987\n"
            result = await session.client.call("frontend.process", {
                "file": source, "invocations": True, "sorrys": False,
            }, timeout=60)
            units = result["units"]
            assert not [message for unit in units for message in unit.get("messages", [])], result
            invocation = next(inv for unit in units for inv in unit.get("invocations", [])
                              if inv["tactic"].strip() == "exact h_fresh_987")
            row = DatasetRow(invocation["goalBefore"], "liveParity", "exact h_fresh_987", "val", 0)
            record = _make_model_record(row, {"invocation": invocation}, {})
            offline = prepare_example(row, sexpr_data=record, use_sexpr=True)
            state = await session.start_declaration("theorem liveParity (P : Prop) (h_fresh_987 : P) : P")
            live = prepare_live_goal(state.goals[0], {"<UNK>": 0}, edge_mode="bidirectional")
            assert [node.as_dict() for node in live.dag.nodes] == [node.as_dict() for node in offline.dag.nodes]
            assert live.dag.edges == offline.dag.edges
            # A source declaration has a hidden local slot. Keep the resulting
            # nonzero/gapped indices, rather than compacting them to 0, 1.
            assert [entry["context_index"] for entry in live.normalized["hyp_sexps"]] == [1, 2]
            done = await session.step(state, "exact «h_fresh_987»")
            assert done.closed

    asyncio.run(run())


def test_smoke_reports_native_closure_as_unverified():
    root = native_root()

    async def run():
        async with LiveLeanSession(source_root=root, pantograph_repl=root / ".lake/build/bin/repl",
                                   imports=("Init",)) as session:
            goal = {"name": "g", "isConversion": False, "vars": [], "target": expression("(:c True)", "True")}
            report = await run_live_smoke(fixture_policy(goal, library=True), session, "True", top_k=1)
            assert report["status"] == "closed_unverified"
            assert report["proof_verified"] is False
            assert report["pending_goals"] == 0
            assert report["accepted_path"] == ["exact _root_.«True».«intro»"]

    asyncio.run(run())
