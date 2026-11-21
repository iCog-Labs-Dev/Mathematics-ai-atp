"""Global objectives and one optimizer step under bounded graph execution."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
import torch
from torch_geometric.data import Batch

from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
    EdgeAction,
    PolicyObservation,
    ObservationOutcome,
    prepare_onpolicy_update,
    prepare_htps_update,
    prepare_transition_update,
    backward_update,
    _execute_update,
    train_step_htps_style,
    train_step,
)
from maths_ai.gnn_inference.atp_lean_gnn.update_batching import (
    UpdateLimits,
    UpdateBudgetError,
    candidate_cost,
)
from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    extract_transitions,
    extract_critic_samples,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.tests import test_pln_rl_training as fixtures
from maths_ai.gnn_inference.tests.rl_update_reference import (
    compute_onpolicy_loss,
    compute_transition_loss,
)


def setup_rows(architecture="graphsage"):
    model, feat, examples, critics = fixtures.HTPSStyleStepTests()._setup()
    if architecture == "gatv2":
        spec = model.model_spec.to_dict()
        spec.update(
            architecture="gatv2",
            encoder={
                "num_layers": 2,
                "heads": 2,
                "readout": "state_mean_max_attention",
            },
        )
        model = build_actor_critic_model(
            model_spec=ModelSpec.from_dict(spec),
            num_node_labels=model.encoder.node_features.label_embedding.num_embeddings,
            num_tactics=3,
        )
    # Configure nonzero dropout even though RL forwards must disable it.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.4
    examples = [examples[0], examples[1], examples[0], examples[0], examples[1]]
    return model, feat, examples, critics


def actor_observations(examples):
    from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import ActorTransition
    from maths_ai.data_models.proof_components import TacticCandidate

    observations, transitions = [], []
    for i, row in enumerate(examples):
        accepted = i in (0, 3)
        action = EdgeAction(
            row.tactic_id,
            row.candidate_pool,
            row.graph_fingerprint,
            row.arguments,
            multiplicity=i + 1,
        )
        observations.append(
            PolicyObservation(
                0,
                row.goal,
                action,
                (
                    ObservationOutcome.ACCEPTED
                    if accepted
                    else ObservationOutcome.REJECTED
                ),
                0,
                edge_id=i if accepted else None,
                log_prior=-1.0,
            )
        )
        if accepted:
            # Harvested transitions need only these fields in this objective.
            from types import SimpleNamespace

            transitions.append(
                SimpleNamespace(edge_id=i, goal=row.goal, return_=1.0 - i * 0.2)
            )
    return transitions, observations


def assert_grads_equal(first, second):
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-4, msg=name)


@pytest.mark.parametrize("architecture", ["graphsage", "gatv2"])
@pytest.mark.parametrize("phase", ["onpolicy", "transition", "htps"])
@pytest.mark.parametrize("chunk_rows", [1, 2, 100])
def test_partition_matches_full_batch_loss_gradients_and_adamw(
    phase, chunk_rows, architecture
):
    model, feat, examples, critics = setup_rows(architecture)
    reference = deepcopy(model)
    rowsize = max(int(feat(r.goal).num_nodes) for r in examples)
    limits = dict(max_update_nodes=rowsize * chunk_rows)
    transitions, observations = actor_observations(examples)
    if phase == "onpolicy":
        plan = prepare_onpolicy_update(
            model, transitions, critics, observations, feat, **limits
        )
        loss, old = compute_onpolicy_loss(
            reference,
            transitions,
            critics,
            observations,
            feat,
            critic_weight=0.7,
            entropy_weight=0.03,
            bc_weight=0.2,
        )
        options = dict(critic_weight=0.7, entropy_weight=0.03, bc_weight=0.2)
    elif phase == "transition":
        graph, feat, model, mapping = fixtures.PLNRLTrainingTests()._setup()
        if architecture == "gatv2":
            specification = model.model_spec.to_dict()
            specification.update(
                architecture="gatv2",
                encoder={
                    "num_layers": 2,
                    "heads": 2,
                    "readout": "state_mean_max_attention",
                },
            )
            model = build_actor_critic_model(
                model_spec=ModelSpec.from_dict(specification),
                num_node_labels=model.encoder.node_features.label_embedding.num_embeddings,
                num_tactics=3,
            )
        reference = deepcopy(model)
        ts, cs = extract_transitions(graph, RewardConfig()), extract_critic_samples(
            graph
        )
        limits = dict(
            max_update_nodes=max(int(feat(t.goal).num_nodes) for t in ts) * chunk_rows
        )
        plan = prepare_transition_update(model, ts, cs, feat, mapping, **limits)
        loss, old = compute_transition_loss(
            reference,
            ts,
            cs,
            feat,
            mapping,
            critic_weight=0.7,
            entropy_weight=0.03,
            bc_weight=0.2,
        )
        options = dict(critic_weight=0.7, entropy_weight=0.03, bc_weight=0.2)
    else:
        plan = prepare_htps_update(model, examples, critics, feat, **limits)
        reference.eval()
        evaluated = reference.evaluate_actions(
            Batch.from_data_list([feat(r.goal) for r in examples]),
            torch.tensor([r.tactic_id for r in examples]),
            [r.arguments for r in examples],
            candidate_pools=[r.candidate_pool for r in examples],
        )
        imitation = -evaluated.tactic_logp.mean() - 0.3 * evaluated.arg_logp.mean()
        values = reference.encode(
            Batch.from_data_list([feat(r.goal) for r in critics])
        )[-1].reshape(-1)
        critic = (
            (values - values.new_tensor([r.target for r in critics])).square().mean()
        )
        loss = imitation + 0.7 * critic
        old = {
            "tactic_imitation_loss": float(imitation.detach()),
            "critic_soft_loss": float(critic.detach()),
            "htps_total_loss": float(loss.detach()),
        }
        options = dict(critic_weight=0.7, arg_loss_weight=0.3)
    loss.backward()
    with torch.no_grad():
        metrics = backward_update(model, plan, feat, **options)
    for key in (
        "actor_loss",
        "critic_loss",
        "entropy",
        "bc_loss",
        "total_loss",
        "mean_return",
        "mean_arg_logp",
        "num_transitions",
        "num_failures",
        "num_critic_samples",
        "tactic_imitation_loss",
        "critic_soft_loss",
        "htps_total_loss",
    ):
        if key in old:
            assert metrics[key] == pytest.approx(old[key], abs=3e-6, rel=1e-4), key
    assert_grads_equal(model, reference)
    for m in (model, reference):
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.7)
        torch.optim.AdamW(m.parameters(), lr=1e-5).step()
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
    assert metrics["update_peak_nodes"] <= limits["max_update_nodes"]
    if chunk_rows == 1:
        assert metrics["update_nodes"] > limits["max_update_nodes"]


def test_one_step_clip_and_clear_under_outer_no_grad():
    model, feat, examples, critics = setup_rows()
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.step = Mock(wraps=optimizer.step)
    optimizer.zero_grad = Mock(wraps=optimizer.zero_grad)
    with patch(
        "torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_
    ) as clip:
        with torch.no_grad():
            metrics = train_step_htps_style(
                model,
                optimizer,
                examples,
                critics,
                feat,
                max_update_nodes=int(feat(examples[0].goal).num_nodes),
            )
    assert metrics["update_backward_batches"] == 7
    assert metrics["htps_optimizer_step"] == 1
    optimizer.step.assert_called_once()
    optimizer.zero_grad.assert_called_once()
    clip.assert_called_once()


def test_late_invalid_row_and_single_row_budget_fail_before_forward():
    model, feat, examples, critics = setup_rows()
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.step = Mock(wraps=optimizer.step)
    model.encode_graph = Mock(side_effect=AssertionError("forward before validation"))
    for kwargs in (
        {"max_update_nodes": 1},
        {"max_update_edges": 1},
        {"max_update_candidates": 1},
        {"max_update_candidate_bytes": 1},
    ):
        with pytest.raises(UpdateBudgetError, match="row 0"):
            train_step_htps_style(model, optimizer, examples, critics, feat, **kwargs)
    bad = replace(
        examples[-1],
        candidate_pool=replace(examples[-1].candidate_pool, query_fingerprint="wrong"),
    )
    with pytest.raises(ValueError, match="query text changed"):
        train_step_htps_style(model, optimizer, [*examples[:-1], bad], critics, feat)
    optimizer.step.assert_not_called()
    model.encode_graph.assert_not_called()
    assert not optimizer.state


def test_later_backward_failure_clears_partial_gradients_without_step():
    model, feat, examples, critics = setup_rows()
    optimizer = torch.optim.AdamW(model.parameters())
    before = deepcopy(model.state_dict())
    original = model.evaluate_actions
    count = 0

    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            assert any(p.grad is not None for p in model.parameters())
            raise RuntimeError("injected later batch failure")
        return original(*args, **kwargs)

    model.evaluate_actions = fail
    with pytest.raises(RuntimeError, match="injected"):
        train_step_htps_style(
            model,
            optimizer,
            examples,
            critics,
            feat,
            max_update_nodes=int(feat(examples[0].goal).num_nodes),
        )
    assert all(p.grad is None for p in model.parameters())
    assert not optimizer.state
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["nodes", "edges", "candidates", "candidate_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_limits_are_positive_integers(field, value):
    with pytest.raises(ValueError):
        UpdateLimits(**{field: value})


def test_nonfinite_targets_and_nonpositive_multiplicity_fail():
    model, feat, examples, critics = setup_rows()
    ts, observations = actor_observations(examples)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="target"):
            prepare_htps_update(
                model, examples, [replace(critics[0], target=bad)], feat
            )
    # EdgeAction itself rejects some invalid multiplicities; prepared rows also validate them.
    from maths_ai.gnn_inference.atp_lean_gnn.update_batching import (
        UpdateRow,
        prepare_plan,
    )

    row = UpdateRow(examples[0].goal, examples[0].graph_fingerprint, 1.0, weight=0)
    with pytest.raises(ValueError, match="multiplicity"):
        prepare_plan(
            model, [row], [], feat, Mock(), phase="test", limits=UpdateLimits()
        )


@pytest.mark.parametrize("phase", ["onpolicy", "htps"])
def test_library_actions_partition_by_candidate_cost_and_keep_binding(phase):
    import numpy as np
    from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import LibraryMemory
    from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import ArgumentReference

    model, feat, examples, critics = setup_rows()
    library = LibraryMemory(
        [10, 20],
        ["Test.one", "Test.two"],
        np.ones((2, model.decoder.text_dim), dtype=np.float32),
        "fixture-library",
    )
    model.retrieval_binding = library.fingerprint
    examples = [
        replace(
            r,
            candidate_pool=replace(
                r.candidate_pool,
                lemma_ids=(10, 20),
                lemma_scores=(0.8, 0.2),
                library=library,
                artifact_fingerprint=library.fingerprint,
            ),
            arguments=(ArgumentReference("library", 20),),
        )
        for r in examples
    ]
    reference = deepcopy(model)
    count, size = candidate_cost(model, examples[0])
    options = dict(max_update_candidates=count, max_update_candidate_bytes=size)
    ts, observations = actor_observations(examples)
    if phase == "onpolicy":
        plan = prepare_onpolicy_update(
            model, ts, critics, observations, feat, **options
        )
        loss, old = compute_onpolicy_loss(reference, ts, critics, observations, feat)
        loss.backward()
        metrics = backward_update(model, plan, feat)
        assert metrics["total_loss"] == pytest.approx(old["total_loss"], abs=3e-6)
        assert_grads_equal(model, reference)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        metrics = train_step_htps_style(
            model, optimizer, examples, critics, feat, **options
        )
    assert metrics["update_actor_batches"] == len(examples)
    assert metrics["update_peak_candidate_bytes"] == size
    assert metrics["update_candidate_bytes"] == size * len(examples)
    assert metrics["update_peak_candidate_count"] == count
    assert (
        size
        == (
            len(examples[0].candidate_pool.local_ids) * model.hidden_dim
            + 2 * (model.decoder.text_dim + model.hidden_dim)
        )
        * 4
    )


def test_backward_releases_previous_batch_outputs_and_keeps_weights_fixed():
    import weakref

    model, feat, examples, critics = setup_rows()
    optimizer = torch.optim.AdamW(model.parameters())
    original = model.evaluate_actions
    previous, calls = [], []
    before = [p.detach().clone() for p in model.parameters()]

    def observed(*args, **kwargs):
        assert not model.training
        assert torch.is_grad_enabled()
        assert all(ref() is None for ref in previous)
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
        result = original(*args, **kwargs)
        previous.append(weakref.ref(result.tactic_logits))
        calls.append(1)
        return result

    model.evaluate_actions = observed
    ts, observations = actor_observations(examples)
    metrics = _execute_update(
        model,
        optimizer,
        lambda: prepare_onpolicy_update(
            model,
            ts,
            critics,
            observations,
            feat,
            max_update_nodes=int(feat(examples[0].goal).num_nodes),
        ),
        feat,
        grad_clip=1.0,
    )
    assert len(calls) == len(examples)
    assert metrics["onpolicy_optimizer_step"] == 1
    assert all(ref() is None for ref in previous)


@pytest.mark.parametrize("field", ["max_update_nodes", "max_update_edges"])
def test_graph_budget_boundaries_pack_without_dropping_rows(field):
    model, feat, examples, critics = setup_rows()
    data = feat(examples[0].goal)
    boundary = (
        int(data.num_nodes) if field.endswith("nodes") else int(data.edge_index.size(1))
    )
    plan = prepare_htps_update(model, examples, critics, feat, **{field: boundary})
    assert len(plan.actor_batches) == len(examples)
    assert sum(map(len, plan.actor_batches)) == len(examples)
    assert len(plan.critic_batches) == len(critics)
    with pytest.raises(UpdateBudgetError):
        prepare_htps_update(model, examples, critics, feat, **{field: boundary - 1})


def test_critic_and_actor_only_and_empty_updates():
    model, feat, examples, critics = setup_rows()
    optimizer = torch.optim.AdamW(model.parameters())
    for actors, values, expected in (([], [], 0), (examples, [], 1), ([], critics, 1)):
        metrics = train_step_htps_style(model, optimizer, actors, values, feat)
        assert metrics["htps_optimizer_step"] == expected
        assert metrics["update_backward_batches"] == bool(actors) + bool(values)
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA memory measurement requires a GPU"
)
@pytest.mark.parametrize("phase", ["htps", "onpolicy"])
def test_cuda_bounded_update_peak_memory(phase):
    model, feat, examples, critics = setup_rows()
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    limit = int(feat(examples[0].goal).num_nodes)

    def update(repeat):
        if phase == "htps":
            return train_step_htps_style(
                model,
                optimizer,
                examples * repeat,
                critics * repeat,
                feat,
                max_update_nodes=limit,
            )
        ts, observations = actor_observations(examples * repeat)
        return _execute_update(
            model,
            optimizer,
            lambda: prepare_onpolicy_update(
                model, ts, critics * repeat, observations, feat, max_update_nodes=limit
            ),
            feat,
            grad_clip=1.0,
        )

    update(1)
    peaks = []
    for repeat in (1, 20):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        metrics = update(repeat)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
        assert metrics["update_peak_nodes"] <= limit
    # Parameters, Adam state and activations are measured after optimizer warmup.
    assert peaks[1] <= peaks[0] * 1.25 + 2 * 1024 * 1024


def test_driver_large_logical_batch_checkpoint_and_resume_contract(tmp_path):
    import asyncio
    import json
    from maths_ai.gnn_inference.tests import test_rl_training_driver as driver_fixtures
    from maths_ai.gnn_inference.atp_lean_gnn import rl_training_driver as driver

    cfg = driver_fixtures._write_config(
        tmp_path,
        num_rounds=1,
        theorems_per_round=16,
        htps_steps_per_round=2,
        htps_batch_size=16,
        max_update_nodes=16,
        max_update_edges=100,
        eval_every=1,
    )
    pool = driver_fixtures._pool(eval_size=2)
    asyncio.run(
        driver.run_rl_training(
            cfg, reasoner_factory=driver_fixtures._qed_factory, pool=pool
        )
    )
    run_dir = next((tmp_path / "runs").iterdir())
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    training = rows[0]
    assert training["update_nodes"] > cfg.max_update_nodes
    assert training["update_peak_nodes"] <= cfg.max_update_nodes
    assert training["onpolicy_optimizer_step"] == 1
    assert training["htps_optimizer_steps"] == 2
    assert training["parameter_updates_this_round"] == 3
    assert training["htps_update_backward_batches"] > 2
    assert training["htps_update_peak_nodes"] <= cfg.max_update_nodes
    assert list((run_dir / "evaluation").glob("*/summary.json"))
    checkpoint = torch.load(run_dir / "last.pt", weights_only=False)
    assert checkpoint["training_update_contract"] == driver.training_update_contract(
        cfg
    )
    assert checkpoint["data_contract"]["training_pool"]
    # Same contract resumes without applying an additional update when rounds are complete.
    asyncio.run(
        driver.run_rl_training(
            cfg,
            reasoner_factory=driver_fixtures._qed_factory,
            pool=driver_fixtures._pool(eval_size=2),
            resume_run_dir=run_dir,
        )
    )
    cfg.max_update_nodes += 1
    with pytest.raises(ValueError, match="training update contract"):
        asyncio.run(
            driver.run_rl_training(
                cfg,
                reasoner_factory=driver_fixtures._qed_factory,
                pool=driver_fixtures._pool(eval_size=2),
                resume_run_dir=run_dir,
            )
        )
    cfg.max_update_nodes -= 1
    checkpoint.pop("training_update_contract")
    torch.save(checkpoint, run_dir / "last.pt")
    with pytest.raises(ValueError, match="warm start"):
        asyncio.run(
            driver.run_rl_training(
                cfg,
                reasoner_factory=driver_fixtures._qed_factory,
                pool=driver_fixtures._pool(eval_size=2),
                resume_run_dir=run_dir,
            )
        )
    # A weight-only warm start does not consume the resume contract.
    cfg.warmstart_checkpoint = run_dir / "last.pt"
    cfg.run_root = tmp_path / "warm-start-run"
    cfg.num_rounds = 0
    asyncio.run(
        driver.run_rl_training(
            cfg,
            reasoner_factory=driver_fixtures._qed_factory,
            pool=driver_fixtures._pool(eval_size=2),
        )
    )


def test_nonfinite_gradients_clear_without_optimizer_mutation():
    model, feat, examples, critics = setup_rows()
    optimizer = torch.optim.AdamW(model.parameters())
    before = deepcopy(model.state_dict())
    hook = model.actor.base.weight.register_hook(lambda grad: grad * float("nan"))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            train_step_htps_style(model, optimizer, examples, critics, feat)
    finally:
        hook.remove()
    assert not optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_constant_global_advantages_are_zero():
    from maths_ai.gnn_inference.atp_lean_gnn.update_batching import (
        normalized_advantages,
    )

    assert normalized_advantages([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 5.0, 2.0]) == [
        0.0,
        0.0,
        0.0,
    ]
