import itertools
import numpy as np
import pytest
import torch
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import (
    ArgumentReference,
    CandidatePoolSnapshot,
)
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import LibraryMemory
from maths_ai.gnn_inference.atp_lean_gnn.source_aware_decoder import SourceAwareDecoder


def inputs():
    torch.manual_seed(31)
    model = SourceAwareDecoder(8, 3, max_args=2, text_dim=4)
    library = LibraryMemory(
        [10, 11],
        ["Nat.add_comm", "Nat.add_zero"],
        np.eye(2, 4, dtype=np.float32),
        "fixture",
    )
    pool = CandidatePoolSnapshot(
        (0, 2), (10, 11), (0.8, 0.4), artifact_fingerprint="fixture", library=library
    )
    return model, torch.randn(8), torch.randn(3, 8), torch.tensor(1), pool


def test_sample_and_forced_probabilities_match_and_mixed_gradients_flow():
    model, state, nodes, tactic, pool = inputs()
    for _ in range(20):
        sampled = model.run(state, nodes, tactic, pool)
        forced = model.run(state, nodes, tactic, pool, forced=sampled.arguments)
        torch.testing.assert_close(sampled.logp, forced.logp)
    with torch.no_grad():
        model.library_residual_scale.fill_(0.4)
    args = (ArgumentReference("library", 11), ArgumentReference("local", 2))
    result = model.run(state, nodes, tactic, pool, forced=args)
    (-result.logp).backward()
    for component in (
        model.arg_count_head,
        model.source_head,
        model.argument_selector,
        model.lemma_adapter,
    ):
        assert (
            sum(
                p.grad.abs().sum() for p in component.parameters() if p.grad is not None
            )
            > 0
        )
    assert model.log_inverse_text_temperature.grad.abs() > 0


def test_joint_distribution_normalizes_and_entropy_surrogate_gradient():
    model, state, nodes, tactic, pool = inputs()
    choices = [ArgumentReference("local", x) for x in pool.local_ids] + [
        ArgumentReference("library", x) for x in pool.lemma_ids
    ]
    logs = torch.stack(
        [
            model.run(state, nodes, tactic, pool, forced=seq).logp
            for count in range(3)
            for seq in itertools.product(choices, repeat=count)
        ]
    )
    probs = logs.exp()
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
    exact = (probs * logs).sum()
    surrogate = (probs.detach() * (logs.detach() + 1) * logs).sum()
    first = torch.autograd.grad(
        exact, tuple(model.parameters()), retain_graph=True, allow_unused=True
    )
    second = torch.autograd.grad(
        surrogate, tuple(model.parameters()), allow_unused=True
    )
    for a, b in zip(first, second):
        if a is not None:
            torch.testing.assert_close(a, b)


def test_empty_support_allows_only_zero_arguments_and_rejects_drift():
    model, state, nodes, tactic, pool = inputs()
    result = model.run(state, nodes, tactic, CandidatePoolSnapshot(()))
    assert result.arguments == () and result.logp == 0
    with pytest.raises(ValueError, match="outside"):
        model.run(
            state, nodes, tactic, pool, forced=(ArgumentReference("library", 99),)
        )


def test_pool_fingerprint_includes_order_query_and_artifacts():
    from dataclasses import replace

    _, _, _, _, pool = inputs()
    assert pool.fingerprint != replace(pool, query_fingerprint="other").fingerprint
    assert pool.fingerprint != replace(pool, local_ids=(2, 0)).fingerprint
    with pytest.raises(ValueError):
        replace(pool, lemma_scores=(float("nan"), 0.4))


def test_pinned_source_logits_and_recurrent_updates():
    import json
    from pathlib import Path

    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/source_aware_decoder/pinned_outputs.json"
        ).read_text()
    )
    model = SourceAwareDecoder(4, 3, max_args=3, text_dim=3)
    model.load_state_dict(
        {key: torch.tensor(value) for key, value in fixture["weights"].items()},
        strict=True,
    )
    state, nodes, vectors, scores = (
        torch.tensor(fixture[key]) for key in ("state", "nodes", "vectors", "scores")
    )
    for record in fixture["records"]:
        model.local_type_bonus_strength = record["bonus"]
        with torch.no_grad():
            model.library_residual_scale.fill_(record["residual"])
        hidden = model.argument_selector.initial_state(
            state.view(1, -1), model.tactic_embedding(torch.tensor([1]))
        )
        adapted = model.lemma_adapter(vectors)
        torch.testing.assert_close(
            hidden, torch.tensor(record["initial"]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            model.arg_count_head(hidden).flatten(),
            torch.tensor(record["count"]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            adapted, torch.tensor(record["adapted"]), rtol=0, atol=0
        )
        for step, (source, position) in enumerate(
            [("library", 1), ("local", 0), ("library", 0)]
        ):
            actual = model.score_step(
                hidden, nodes, adapted, scores, tuple(fixture["signal"]), step
            )
            for result, expected in zip(actual, record["steps"][step]["logits"]):
                torch.testing.assert_close(
                    result, torch.tensor(expected), rtol=1e-6, atol=1e-6
                )
            hidden = model.argument_selector.gru(
                (adapted if source == "library" else nodes)[position : position + 1],
                hidden,
            )
            torch.testing.assert_close(
                hidden, torch.tensor(record["steps"][step]["hidden"]), rtol=0, atol=0
            )


def test_library_adaptation_is_shared_within_forward_and_recomputed_next_time():
    model, state, nodes, tactic, pool = inputs()
    calls = []
    handle = model.lemma_adapter.register_forward_hook(lambda *args: calls.append(1))
    rows = [(state, nodes, tactic, pool)] * 3
    forced = [(ArgumentReference("library", 11), ArgumentReference("local", 0))] * 3
    first = model.run_many(rows, forced_rows=forced)
    assert len(calls) == 1
    torch.stack([item.logp for item in first]).sum().backward()
    model.run_many(rows, forced_rows=forced)
    assert len(calls) == 2
    handle.remove()
