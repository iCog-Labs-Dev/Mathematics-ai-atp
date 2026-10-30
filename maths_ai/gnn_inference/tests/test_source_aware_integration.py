import torch
from torch_geometric.data import Data, Batch
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import local_pools
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import (
    checkpoint_payload,
    build_model_from_checkpoint,
)
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC


def fixture():
    spec = ModelSpec.from_dict(
        dict(
            architecture="gatv2",
            hidden_dim=8,
            dropout=0.5,
            encoder=dict(heads=2, num_layers=1, readout="state_mean_attention"),
            text_dim=4,
        )
    )
    model = build_actor_critic_model(model_spec=spec, num_node_labels=5, num_tactics=3)
    data = Data(
        x=torch.tensor([1, 2, 3]),
        node_type=torch.tensor([0, 0, 0]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        premise_mask=torch.tensor([True, False, True]),
        state_node_index=torch.tensor([2]),
    )
    batch = Batch.from_data_list([data, data])
    return model, batch, local_pools(batch)


def test_dropout_disabled_recomputation_preserves_full_policy_and_gradients():
    model, batch, pools = fixture()
    model.train()
    with torch.no_grad():
        sampled = model.act(batch, candidate_pools=pools)
    model.train()
    evaluated = model.evaluate_actions(
        batch, sampled.tactic_action, sampled.arguments, candidate_pools=pools
    )
    torch.testing.assert_close(sampled.logp, evaluated.logp, rtol=0, atol=0)
    (-evaluated.logp.mean() + evaluated.value.square().mean()).backward()
    assert (
        sum(
            p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None
        )
        > 0
    )
    assert (
        sum(p.grad.abs().sum() for p in model.critic.parameters() if p.grad is not None)
        > 0
    )


def test_version_four_roundtrip_has_identical_actions():
    model, batch, pools = fixture()
    nodes = {str(i): i for i in range(5)}
    tactics = {str(i): i for i in range(3)}
    payload = checkpoint_payload(
        model_kind="actor_critic_with_args",
        model_spec=model.model_spec,
        node_vocab=nodes,
        tactic_vocab=tactics,
        model=model,
        graph_representation=MODEL_SEXPR_GRAPH_SPEC,
    )
    assert payload["manifest"]["checkpoint_format_version"] == 4
    loaded, _, _ = build_model_from_checkpoint(
        payload,
        node_vocab=nodes,
        tactic_vocab=tactics,
        graph_representation=MODEL_SEXPR_GRAPH_SPEC,
    )
    first = model.act(batch, candidate_pools=pools, greedy=True)
    second = loaded.act(batch, candidate_pools=pools, greedy=True)
    assert first.arguments == second.arguments
    torch.testing.assert_close(first.logp, second.logp)


def test_offline_candidates_resolve_original_rows_after_pyg_offsets():
    import pytest
    from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import (
        OfflineCandidateProvider,
    )

    model, batch, pools = fixture()
    graphs = batch.to_data_list()
    for index, data in enumerate(graphs):
        data.split = "train"
        data.row_index = index + 9
        data.candidate_prepared_identity = "prepared"
    batch = Batch.from_data_list(graphs)
    provider = OfflineCandidateProvider(
        {("train", 9): pools[0], ("train", 10): pools[1]}, prepared_identity="prepared"
    )
    assert provider(batch) == pools
    batch.candidate_prepared_identity = ["changed", "prepared"]
    with pytest.raises(ValueError, match="differ"):
        provider(batch)


def test_supervised_retrieval_miss_still_trains_count_without_fabricated_identity():
    import numpy as np
    from dataclasses import replace
    from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_pointer_model
    from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import supervised_loss
    from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import CandidatePoolSnapshot
    from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import LibraryMemory

    actor, batch, _ = fixture()
    model = build_pointer_model(
        model_spec=replace(actor.model_spec, retrieval_binding="library"),
        num_node_labels=5,
        num_tactics=3,
    )
    batch.y = torch.tensor([1, 1])
    batch.arg_count = torch.tensor([1, 1])
    batch.arg_node_indices = torch.tensor([-1, -1])
    batch.arg_lemma_ids = torch.tensor([11, 11])
    batch.action_target_supported = torch.tensor([True, True])
    library = LibraryMemory(
        [10, 11],
        ["Fixture.one", "Fixture.missing"],
        np.eye(2, 4, dtype=np.float32),
        "library",
    )
    pool = CandidatePoolSnapshot(
        (), (10,), (0.5,), artifact_fingerprint="library", library=library
    )
    loss, _, _ = supervised_loss(model, batch, candidate_pools=(pool, pool))
    loss.backward()
    assert model.decoder.arg_count_head.weight.grad.abs().sum() > 0
    assert model.decoder.argument_selector.out_proj.weight.grad is None
