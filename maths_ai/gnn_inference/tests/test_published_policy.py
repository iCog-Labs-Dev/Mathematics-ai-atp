import torch
from torch_geometric.data import Data

from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import (
    PublishedPointerClassifier,
)


def test_published_policy_returns_tactic_arguments_and_stop_scores() -> None:
    model = PublishedPointerClassifier(
        num_node_labels=8,
        num_tactics=3,
        num_node_types=11,
        hidden_dim=8,
        num_layers=1,
        heads=2,
        dropout=0.0,
        use_node_type=True,
        max_args=2,
    )
    data = Data(
        x=torch.tensor([1, 2, 3]),
        node_type=torch.tensor([0, 1, 5]),
        is_bound=torch.tensor([0, 0, 0]),
        binder_depth=torch.tensor([0, 0, 0]),
        binder_kind=torch.tensor([0, 0, 0]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        premise_mask=torch.tensor([True, True, False]),
        state_node_index=torch.tensor([0]),
    )
    data.batch = torch.zeros(3, dtype=torch.long)

    tactic_logits, arg_logits, stop_logits = model(data)

    assert tactic_logits.shape == (1, 3)
    assert len(arg_logits) == 2
    assert all(logits.shape == (1, 3) for logits in arg_logits)
    assert stop_logits.shape == (1, 2)
    assert torch.isneginf(arg_logits[0][0, 2])
