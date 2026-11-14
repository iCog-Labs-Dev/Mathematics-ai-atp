import torch

from maths_ai.gnn_inference.atp_lean_gnn.joint_loader import mean_pool_embeddings


def test_mean_pool_embeddings_masks_padding_and_normalizes() -> None:
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 9.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    result = mean_pool_embeddings(hidden, mask)

    expected = torch.tensor([[0.31622776, 0.9486833]])
    assert torch.allclose(result, expected)


def test_mean_pool_embeddings_can_preserve_scale() -> None:
    hidden = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
    mask = torch.tensor([[1, 1]])

    result = mean_pool_embeddings(hidden, mask, normalize=False)

    assert torch.equal(result, torch.tensor([[1.0, 2.0]]))
