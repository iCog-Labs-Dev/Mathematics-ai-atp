from __future__ import annotations

import math

import torch
from torch import Tensor

from .architectures import amp_dtype_for_architecture


def resolve_amp_dtype(
    *,
    architecture: str,
    device: torch.device,
    requested: bool,
) -> torch.dtype | None:
    if not requested:
        return None
    return amp_dtype_for_architecture(architecture, device)


def require_finite_loss(
    loss: Tensor,
    *,
    architecture: str,
    amp_dtype: torch.dtype | None,
    epoch: int,
    batch_index: int,
    components: dict[str, float] | None = None,
) -> None:
    if bool(torch.isfinite(loss).all()):
        return
    precision = "fp32" if amp_dtype is None else str(amp_dtype).removeprefix("torch.")
    component_text = ""
    if components:
        rendered = ", ".join(
            f"{name}={value}" for name, value in sorted(components.items())
        )
        component_text = f", components=({rendered})"
    raise FloatingPointError(
        "Non-finite training loss: "
        f"architecture={architecture}, precision={precision}, epoch={epoch}, "
        f"batch={batch_index}, loss={float(loss.detach().item())}{component_text}."
    )
