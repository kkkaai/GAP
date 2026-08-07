from __future__ import annotations

import torch

from .model import QGoalCFMModel


@torch.no_grad()
def euler_sample(
    model: QGoalCFMModel,
    batch: dict[str, torch.Tensor],
    *,
    q_dim: int,
    num_steps: int = 20,
    num_candidates: int = 1,
    clamp: bool = True,
) -> torch.Tensor:
    """Sample normalized q_goal candidates with fixed-step Euler integration."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    base_batch = next(tensor for tensor in batch.values() if isinstance(tensor, torch.Tensor))
    batch_size = base_batch.shape[0]
    device = base_batch.device
    total = batch_size * num_candidates

    expanded = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
            expanded[key] = value.repeat_interleave(num_candidates, dim=0)
        elif isinstance(value, list) and len(value) == batch_size:
            expanded[key] = [item for item in value for _ in range(num_candidates)]
        else:
            expanded[key] = value

    x = torch.randn(total, q_dim, device=device)
    dt = 1.0 / float(num_steps)
    for step in range(num_steps):
        t = torch.full((total,), step / float(num_steps), dtype=x.dtype, device=device)
        x = x + model(x, t, expanded) * dt
        if clamp:
            x = torch.clamp(x, -1.25, 1.25)
    return x.view(batch_size, num_candidates, q_dim)
