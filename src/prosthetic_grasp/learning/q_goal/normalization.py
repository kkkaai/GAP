from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class JointNormalizer:
    """Normalize robot joint vectors to a stable training range."""

    q_min: torch.Tensor
    q_max: torch.Tensor
    eps: float = 1e-6

    @classmethod
    def from_bounds(
        cls,
        q_min: Sequence[float],
        q_max: Sequence[float],
        *,
        device: torch.device | str | None = None,
    ) -> "JointNormalizer":
        lo = torch.as_tensor(q_min, dtype=torch.float32, device=device)
        hi = torch.as_tensor(q_max, dtype=torch.float32, device=device)
        if lo.shape != hi.shape:
            raise ValueError(f"q_min/q_max must have the same shape, got {lo.shape} and {hi.shape}.")
        if torch.any(hi <= lo):
            raise ValueError("Every q_max entry must be greater than q_min.")
        return cls(q_min=lo, q_max=hi)

    @classmethod
    def identity(cls, q_dim: int, *, device: torch.device | str | None = None) -> "JointNormalizer":
        return cls(
            q_min=torch.full((q_dim,), -1.0, dtype=torch.float32, device=device),
            q_max=torch.full((q_dim,), 1.0, dtype=torch.float32, device=device),
        )

    def to(self, device: torch.device | str) -> "JointNormalizer":
        return JointNormalizer(self.q_min.to(device), self.q_max.to(device), self.eps)

    @property
    def q_dim(self) -> int:
        return int(self.q_min.numel())

    def normalize(self, q: torch.Tensor) -> torch.Tensor:
        lo = self.q_min.to(q.device)
        hi = self.q_max.to(q.device)
        return 2.0 * (q - lo) / torch.clamp(hi - lo, min=self.eps) - 1.0

    def denormalize(self, q_norm: torch.Tensor) -> torch.Tensor:
        lo = self.q_min.to(q_norm.device)
        hi = self.q_max.to(q_norm.device)
        return 0.5 * (q_norm + 1.0) * (hi - lo) + lo

    def clamp_normalized(self, q_norm: torch.Tensor) -> torch.Tensor:
        return torch.clamp(q_norm, -1.0, 1.0)

