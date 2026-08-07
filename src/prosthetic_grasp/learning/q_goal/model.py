from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from .encoders import ConditionEncoder, ConditionEncoderConfig, mlp


@dataclass
class QGoalCFMModelConfig:
    q_dim: int
    hidden_dim: int = 512
    depth: int = 4
    time_dim: int = 128
    dropout: float = 0.0
    condition: ConditionEncoderConfig = field(default_factory=ConditionEncoderConfig)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class QGoalCFMModel(nn.Module):
    """Conditional vector field v_theta(x_t, t, condition) for q_goal CFM."""

    def __init__(self, config: QGoalCFMModelConfig) -> None:
        super().__init__()
        if config.q_dim <= 0:
            raise ValueError("q_dim must be positive.")
        config.condition.q_dim = config.q_dim
        config.condition.dropout = config.dropout
        self.config = config
        self.condition_encoder = ConditionEncoder(config.condition)
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(config.time_dim),
            nn.Linear(config.time_dim, config.time_dim),
            nn.SiLU(),
        )

        sizes = [config.q_dim + self.condition_encoder.out_dim + config.time_dim]
        sizes.extend([config.hidden_dim] * max(config.depth, 1))
        sizes.append(config.q_dim)
        self.vector_field = mlp(sizes, dropout=config.dropout)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.condition_encoder(batch)
        time = self.time_encoder(t)
        if time.shape[0] != x_t.shape[0]:
            time = time.expand(x_t.shape[0], -1)
        return self.vector_field(torch.cat([x_t, time, cond], dim=-1))

