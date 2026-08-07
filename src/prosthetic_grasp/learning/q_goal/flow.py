from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlowTrainingBatch:
    x_0: torch.Tensor
    x_1: torch.Tensor
    t: torch.Tensor
    x_t: torch.Tensor
    target_velocity: torch.Tensor


class SimpleLinearFlowMatcher:
    """Minimal CFM path: x_t = (1 - t) x_0 + t x_1."""

    name = "simple"

    def sample(self, x_1: torch.Tensor) -> FlowTrainingBatch:
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)
        view = (x_1.shape[0],) + (1,) * (x_1.ndim - 1)
        t_view = t.view(view)
        x_t = (1.0 - t_view) * x_0 + t_view * x_1
        target_velocity = x_1 - x_0
        return FlowTrainingBatch(x_0=x_0, x_1=x_1, t=t, x_t=x_t, target_velocity=target_velocity)


class MetaFlowMatchingAdapter:
    """Adapter around Meta's flow_matching AffineProbPath + CondOTScheduler.

    This keeps GAP's model/data code independent from the external library API.
    """

    name = "meta"

    def __init__(self) -> None:
        try:
            from flow_matching.path.affine import AffineProbPath
            from flow_matching.path.scheduler.scheduler import CondOTScheduler
        except ImportError:
            try:
                from flow_matching.path import AffineProbPath
                from flow_matching.path.scheduler import CondOTScheduler
            except ImportError as exc:
                raise ImportError(
                    "backend='meta' requires Meta flow_matching. Install it later with "
                    "`pip install flow_matching` in the CFM conda environment."
                ) from exc

        self.path = AffineProbPath(scheduler=CondOTScheduler())

    def sample(self, x_1: torch.Tensor) -> FlowTrainingBatch:
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = getattr(path_sample, "x_t")
        dx_t = getattr(path_sample, "dx_t")
        return FlowTrainingBatch(x_0=x_0, x_1=x_1, t=t, x_t=x_t, target_velocity=dx_t)


def build_flow_matcher(backend: str):
    backend = backend.strip().lower()
    if backend == "simple":
        return SimpleLinearFlowMatcher()
    if backend == "meta":
        return MetaFlowMatchingAdapter()
    raise ValueError(f"Unknown CFM backend {backend!r}. Expected 'meta' or 'simple'.")

