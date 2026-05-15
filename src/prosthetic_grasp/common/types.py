from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass
class SensorFrame:
    rgb: Array
    depth: Array | None = None
    timestamp: float = 0.0
    rgb_path: str | None = None
    depth_path: str | None = None


@dataclass
class Phase1MaskResult:
    mask: Array
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase2LollipopResult:
    lollipop_mask: Array
    lolli_params: tuple[float, float, float, float, float]
    wrist_point: tuple[float, float]
    tip_point: tuple[float, float]


@dataclass
class ROIBox:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class Phase3InpaintResult:
    roi_box: ROIBox
    rgb_crop: Array
    mask_crop: Array
    inpaint_crop: Array
    inpaint_full: Array
    prompt: str


@dataclass
class Phase4FluxFillResult:
    roi_box: ROIBox
    flux_crop: Array
    flux_full: Array
    prompt: str


@dataclass
class PhasePlaceholderResult:
    status: str
    message: str


@dataclass
class PipelineResult:
    status: str
    frame: SensorFrame | None = None
    phase1_mask: Phase1MaskResult | None = None
    phase2_lollipop: Phase2LollipopResult | None = None
    phase3_inpaint: Phase3InpaintResult | None = None
    phase4_flux_fill: Phase4FluxFillResult | None = None
    phase5_mano: PhasePlaceholderResult | None = None
    phase6_prosthetic_action: PhasePlaceholderResult | None = None

    def to_json_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return {"shape": list(value.shape), "dtype": str(value.dtype)}
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, "__dataclass_fields__"):
                return {k: convert(v) for k, v in asdict(value).items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        return convert(self)
