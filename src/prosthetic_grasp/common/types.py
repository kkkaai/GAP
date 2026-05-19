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
class Phase3EraseResult:
    roi_box: ROIBox
    rgb_crop: Array
    mask_crop: Array
    erased_crop: Array
    erased_full: Array
    mode: str
    model_name: str
    prompt: str


@dataclass
class Phase4InpaintResult:
    roi_box: ROIBox
    rgb_crop: Array
    mask_crop: Array
    inpaint_crop: Array
    inpaint_full: Array
    mode: str
    model_name: str
    prompt: str


@dataclass
class PhasePlaceholderResult:
    status: str
    message: str


@dataclass
class Phase5HandPrediction:
    hand_index: int
    is_right: bool
    bbox_xyxy: Array
    keypoints_2d: Array
    keypoint_score_mean: float
    vertices: Array
    keypoints_3d: Array
    pred_cam: Array
    pred_cam_t_crop: Array
    pred_cam_t_full: Array
    focal_length: float
    mano_params: dict[str, Array]


@dataclass
class Phase5ManoResult:
    status: str
    message: str
    faces: Array | None = None
    hands: list[Phase5HandPrediction] = field(default_factory=list)


@dataclass
class Phase6ProstheticActionResult:
    status: str
    message: str
    selected_hand_index: int | None
    action_names: list[str]
    action: Array
    mano_wrist: Array | None = None
    mano_fingertips: Array | None = None
    target_fingertips_wrist: Array | None = None
    prosthetic_fingertips_wrist: Array | None = None
    fingertip_error: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    status: str
    frame: SensorFrame | None = None
    phase1_mask: Phase1MaskResult | None = None
    phase2_lollipop: Phase2LollipopResult | None = None
    phase3_erase: Phase3EraseResult | None = None
    phase4_inpaint: Phase4InpaintResult | None = None
    phase5_mano: Phase5ManoResult | PhasePlaceholderResult | None = None
    phase6_prosthetic_action: Phase6ProstheticActionResult | PhasePlaceholderResult | None = None

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
