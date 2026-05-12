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
class Burst:
    frames: list[SensorFrame]


@dataclass
class IntentSpec:
    raw_text: str
    target: str = "unknown"
    task: str = "pick_up"
    grasp_part: str = "unspecified"
    constraints: list[str] = field(default_factory=list)


@dataclass
class ProsthesisMaskState:
    fine_mask: Array
    lollipop_mask: Array
    lolli_params: tuple[float, float, float, float, float]


@dataclass
class InteractionROI:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class SceneLite:
    frame: SensorFrame
    prosthesis: ProsthesisMaskState
    interaction_roi: InteractionROI
    object_mask: Array | None
    intent: IntentSpec


@dataclass
class GeneratedCandidate:
    image: Array
    prompt: str
    seed: int
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanPrior2D:
    anchor_uv: tuple[int, int]
    approach_theta: float
    human_lolli: tuple[float, float, float, float, float]
    contact_patch: Array
    hand_span_px: float
    object_width_px: float | None
    grasp_family_hint: str
    confidence: float


@dataclass
class ProstheticAction:
    grasp_type: str
    thumb_mode: str
    wrist_rotation: float
    aperture: float
    closure_speed: float
    force_level: float
    confidence: float


@dataclass
class ExecutionResult:
    status: str
    message: str
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    status: str
    scene: SceneLite | None = None
    candidates: list[GeneratedCandidate] = field(default_factory=list)
    prior: HumanPrior2D | None = None
    action: ProstheticAction | None = None
    execution: ExecutionResult | None = None

    def to_json_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
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

