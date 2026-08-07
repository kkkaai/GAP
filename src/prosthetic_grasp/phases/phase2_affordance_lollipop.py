from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from prosthetic_grasp.common.secrets import get_secret


QWEN3_7_PLUS_MODEL_ID = "qwen/qwen3.7-plus"
ZENMUX_CHAT_BASE_URL = "https://zenmux.ai/api/v1"


@dataclass(frozen=True)
class AffordanceLollipopCandidate:
    candidate_id: str
    affordance_part: str
    task: str
    grasp_type: str
    palm_center_xy: tuple[float, float]
    tip_xy: tuple[float, float]
    palm_radius: float
    arm_width: float
    theta_rad: float
    priority: int = 1
    reason: str = ""

    @property
    def strategy(self) -> str:
        part = self.affordance_part.strip().lower().replace(" ", "_") or "affordance"
        grasp = self.grasp_type.strip().lower().replace(" ", "_") or "grasp"
        return f"{part}_{grasp}"


@dataclass
class Phase2AffordanceLollipopResult:
    target_object: str
    global_task: str
    candidates: list[AffordanceLollipopCandidate]
    model_id: str
    prompt: str
    raw_response: str
    parsed_json: dict[str, Any]


@dataclass
class Phase2AffordanceLollipopConfig:
    model_id: str = QWEN3_7_PLUS_MODEL_ID
    zenmux_base_url: str = ZENMUX_CHAT_BASE_URL
    temperature: float = 0.2
    max_tokens: int = 1200
    num_candidates: int = 4
    min_palm_radius_norm: float = 0.045
    max_palm_radius_norm: float = 0.14
    min_arm_width_norm: float = 0.028
    max_arm_width_norm: float = 0.09
    default_palm_radius_norm: float = 0.08
    default_arm_width_norm: float = 0.05
    use_coordinate_grid: bool = True

    def __post_init__(self) -> None:
        self.model_id = self.model_id.strip()
        self.zenmux_base_url = self.zenmux_base_url.strip().rstrip("/")
        if not self.model_id:
            raise ValueError("model_id must be non-empty.")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}.")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}.")
        if self.num_candidates <= 0:
            raise ValueError(f"num_candidates must be positive, got {self.num_candidates}.")


def build_affordance_lollipop_prompt(
    *,
    object_name: str | None,
    task_instruction: str | None,
    num_candidates: int,
) -> str:
    object_block = (object_name or "").strip() or "unknown target object"
    task_block = (task_instruction or "").strip() or "No user task was provided. Infer plausible daily-use grasp tasks."
    return f"""
You are an affordance planner for first-person prosthetic grasp data generation.

You receive images:
1. RGB scene image.
2. Binary object mask image. White pixels indicate the target object.
3. Optional RGB scene image with a 10x10 coordinate grid. Use it to estimate normalized coordinates accurately.

Known object name: {object_block}
User task instruction: {task_block}

Goal:
Generate {num_candidates} lollipop hand-placement candidates for synthesizing a healthy adult right human hand interacting with the target object.

A lollipop candidate is a coarse hand proxy:
- palm_center_xy_norm: center of the palm/contact region, normalized image coordinates [x, y].
- approach_from_xy_norm: a point away from the object indicating where the wrist or forearm approaches from.
- palm_radius_norm: radius relative to max(image_width, image_height).
- arm_width_norm: arm strip width relative to max(image_width, image_height).

Rules:
- The palm center should be near a functional contact part, not a random object center.
- The approach point should extend outward from the palm toward a plausible wrist/forearm direction.
- Prefer functional affordances: handle, lid, cap, rim, body grip surface, side wall, top button, stable support region.
- For objects with handles, include at least one handle grasp candidate.
- For a handle candidate, palm_center_xy_norm must lie on or immediately outside the visible handle region, not on the object body, spout, rim, or lid.
- For a lid/cap/top candidate, palm_center_xy_norm must lie near the visible lid/cap/top region.
- For a body/side candidate, palm_center_xy_norm must lie near the side wall or main body grip surface.
- For containers, include candidates such as handle grasp, body power grasp, lid/cap pinch, rim grasp, or top stabilization if visible.
- For cylindrical objects, include side power grasp and top stabilization if useful.
- For round objects, include spherical power grasp and top/side stabilization.
- Avoid candidates that require impossible right-hand geometry or severe object penetration.
- Keep task short and action-specific, for example "grasp the handle", "hold the body", "pinch the lid".
- Do not mention image editing, inpainting, masks, prompts, or VLMs in the candidate task.
- Use normalized coordinates in [0, 1]. The origin is the top-left image corner.
- Output only valid JSON. Do not output markdown.

Output schema:
{{
  "target_object": "short English object name",
  "global_task": "short English daily-use task",
  "candidates": [
    {{
      "id": "lollipop_00",
      "affordance_part": "handle | lid | cap | body | rim | side | top | other",
      "task": "short action phrase",
      "grasp_type": "short right-hand grasp type",
      "palm_center_xy_norm": [0.0, 0.0],
      "approach_from_xy_norm": [0.0, 0.0],
      "palm_radius_norm": 0.08,
      "arm_width_norm": 0.05,
      "priority": 1,
      "reason": "brief spatial and functional reason"
    }}
  ]
}}
""".strip()


def rgb_to_data_url(image_rgb: np.ndarray) -> str:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_rgb.shape}.")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    handle = io.BytesIO()
    Image.fromarray(image_rgb).save(handle, format="PNG")
    encoded = base64.b64encode(handle.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def mask_to_data_url(mask: np.ndarray) -> str:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}.")
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    return rgb_to_data_url(np.repeat(mask_u8[..., None], 3, axis=2))


def coordinate_grid_data_url(image_rgb: np.ndarray) -> str:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_rgb.shape}.")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    grid = image_rgb.copy()
    height, width = grid.shape[:2]
    color = (255, 255, 255)
    shadow = (0, 0, 0)
    for i in range(11):
        x = int(round(i * (width - 1) / 10))
        y = int(round(i * (height - 1) / 10))
        cv2.line(grid, (x, 0), (x, height - 1), color, 1, lineType=cv2.LINE_AA)
        cv2.line(grid, (0, y), (width - 1, y), color, 1, lineType=cv2.LINE_AA)
        label = f"{i / 10:.1f}"
        cv2.putText(grid, label, (min(x + 3, width - 42), 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, shadow, 2)
        cv2.putText(grid, label, (min(x + 3, width - 42), 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        cv2.putText(grid, label, (3, min(y + 14, height - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, shadow, 2)
        cv2.putText(grid, label, (3, min(y + 14, height - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    return rgb_to_data_url(grid)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from VLM, got {type(parsed).__name__}.")
    return parsed


def _number_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list.")
    return float(value[0]), float(value[1])


def _clamp_norm_pair(value: tuple[float, float]) -> tuple[float, float]:
    return float(np.clip(value[0], 0.0, 1.0)), float(np.clip(value[1], 0.0, 1.0))


def _norm_to_pixel(value: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    x = float(np.clip(value[0], 0.0, 1.0) * (width - 1))
    y = float(np.clip(value[1], 0.0, 1.0) * (height - 1))
    return x, y


def extend_ray_to_image_edge(
    palm_xy: tuple[float, float],
    approach_xy: tuple[float, float],
    width: int,
    height: int,
) -> tuple[float, float]:
    """Extend the wrist/forearm side of the lollipop to the image border."""
    px, py = palm_xy
    ax, ay = approach_xy
    dx = ax - px
    dy = ay - py
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        dx = px - cx
        dy = py - cy
        norm = math.hypot(dx, dy)
    if norm < 1e-6:
        dx, dy = 0.0, 1.0
        norm = 1.0
    dx /= norm
    dy /= norm

    ts: list[float] = []
    eps = 1e-6
    if abs(dx) > eps:
        ts.append((0.0 - px) / dx)
        ts.append((float(width - 1) - px) / dx)
    if abs(dy) > eps:
        ts.append((0.0 - py) / dy)
        ts.append((float(height - 1) - py) / dy)
    valid_points: list[tuple[float, float, float]] = []
    for t in ts:
        if t <= 0:
            continue
        x = px + dx * t
        y = py + dy * t
        if -0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5:
            valid_points.append((t, float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))))
    if not valid_points:
        return float(np.clip(approach_xy[0], 0, width - 1)), float(np.clip(approach_xy[1], 0, height - 1))
    _, x_edge, y_edge = min(valid_points, key=lambda item: item[0])
    return x_edge, y_edge


def _scalar_norm(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return float(np.clip(number, low, high))


def parse_affordance_lollipop_candidates(
    parsed: dict[str, Any],
    *,
    image_shape: tuple[int, int],
    config: Phase2AffordanceLollipopConfig,
) -> list[AffordanceLollipopCandidate]:
    height, width = image_shape
    scale = float(max(width, height))
    raw_candidates = parsed.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"VLM response missing non-empty candidates list: {parsed}")

    candidates: list[AffordanceLollipopCandidate] = []
    for index, item in enumerate(raw_candidates[: config.num_candidates]):
        if not isinstance(item, dict):
            continue
        palm_norm = _clamp_norm_pair(_number_pair(item.get("palm_center_xy_norm"), "palm_center_xy_norm"))
        approach_norm = _clamp_norm_pair(_number_pair(item.get("approach_from_xy_norm"), "approach_from_xy_norm"))
        palm_xy = _norm_to_pixel(palm_norm, width, height)
        approach_xy = _norm_to_pixel(approach_norm, width, height)
        tip_xy = extend_ray_to_image_edge(palm_xy, approach_xy, width, height)
        palm_radius_norm = _scalar_norm(
            item.get("palm_radius_norm"),
            config.default_palm_radius_norm,
            config.min_palm_radius_norm,
            config.max_palm_radius_norm,
        )
        arm_width_norm = _scalar_norm(
            item.get("arm_width_norm"),
            config.default_arm_width_norm,
            config.min_arm_width_norm,
            config.max_arm_width_norm,
        )
        theta = math.atan2(palm_xy[1] - tip_xy[1], palm_xy[0] - tip_xy[0])
        candidate_id = str(item.get("id") or f"lollipop_{index:02d}").strip() or f"lollipop_{index:02d}"
        candidates.append(
            AffordanceLollipopCandidate(
                candidate_id=candidate_id,
                affordance_part=str(item.get("affordance_part", "affordance")).strip(),
                task=str(item.get("task", "grasp the object")).strip(),
                grasp_type=str(item.get("grasp_type", "right-hand grasp")).strip(),
                palm_center_xy=palm_xy,
                tip_xy=tip_xy,
                palm_radius=palm_radius_norm * scale,
                arm_width=arm_width_norm * scale,
                theta_rad=theta,
                priority=int(item.get("priority") or index + 1),
                reason=str(item.get("reason", "")).strip(),
            )
        )
    if not candidates:
        raise ValueError(f"No valid VLM lollipop candidates could be parsed: {parsed}")
    return candidates


def draw_lollipop_mask(shape: tuple[int, int], candidate: AffordanceLollipopCandidate) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    palm = tuple(int(round(v)) for v in candidate.palm_center_xy)
    tip = tuple(int(round(v)) for v in candidate.tip_xy)
    cv2.line(mask, tip, palm, 255, int(round(candidate.arm_width)), lineType=cv2.LINE_AA)
    cv2.circle(mask, palm, int(round(candidate.palm_radius)), 255, thickness=-1, lineType=cv2.LINE_AA)
    return mask


def overlay_lollipop_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    color = np.zeros_like(rgb)
    color[..., 0] = 255
    color[..., 1] = 80
    alpha = (mask.astype(np.float32) / 255.0) * 0.42
    return (overlay * (1.0 - alpha[..., None]) + color * alpha[..., None]).astype(np.uint8)


def candidate_to_json(candidate: AffordanceLollipopCandidate) -> dict[str, Any]:
    return asdict(candidate) | {"strategy": candidate.strategy}


class Phase2AffordanceLollipop:
    """Use a VLM to generate affordance-aware lollipop masks and short tasks."""

    def __init__(self, config: Phase2AffordanceLollipopConfig) -> None:
        self.config = config

    def run(
        self,
        image_rgb: np.ndarray,
        object_mask: np.ndarray,
        *,
        object_name: str | None = None,
        task_instruction: str | None = None,
    ) -> Phase2AffordanceLollipopResult:
        import requests

        prompt = build_affordance_lollipop_prompt(
            object_name=object_name,
            task_instruction=task_instruction,
            num_candidates=self.config.num_candidates,
        )
        api_key = get_secret("ZENMUX_API_KEY") or get_secret("OPENAI_API_KEY", required=True)
        base_url = get_secret("ZENMUX_BASE_URL") or self.config.zenmux_base_url
        url = base_url.rstrip("/") + "/chat/completions"
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": rgb_to_data_url(image_rgb)}},
            {"type": "image_url", "image_url": {"url": mask_to_data_url(object_mask)}},
        ]
        if self.config.use_coordinate_grid:
            content.append({"type": "image_url", "image_url": {"url": coordinate_grid_data_url(image_rgb)}})
        payload = {
            "model": self.config.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        if response.status_code in {400, 422}:
            retry_payload = dict(payload)
            retry_payload.pop("response_format", None)
            response = requests.post(url, headers=headers, json=retry_payload, timeout=180)
        if not response.ok:
            try:
                error_text = str(response.json())
            except ValueError:
                error_text = response.text
            raise RuntimeError(
                f"ZenMux affordance lollipop API failed for model {self.config.model_id!r} "
                f"with HTTP {response.status_code}: {error_text}"
            )
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"ZenMux affordance lollipop API returned no choices: {body}")
        message = choices[0].get("message") or {}
        raw_response = str(message.get("content", "")).strip()
        if not raw_response:
            raise RuntimeError(f"ZenMux affordance lollipop API returned empty content: {body}")
        parsed = extract_json_object(raw_response)
        candidates = parse_affordance_lollipop_candidates(
            parsed,
            image_shape=image_rgb.shape[:2],
            config=self.config,
        )
        return Phase2AffordanceLollipopResult(
            target_object=str(parsed.get("target_object") or object_name or "").strip(),
            global_task=str(parsed.get("global_task") or task_instruction or "").strip(),
            candidates=candidates,
            model_id=self.config.model_id,
            prompt=prompt,
            raw_response=raw_response,
            parsed_json=parsed,
        )
