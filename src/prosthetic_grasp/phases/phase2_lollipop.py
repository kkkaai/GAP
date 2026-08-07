from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from prosthetic_grasp.common.types import Phase2LollipopResult


@dataclass
class Phase2LollipopConfig:
    min_mask_points: int = 10
    min_direction_norm_px: float = 1.0
    line_length_scale: float = 2.0
    min_iou: float = 0.5
    palm_radius_scale: float = 1.1
    strip_width_scale: float = 1.1

    def __post_init__(self) -> None:
        if self.min_mask_points <= 0:
            raise ValueError(f"min_mask_points must be positive, got {self.min_mask_points}.")
        for name, value in {
            "min_direction_norm_px": self.min_direction_norm_px,
            "line_length_scale": self.line_length_scale,
            "palm_radius_scale": self.palm_radius_scale,
            "strip_width_scale": self.strip_width_scale,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if not 0.0 <= self.min_iou <= 1.0:
            raise ValueError(f"min_iou must be in [0, 1], got {self.min_iou}.")


def _fit_lollipop_from_mask(mask: np.ndarray, config: Phase2LollipopConfig):
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D hand mask, got {mask.shape}.")
    points = np.argwhere(mask)
    h, w = mask.shape
    if len(points) < config.min_mask_points:
        raise ValueError("Mask too small to fit lollipop.")

    xy = points[:, ::-1].astype(np.float32)
    x_min = float(xy[:, 0].min())
    x_max = float(xy[:, 0].max())
    y_min = float(xy[:, 1].min())
    y_max = float(xy[:, 1].max())
    center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float32)

    direction = np.mean(xy - center[None], axis=0)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < config.min_direction_norm_px:
        centroid = xy.mean(axis=0)
        direction = centroid - center
        direction_norm = float(np.linalg.norm(direction))
    if direction_norm < config.min_direction_norm_px:
        centered = xy - xy.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
        direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-6:
        raise ValueError("Mask direction is degenerate; cannot fit lollipop.")

    direction = direction / direction_norm
    size = max((x_max - x_min) / 2.0, (y_max - y_min) / 2.0)
    params = (float(center[0]), float(center[1]), float(size), float(direction[0]), float(direction[1]))
    p_wrist = center
    p_tip = center + direction * size
    return params, p_wrist, p_tip


def _intersect_ray_with_image_border(origin, direction, h, w):
    x, y = origin
    dx, dy = direction
    ts = []
    eps = 1e-6
    if abs(dx) > eps:
        ts.append((0 - x) / dx)
        ts.append((w - 1 - x) / dx)
    if abs(dy) > eps:
        ts.append((0 - y) / dy)
        ts.append((h - 1 - y) / dy)
    ts = [t for t in ts if t > 0]
    if not ts:
        return np.array([x, y], dtype=np.float32)
    t = min(ts)
    return np.array([x + t * dx, y + t * dy], dtype=np.float32)


def rotate_lollipop_params(params, angle_degrees: float):
    """Rotate the lollipop direction around its circle center in image coordinates."""
    x, y, size, dir_x, dir_y = params
    theta = np.deg2rad(angle_degrees)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    rotated_x = cos_t * dir_x - sin_t * dir_y
    rotated_y = sin_t * dir_x + cos_t * dir_y
    direction = np.array([rotated_x, rotated_y], dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    return (float(x), float(y), float(size), float(direction[0]), float(direction[1]))


def mirror_lollipop_params(params, axis: str = "horizontal"):
    """Mirror the lollipop direction around its circle center."""
    x, y, size, dir_x, dir_y = params
    axis = axis.strip().lower()
    if axis == "none":
        mirrored_x = dir_x
        mirrored_y = dir_y
    elif axis == "horizontal":
        mirrored_x = -dir_x
        mirrored_y = dir_y
    elif axis == "vertical":
        mirrored_x = dir_x
        mirrored_y = -dir_y
    elif axis == "both":
        mirrored_x = -dir_x
        mirrored_y = -dir_y
    else:
        raise ValueError(f"Unsupported lollipop mirror axis: {axis!r}.")
    direction = np.array([mirrored_x, mirrored_y], dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    return (float(x), float(y), float(size), float(direction[0]), float(direction[1]))


def render_lollipop_mask(
    mask_shape: tuple[int, int],
    params,
    config: Phase2LollipopConfig | None = None,
) -> np.ndarray:
    config = config or Phase2LollipopConfig()
    h, w = mask_shape
    x, y, size, dir_x, dir_y = params
    canvas_image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas_image)
    palm_radius = int(round(size * config.palm_radius_scale))
    cx = int(round(x))
    cy = int(round(y))
    draw.ellipse((cx - palm_radius, cy - palm_radius, cx + palm_radius, cy + palm_radius), fill=255)

    direction = np.array([dir_x, dir_y], dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    p_start = np.array([x, y], dtype=np.float32)
    p_end = p_start + direction * (config.line_length_scale * h)
    strip_half_width = int(round(max((size / 2.0) * config.strip_width_scale, 1.0)))
    orth = np.array([-direction[1], direction[0]], dtype=np.float32)
    p1 = p_start + orth * strip_half_width
    p2 = p_start - orth * strip_half_width
    p3 = p_end - orth * strip_half_width
    p4 = p_end + orth * strip_half_width
    poly = np.round(np.stack([p1, p2, p3, p4], axis=0)).astype(np.int32)
    draw.polygon([tuple(point) for point in poly], fill=255)
    return np.array(canvas_image) > 127


def _render_lollipop(mask: np.ndarray, params, config: Phase2LollipopConfig):
    lollipop_mask = render_lollipop_mask(mask.shape, params, config)
    iou = float(np.logical_and(lollipop_mask, mask).sum() / np.logical_or(lollipop_mask, mask).sum())
    if iou < config.min_iou:
        raise ValueError(f"Lollipop IoU below threshold: {iou:.4f} < {config.min_iou:.4f}.")
    return lollipop_mask


class Phase2Lollipop:
    def __init__(self, config: Phase2LollipopConfig | None = None) -> None:
        self.config = config or Phase2LollipopConfig()

    def run(self, hand_mask: np.ndarray) -> Phase2LollipopResult:
        params, p_wrist, p_tip = _fit_lollipop_from_mask(hand_mask, self.config)
        lollipop_mask = _render_lollipop(hand_mask, params, self.config)
        return Phase2LollipopResult(
            lollipop_mask=lollipop_mask,
            lolli_params=params,
            wrist_point=(float(p_wrist[0]), float(p_wrist[1])),
            tip_point=(float(p_tip[0]), float(p_tip[1])),
        )
