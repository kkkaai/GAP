from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from prosthetic_grasp.common.types import Phase2LollipopResult


@dataclass
class Phase2LollipopConfig:
    min_mask_points: int = 10
    palm_center_ratio: float = 0.30
    min_axis_band_px: float = 20.0
    axis_band_ratio: float = 0.12
    palm_width_std_scale: float = 1.8
    min_palm_half_width: float = 8.0
    palm_radius_scale: float = 1.25
    strip_half_width_scale: float = 0.55
    min_strip_half_width: float = 4.0

    def __post_init__(self) -> None:
        if self.min_mask_points <= 0:
            raise ValueError(f"min_mask_points must be positive, got {self.min_mask_points}.")
        if not 0.0 <= self.palm_center_ratio <= 1.0:
            raise ValueError(f"palm_center_ratio must be in [0, 1], got {self.palm_center_ratio}.")
        positive_fields = {
            "min_axis_band_px": self.min_axis_band_px,
            "axis_band_ratio": self.axis_band_ratio,
            "palm_width_std_scale": self.palm_width_std_scale,
            "min_palm_half_width": self.min_palm_half_width,
            "palm_radius_scale": self.palm_radius_scale,
            "strip_half_width_scale": self.strip_half_width_scale,
            "min_strip_half_width": self.min_strip_half_width,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")


def _fit_lollipop_from_mask(mask: np.ndarray, config: Phase2LollipopConfig):
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D hand mask, got {mask.shape}.")
    points = np.argwhere(mask)
    h, w = mask.shape
    if len(points) < config.min_mask_points:
        raise ValueError("Mask too small to fit lollipop.")

    xy = points[:, ::-1].astype(np.float32)
    center = xy.mean(axis=0)
    centered = xy - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]

    proj = centered @ axis
    p0 = center + axis * proj.min()
    p1 = center + axis * proj.max()

    def border_distance(p):
        x, y = p
        return min(x, y, w - 1 - x, h - 1 - y)

    if border_distance(p0) < border_distance(p1):
        p_wrist = p0
        p_tip = p1
    else:
        p_wrist = p1
        p_tip = p0

    c_palm = p_wrist + config.palm_center_ratio * (p_tip - p_wrist)
    orth = np.array([-axis[1], axis[0]], dtype=np.float32)

    rel = xy - c_palm[None]
    along = rel @ axis
    across = rel @ orth
    keep = np.abs(along) < max(config.min_axis_band_px, config.axis_band_ratio * np.linalg.norm(p_tip - p_wrist))
    if keep.sum() < config.min_mask_points:
        keep = np.ones(len(xy), dtype=bool)

    palm_half_width = max(float(np.std(across[keep]) * config.palm_width_std_scale), config.min_palm_half_width)
    theta = math.atan2(float(axis[1]), float(axis[0]))
    b1 = math.cos(theta)
    b2 = math.sin(theta)

    params = (float(c_palm[0]), float(c_palm[1]), float(palm_half_width), float(b1), float(b2))
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


def _render_lollipop(mask_shape, params, p_wrist, config: Phase2LollipopConfig):
    h, w = mask_shape
    x, y, a, b1, b2 = params
    canvas_image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas_image)
    palm_radius = int(round(config.palm_radius_scale * a))
    cx = int(round(x))
    cy = int(round(y))
    draw.ellipse((cx - palm_radius, cy - palm_radius, cx + palm_radius, cy + palm_radius), fill=255)

    wrist_dir = np.array(p_wrist, dtype=np.float32) - np.array([x, y], dtype=np.float32)
    norm = np.linalg.norm(wrist_dir)
    if norm < 1e-6:
        wrist_dir = np.array([b1, b2], dtype=np.float32)
        norm = np.linalg.norm(wrist_dir)
    wrist_dir = wrist_dir / max(norm, 1e-6)

    p_start = np.array([x, y], dtype=np.float32)
    p_end = _intersect_ray_with_image_border(p_start, wrist_dir, h, w)
    strip_half_width = int(round(max(config.strip_half_width_scale * a, config.min_strip_half_width)))
    orth = np.array([-wrist_dir[1], wrist_dir[0]], dtype=np.float32)
    p1 = p_start + orth * strip_half_width
    p2 = p_start - orth * strip_half_width
    p3 = p_end - orth * strip_half_width
    p4 = p_end + orth * strip_half_width
    poly = np.round(np.stack([p1, p2, p3, p4], axis=0)).astype(np.int32)
    draw.polygon([tuple(point) for point in poly], fill=255)
    return np.array(canvas_image) > 127


class Phase2Lollipop:
    def __init__(self, config: Phase2LollipopConfig | None = None) -> None:
        self.config = config or Phase2LollipopConfig()

    def run(self, hand_mask: np.ndarray) -> Phase2LollipopResult:
        params, p_wrist, p_tip = _fit_lollipop_from_mask(hand_mask, self.config)
        lollipop_mask = _render_lollipop(hand_mask.shape, params, p_wrist, self.config)
        return Phase2LollipopResult(
            lollipop_mask=lollipop_mask,
            lolli_params=params,
            wrist_point=(float(p_wrist[0]), float(p_wrist[1])),
            tip_point=(float(p_tip[0]), float(p_tip[1])),
        )
