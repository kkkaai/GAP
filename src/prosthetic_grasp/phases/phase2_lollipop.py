from __future__ import annotations

import math

import cv2
import numpy as np

from prosthetic_grasp.common.types import Phase2LollipopResult


def _fit_lollipop_from_mask(mask: np.ndarray):
    points = np.argwhere(mask)
    h, w = mask.shape
    if len(points) < 10:
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

    c_palm = p_wrist + 0.30 * (p_tip - p_wrist)
    orth = np.array([-axis[1], axis[0]], dtype=np.float32)

    rel = xy - c_palm[None]
    along = rel @ axis
    across = rel @ orth
    keep = np.abs(along) < max(20.0, 0.12 * np.linalg.norm(p_tip - p_wrist))
    if keep.sum() < 10:
        keep = np.ones(len(xy), dtype=bool)

    palm_half_width = max(float(np.std(across[keep]) * 1.8), 8.0)
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


def _render_lollipop(mask_shape, params, p_wrist):
    h, w = mask_shape
    x, y, a, b1, b2 = params
    canvas = np.zeros((h, w), dtype=np.float32)
    palm_radius = int(round(1.25 * a))
    cv2.circle(canvas, (int(round(x)), int(round(y))), palm_radius, 1.0, -1)

    wrist_dir = np.array(p_wrist, dtype=np.float32) - np.array([x, y], dtype=np.float32)
    norm = np.linalg.norm(wrist_dir)
    if norm < 1e-6:
        wrist_dir = np.array([b1, b2], dtype=np.float32)
        norm = np.linalg.norm(wrist_dir)
    wrist_dir = wrist_dir / max(norm, 1e-6)

    p_start = np.array([x, y], dtype=np.float32)
    p_end = _intersect_ray_with_image_border(p_start, wrist_dir, h, w)
    strip_half_width = int(round(max(0.55 * a, 4)))
    orth = np.array([-wrist_dir[1], wrist_dir[0]], dtype=np.float32)
    p1 = p_start + orth * strip_half_width
    p2 = p_start - orth * strip_half_width
    p3 = p_end - orth * strip_half_width
    p4 = p_end + orth * strip_half_width
    poly = np.round(np.stack([p1, p2, p3, p4], axis=0)).astype(np.int32)
    cv2.fillConvexPoly(canvas, poly, 1.0)
    return canvas > 0.5


class Phase2Lollipop:
    def run(self, hand_mask: np.ndarray) -> Phase2LollipopResult:
        params, p_wrist, p_tip = _fit_lollipop_from_mask(hand_mask)
        lollipop_mask = _render_lollipop(hand_mask.shape, params, p_wrist)
        return Phase2LollipopResult(
            lollipop_mask=lollipop_mask,
            lolli_params=params,
            wrist_point=(float(p_wrist[0]), float(p_wrist[1])),
            tip_point=(float(p_tip[0]), float(p_tip[1])),
        )
