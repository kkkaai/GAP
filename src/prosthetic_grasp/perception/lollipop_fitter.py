from __future__ import annotations

import math

import numpy as np


def _clip_int(value: float, low: int, high: int) -> int:
    return max(low, min(int(round(value)), high))


def fit_lollipop(mask: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float, float]]:
    points = np.argwhere(mask)
    height, width = mask.shape

    if len(points) < 10:
        empty = np.zeros_like(mask, dtype=bool)
        return empty, (0.0, 0.0, 0.0, 1.0, 0.0)

    xy = points[:, ::-1].astype(np.float32)
    center = xy.mean(axis=0)
    centered = xy - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]

    proj = centered @ axis
    p_min = center + axis * proj.min()
    p_max = center + axis * proj.max()
    p_wrist = p_min
    p_tip = p_max
    c_palm = p_wrist + 0.65 * (p_tip - p_wrist)

    orth = np.array([-axis[1], axis[0]], dtype=np.float32)
    transverse = centered @ orth
    a = max(float(np.std(transverse) * 2.0), 8.0)

    theta = math.atan2(float(axis[1]), float(axis[0]))
    b1 = math.cos(theta)
    b2 = math.sin(theta)

    lollipop = np.zeros_like(mask, dtype=bool)
    yy, xx = np.indices(mask.shape)
    circle = (xx - c_palm[0]) ** 2 + (yy - c_palm[1]) ** 2 <= (1.3 * a) ** 2
    lollipop |= circle

    num_steps = max(int(np.linalg.norm(c_palm - p_wrist)), 1)
    for t in np.linspace(0.0, 1.0, num_steps):
        point = p_wrist * (1 - t) + c_palm * t
        radius = max(int(round(0.6 * a)), 2)
        x = _clip_int(point[0], 0, width - 1)
        y = _clip_int(point[1], 0, height - 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        lollipop[y0:y1, x0:x1] = True

    params = (float(c_palm[0]), float(c_palm[1]), float(a), float(b1), float(b2))
    return lollipop, params

