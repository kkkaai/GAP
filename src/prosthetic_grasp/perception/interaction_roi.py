from __future__ import annotations

from prosthetic_grasp.common.types import InteractionROI


def build_interaction_roi(
    image_shape: tuple[int, int, int],
    lolli_params: tuple[float, float, float, float, float],
    forward_scale: float,
    side_scale: float,
    min_size_px: int,
) -> InteractionROI:
    height, width = image_shape[:2]
    x, y, a, b1, b2 = lolli_params

    forward = max(int(round(forward_scale * max(a, 32.0))), min_size_px // 2)
    side = max(int(round(side_scale * max(a, 32.0))), min_size_px // 2)

    cx = int(round(x + b1 * forward))
    cy = int(round(y + b2 * forward))

    half_w = max(side, min_size_px // 2)
    half_h = max(side, min_size_px // 2)

    x0 = max(0, cx - half_w)
    y0 = max(0, cy - half_h)
    x1 = min(width, cx + half_w)
    y1 = min(height, cy + half_h)

    return InteractionROI(x0=x0, y0=y0, x1=x1, y1=y1)

