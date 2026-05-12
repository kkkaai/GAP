from __future__ import annotations

import math

import numpy as np

from prosthetic_grasp.common.types import HumanPrior2D, IntentSpec, InteractionROI


class HandProxyExtractor:
    """Stub human-prior extractor.

    Replace with MediaPipe or HaMeR-backed extraction later.
    """

    def __init__(self, model_name: str = "stub") -> None:
        self.model_name = model_name

    def extract(
        self,
        generated_rgb: np.ndarray,
        roi: InteractionROI,
        intent: IntentSpec,
    ) -> HumanPrior2D:
        anchor = ((roi.x0 + roi.x1) // 2, (roi.y0 + roi.y1) // 2)
        theta = -math.pi / 2.0
        hand_span_px = float(max(roi.x1 - roi.x0, roi.y1 - roi.y0) * 0.35)
        contact = np.zeros(generated_rgb.shape[:2], dtype=bool)
        cy, cx = anchor[1], anchor[0]
        radius = max(int(hand_span_px * 0.18), 6)
        yy, xx = np.indices(contact.shape)
        contact |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2

        grasp_hint = "power_wrap"
        if intent.grasp_part in {"handle", "cap"}:
            grasp_hint = "pinch"
        elif intent.task == "use":
            grasp_hint = "tripod"

        return HumanPrior2D(
            anchor_uv=anchor,
            approach_theta=theta,
            human_lolli=(float(anchor[0]), float(anchor[1]), hand_span_px / 3.0, 0.0, -1.0),
            contact_patch=contact,
            hand_span_px=hand_span_px,
            object_width_px=None,
            grasp_family_hint=grasp_hint,
            confidence=0.8,
        )

