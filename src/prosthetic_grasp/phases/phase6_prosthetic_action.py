from __future__ import annotations

from prosthetic_grasp.common.types import PhasePlaceholderResult


class Phase6ProstheticAction:
    """Placeholder for future MANO-to-prosthetic retargeting."""

    def run(self, *_args, **_kwargs) -> PhasePlaceholderResult:
        return PhasePlaceholderResult(
            status="not_implemented",
            message=(
                "phase6_prosthetic_action is intentionally left blank. "
                "MANO / fingertip extraction has not yet been converted into "
                "prosthetic control commands in src/."
            ),
        )
