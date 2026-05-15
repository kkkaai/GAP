from __future__ import annotations

from prosthetic_grasp.common.types import PhasePlaceholderResult


class Phase5Mano:
    """Placeholder for future HaMeR integration into src/."""

    def run(self, *_args, **_kwargs) -> PhasePlaceholderResult:
        return PhasePlaceholderResult(
            status="not_implemented",
            message=(
                "phase5_mano is intentionally left blank in src/. "
                "The validated notebook currently runs HaMeR separately, "
                "but this has not yet been integrated into the local pipeline."
            ),
        )
