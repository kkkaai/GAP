from __future__ import annotations

from prosthetic_grasp.common.types import ExecutionResult, ProstheticAction


class HandExecutor:
    """Stub executor.

    Replace with the real prosthesis hardware interface.
    """

    def __init__(self, model_name: str = "stub") -> None:
        self.model_name = model_name

    def execute(self, action: ProstheticAction) -> ExecutionResult:
        return ExecutionResult(
            status="simulated_success",
            message="Stub executor accepted the action.",
            telemetry={
                "grasp_type": action.grasp_type,
                "aperture": action.aperture,
                "force_level": action.force_level,
            },
        )

