from __future__ import annotations

from prosthetic_grasp.common.types import PipelineResult, SensorFrame
from prosthetic_grasp.phases.phase1_mask import Phase1Mask, Phase1MaskConfig
from prosthetic_grasp.phases.phase2_lollipop import Phase2Lollipop
from prosthetic_grasp.phases.phase3_erase import Phase3Erase, Phase3EraseConfig
from prosthetic_grasp.phases.phase4_inpaint import Phase4Inpaint, Phase4InpaintConfig
from prosthetic_grasp.phases.phase5_mano import Phase5Mano
from prosthetic_grasp.phases.phase6_prosthetic_action import Phase6ProstheticAction


class ProstheticGraspPipeline:
    def __init__(self, settings: dict) -> None:
        self.settings = settings
        phase1 = settings.get("phase1_mask", {})
        phase3 = settings.get("phase3_erase", {})
        phase4 = settings.get("phase4_inpaint", {})

        self.phase1_mask = Phase1Mask(Phase1MaskConfig(**phase1))
        self.phase2_lollipop = Phase2Lollipop()
        self.phase3_erase = Phase3Erase(Phase3EraseConfig(**phase3))
        self.phase4_inpaint = Phase4Inpaint(Phase4InpaintConfig(**phase4))
        self.phase5_mano = Phase5Mano()
        self.phase6_prosthetic_action = Phase6ProstheticAction()

    def run(self, frame: SensorFrame) -> PipelineResult:
        phase1_result = self.phase1_mask.run(frame.rgb)
        phase2_result = self.phase2_lollipop.run(phase1_result.mask)
        phase3_result = self.phase3_erase.run(frame.rgb, phase2_result.lollipop_mask)
        phase4_result = self.phase4_inpaint.run(phase3_result.erased_full, phase2_result.lollipop_mask)
        phase5_result = self.phase5_mano.run(phase4_result.inpaint_full)
        phase6_result = self.phase6_prosthetic_action.run(phase5_result)
        return PipelineResult(
            status="ok",
            frame=frame,
            phase1_mask=phase1_result,
            phase2_lollipop=phase2_result,
            phase3_erase=phase3_result,
            phase4_inpaint=phase4_result,
            phase5_mano=phase5_result,
            phase6_prosthetic_action=phase6_result,
        )
