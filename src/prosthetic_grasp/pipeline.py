from __future__ import annotations

from prosthetic_grasp.common.types import PipelineResult, SensorFrame
from prosthetic_grasp.phases.phase1_mask import Phase1Mask, Phase1MaskConfig
from prosthetic_grasp.phases.phase2_lollipop import Phase2Lollipop, Phase2LollipopConfig
from prosthetic_grasp.phases.phase3_erase import Phase3Erase, Phase3EraseConfig
from prosthetic_grasp.phases.phase4_intention import Phase4Intention, Phase4IntentionConfig
from prosthetic_grasp.phases.phase4_inpaint import Phase4Inpaint, Phase4InpaintConfig
from prosthetic_grasp.phases.phase5_mano import Phase5Mano, Phase5ManoConfig
from prosthetic_grasp.phases.phase6_prosthetic_action import Phase6ProstheticAction, Phase6ProstheticActionConfig


def _phase4_config_with_intention(config: Phase4InpaintConfig, intention: str) -> Phase4InpaintConfig:
    phase4_config = Phase4InpaintConfig(**config.__dict__)
    phase4_config.intention = intention
    if "{intention}" in phase4_config.prompt:
        phase4_config.prompt = phase4_config.prompt.format(intention=intention)
    return phase4_config


class ProstheticGraspPipeline:
    def __init__(self, settings: dict) -> None:
        self.settings = settings
        phase1 = settings.get("phase1_mask", {})
        phase2 = settings.get("phase2_lollipop", {})
        phase3 = settings.get("phase3_erase", {})
        phase4_intention = settings.get("phase4_intention")
        phase4 = settings.get("phase4_inpaint", {})
        phase5 = settings.get("phase5_mano", {})
        phase6 = settings.get("phase6_prosthetic_action", {})

        self.phase1_mask = Phase1Mask(Phase1MaskConfig(**phase1))
        self.phase2_lollipop = Phase2Lollipop(Phase2LollipopConfig(**phase2))
        self.phase3_erase = Phase3Erase(Phase3EraseConfig(**phase3))
        self.phase4_intention = None
        if phase4_intention is not None:
            phase4_intention_config = dict(phase4_intention)
            enabled = bool(phase4_intention_config.pop("enabled", True))
            if enabled:
                self.phase4_intention = Phase4Intention(Phase4IntentionConfig(**phase4_intention_config))
        self.phase4_inpaint = Phase4Inpaint(Phase4InpaintConfig(**phase4))
        self.phase5_mano = Phase5Mano(Phase5ManoConfig(**phase5))
        self.phase6_prosthetic_action = Phase6ProstheticAction(Phase6ProstheticActionConfig(**phase6))

    def run(self, frame: SensorFrame, task_instruction: str | None = None) -> PipelineResult:
        phase1_result = self.phase1_mask.run(frame.rgb)
        phase2_result = self.phase2_lollipop.run(phase1_result.mask)
        phase3_result = self.phase3_erase.run(frame.rgb, phase1_result.mask)
        phase4_intention_result = None
        phase4_inpaint = self.phase4_inpaint
        if self.phase4_intention is not None:
            phase4_intention_result = self.phase4_intention.run(
                phase3_result.erased_full,
                task_instruction=task_instruction,
            )
            phase4_config = _phase4_config_with_intention(
                self.phase4_inpaint.config,
                phase4_intention_result.phase4_intention,
            )
            phase4_inpaint = Phase4Inpaint(phase4_config)
        phase4_result = phase4_inpaint.run(phase3_result.erased_full, phase2_result.lollipop_mask)
        phase5_result = self.phase5_mano.run(phase4_result.inpaint_full)
        phase6_result = self.phase6_prosthetic_action.run(phase5_result)
        status = "ok"
        if phase5_result.status != "ok" or phase6_result.status != "ok":
            status = "partial"
        return PipelineResult(
            status=status,
            frame=frame,
            phase1_mask=phase1_result,
            phase2_lollipop=phase2_result,
            phase3_erase=phase3_result,
            phase4_intention=phase4_intention_result,
            phase4_inpaint=phase4_result,
            phase5_mano=phase5_result,
            phase6_prosthetic_action=phase6_result,
        )
