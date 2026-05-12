from __future__ import annotations

from prosthetic_grasp.common.types import (
    Burst,
    PipelineResult,
    ProsthesisMaskState,
    SceneLite,
    SensorFrame,
)
from prosthetic_grasp.control.hand_executor import HandExecutor
from prosthetic_grasp.extraction.hand_proxy_extractor import HandProxyExtractor
from prosthetic_grasp.generation.candidate_ranker import rank_candidates
from prosthetic_grasp.generation.clean_inpainter import CleanInpainter
from prosthetic_grasp.generation.flux_fill_client import FluxFillClient
from prosthetic_grasp.generation.prompt_builder import build_generation_prompt
from prosthetic_grasp.perception.frame_stability import is_stable_burst, select_best_frame
from prosthetic_grasp.perception.intent_parser import parse_intent
from prosthetic_grasp.perception.interaction_roi import build_interaction_roi
from prosthetic_grasp.perception.lollipop_fitter import fit_lollipop
from prosthetic_grasp.perception.prosthesis_segmentor import ProsthesisSegmentor
from prosthetic_grasp.retarget.local_optimizer import optimize_action
from prosthetic_grasp.retarget.rule_initializer import initialize_action


def _build_removal_mask(fine_mask, lollipop_mask):
    return fine_mask | lollipop_mask


class ProstheticGraspPipeline:
    def __init__(self, settings: dict) -> None:
        self.settings = settings
        model_settings = settings["models"]
        self.segmentor = ProsthesisSegmentor(model_settings["prosthesis_segmentor"])
        self.inpainter = CleanInpainter(model_settings["clean_inpainter"])
        self.generator = FluxFillClient(model_settings["grasp_generator"])
        self.extractor = HandProxyExtractor(model_settings["hand_prior_extractor"])
        self.executor = HandExecutor(model_settings["hand_executor"])

    def run(self, frames: list[SensorFrame], instruction: str) -> PipelineResult:
        burst = Burst(frames=frames)
        if not is_stable_burst(burst, max_blur_score=self.settings["stability"]["max_blur_score"]):
            return PipelineResult(status="unstable_scene")

        frame = select_best_frame(burst)
        intent = parse_intent(instruction)

        fine_mask = self.segmentor.predict(frame.rgb)
        lollipop_mask, lolli_params = fit_lollipop(fine_mask)
        roi = build_interaction_roi(
            image_shape=frame.rgb.shape,
            lolli_params=lolli_params,
            forward_scale=self.settings["roi"]["forward_scale"],
            side_scale=self.settings["roi"]["side_scale"],
            min_size_px=self.settings["roi"]["min_size_px"],
        )

        prosthesis = ProsthesisMaskState(
            fine_mask=fine_mask,
            lollipop_mask=lollipop_mask,
            lolli_params=lolli_params,
        )
        scene = SceneLite(
            frame=frame,
            prosthesis=prosthesis,
            interaction_roi=roi,
            object_mask=None,
            intent=intent,
        )

        removal_mask = _build_removal_mask(fine_mask, lollipop_mask)
        rgb_clean = self.inpainter.inpaint(frame.rgb, removal_mask)

        prompt = build_generation_prompt(intent)
        candidates = self.generator.generate(
            rgb_clean=rgb_clean,
            prompt=prompt,
            num_samples=self.settings["pipeline"]["num_generation_candidates"],
            roi=(roi.x0, roi.y0, roi.x1, roi.y1),
        )
        ranked = rank_candidates(candidates, rgb_clean, intent)

        if not ranked:
            return PipelineResult(status="no_candidates", scene=scene)

        prior = self.extractor.extract(ranked[0].image, roi, intent)
        action = initialize_action(
            prior=prior,
            intent=intent,
            default_force_level=self.settings["retarget"]["default_force_level"],
            default_closure_speed=self.settings["retarget"]["default_closure_speed"],
            min_aperture=self.settings["retarget"]["min_aperture"],
            max_aperture=self.settings["retarget"]["max_aperture"],
        )
        action = optimize_action(action)

        if action.confidence < self.settings["pipeline"]["candidate_confidence_threshold"]:
            return PipelineResult(
                status="low_confidence_action",
                scene=scene,
                candidates=ranked,
                prior=prior,
                action=action,
            )

        execution = self.executor.execute(action)
        return PipelineResult(
            status="ok",
            scene=scene,
            candidates=ranked,
            prior=prior,
            action=action,
            execution=execution,
        )

