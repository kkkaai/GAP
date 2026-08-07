from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from prosthetic_grasp.common.prompts import build_vlm_quality_prompt
from prosthetic_grasp.common.secrets import get_secret
from prosthetic_grasp.phases.phase2_affordance_lollipop import (
    QWEN3_7_PLUS_MODEL_ID,
    ZENMUX_CHAT_BASE_URL,
    extract_json_object,
    rgb_to_data_url,
)


RISK_METRICS = {"flip_risk", "penetration_visual_risk"}


DEFAULT_STAGE_THRESHOLDS: dict[str, dict[str, float]] = {
    "phase2_lollipop": {
        "overall_score": 3.5,
        "mask_affordance_score": 3.0,
        "tail_direction_score": 3.0,
        "task_mask_alignment": 3.0,
        "object_relevance": 3.0,
    },
    "phase4_generation": {
        "overall_score": 3.5,
        "hand_completeness": 3.0,
        "contact_plausibility": 3.0,
        "task_alignment": 3.0,
        "egocentric_consistency": 3.0,
    },
    "phase5_mano": {
        "overall_score": 3.0,
        "mesh_image_alignment": 3.0,
        "hand_orientation": 3.0,
        "finger_pose_consistency": 3.0,
        "flip_risk_max": 3.0,
    },
    "phase6_object_pose": {
        "overall_score": 3.0,
        "object_mask_alignment": 3.0,
        "pose_overlay_alignment": 3.0,
        "scale_consistency": 3.0,
    },
    "phase6_retarget": {
        "overall_score": 3.0,
        "contact_geometry_plausibility": 3.0,
        "retarget_task_alignment": 3.0,
        "penetration_visual_risk_max": 3.0,
    },
}


@dataclass
class VLMQualityScoreConfig:
    model_id: str = QWEN3_7_PLUS_MODEL_ID
    zenmux_base_url: str = ZENMUX_CHAT_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 900
    timeout_seconds: int = 180
    thresholds: dict[str, dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_STAGE_THRESHOLDS))

    def __post_init__(self) -> None:
        self.model_id = self.model_id.strip()
        self.zenmux_base_url = self.zenmux_base_url.strip().rstrip("/")
        if not self.model_id:
            raise ValueError("model_id must be non-empty.")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}.")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}.")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}.")


@dataclass
class VLMQualityScoreResult:
    stage: str
    overall_score: float
    pass_: bool
    scores: dict[str, float]
    failure_tags: list[str]
    reason: str
    model_id: str
    prompt: str
    raw_response: str
    parsed_json: dict[str, Any]
    thresholds: dict[str, float]
    image_labels: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pass"] = payload.pop("pass_")
        return payload


def normalize_score(value: Any, *, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return float(np.clip(score, 0.0, 5.0))


def normalize_scores(raw_scores: Any) -> dict[str, float]:
    if not isinstance(raw_scores, dict):
        return {}
    return {str(key): normalize_score(value) for key, value in raw_scores.items()}


def normalize_failure_tags(raw_tags: Any) -> list[str]:
    if not isinstance(raw_tags, list):
        return []
    tags: list[str] = []
    for item in raw_tags:
        tag = str(item).strip().lower().replace(" ", "_")
        if tag:
            tags.append(tag)
    return tags


def evaluate_thresholds(
    *,
    stage: str,
    overall_score: float,
    scores: dict[str, float],
    thresholds_by_stage: dict[str, dict[str, float]],
) -> tuple[bool, list[str], dict[str, float]]:
    thresholds = thresholds_by_stage.get(stage, {})
    failures: list[str] = []
    overall_min = thresholds.get("overall_score")
    if overall_min is not None and overall_score < overall_min:
        failures.append("overall_score_below_threshold")
    for metric, threshold in thresholds.items():
        if metric == "overall_score":
            continue
        if metric.endswith("_max"):
            score_name = metric[: -len("_max")]
            if scores.get(score_name, 0.0) > threshold:
                failures.append(f"{score_name}_above_threshold")
            continue
        if scores.get(metric, 0.0) < threshold:
            failures.append(f"{metric}_below_threshold")
    return not failures, failures, dict(thresholds)


class PhaseQualityVLM:
    """Score visual artifacts for teacher-data filtering using an OpenAI-compatible VLM."""

    def __init__(self, config: VLMQualityScoreConfig | None = None) -> None:
        self.config = config or VLMQualityScoreConfig()

    def run(
        self,
        *,
        stage: str,
        images: list[tuple[str, np.ndarray]],
        object_name: str | None = None,
        task_instruction: str | None = None,
    ) -> VLMQualityScoreResult:
        import requests

        if not images:
            raise ValueError("At least one image is required for VLM quality scoring.")
        prompt = build_vlm_quality_prompt(
            stage=stage,
            object_name=object_name,
            task_instruction=task_instruction,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for label, image_rgb in images:
            content.append({"type": "text", "text": f"Image label: {label}"})
            content.append({"type": "image_url", "image_url": {"url": rgb_to_data_url(image_rgb)}})

        api_key = get_secret("ZENMUX_API_KEY") or get_secret("OPENAI_API_KEY", required=True)
        base_url = get_secret("ZENMUX_BASE_URL") or self.config.zenmux_base_url
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout_seconds)
        if response.status_code in {400, 422}:
            retry_payload = dict(payload)
            retry_payload.pop("response_format", None)
            response = requests.post(url, headers=headers, json=retry_payload, timeout=self.config.timeout_seconds)
        if not response.ok:
            try:
                error_text = str(response.json())
            except ValueError:
                error_text = response.text
            raise RuntimeError(
                f"ZenMux VLM quality API failed for model {self.config.model_id!r} "
                f"with HTTP {response.status_code}: {error_text}"
            )

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"ZenMux VLM quality API returned no choices: {body}")
        message = choices[0].get("message") or {}
        raw_response = str(message.get("content", "")).strip()
        if not raw_response:
            raise RuntimeError(f"ZenMux VLM quality API returned empty content: {body}")
        parsed = extract_json_object(raw_response)
        scores = normalize_scores(parsed.get("scores"))
        overall_score = normalize_score(parsed.get("overall_score"))
        threshold_pass, threshold_failures, thresholds = evaluate_thresholds(
            stage=stage,
            overall_score=overall_score,
            scores=scores,
            thresholds_by_stage=self.config.thresholds,
        )
        model_pass = bool(parsed.get("pass", False))
        failure_tags = normalize_failure_tags(parsed.get("failure_tags"))
        for failure in threshold_failures:
            if failure not in failure_tags:
                failure_tags.append(failure)
        return VLMQualityScoreResult(
            stage=stage,
            overall_score=overall_score,
            pass_=model_pass and threshold_pass,
            scores=scores,
            failure_tags=failure_tags,
            reason=str(parsed.get("reason", "")).strip(),
            model_id=self.config.model_id,
            prompt=prompt,
            raw_response=raw_response,
            parsed_json=parsed,
            thresholds=thresholds,
            image_labels=[label for label, _ in images],
        )
