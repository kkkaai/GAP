from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from prosthetic_grasp.common.prompts import build_phase4_intention_prompt
from prosthetic_grasp.common.secrets import get_secret
from prosthetic_grasp.common.types import Phase4IntentionResult


QWEN3_VL_PLUS_MODEL_ID = "qwen/qwen3-vl-plus"
QWEN3_VL_FLASH_MODEL_ID = "qwen/qwen3-vl-flash"
ZENMUX_CHAT_BASE_URL = "https://zenmux.ai/api/v1"


@dataclass
class Phase4IntentionConfig:
    model_id: str = QWEN3_VL_PLUS_MODEL_ID
    fast_model_id: str = QWEN3_VL_FLASH_MODEL_ID
    use_fast_model: bool = False
    zenmux_base_url: str = ZENMUX_CHAT_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 400

    def __post_init__(self) -> None:
        self.model_id = self.model_id.strip()
        self.fast_model_id = self.fast_model_id.strip()
        self.zenmux_base_url = self.zenmux_base_url.strip().rstrip("/")
        if not self.model_id:
            raise ValueError("model_id must be non-empty.")
        if not self.fast_model_id:
            raise ValueError("fast_model_id must be non-empty.")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}.")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}.")

    @property
    def selected_model_id(self) -> str:
        return self.fast_model_id if self.use_fast_model else self.model_id


def _rgb_to_data_url(image_rgb: np.ndarray) -> str:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_rgb.shape}.")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    handle = io.BytesIO()
    Image.fromarray(image_rgb).save(handle, format="PNG")
    encoded = base64.b64encode(handle.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from VLM, got {type(parsed).__name__}.")
    return parsed


class Phase4Intention:
    """Generate a Phase4 grasp intention from a scene image and optional speech transcription."""

    def __init__(self, config: Phase4IntentionConfig) -> None:
        self.config = config

    def run(self, image_rgb: np.ndarray, task_instruction: str | None = None) -> Phase4IntentionResult:
        import requests

        prompt = build_phase4_intention_prompt(task_instruction)
        api_key = get_secret("ZENMUX_API_KEY", required=True)
        base_url = get_secret("ZENMUX_BASE_URL") or self.config.zenmux_base_url
        url = base_url.rstrip("/") + "/chat/completions"
        model_id = self.config.selected_model_id
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _rgb_to_data_url(image_rgb)}},
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )
        if not response.ok:
            try:
                error_text = str(response.json())
            except ValueError:
                error_text = response.text
            raise RuntimeError(
                f"ZenMux VLM intention API failed for model {model_id!r} with HTTP "
                f"{response.status_code}: {error_text}"
            )
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"ZenMux VLM intention API returned no choices: {body}")
        message = choices[0].get("message") or {}
        raw_response = str(message.get("content", "")).strip()
        if not raw_response:
            raise RuntimeError(f"ZenMux VLM intention API returned empty content: {body}")
        parsed = _extract_json_object(raw_response)

        target_object = str(parsed.get("target_object", "")).strip()
        daily_task = str(parsed.get("daily_task", "")).strip()
        grasp_type = str(parsed.get("grasp_type", "")).strip()
        phase4_intention = str(parsed.get("phase4_intention", "")).strip()
        if not phase4_intention:
            raise RuntimeError(f"VLM response missing phase4_intention: {raw_response}")

        return Phase4IntentionResult(
            target_object=target_object,
            daily_task=daily_task,
            grasp_type=grasp_type,
            phase4_intention=phase4_intention,
            model_id=model_id,
            prompt=prompt,
            raw_response=raw_response,
            parsed_json=parsed,
        )
