from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from prosthetic_grasp.common.secrets import get_secret
from prosthetic_grasp.common.types import ROIBox


@dataclass
class ImageEditConfig:
    mode: str = "api"
    model_name: str = "stable-diffusion-3.5-large-turbo"
    model_id: str = "black-forest-labs/FLUX.1-Fill-dev"
    prompt: str = ""
    guidance_scale: float = 30.0
    num_inference_steps: int = 50
    pad_ratio: float = 0.25
    preserve_unmasked_pixels: bool = True
    openai_model: str = "gpt-image-1"
    stability_endpoint: str = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"


def crop_from_mask(image: np.ndarray, mask: np.ndarray, pad_ratio: float = 0.25) -> tuple[np.ndarray, np.ndarray, ROIBox]:
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = image.shape[:2]
    bh = y1 - y0 + 1
    bw = x1 - x0 + 1
    pad_y = max(int(bh * pad_ratio), 16)
    pad_x = max(int(bw * pad_ratio), 16)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1], ROIBox(x0=x0, y0=y0, x1=x1, y1=y1)


def paste_crop_back(full_rgb: np.ndarray, crop_rgb: np.ndarray, crop_box: ROIBox) -> np.ndarray:
    out = full_rgb.copy()
    out[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1] = crop_rgb
    return out


def preserve_unmasked_pixels(source_rgb: np.ndarray, edited_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if source_rgb.shape != edited_rgb.shape:
        raise ValueError("Source crop and edited crop must have identical shapes.")
    out = edited_rgb.copy()
    out[~mask] = source_rgb[~mask]
    return out


def _array_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    handle = io.BytesIO()
    Image.fromarray(image_rgb).save(handle, format="PNG")
    return handle.getvalue()


def _binary_mask_to_png_bytes(mask: np.ndarray) -> bytes:
    handle = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(handle, format="PNG")
    return handle.getvalue()


def _openai_mask_to_png_bytes(mask: np.ndarray) -> bytes:
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.where(mask, 0, 255).astype(np.uint8)
    handle = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(handle, format="PNG")
    return handle.getvalue()


def _resize_rgb_and_mask(image_rgb: np.ndarray, mask: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    resized_rgb = np.array(Image.fromarray(image_rgb).resize((width, height), Image.Resampling.LANCZOS).convert("RGB"))
    resized_mask = np.array(
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize((width, height), Image.Resampling.NEAREST)
    ) > 127
    return resized_rgb, resized_mask


def _pick_stability_sdxl_size(image_rgb: np.ndarray) -> tuple[int, int]:
    valid_sizes = [
        (1024, 1024),
        (1152, 896),
        (1216, 832),
        (1344, 768),
        (1536, 640),
        (640, 1536),
        (768, 1344),
        (832, 1216),
        (896, 1152),
    ]
    h, w = image_rgb.shape[:2]
    aspect = w / max(h, 1)
    return min(valid_sizes, key=lambda size: abs((size[0] / size[1]) - aspect))


def _decode_api_image(payload: dict) -> np.ndarray:
    import requests

    data = payload.get("data") or []
    if not data:
        raise RuntimeError(f"Image API returned no data: {payload}")
    item = data[0]
    if "b64_json" in item:
        image_bytes = base64.b64decode(item["b64_json"])
        return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    if "url" in item:
        response = requests.get(item["url"], timeout=120)
        response.raise_for_status()
        return np.array(Image.open(io.BytesIO(response.content)).convert("RGB"))
    raise RuntimeError(f"Unsupported image API response payload: {payload}")


def _run_local_flux_fill(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch
    from diffusers import FluxFillPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxFillPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    result = pipe(
        prompt=config.prompt,
        image=Image.fromarray(image_rgb),
        mask_image=Image.fromarray(mask.astype(np.uint8) * 255),
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        max_sequence_length=512,
    ).images[0]
    return np.array(result.convert("RGB"))


def _run_bfl_flux_fill_api(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import requests

    api_key = get_secret("BFL_API_KEY", required=True)
    submit_url = "https://api.us1.bfl.ai/v1/flux-pro-1.0-fill"
    headers = {"x-key": api_key, "Content-Type": "application/json"}
    payload = {
        "prompt": config.prompt,
        "image": base64.b64encode(_array_to_png_bytes(image_rgb)).decode("utf-8"),
        "mask": base64.b64encode(_binary_mask_to_png_bytes(mask)).decode("utf-8"),
        "guidance": config.guidance_scale,
        "steps": config.num_inference_steps,
        "output_format": "png",
        "safety_tolerance": 2,
    }
    response = requests.post(submit_url, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    job = response.json()
    polling_url = job.get("polling_url")
    if not polling_url:
        raise RuntimeError(f"BFL API response missing polling_url: {job}")

    for _ in range(120):
        status_response = requests.get(polling_url, headers={"x-key": api_key}, timeout=60)
        status_response.raise_for_status()
        status_payload = status_response.json()
        status = str(status_payload.get("status", "")).lower()
        if status in {"ready", "succeeded", "success"}:
            result = status_payload.get("result") or {}
            sample_url = result.get("sample")
            if not sample_url:
                raise RuntimeError(f"BFL ready response missing sample URL: {status_payload}")
            image_response = requests.get(sample_url, timeout=180)
            image_response.raise_for_status()
            return np.array(Image.open(io.BytesIO(image_response.content)).convert("RGB"))
        if status in {"failed", "error"}:
            raise RuntimeError(f"BFL API job failed: {status_payload}")
        time.sleep(2.0)

    raise TimeoutError("Timed out waiting for BFL FLUX Fill API job.")


def _run_openai_image_edit_api(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import requests

    api_key = get_secret("OPENAI_API_KEY", required=True)
    url = "https://api.openai.com/v1/images/edits"
    files = {
        "image": ("image.png", _array_to_png_bytes(image_rgb), "image/png"),
        "mask": ("mask.png", _openai_mask_to_png_bytes(mask), "image/png"),
    }
    data = {
        "model": config.openai_model,
        "prompt": config.prompt,
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        files=files,
        data=data,
        timeout=300,
    )
    response.raise_for_status()
    return _decode_api_image(response.json())


def _run_stability_api(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import requests

    api_key = get_secret("STABILITY_API_KEY", required=True)
    modern_files = {
        "image": ("image.png", _array_to_png_bytes(image_rgb), "image/png"),
        "mask": ("mask.png", _binary_mask_to_png_bytes(mask), "image/png"),
    }
    modern_data = {
        "prompt": config.prompt,
        "output_format": "png",
    }
    modern_response = requests.post(
        config.stability_endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "image/*",
        },
        files=modern_files,
        data=modern_data,
        timeout=300,
    )
    if modern_response.ok:
        return np.array(Image.open(io.BytesIO(modern_response.content)).convert("RGB"))

    engine_id = config.model_id or "stable-diffusion-xl-1024-v1-0"
    legacy_url = f"https://api.stability.ai/v1/generation/{engine_id}/image-to-image/masking"
    legacy_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    legacy_files = {
        "init_image": ("init_image.png", _array_to_png_bytes(image_rgb), "image/png"),
        "mask_image": ("mask_image.png", _binary_mask_to_png_bytes(mask), "image/png"),
    }
    legacy_data = {
        "mask_source": "MASK_IMAGE_WHITE",
        "text_prompts[0][text]": config.prompt,
        "text_prompts[0][weight]": "1",
        "cfg_scale": str(config.guidance_scale),
        "samples": "1",
        "steps": str(config.num_inference_steps),
    }
    legacy_response = requests.post(legacy_url, headers=legacy_headers, files=legacy_files, data=legacy_data, timeout=300)
    legacy_response.raise_for_status()
    payload = legacy_response.json()
    artifacts = payload.get("artifacts") or []
    if not artifacts:
        raise RuntimeError(
            f"Stability API returned no artifacts. Modern error={modern_response.text!r}, legacy payload={payload!r}"
        )
    artifact = artifacts[0]
    image_bytes = base64.b64decode(artifact["base64"])
    return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


def run_masked_image_edit(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    model_name = config.model_name.strip().lower()
    mode = config.mode.strip().lower()

    if mode == "local":
        if model_name != "flux-fill":
            raise NotImplementedError(
                f"Local deployment currently only supports 'flux-fill', got {config.model_name!r}."
            )
        edited = _run_local_flux_fill(config, image_rgb, mask)
    elif mode == "api":
        if model_name == "flux-fill":
            edited = _run_bfl_flux_fill_api(config, image_rgb, mask)
        elif model_name in {"gpt", "openai", "gpt-image-1"}:
            edited = _run_openai_image_edit_api(config, image_rgb, mask)
        elif model_name == "stable-diffusion-3.5-large-turbo":
            edited = _run_stability_api(config, image_rgb, mask)
        else:
            raise ValueError(f"Unsupported API image-edit model_name: {config.model_name}")
    else:
        raise ValueError(f"Unsupported image-edit mode: {config.mode}")

    if config.preserve_unmasked_pixels:
        edited = preserve_unmasked_pixels(image_rgb, edited, mask)
    return edited
