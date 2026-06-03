from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from prosthetic_grasp.common.secrets import get_secret
from prosthetic_grasp.common.types import ROIBox


FLUX_FILL_MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
STABILITY_SDXL_MODEL_ID = "stable-diffusion-xl-1024-v1-0"
STABILITY_ERASE_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/edit/erase"
STABILITY_INPAINT_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"
STABILITY_LEGACY_INPAINT_MODEL_NAME = "stable-diffusion-3.5-large-turbo"
ZENMUX_BASE_URL = "https://zenmux.ai/api/v1"


@dataclass
class ImageEditConfig:
    mode: str = "local"
    model_name: str = "flux-fill"
    model_id: str = FLUX_FILL_MODEL_ID
    prompt: str = ""
    guidance_scale: float = 30.0
    num_inference_steps: int = 50
    pad_ratio: float = 0.25
    preserve_unmasked_pixels: bool = False
    openai_model: str = "gpt-image-1"
    stability_endpoint: str = ""
    stability_output_format: str = "png"
    zenmux_base_url: str = ZENMUX_BASE_URL

    def __post_init__(self) -> None:
        self.mode = self.mode.strip().lower()
        self.model_name = self.model_name.strip().lower()
        if self.guidance_scale < 0:
            raise ValueError(f"guidance_scale must be non-negative, got {self.guidance_scale}.")
        if self.num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {self.num_inference_steps}.")
        if self.pad_ratio < 0:
            raise ValueError(f"pad_ratio must be non-negative, got {self.pad_ratio}.")
        if self.mode == "local":
            if self.model_name != "flux-fill":
                raise ValueError(f"Local image editing supports only model_name='flux-fill', got {self.model_name!r}.")
            if not self.model_id:
                raise ValueError("Local flux-fill requires a non-empty model_id.")
            return
        if self.mode == "api":
            supported_api_models = {
                "flux-fill",
                "gpt",
                "openai",
                "gpt-image-1",
                "stability-erase",
                "stability-inpaint",
                "zenmux",
                STABILITY_LEGACY_INPAINT_MODEL_NAME,
            }
            if self.model_name not in supported_api_models:
                raise ValueError(f"Unsupported API image-edit model_name: {self.model_name!r}.")
            if self.model_name == STABILITY_LEGACY_INPAINT_MODEL_NAME:
                self.model_name = "stability-inpaint"
            if self.model_name == "stability-inpaint" and self.model_id == FLUX_FILL_MODEL_ID:
                self.model_id = STABILITY_SDXL_MODEL_ID
            if self.stability_output_format not in {"png", "jpeg", "webp"}:
                raise ValueError(
                    f"stability_output_format must be 'png', 'jpeg', or 'webp', got {self.stability_output_format!r}."
                )
            if self.model_name == "zenmux" and not self.model_id:
                raise ValueError("ZenMux image editing requires model_id, for example 'openai/gpt-image-2'.")
            return
        raise ValueError(f"Unsupported image-edit mode: {self.mode!r}.")


_LOCAL_FLUX_FILL_PIPELINES: dict[tuple[str, str, str], object] = {}


def _validate_rgb_and_mask(image: np.ndarray, mask: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}.")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask with shape (H, W), got {mask.shape}.")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask size mismatch: image={image.shape[:2]}, mask={mask.shape}.")


def crop_from_mask(image: np.ndarray, mask: np.ndarray, pad_ratio: float = 0.25) -> tuple[np.ndarray, np.ndarray, ROIBox]:
    _validate_rgb_and_mask(image, mask)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Mask is empty; cannot compute ROI crop.")
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
    expected_shape = (crop_box.y1 - crop_box.y0, crop_box.x1 - crop_box.x0)
    if crop_rgb.shape[:2] != expected_shape:
        raise ValueError(f"Crop size mismatch: expected {expected_shape}, got {crop_rgb.shape[:2]}.")
    out = full_rgb.copy()
    out[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1] = crop_rgb
    return out


def preserve_unmasked_pixels(source_rgb: np.ndarray, edited_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _validate_rgb_and_mask(source_rgb, mask)
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


def _ensure_rgb_size(image_rgb: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    target_h, target_w = target_shape[:2]
    if image_rgb.ndim == 2:
        image_rgb = np.stack([image_rgb] * 3, axis=-1)
    if image_rgb.ndim != 3:
        raise ValueError(f"Expected edited RGB image with 2 or 3 dimensions, got {image_rgb.shape}.")
    if image_rgb.shape[-1] == 1:
        image_rgb = np.repeat(image_rgb, 3, axis=-1)
    if image_rgb.shape[-1] == 4:
        image_rgb = image_rgb[..., :3]
    if image_rgb.shape[-1] != 3:
        raise ValueError(f"Expected edited RGB image with 3 channels, got {image_rgb.shape}.")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    if image_rgb.shape[:2] != (target_h, target_w):
        image_rgb = np.array(
            Image.fromarray(image_rgb).resize((target_w, target_h), Image.Resampling.LANCZOS).convert("RGB")
        )
    return image_rgb


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


def _decode_stability_error(response) -> str:
    try:
        return str(response.json())
    except ValueError:
        return response.text


def _decode_response_error(response) -> str:
    try:
        return str(response.json())
    except ValueError:
        return response.text


def _raise_for_stability_error(response, operation: str) -> None:
    if response.ok:
        return
    raise RuntimeError(
        f"Stability {operation} API failed with HTTP {response.status_code}: {_decode_stability_error(response)}"
    )


def _get_stability_api_key() -> str:
    api_key = get_secret("STABILITY_API_KEY", required=True)
    if not api_key.startswith("sk-"):
        raise RuntimeError("STABILITY_API_KEY must be a Stability API key beginning with 'sk-'.")
    return api_key


def _run_local_flux_fill(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch
    from diffusers import FluxFillPipeline

    if not config.model_id:
        raise ValueError(f"Local flux-fill requires config.model_id, for example {FLUX_FILL_MODEL_ID!r}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    cache_key = (config.model_id, str(torch_dtype), device)
    pipe = _LOCAL_FLUX_FILL_PIPELINES.get(cache_key)
    if pipe is None:
        pipe = FluxFillPipeline.from_pretrained(config.model_id, torch_dtype=torch_dtype)
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        _LOCAL_FLUX_FILL_PIPELINES[cache_key] = pipe

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


def _run_zenmux_image_edit_api(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import requests

    if "/" in config.model_id and not config.model_id.startswith("openai/"):
        return _run_zenmux_vertex_image_edit_api(config, image_rgb, mask)

    api_key = get_secret("ZENMUX_API_KEY", required=True)
    base_url = get_secret("ZENMUX_BASE_URL") or config.zenmux_base_url or ZENMUX_BASE_URL
    url = base_url.rstrip("/") + "/images/edits"
    data = {
        "model": config.model_id,
        "prompt": config.prompt,
        "n": "1",
        "output_format": config.stability_output_format,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    def post_with_image_field(field_name: str):
        files = {
            field_name: ("image.png", _array_to_png_bytes(image_rgb), "image/png"),
            "mask": ("mask.png", _openai_mask_to_png_bytes(mask), "image/png"),
        }
        return requests.post(url, headers=headers, files=files, data=data, timeout=600)

    response = post_with_image_field("image")
    if not response.ok and response.status_code in {400, 422}:
        response = post_with_image_field("image[]")
    if not response.ok:
        raise RuntimeError(
            f"ZenMux image edit API failed for model {config.model_id!r} with HTTP "
            f"{response.status_code}: {_decode_response_error(response)}"
        )
    return _decode_api_image(response.json())


def _decode_vertex_image(payload: dict) -> np.ndarray:
    predictions = payload.get("predictions") or []
    if not predictions:
        raise RuntimeError(f"Vertex image API returned no predictions: {payload}")
    item = predictions[0]
    if "bytesBase64Encoded" in item:
        image_bytes = base64.b64decode(item["bytesBase64Encoded"])
        return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    if "image" in item and isinstance(item["image"], dict) and "bytesBase64Encoded" in item["image"]:
        image_bytes = base64.b64decode(item["image"]["bytesBase64Encoded"])
        return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    if "gcsUri" in item:
        raise RuntimeError(f"Vertex image API returned gcsUri instead of image bytes: {item['gcsUri']}")
    raise RuntimeError(f"Unsupported Vertex image API response payload: {payload}")


def _run_zenmux_vertex_image_edit_api(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import requests

    api_key = get_secret("ZENMUX_API_KEY", required=True)
    provider, model = config.model_id.split("/", 1)
    url = f"https://zenmux.ai/api/vertex-ai/v1/publishers/{provider}/models/{model}:predict"
    raw_image_b64 = base64.b64encode(_array_to_png_bytes(image_rgb)).decode("utf-8")
    mask_b64 = base64.b64encode(_openai_mask_to_png_bytes(mask)).decode("utf-8")
    payload = {
        "instances": [
            {
                "prompt": config.prompt,
                "referenceImages": [
                    {
                        "referenceType": "REFERENCE_TYPE_RAW",
                        "referenceId": 1,
                        "referenceImage": {
                            "bytesBase64Encoded": raw_image_b64,
                            "mimeType": "image/png",
                        },
                    },
                    {
                        "referenceType": "REFERENCE_TYPE_MASK",
                        "referenceId": 2,
                        "referenceImage": {
                            "bytesBase64Encoded": mask_b64,
                            "mimeType": "image/png",
                        },
                        "maskImageConfig": {
                            "maskMode": "MASK_MODE_USER_PROVIDED",
                            "dilation": 0,
                        },
                    },
                ],
            }
        ],
        "parameters": {
            "sampleCount": 1,
            "outputOptions": {
                "mimeType": "image/png",
            },
            "addWatermark": False,
        },
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=600,
    )
    if not response.ok:
        raise RuntimeError(
            f"ZenMux Vertex image edit API failed for model {config.model_id!r} with HTTP "
            f"{response.status_code}: {_decode_response_error(response)}"
        )
    return _decode_vertex_image(response.json())


def _stability_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
    }


def _run_stability_erase_api(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import requests

    api_key = _get_stability_api_key()
    endpoint = config.stability_endpoint or STABILITY_ERASE_ENDPOINT
    files = {
        "image": ("image.png", _array_to_png_bytes(image_rgb), "image/png"),
        "mask": ("mask.png", _binary_mask_to_png_bytes(mask), "image/png"),
    }
    data = {
        "output_format": config.stability_output_format,
    }
    response = requests.post(
        endpoint,
        headers=_stability_headers(api_key),
        files=files,
        data=data,
        timeout=300,
    )
    _raise_for_stability_error(response, "erase")
    return np.array(Image.open(io.BytesIO(response.content)).convert("RGB"))


def _run_stability_inpaint_api(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import requests

    api_key = _get_stability_api_key()
    endpoint = config.stability_endpoint or STABILITY_INPAINT_ENDPOINT
    files = {
        "image": ("image.png", _array_to_png_bytes(image_rgb), "image/png"),
        "mask": ("mask.png", _binary_mask_to_png_bytes(mask), "image/png"),
    }
    data = {
        "prompt": config.prompt,
        "output_format": config.stability_output_format,
    }
    response = requests.post(
        endpoint,
        headers=_stability_headers(api_key),
        files=files,
        data=data,
        timeout=300,
    )
    _raise_for_stability_error(response, "inpaint")
    return np.array(Image.open(io.BytesIO(response.content)).convert("RGB"))


def run_masked_image_edit(config: ImageEditConfig, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _validate_rgb_and_mask(image_rgb, mask)
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
        elif model_name == "zenmux":
            edited = _run_zenmux_image_edit_api(config, image_rgb, mask)
        elif model_name == "stability-erase":
            edited = _run_stability_erase_api(config, image_rgb, mask)
        elif model_name in {"stability-inpaint", STABILITY_LEGACY_INPAINT_MODEL_NAME}:
            edited = _run_stability_inpaint_api(config, image_rgb, mask)
        else:
            raise ValueError(f"Unsupported API image-edit model_name: {config.model_name}")
    else:
        raise ValueError(f"Unsupported image-edit mode: {config.mode}")

    edited = _ensure_rgb_size(edited, image_rgb.shape)
    if config.preserve_unmasked_pixels:
        edited = preserve_unmasked_pixels(image_rgb, edited, mask)
    return edited
