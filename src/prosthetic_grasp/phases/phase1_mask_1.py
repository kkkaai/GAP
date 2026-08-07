from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from prosthetic_grasp.common.io import load_image
from prosthetic_grasp.common.types import Phase1MaskResult

@dataclass
class Phase1MaskConfig:
    text_prompt: str = "hand. coffee cup."
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    mode: str = "seggpt"
    precomputed_mask_path: str | None = None
    model_id: str = "BAAI/seggpt-vit-large"
    support_image_path: str | None = None
    support_mask_path: str | None = None
    support_image_paths: list[str] | None = None
    support_mask_paths: list[str] | None = None
    threshold: float = 0.5
    device: str = "auto"

    def __post_init__(self) -> None:
        self.mode = self.mode.strip().lower()
        if not 0.0 <= self.box_threshold <= 1.0:
            raise ValueError(f"box_threshold must be in [0, 1], got {self.box_threshold}.")
        if not 0.0 <= self.text_threshold <= 1.0:
            raise ValueError(f"text_threshold must be in [0, 1], got {self.text_threshold}.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}.")

class Phase1Mask:
    """Phase 1 mask extraction.

    `precomputed` keeps the notebook-exported mask path available for quick
    debugging. `seggpt` adds a few-shot backend for custom prosthetic hand masks.
    """

    def __init__(self, config: Phase1MaskConfig) -> None:
        self.config = config
        self._processor = None
        self._model = None
        self._device = None

    def run(self, image_rgb: np.ndarray) -> Phase1MaskResult:
        if self.config.mode == "precomputed":
            return self._run_precomputed(image_rgb)
        if self.config.mode == "seggpt":
            return self._run_seggpt(image_rgb)
        raise NotImplementedError(
            f"Unsupported phase1_mask mode: {self.config.mode!r}. "
            "Use mode='precomputed' or mode='seggpt'."
        )

    def _run_precomputed(self, image_rgb: np.ndarray) -> Phase1MaskResult:
        if not self.config.precomputed_mask_path:
            raise NotImplementedError(
                "phase1_mask mode='precomputed' requires precomputed_mask_path."
            )

        mask = load_image(self.config.precomputed_mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != image_rgb.shape[:2]:
            raise ValueError(
                f"Precomputed mask shape {mask.shape} does not match input RGB shape {image_rgb.shape[:2]}."
            )
        return Phase1MaskResult(mask=mask > 127, metadata={"source": str(Path(self.config.precomputed_mask_path))})

    def _resolve_support_pairs(self) -> list[tuple[str, str]]:
        image_paths = list(self.config.support_image_paths or [])
        mask_paths = list(self.config.support_mask_paths or [])
        if self.config.support_image_path:
            image_paths.append(self.config.support_image_path)
        if self.config.support_mask_path:
            mask_paths.append(self.config.support_mask_path)
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                "SegGPT support image and mask counts must match: "
                f"{len(image_paths)} image(s), {len(mask_paths)} mask(s)."
            )
        if not image_paths:
            raise ValueError(
                "phase1_mask mode='seggpt' requires support_image_path/support_mask_path "
                "or support_image_paths/support_mask_paths."
            )
        return list(zip(image_paths, mask_paths))

    def _ensure_seggpt(self):
        if self._processor is not None and self._model is not None:
            return self._processor, self._model, self._device

        import torch
        from transformers import SegGptForImageSegmentation, SegGptImageProcessor

        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device

        processor = SegGptImageProcessor.from_pretrained(self.config.model_id)
        model = SegGptForImageSegmentation.from_pretrained(self.config.model_id)
        model.to(device)
        model.eval()

        self._processor = processor
        self._model = model
        self._device = device
        return processor, model, device

    @staticmethod
    def _rgb_to_pil(image_rgb: np.ndarray) -> Image.Image:
        return Image.fromarray(image_rgb).convert("RGB")

    @staticmethod
    def _mask_to_label_map(mask_path: str | Path) -> Image.Image:
        mask = load_image(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        label_map = (mask > 127).astype(np.uint8)
        return Image.fromarray(label_map, mode="L")

    @staticmethod
    def _result_to_bool_mask(result, threshold: float) -> np.ndarray:
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        array = np.asarray(result)
        if array.ndim == 3:
            array = array.squeeze()
        if array.dtype == np.bool_:
            return array
        if np.issubdtype(array.dtype, np.integer):
            return array > 0
        return array > threshold

    def _run_seggpt(self, image_rgb: np.ndarray) -> Phase1MaskResult:
        import torch

        processor, model, device = self._ensure_seggpt()
        support_pairs = self._resolve_support_pairs()

        query_images = [self._rgb_to_pil(image_rgb) for _ in support_pairs]
        prompt_images = [
            self._rgb_to_pil(load_image(image_path))
            for image_path, _ in support_pairs
        ]
        segmentation_maps = [self._mask_to_label_map(mask_path) for _, mask_path in support_pairs]

        inputs = processor(
            images=query_images,
            prompt_images=prompt_images,
            prompt_masks=segmentation_maps,
            num_labels=1,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, feature_ensemble=len(support_pairs) > 1)

        post_processed = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image_rgb.shape[:2]] * len(support_pairs),
            num_labels=1,
        )
        masks = [self._result_to_bool_mask(result, self.config.threshold) for result in post_processed]
        if len(masks) == 1:
            mask = masks[0]
        else:
            mask = np.stack(masks, axis=0).mean(axis=0) >= 0.5
        return Phase1MaskResult(
            mask=mask,
            metadata={
                "source": "seggpt",
                "model_id": self.config.model_id,
                "support_count": len(support_pairs),
                "device": device,
            },
        )

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_image = base_dir / "coffeecup.png"
    support_dir = base_dir / "src_pic"
    support_mask_dir = support_dir / "mask"
    output_mask = base_dir / "output_hand_mask.png"

    def find_support_image(stem: str) -> Path:
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = support_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Support image not found: {stem}.* in {support_dir}")

    support_image_paths: list[str] = []
    support_mask_paths: list[str] = []
    for name in ["1", "2", "3", "4"]:
        support_image_paths.append(str(find_support_image(name)))
        support_mask_paths.append(str(support_mask_dir / f"{name}.png"))

    required_paths = [input_image, *support_image_paths, *support_mask_paths]
    missing_files = [str(path) for path in required_paths if not Path(path).exists()]
    if missing_files:
        print("Error: missing required files. Check src_pic and src_pic/mask:")
        for path in missing_files:
            print(f"  - {path}")
    else:
        img_rgb = np.array(Image.open(input_image).convert("RGB"))
        config = Phase1MaskConfig(
            mode="seggpt",
            support_image_paths=support_image_paths,
            support_mask_paths=support_mask_paths,
            threshold=0.5,
        )

        extractor = Phase1Mask(config)
        print("Running SegGPT mask extraction...")
        result = extractor.run(img_rgb)

        visual_mask = (result.mask * 255).astype(np.uint8)
        Image.fromarray(visual_mask, mode="L").save(output_mask)
        print(f"Saved mask to: {output_mask}")
