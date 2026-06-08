from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class SegFormerDistillConfig:
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    num_labels: int = 2
    img_size: int = 448
    mask_threshold: int = 127
    local_files_only: bool = False


@dataclass
class SegFormerTrainMetrics:
    loss: float
    pixel_acc: float
    fg_iou: float
    fg_dice: float
    fg_precision: float
    fg_recall: float
    pred_fg_ratio: float
    gt_fg_ratio: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_split_txt(path: str | Path) -> list[str]:
    stamps: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stamp = line.split(",")[-1].strip()
            if stamp:
                stamps.append(stamp)
    return stamps


def parse_multi_arg(value: str | None) -> list[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def broadcast_arg(items: list[str], n: int, name: str) -> list[str | None]:
    if len(items) == 0:
        return [None] * n
    if len(items) == 1:
        return list(items) * n
    if len(items) != n:
        raise ValueError(f"{name} expects 1 or {n} values, got {len(items)}.")
    return list(items)


def resolve_maybe_relative(path: str | Path, root: str | Path | None = None) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute() or root is None:
        return resolved
    return Path(root).expanduser() / resolved


def find_image_path(image_dir: str | Path, stamp: str) -> Path:
    image_dir = Path(image_dir)
    candidate = image_dir / stamp
    if candidate.suffix and candidate.exists():
        return candidate
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stamp}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for stamp {stamp!r} in {image_dir}.")


def image_stem(path: str | Path) -> str:
    return Path(path).stem


def iter_image_files(image_dir: str | Path, recursive: bool = False) -> list[Path]:
    image_dir = Path(image_dir)
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in image_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def load_mask_bool(path: str | Path, threshold: int = 127) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    return mask > threshold


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_overlay(path: str | Path, image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image_rgb.astype(np.float32).copy()
    color = np.array([255, 40, 40], dtype=np.float32)
    rgb[mask] = rgb[mask] * (1.0 - alpha) + color * alpha
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(path)


def build_image_transform(img_size: int, *, is_training: bool, color_jitter: bool = False,
                          brightness: float = 0.35, contrast: float = 0.30,
                          saturation: float = 0.15, hue: float = 0.03,
                          gray_prob: float = 0.05) -> transforms.Compose:
    transform_list: list[Any] = [transforms.Resize((img_size, img_size))]
    if is_training and color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )
        if gray_prob > 0:
            transform_list.append(transforms.RandomGrayscale(p=gray_prob))
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(transform_list)


def build_mask_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
        ]
    )


class MaskDistillDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        img_subdir: str,
        mask_dir: str | Path,
        split_txt: str | Path,
        *,
        img_size: int = 448,
        mask_threshold: int = 127,
        is_training: bool = False,
        color_jitter: bool = False,
        brightness: float = 0.35,
        contrast: float = 0.30,
        saturation: float = 0.15,
        hue: float = 0.03,
        gray_prob: float = 0.05,
    ) -> None:
        self.data_root = Path(data_root)
        self.img_dir = resolve_maybe_relative(img_subdir, self.data_root)
        self.mask_dir = resolve_maybe_relative(mask_dir, self.data_root)
        self.split_txt = resolve_maybe_relative(split_txt, self.data_root)
        self.stamps = parse_split_txt(self.split_txt)
        self.img_size = img_size
        self.mask_threshold = mask_threshold
        self.img_tf = build_image_transform(
            img_size,
            is_training=is_training,
            color_jitter=color_jitter,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            gray_prob=gray_prob,
        )
        self.mask_tf = build_mask_transform(img_size)

    def __len__(self) -> int:
        return len(self.stamps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stamp = self.stamps[idx]
        image_path = find_image_path(self.img_dir, stamp)
        mask_path = self.mask_dir / f"{stamp}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Could not find mask for stamp {stamp!r}: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image_tensor = self.img_tf(image)
        mask = self.mask_tf(mask)
        mask_array = np.array(mask, dtype=np.uint8)
        mask_array = (mask_array > self.mask_threshold).astype(np.int64)
        return image_tensor, torch.from_numpy(mask_array)


def load_segformer_model(
    *,
    model_name: str,
    num_labels: int = 2,
    local_files_only: bool = False,
    device: str | torch.device = "cpu",
) -> Any:
    try:
        from transformers import SegformerForSemanticSegmentation
    except Exception as exc:
        raise RuntimeError("transformers is required for SegFormer. Install transformers first.") from exc

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )
    return model.to(device)


def load_segformer_checkpoint(model: Any, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": state}


def preprocess_image_array(image_rgb: np.ndarray, img_size: int) -> torch.Tensor:
    image = Image.fromarray(image_rgb).convert("RGB")
    tensor = build_image_transform(img_size, is_training=False)(image)
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_segformer_mask(
    model: Any,
    image_rgb: np.ndarray,
    *,
    img_size: int = 448,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    original_h, original_w = image_rgb.shape[:2]
    pixel_values = preprocess_image_array(image_rgb, img_size).to(device)
    logits = model(pixel_values=pixel_values).logits
    logits = F.interpolate(logits, size=(original_h, original_w), mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)[:, 1]
    prob_map = probs[0].detach().cpu().numpy()
    return prob_map >= threshold, prob_map


def metric_from_logits(logits: torch.Tensor, masks: torch.Tensor, loss: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    gt = masks
    eps = 1e-7
    pred_fg = pred == 1
    gt_fg = gt == 1
    tp = torch.logical_and(pred_fg, gt_fg).sum().float()
    fp = torch.logical_and(pred_fg, ~gt_fg).sum().float()
    fn = torch.logical_and(~pred_fg, gt_fg).sum().float()
    tn = torch.logical_and(~pred_fg, ~gt_fg).sum().float()
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "loss": float(loss.item()),
        "pixel_acc": float(pixel_acc.item()),
        "fg_iou": float(iou.item()),
        "fg_dice": float(dice.item()),
        "fg_precision": float(precision.item()),
        "fg_recall": float(recall.item()),
        "pred_fg_ratio": float(pred_fg.float().mean().item()),
        "gt_fg_ratio": float(gt_fg.float().mean().item()),
    }


def average_metric_dicts(metrics: list[dict[str, float]]) -> SegFormerTrainMetrics:
    if not metrics:
        return SegFormerTrainMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    keys = metrics[0].keys()
    averaged = {key: float(np.mean([item[key] for item in metrics])) for key in keys}
    return SegFormerTrainMetrics(**averaged)


def save_training_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_csv_row(path: str | Path, row: dict[str, Any], *, header: list[str]) -> None:
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
