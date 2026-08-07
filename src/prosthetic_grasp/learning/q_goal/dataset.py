from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .normalization import JointNormalizer


_LOLLIPOP_KEYS = ("center_x", "center_y", "theta", "palm_radius", "arm_length", "confidence")


@dataclass
class QGoalDatasetConfig:
    jsonl_path: Path
    image_size: int = 224
    mask_size: int = 224
    require_rgb: bool = True
    use_lollipop_mask: bool = True
    use_lollipop_params: bool = True
    use_q_current: bool = True
    task_vocab: dict[str, int] = field(default_factory=dict)
    task_embedding_dim: int = 0
    image_embedding_dim: int = 0
    quality_min: float | None = None
    normalizer: JointNormalizer | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_task_vocab(samples: Iterable[dict[str, Any]], *, min_count: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        task = task_text(sample)
        counts[task] = counts.get(task, 0) + 1
    vocab = {"<unk>": 0}
    for task, count in sorted(counts.items()):
        if count >= min_count and task not in vocab:
            vocab[task] = len(vocab)
    return vocab


def infer_joint_bounds(samples: list[dict[str, Any]], q_dim: int) -> JointNormalizer:
    for sample in samples:
        q_min = sample.get("q_min")
        q_max = sample.get("q_max")
        if q_min is not None and q_max is not None:
            return JointNormalizer.from_bounds(q_min, q_max)
    return JointNormalizer.identity(q_dim)


def task_text(sample: dict[str, Any]) -> str:
    for key in ("task", "task_text", "instruction", "daily_task", "grasp_type", "task_id"):
        value = sample.get(key)
        if value:
            return str(value)
    teacher = sample.get("teacher")
    if isinstance(teacher, dict) and teacher.get("grasp_type"):
        return str(teacher["grasp_type"])
    return "<unk>"


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _load_rgb(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_mask(path: Path | None, size: int) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros(1, size, size, dtype=torch.float32)
    image = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).contiguous()


def _lollipop_params(sample: dict[str, Any]) -> torch.Tensor:
    value = sample.get("lollipop") or sample.get("lollipop_params") or sample.get("lolli_params")
    if isinstance(value, dict):
        params = [float(value.get(key, 0.0)) for key in _LOLLIPOP_KEYS]
    elif isinstance(value, (list, tuple)):
        params = [float(v) for v in value[: len(_LOLLIPOP_KEYS)]]
        params.extend([0.0] * (len(_LOLLIPOP_KEYS) - len(params)))
    else:
        params = [0.0] * len(_LOLLIPOP_KEYS)
    return torch.tensor(params, dtype=torch.float32)


class QGoalDataset(Dataset[dict[str, torch.Tensor | str | dict[str, Any]]]):
    """JSONL dataset for q_goal conditional flow matching.

    Expected sample keys are intentionally flexible. The canonical keys are:
    rgb_path, lollipop_mask_path, task, q_current, q_goal, q_min, q_max.
    Precomputed image/task embeddings can be supplied as image_embedding and
    task_embedding when the corresponding encoder mode is set to "precomputed".
    """

    def __init__(self, config: QGoalDatasetConfig) -> None:
        self.config = config
        self.path = Path(config.jsonl_path)
        self.base_dir = self.path.parent
        samples = read_jsonl(self.path)
        if config.quality_min is not None:
            samples = [sample for sample in samples if self._quality(sample) >= float(config.quality_min)]
        if not samples:
            raise ValueError(f"No q_goal samples found in {self.path}.")
        self.samples = samples
        first_q = torch.as_tensor(samples[0]["q_goal"], dtype=torch.float32)
        self.q_dim = int(first_q.numel())
        self.normalizer = config.normalizer or infer_joint_bounds(samples, self.q_dim)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | dict[str, Any]]:
        sample = self.samples[index]
        q_goal_raw = torch.as_tensor(sample["q_goal"], dtype=torch.float32)
        if q_goal_raw.numel() != self.q_dim:
            raise ValueError(f"Sample {index} q_goal has dim {q_goal_raw.numel()}, expected {self.q_dim}.")

        q_current = torch.as_tensor(sample.get("q_current", [0.0] * self.q_dim), dtype=torch.float32)
        if q_current.numel() != self.q_dim:
            q_current = torch.zeros(self.q_dim, dtype=torch.float32)

        item: dict[str, torch.Tensor | str | dict[str, Any]] = {
            "q_goal_raw": q_goal_raw,
            "q_goal": self.normalizer.normalize(q_goal_raw),
            "q_current_raw": q_current,
            "q_current": self.normalizer.normalize(q_current),
            "lollipop_params": _lollipop_params(sample),
            "task_text": task_text(sample),
            "metadata": json.dumps(sample.get("metadata", {}), ensure_ascii=False),
        }

        task_id = self.config.task_vocab.get(str(item["task_text"]), 0) if self.config.task_vocab else 0
        item["task_id"] = torch.tensor(task_id, dtype=torch.long)

        if self.config.task_embedding_dim > 0:
            emb = sample.get("task_embedding") or sample.get("text_embedding")
            item["task_embedding"] = _fixed_vector(emb, self.config.task_embedding_dim)

        if self.config.image_embedding_dim > 0:
            emb = sample.get("image_embedding") or sample.get("rgb_embedding")
            item["image_embedding"] = _fixed_vector(emb, self.config.image_embedding_dim)

        rgb_path = _resolve_path(sample.get("rgb_path") or sample.get("image_path"), self.base_dir)
        if rgb_path is not None and rgb_path.exists():
            item["rgb"] = _load_rgb(rgb_path, self.config.image_size)
        elif self.config.require_rgb:
            raise FileNotFoundError(f"Sample {index} has no readable rgb_path: {rgb_path}")
        else:
            item["rgb"] = torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32)

        mask_path = _resolve_path(
            sample.get("lollipop_mask_path") or sample.get("lolli_mask_path") or sample.get("mask_path"),
            self.base_dir,
        )
        item["lollipop_mask"] = _load_mask(mask_path, self.config.mask_size)
        return item

    @staticmethod
    def _quality(sample: dict[str, Any]) -> float:
        for key in ("quality", "quality_score"):
            if key in sample:
                return float(sample[key])
        teacher = sample.get("teacher")
        if isinstance(teacher, dict):
            for key in ("quality", "quality_score"):
                if key in teacher:
                    return float(teacher[key])
        return 1.0


def _fixed_vector(value: Any, dim: int) -> torch.Tensor:
    out = torch.zeros(dim, dtype=torch.float32)
    if value is None:
        return out
    src = torch.as_tensor(value, dtype=torch.float32).flatten()
    n = min(dim, int(src.numel()))
    if n > 0:
        out[:n] = src[:n]
    return out
