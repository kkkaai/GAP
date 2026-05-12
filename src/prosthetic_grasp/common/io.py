from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def load_depth(path: str | Path) -> np.ndarray:
    image = Image.open(path)
    return np.array(image)


def save_image(path: str | Path, array: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    save_image(path, (mask.astype(np.uint8) * 255))


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

