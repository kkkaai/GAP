from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_settings(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)

