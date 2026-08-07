from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility.
    import tomli as tomllib


def load_settings(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)
