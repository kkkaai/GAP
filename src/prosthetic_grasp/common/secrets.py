from __future__ import annotations

import os


_DOTENV_LOADED = False


def load_secrets() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    from dotenv import load_dotenv

    load_dotenv()
    _DOTENV_LOADED = True


def get_secret(name: str, required: bool = False) -> str:
    load_secrets()
    value = os.getenv(name, "").strip()
    if required and not value:
        raise RuntimeError(f"Missing required secret: {name}")
    return value
