"""Shared utility helpers for the music recommender package."""

import os
import tempfile
from pathlib import Path
from typing import Any

import joblib


def atomic_joblib_dump(value: Any, path: str | Path) -> None:
    """Serialize to a temporary sibling and atomically replace the target."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(value, temporary_path)
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)
