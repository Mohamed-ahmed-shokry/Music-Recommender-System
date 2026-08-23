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
        # Ensure data is flushed to disk before atomic replacement.
        with temporary_path.open("rb") as handle:
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        temporary_path.replace(target)
        # Sync parent directory to guarantee rename durability on POSIX.
        dir_fd: int | None = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            dir_fd = os.open(target.parent, flags)
        except (OSError, AttributeError, ValueError):
            dir_fd = None  # Windows or unsupported platform
        else:
            try:
                os.fsync(dir_fd)
            finally:
                if dir_fd is not None:
                    os.close(dir_fd)
    finally:
        temporary_path.unlink(missing_ok=True)
