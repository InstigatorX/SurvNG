"""Helpers for launching identifiable external FFmpeg processes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .runtime_directory import SERVICE_RUNTIME_DIRECTORY, prepare_private_directory


def named_ffmpeg_executable(
    path: str,
    name: str,
    *,
    runtime_dir: Path = SERVICE_RUNTIME_DIRECTORY,
) -> str:
    """Return a stable process-name alias without breaking PATH resolution."""
    resolved = shutil.which(path)
    if resolved is None:
        return path
    target = os.path.realpath(resolved)
    try:
        prepare_private_directory(runtime_dir, repair_mode=True)
        link = runtime_dir / name
        if not link.is_symlink() or os.path.realpath(link) != target:
            link.unlink(missing_ok=True)
            link.symlink_to(target)
        return str(link)
    except OSError:
        return path
