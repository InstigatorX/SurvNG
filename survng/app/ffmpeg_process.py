"""Helpers for launching identifiable external FFmpeg processes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def named_ffmpeg_executable(
    path: str,
    name: str,
    *,
    runtime_dir: Path = Path("/run/survng"),
) -> str:
    """Return a stable process-name alias without breaking PATH resolution."""
    resolved = shutil.which(path)
    if resolved is None:
        return path
    target = os.path.realpath(resolved)
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        link = runtime_dir / name
        if not link.is_symlink() or os.path.realpath(link) != target:
            link.unlink(missing_ok=True)
            link.symlink_to(target)
        return str(link)
    except OSError:
        return path
