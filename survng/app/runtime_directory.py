"""Owner-only directories shared by the service's runtime helpers."""

from __future__ import annotations

import os
import stat
from pathlib import Path


SERVICE_RUNTIME_DIRECTORY = Path("/run/survng")


def prepare_private_directory(path: Path, *, repair_mode: bool = False) -> None:
    """Create or validate a private directory without following its final symlink.

    Known service directories may repair permissions from older releases. Never
    take ownership of another user's directory or change a symlink's target.
    Use a descriptor so validation and chmod operate on the same directory.
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.geteuid():
            raise PermissionError(f"runtime directory is not owned by this service: {path}")
        mode = stat.S_IMODE(info.st_mode)
        if repair_mode:
            if mode != 0o700:
                os.fchmod(descriptor, 0o700)
        elif mode & 0o077:
            raise PermissionError(f"runtime directory must be mode 0700: {path}")
    finally:
        os.close(descriptor)
