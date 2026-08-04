from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import cv2

from .config import ImageStorageConfig
from .incident_utils import SNAPSHOT_SUFFIXES


class DurableImageWriter:
    """Thread-safe, hot-reconfigurable writer for durable evidence images."""

    def __init__(self, config: ImageStorageConfig) -> None:
        self._lock = threading.Lock()
        self._config = config.model_copy(deep=True)

    def reconfigure(self, config: ImageStorageConfig) -> None:
        with self._lock:
            self._config = config.model_copy(deep=True)

    def configuration(self) -> ImageStorageConfig:
        with self._lock:
            return self._config.model_copy(deep=True)

    def write(self, directory: Path, stem: str, frame: Any) -> Path | None:
        config = self.configuration()
        suffix = ".webp" if config.format == "webp" else ".jpg"
        parameter = cv2.IMWRITE_WEBP_QUALITY if config.format == "webp" else cv2.IMWRITE_JPEG_QUALITY
        try:
            success, encoded = cv2.imencode(suffix, frame, [parameter, config.quality])
        except cv2.error:
            return None
        if not success:
            return None
        temporary_path: Path | None = None
        descriptor: int | None = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{stem}{suffix}"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=directory,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as output:
                descriptor = None
                output.write(encoded.tobytes())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
            return path
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
            return None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def stored_images(directory: Path) -> list[Path]:
        try:
            return [
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in SNAPSHOT_SUFFIXES
            ]
        except OSError:
            return []
