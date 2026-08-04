from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

import cv2

from .config import ImageStorageConfig
from .incident_utils import SNAPSHOT_SUFFIXES


LOGGER = logging.getLogger(__name__)
SAFE_STEM_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
MAX_STEM_LENGTH = 160


class DurableImageWriter:
    """Thread-safe, hot-reconfigurable writer for durable evidence images."""

    def __init__(self, config: ImageStorageConfig) -> None:
        self._lock = threading.Lock()
        self._config = config.model_copy(deep=True)
        self._webp_fallback_logged = False

    def reconfigure(self, config: ImageStorageConfig) -> None:
        with self._lock:
            self._config = config.model_copy(deep=True)

    def configuration(self) -> ImageStorageConfig:
        with self._lock:
            return self._config.model_copy(deep=True)

    def write(self, directory: Path, stem: str, frame: Any) -> Path | None:
        config = self.configuration()
        image_format = config.format
        encoded = self._encode(image_format, frame, config.quality)
        if encoded is None and image_format == "webp":
            image_format = "jpeg"
            encoded = self._encode(image_format, frame, config.quality)
            if encoded is not None:
                with self._lock:
                    if not self._webp_fallback_logged:
                        LOGGER.warning(
                            "WebP evidence encoding is unavailable; falling back to JPEG"
                        )
                        self._webp_fallback_logged = True
        if encoded is None:
            return None
        suffix = ".webp" if image_format == "webp" else ".jpg"
        safe_stem = self._safe_stem(stem)
        temporary_path: Path | None = None
        descriptor: int | None = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            directory_mode = directory.stat().st_mode
            file_mode = 0o600 | (directory_mode & 0o044)
            path = directory / f"{safe_stem}{suffix}"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=directory,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as output:
                descriptor = None
                try:
                    os.fchmod(output.fileno(), file_mode)
                except OSError:
                    # Retain mkstemp's safe 0600 mode if a filesystem does
                    # not permit chmod; evidence availability takes priority.
                    pass
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
            self._sync_directory(directory)
            return path
        except OSError:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            return None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _encode(image_format: str, frame: Any, quality: int) -> bytes | None:
        suffix = ".webp" if image_format == "webp" else ".jpg"
        parameter = (
            cv2.IMWRITE_WEBP_QUALITY
            if image_format == "webp"
            else cv2.IMWRITE_JPEG_QUALITY
        )
        try:
            success, encoded = cv2.imencode(suffix, frame, [parameter, quality])
            if not success or encoded is None:
                return None
            return encoded.tobytes()
        except (cv2.error, TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _safe_stem(stem: str) -> str:
        normalized = SAFE_STEM_PATTERN.sub("-", str(stem or "")).strip("._-")
        return (normalized or "image")[:MAX_STEM_LENGTH]

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(directory, flags)
            os.fsync(descriptor)
        except OSError:
            # File data is already fsynced and atomically visible. Some NFS
            # servers do not support fsync on directory handles.
            pass
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

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
