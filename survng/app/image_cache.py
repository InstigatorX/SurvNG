from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse, Response


IMMUTABLE_IMAGE = "private, max-age=31536000, immutable"


class LocalImageCache:
    """Thread-safe, bounded cache for derived JPEG interface images."""

    def __init__(self, root: Path, max_entries: int = 5000) -> None:
        self.root = root
        self.max_entries = max(100, int(max_entries))
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
        self._locks_guard = threading.Lock()
        self._maintenance_lock = threading.Lock()
        self._last_maintenance: dict[Path, float] = {}

    def get_or_create(self, namespace: str, identity: str, builder: Callable[[], bytes]) -> Path:
        safe_namespace = "".join(character for character in namespace if character.isalnum() or character in "-_")
        if not safe_namespace:
            raise ValueError("image cache namespace is required")
        digest = hashlib.sha256(identity.encode("utf-8", errors="replace")).hexdigest()
        directory = self.root / safe_namespace
        path = directory / f"{digest}.jpg"
        if self._is_usable(path):
            return path
        lock = self._key_lock(f"{safe_namespace}:{digest}")
        with lock:
            if self._is_usable(path):
                return path
            payload = builder()
            if not payload:
                raise ValueError("derived image is empty")
            directory.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{digest}-", suffix=".tmp", dir=directory)
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(payload)
                os.replace(temporary_path, path)
            finally:
                temporary_path.unlink(missing_ok=True)
        self._maintain(directory)
        return path

    @staticmethod
    def _is_usable(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def _key_lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _maintain(self, directory: Path) -> None:
        now = time.monotonic()
        with self._maintenance_lock:
            if now - self._last_maintenance.get(directory, 0.0) < 600:
                return
            self._last_maintenance[directory] = now
        try:
            files = sorted(
                (path for path in directory.glob("*.jpg") if path.is_file()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        for path in files[self.max_entries:]:
            try:
                path.unlink()
            except OSError:
                continue


def image_version(path: Path) -> str:
    """A short token naming one version of an image file.

    It changes whenever the file is replaced or rewritten, so a URL that
    carries it can be kept by the browser without asking again.
    """
    stat = path.stat()
    return hashlib.sha1(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:16]


def revalidated_image_response(
    path: Path, media_type: str, request: Any = None, filename: str | None = None,
    *, version: str = "", source: Path | None = None,
) -> Response:
    """Serve an interface image that the browser must revalidate on every use.

    Evidence can change behind a URL, so the response stays ``no-cache`` unless
    the request pins a file version (see below). The validator lets an
    unchanged image answer with 304 and no body instead of being sent again
    each time a page shows it.
    """
    stat = path.stat()
    etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
    headers = {"Cache-Control": "private, no-cache", "ETag": etag}
    # A request that names the exact version of the source file it wants may
    # keep the answer for good. Any other request, or one whose version is no
    # longer current, is revalidated as before.
    if version and filename is None:
        try:
            if version == image_version(source if source is not None else path):
                headers["Cache-Control"] = IMMUTABLE_IMAGE
        except OSError:
            pass
    presented = request.headers.get("if-none-match", "") if request is not None else ""
    if filename is None and presented:
        candidates = {item.strip().removeprefix("W/") for item in presented.split(",")}
        if etag in candidates or "*" in candidates:
            return Response(status_code=304, headers=headers)
    return FileResponse(path, media_type=media_type, filename=filename, headers=headers)
