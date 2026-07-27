from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
import weakref
from collections.abc import Callable
from pathlib import Path


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
        if path.is_file():
            return path
        lock = self._key_lock(f"{safe_namespace}:{digest}")
        with lock:
            if path.is_file():
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
