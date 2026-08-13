"""Role-aware placement across independently managed media filesystems."""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import MediaStorageConfig, MediaStorageLocationConfig, MediaStorageRole

LocationState = Literal["online", "unavailable", "not_mounted", "read_only", "full"]


@dataclass(frozen=True, slots=True)
class MediaLocationStatus:
    id: str
    name: str
    path: Path
    roles: tuple[MediaStorageRole, ...]
    state: LocationState
    total_bytes: int = 0
    free_bytes: int = 0
    usable_bytes: int = 0
    reserve_percent: float = 0.0
    priority: int = 100
    error: str = ""

    @property
    def writable(self) -> bool:
        return self.state == "online"


class MediaStorageRegistry:
    """Resolve media locations and make stable, bounded placement decisions.

    An empty location list intentionally means the historical single
    ``storage_dir``. This makes the abstraction safe to deploy before an
    operator configures a second filesystem.
    """

    LEGACY_LOCATION_ID = "default"
    ROLE_DIRECTORIES: dict[MediaStorageRole, str] = {
        "recordings": "recordings",
        "snapshots": "snapshots",
        "motion_audits": "motion_samples",
        "clips": "event_clips",
        "exports": "exports",
    }

    def __init__(self, storage_dir: Path, config: MediaStorageConfig) -> None:
        self.storage_dir = storage_dir.expanduser().resolve(strict=False)
        self.config = config
        self._lock = threading.RLock()
        self._assignments: dict[tuple[str, str], str] = {}
        configured = list(config.locations)
        if not configured:
            configured = [MediaStorageLocationConfig(
                id=self.LEGACY_LOCATION_ID,
                name="Primary media",
                path=str(self.storage_dir),
                reserve_percent=0,
            )]
        self._locations = {item.id: item for item in configured}

    @property
    def location_ids(self) -> tuple[str, ...]:
        return tuple(self._locations)

    def status(self, location_id: str) -> MediaLocationStatus:
        location = self._locations[location_id]
        path = Path(location.path).expanduser().resolve(strict=False)
        base = dict(
            id=location.id,
            name=location.name or location.id,
            path=path,
            roles=tuple(location.roles),
            reserve_percent=location.reserve_percent,
            priority=location.priority,
        )
        if not location.enabled:
            return MediaLocationStatus(**base, state="unavailable", error="disabled")
        if location.require_mount and not os.path.ismount(path):
            return MediaLocationStatus(**base, state="not_mounted", error="required mount is absent")
        if not path.is_dir():
            return MediaLocationStatus(**base, state="unavailable", error="directory does not exist")
        if not os.access(path, os.W_OK | os.X_OK):
            return MediaLocationStatus(**base, state="read_only", error="directory is not writable")
        try:
            usage = shutil.disk_usage(path)
        except OSError as error:
            return MediaLocationStatus(**base, state="unavailable", error=str(error))
        reserve = round(usage.total * location.reserve_percent / 100.0)
        usable = max(0, usage.free - reserve)
        return MediaLocationStatus(
            **base,
            state="online" if usable > 0 else "full",
            total_bytes=usage.total,
            free_bytes=usage.free,
            usable_bytes=usable,
        )

    def statuses(self) -> list[MediaLocationStatus]:
        return [self.status(location_id) for location_id in self.location_ids]

    def roots_for(self, role: MediaStorageRole, *, writable_only: bool = False) -> list[Path]:
        statuses = [item for item in self.statuses() if role in item.roles]
        if writable_only:
            statuses = [item for item in statuses if item.writable]
        directory = self.ROLE_DIRECTORIES[role]
        return [item.path / directory for item in statuses]

    def choose(self, role: MediaStorageRole, assignment_key: str) -> MediaLocationStatus:
        cache_key = (role, assignment_key)
        with self._lock:
            previous = self._assignments.get(cache_key)
            if previous is not None:
                status = self.status(previous)
                if role in status.roles and status.writable:
                    return status
            candidates = [
                item for item in self.statuses()
                if role in item.roles and item.writable
            ]
            if not candidates:
                raise OSError(f"no writable media location supports {role}")
            if self.config.placement == "priority":
                selected = max(candidates, key=lambda item: (item.priority, item.usable_bytes, item.id))
            else:
                selected = max(
                    candidates,
                    key=lambda item: (item.usable_bytes * item.priority / 100.0, item.priority, item.id),
                )
            self._assignments[cache_key] = selected.id
            return selected

    def directory(
        self,
        role: MediaStorageRole,
        assignment_key: str,
        *relative: str,
        create: bool = True,
    ) -> Path:
        selected = self.choose(role, assignment_key)
        directory = selected.path / self.ROLE_DIRECTORIES[role]
        if relative:
            directory = directory.joinpath(*relative)
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def contains(self, path: Path, role: MediaStorageRole | None = None) -> bool:
        resolved = path.expanduser().resolve(strict=False)
        for status in self.statuses():
            if role is not None and role not in status.roles:
                continue
            root = status.path / self.ROLE_DIRECTORIES[role] if role is not None else status.path
            try:
                resolved.relative_to(root.resolve(strict=False))
            except ValueError:
                continue
            return True
        return False

    def payload(self) -> dict[str, object]:
        return {
            "placement": self.config.placement,
            "locations": [
                {
                    "id": item.id,
                    "name": item.name,
                    "path": str(item.path),
                    "roles": list(item.roles),
                    "state": item.state,
                    "total_bytes": item.total_bytes,
                    "free_bytes": item.free_bytes,
                    "usable_bytes": item.usable_bytes,
                    "reserve_percent": item.reserve_percent,
                    "priority": item.priority,
                    "error": item.error,
                }
                for item in self.statuses()
            ],
        }
