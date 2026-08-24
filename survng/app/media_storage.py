"""Role-aware placement across independently managed media filesystems."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import threading
import time
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
    filesystem_id: str = ""
    error: str = ""

    @property
    def writable(self) -> bool:
        return self.state == "online"


@dataclass(frozen=True, slots=True)
class MediaStorageSnapshot:
    """One coherent health sample for every configured media location."""

    sampled_at: float
    locations: tuple[MediaLocationStatus, ...]
    shared_filesystems: tuple[tuple[str, ...], ...] = ()

    def status(self, location_id: str) -> MediaLocationStatus:
        for location in self.locations:
            if location.id == location_id:
                return location
        raise KeyError(location_id)


class MediaStorageRegistry:
    """Resolve media locations and make stable, bounded placement decisions.

    At least one configured location is required. Single-disk and multi-disk
    installs share the same placement and health path.
    """

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
            raise ValueError("media_storage.locations requires at least one location")
        self._locations = {item.id: item for item in configured}
        self._roots = {
            item.id: Path(item.path).expanduser().resolve(strict=False)
            for item in configured
        }

    @property
    def location_ids(self) -> tuple[str, ...]:
        return tuple(self._locations)

    def status(
        self,
        location_id: str,
        *,
        snapshot: MediaStorageSnapshot | None = None,
    ) -> MediaLocationStatus:
        if snapshot is not None:
            return snapshot.status(location_id)
        location = self._locations[location_id]
        path = self._roots[location_id]
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
        if location.require_mount and not self._has_mounted_ancestor(path):
            return MediaLocationStatus(
                **base,
                state="not_mounted",
                error="required mount is absent",
            )
        if not path.is_dir():
            return MediaLocationStatus(**base, state="unavailable", error="directory does not exist")
        if not os.access(path, os.W_OK | os.X_OK):
            return MediaLocationStatus(**base, state="read_only", error="directory is not writable")
        try:
            filesystem_id = f"device:{path.stat().st_dev}"
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
            filesystem_id=filesystem_id,
        )

    @staticmethod
    def _has_mounted_ancestor(path: Path) -> bool:
        """Return whether *path* resides below a non-root mountpoint.

        Media roots are normally directories such as ``/mnt/media/SurvNG``
        beneath the actual NFS or block-device mount. Requiring that exact
        directory to be a mountpoint rejects that safe and common layout.
        Walking parents also preserves the fail-closed behavior: if the
        external mount disappears, only the filesystem root remains and the
        configured location is rejected rather than written through.
        """
        current = path.resolve(strict=False)
        filesystem_root = Path(current.anchor)
        while current != filesystem_root:
            if os.path.ismount(current):
                return True
            current = current.parent
        return False

    def health_snapshot(self) -> MediaStorageSnapshot:
        """Sample each location once and report roots sharing one filesystem."""
        locations = tuple(self.status(location_id) for location_id in self.location_ids)
        by_filesystem: dict[str, list[str]] = {}
        for location in locations:
            if location.filesystem_id:
                by_filesystem.setdefault(location.filesystem_id, []).append(location.id)
        shared = tuple(
            tuple(sorted(location_ids))
            for _filesystem_id, location_ids in sorted(by_filesystem.items())
            if len(location_ids) > 1
        )
        return MediaStorageSnapshot(
            sampled_at=time.time(),
            locations=locations,
            shared_filesystems=shared,
        )

    def statuses(
        self,
        *,
        snapshot: MediaStorageSnapshot | None = None,
    ) -> list[MediaLocationStatus]:
        sampled = snapshot or self.health_snapshot()
        return list(sampled.locations)

    def configured_roots_for(self, role: MediaStorageRole) -> list[Path]:
        """Return configured role roots without probing their filesystems."""
        directory = self.ROLE_DIRECTORIES[role]
        return [
            self._roots[location_id] / directory
            for location_id, location in self._locations.items()
            if role in location.roles
        ]

    def roots_for(
        self,
        role: MediaStorageRole,
        *,
        writable_only: bool = False,
        snapshot: MediaStorageSnapshot | None = None,
    ) -> list[Path]:
        if not writable_only:
            return self.configured_roots_for(role)
        statuses = [
            item
            for item in self.statuses(snapshot=snapshot)
            if role in item.roles
        ]
        statuses = [item for item in statuses if item.writable]
        directory = self.ROLE_DIRECTORIES[role]
        return [item.path / directory for item in statuses]

    def choose(self, role: MediaStorageRole, assignment_key: str) -> MediaLocationStatus:
        cache_key = (role, assignment_key)
        with self._lock:
            snapshot = self.health_snapshot()
            previous = self._assignments.get(cache_key)
            if previous is not None:
                status = self.status(previous, snapshot=snapshot)
                if role in status.roles and status.writable:
                    return status
            candidates = [
                item for item in self.statuses(snapshot=snapshot)
                if role in item.roles and item.writable
            ]
            if not candidates:
                details = "; ".join(
                    f"{item.name}: {item.error or item.state}"
                    for item in self.statuses(snapshot=snapshot)
                    if role in item.roles
                )
                suffix = f" ({details})" if details else ""
                raise OSError(f"no writable media location supports {role}{suffix}")
            if self.config.placement == "priority":
                selected = max(candidates, key=lambda item: (item.priority, item.usable_bytes, item.id))
            else:
                def balanced_score(item: MediaLocationStatus) -> tuple[float, int, str]:
                    digest = hashlib.sha256(
                        f"{role}:{assignment_key}:{item.id}".encode()
                    ).digest()
                    unit = (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 1)
                    weight = max(
                        1.0,
                        item.usable_bytes * max(1, item.priority) / 100.0,
                    )
                    return (weight / -math.log(unit), item.priority, item.id)

                selected = max(
                    candidates,
                    key=balanced_score,
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
        for location_id, location in self._locations.items():
            if role is not None and role not in location.roles:
                continue
            root = self._roots[location_id]
            if role is not None:
                root /= self.ROLE_DIRECTORIES[role]
            try:
                resolved.relative_to(root.resolve(strict=False))
            except ValueError:
                continue
            return True
        return False

    def location_id_for(self, path: Path, role: MediaStorageRole | None = None) -> str | None:
        resolved = path.expanduser().resolve(strict=False)
        for location_id, location in self._locations.items():
            if role is not None and role not in location.roles:
                continue
            root = self._roots[location_id]
            if role is not None:
                root /= self.ROLE_DIRECTORIES[role]
            try:
                resolved.relative_to(root.resolve(strict=False))
            except ValueError:
                continue
            return location_id
        return None

    def payload(
        self,
        *,
        snapshot: MediaStorageSnapshot | None = None,
    ) -> dict[str, object]:
        sampled = snapshot or self.health_snapshot()
        return {
            "placement": self.config.placement,
            "sampled_at": sampled.sampled_at,
            "shared_filesystems": [list(group) for group in sampled.shared_filesystems],
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
                    "filesystem_id": item.filesystem_id,
                    "error": item.error,
                }
                for item in sampled.locations
            ],
        }
