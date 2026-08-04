from __future__ import annotations

import copy
import logging
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from threading import Thread

from .config import CameraConfig, RecordingRetentionConfig


LOGGER = logging.getLogger(__name__)
GIB = 1024**3
TIB = 1024**4
SECONDS_PER_DAY = 24 * 60 * 60
RECENT_RECORDING_PROTECTION_SECONDS = 5 * 60
RETENTION_PLAN_INTERVAL_SECONDS = SECONDS_PER_DAY
RETENTION_CLEANUP_INTERVAL_SECONDS = 15 * 60
RETENTION_RETRY_SECONDS = 10
RETENTION_INITIAL_DELAY_SECONDS = 5


class RecordingRetentionService:
    """Index-driven continuous-recording retention with bounded NFS work."""

    def __init__(
        self,
        storage_dir: Path,
        recordings_dir: Path,
        connection_factory: Callable[[], sqlite3.Connection],
        config: RecordingRetentionConfig,
        protected_paths_provider: Callable[[], set[str]] | None = None,
    ) -> None:
        self.storage_dir = storage_dir.resolve()
        self.recordings_dir = recordings_dir.resolve()
        self.connection_factory = connection_factory
        self.config = config
        self.protected_paths_provider = protected_paths_provider or set
        self._cameras: dict[str, CameraConfig] = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Thread | None = None
        self._state_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._requested_apply = False
        self._cleanup_active = False
        self._force_plan = True
        self._last_plan_monotonic = 0.0
        self._capacity_reclaim_remaining = 0
        self._planned_reclaim_remaining = 0
        self._status: dict[str, Any] = {
            "state": "starting",
            "enabled": config.enabled,
            "automatic_cleanup": config.automatic_cleanup,
            "last_plan_at": None,
            "last_run_at": None,
            "plan_interval_seconds": RETENTION_PLAN_INTERVAL_SECONDS,
            "cleanup_interval_seconds": RETENTION_CLEANUP_INTERVAL_SECONDS,
            "cleanup_retry_seconds": RETENTION_RETRY_SECONDS,
            "error": "",
            "plan": None,
            "last_run": None,
        }

    def start(self, cameras: Sequence[CameraConfig]) -> None:
        with self._state_lock:
            self._cameras = {camera.id: camera for camera in cameras}
            self._force_plan = True
        self._stop.clear()
        if self._thread is not None and self._thread.is_alive():
            self._wake.set()
            return
        self._thread = Thread(
            target=self._loop,
            name="recording-retention",
            daemon=True,
        )
        self._thread.start()

    def reconfigure(
        self,
        config: RecordingRetentionConfig,
        cameras: Sequence[CameraConfig],
    ) -> None:
        """Apply policy changes without interrupting recorders or cameras."""
        with self._state_lock:
            self.config = config
            self._cameras = {camera.id: camera for camera in cameras}
            self._status = {
                **self._status,
                "state": "queued",
                "enabled": config.enabled,
                "automatic_cleanup": config.automatic_cleanup,
                "error": "",
            }
            self._force_plan = True
        self._wake.set()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            LOGGER.error("Recording retention worker did not stop")
        else:
            self._thread = None

    def request_run(self, *, apply: bool = False) -> dict[str, Any]:
        with self._state_lock:
            self._force_plan = True
            if apply:
                self._requested_apply = True
                self._cleanup_active = True
            self._status = {**self._status, "state": "queued", "error": ""}
        self._wake.set()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return _copy_payload(self._status)

    def plan(self, *, now_epoch: float | None = None) -> dict[str, Any]:
        now = time.time() if now_epoch is None else float(now_epoch)
        usage = shutil.disk_usage(self.storage_dir)
        protected_paths = self.protected_paths_provider()
        with self.connection_factory() as connection:
            totals = connection.execute(
                """
                SELECT count(*) AS file_count, coalesce(sum(size_bytes), 0) AS bytes,
                       min(start_epoch) AS oldest, max(end_epoch) AS newest
                FROM recordings
                """
            ).fetchone()
            groups = connection.execute(
                """
                SELECT camera_id, source, count(*) AS file_count,
                       coalesce(sum(size_bytes), 0) AS bytes,
                       min(start_epoch) AS oldest, max(end_epoch) AS newest
                FROM recordings
                GROUP BY camera_id, source
                ORDER BY camera_id, source
                """
            ).fetchall()

            per_camera: list[dict[str, Any]] = []
            expired_count = 0
            expired_bytes = 0
            protected_expired_count = 0
            protected_expired_bytes = 0
            for row in groups:
                camera_id = str(row["camera_id"])
                source = "main" if str(row["source"]) == "main" else "live"
                retention_days = self._retention_days(camera_id, source)
                cutoff = now - retention_days * SECONDS_PER_DAY
                expired_rows = connection.execute(
                    """
                    SELECT path, size_bytes
                    FROM recordings
                    WHERE camera_id = ? AND source = ? AND end_epoch < ?
                    """,
                    (camera_id, source, cutoff),
                ).fetchall()
                eligible_expired_files = 0
                eligible_expired_bytes = 0
                group_protected_expired_files = 0
                group_protected_expired_bytes = 0
                for expired_row in expired_rows:
                    row_bytes = int(expired_row["size_bytes"] or 0)
                    if self._path_is_protected(
                        str(expired_row["path"]), protected_paths
                    ):
                        group_protected_expired_files += 1
                        group_protected_expired_bytes += row_bytes
                    else:
                        eligible_expired_files += 1
                        eligible_expired_bytes += row_bytes
                group_bytes = int(row["bytes"] or 0)
                oldest = _optional_float(row["oldest"])
                newest = _optional_float(row["newest"])
                span = max(1.0, (newest or now) - (oldest or now))
                bytes_per_day = group_bytes / span * SECONDS_PER_DAY
                item = {
                    "camera_id": camera_id,
                    "source": source,
                    "file_count": int(row["file_count"] or 0),
                    "bytes": group_bytes,
                    "bytes_per_day": round(bytes_per_day),
                    "retention_days": retention_days,
                    "projected_bytes": round(bytes_per_day * retention_days),
                    "expired_files": eligible_expired_files,
                    "expired_bytes": eligible_expired_bytes,
                    "protected_expired_files": group_protected_expired_files,
                    "protected_expired_bytes": group_protected_expired_bytes,
                }
                expired_count += item["expired_files"]
                expired_bytes += item["expired_bytes"]
                protected_expired_count += item["protected_expired_files"]
                protected_expired_bytes += item["protected_expired_bytes"]
                per_camera.append(item)

        indexed_bytes = int(totals["bytes"] or 0)
        storage_limit_bytes = round(self.config.storage_limit_tb * TIB)
        quota_reclaim = max(0, indexed_bytes - storage_limit_bytes)
        free_percent = usage.free / usage.total * 100 if usage.total else 0.0
        free_reclaim = 0
        if free_percent < self.config.minimum_free_percent:
            target_free = round(usage.total * self.config.target_free_percent / 100)
            free_reclaim = max(0, target_free - usage.free)
        capacity_reclaim = max(quota_reclaim, free_reclaim)
        planned_reclaim = max(expired_bytes, capacity_reclaim)
        total_span = max(
            1.0,
            (_optional_float(totals["newest"]) or now)
            - (_optional_float(totals["oldest"]) or now),
        )
        bytes_per_day = indexed_bytes / total_span * SECONDS_PER_DAY
        days_to_minimum = None
        minimum_free_bytes = round(usage.total * self.config.minimum_free_percent / 100)
        if bytes_per_day > 0:
            days_to_minimum = max(0.0, (usage.free - minimum_free_bytes) / bytes_per_day)

        reasons: list[str] = []
        if expired_bytes:
            reasons.append("age")
        if quota_reclaim:
            reasons.append("quota")
        if free_reclaim:
            reasons.append("free_space")
        return {
            "generated_at": _iso_time(now),
            "enabled": self.config.enabled,
            "automatic_cleanup": self.config.automatic_cleanup,
            "storage": {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "free_percent": round(free_percent, 1),
                "minimum_free_percent": self.config.minimum_free_percent,
                "target_free_percent": self.config.target_free_percent,
                "emergency_free_percent": self.config.emergency_free_percent,
                "emergency": free_percent < self.config.emergency_free_percent,
            },
            "indexed": {
                "file_count": int(totals["file_count"] or 0),
                "bytes": indexed_bytes,
                "oldest_at": _iso_time(_optional_float(totals["oldest"])),
                "newest_at": _iso_time(_optional_float(totals["newest"])),
                "bytes_per_day": round(bytes_per_day),
                "days_to_minimum_free": round(days_to_minimum, 1) if days_to_minimum is not None else None,
            },
            "policy": {
                "storage_limit_bytes": storage_limit_bytes,
                "main_days": self.config.main_days,
                "live_days": self.config.live_days,
                "protected_recent_seconds": RECENT_RECORDING_PROTECTION_SECONDS,
            },
            "reclaim": {
                "expired_files": expired_count,
                "expired_bytes": expired_bytes,
                "protected_expired_files": protected_expired_count,
                "protected_expired_bytes": protected_expired_bytes,
                "quota_bytes": quota_reclaim,
                "free_space_bytes": free_reclaim,
                "planned_bytes": planned_reclaim,
                "reasons": reasons,
            },
            "per_camera": per_camera,
        }

    def run_once(self, *, apply: bool) -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("recording retention is already running")
        started = time.time()
        try:
            before = self.plan(now_epoch=started)
            capacity_reclaim = max(
                int(before["reclaim"]["quota_bytes"]),
                int(before["reclaim"]["free_space_bytes"]),
            )
            result = self._apply_plan(
                before,
                apply=apply,
                now_epoch=started,
                capacity_reclaim_bytes=capacity_reclaim,
                planned_reclaim_bytes=int(before["reclaim"]["planned_bytes"]),
            )
            return {"plan": before, "result": result}
        finally:
            self._run_lock.release()

    def _apply_plan(
        self,
        plan: Mapping[str, Any],
        *,
        apply: bool,
        now_epoch: float,
        capacity_reclaim_bytes: int,
        planned_reclaim_bytes: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "started_at": _iso_time(now_epoch),
            "finished_at": None,
            "apply": apply,
            "selected_files": 0,
            "deleted_files": 0,
            "missing_files": 0,
            "deleted_bytes": 0,
            "failed_files": 0,
            "cancelled": False,
            "remaining_planned_bytes": max(0, int(planned_reclaim_bytes)),
        }
        if apply:
            candidates = self._candidates(
                plan,
                now_epoch=now_epoch,
                capacity_reclaim_bytes=max(0, int(capacity_reclaim_bytes)),
            )
            result["selected_files"] = len(candidates)
            removed_paths: list[str] = []
            empty_directory_candidates: set[Path] = set()
            for row in candidates:
                if self._stop.is_set():
                    result["cancelled"] = True
                    break
                raw_path = str(row["path"])
                try:
                    path = self._safe_recording_path(raw_path)
                except ValueError:
                    result["failed_files"] += 1
                    LOGGER.error("Retention rejected recording path outside storage: %s", raw_path)
                    continue
                try:
                    path.unlink()
                    result["deleted_files"] += 1
                    result["deleted_bytes"] += int(row["size_bytes"] or 0)
                    empty_directory_candidates.add(path.parent)
                    removed_paths.append(raw_path)
                except FileNotFoundError:
                    result["missing_files"] += 1
                    removed_paths.append(raw_path)
                except OSError as error:
                    result["failed_files"] += 1
                    LOGGER.warning("Retention could not delete %s: %s", path, error)
            if removed_paths:
                with self.connection_factory() as connection:
                    connection.executemany(
                        "DELETE FROM recordings WHERE path = ?",
                        ((path,) for path in removed_paths),
                    )
            self._remove_empty_directories(empty_directory_candidates)
            result["remaining_planned_bytes"] = max(
                0,
                int(planned_reclaim_bytes) - int(result["deleted_bytes"]),
            )
        result["finished_at"] = _iso_time(time.time())
        return result

    def _loop(self) -> None:
        wait_seconds = float(RETENTION_INITIAL_DELAY_SECONDS)
        while not self._stop.is_set():
            self._wake.wait(wait_seconds)
            self._wake.clear()
            if self._stop.is_set():
                return
            with self._state_lock:
                requested_apply = self._requested_apply
                self._requested_apply = False
                cleanup_active = self._cleanup_active
                force_plan = self._force_plan
                self._force_plan = False
                cached_plan = copy.deepcopy(self._status.get("plan"))
            apply = requested_apply or cleanup_active or (
                self.config.enabled and self.config.automatic_cleanup
            )
            sampled_monotonic = time.monotonic()
            plan_due = bool(
                force_plan
                or not isinstance(cached_plan, dict)
                or sampled_monotonic - self._last_plan_monotonic
                >= RETENTION_PLAN_INTERVAL_SECONDS
            )
            if not plan_due and self._storage_below_minimum():
                plan_due = True
            if not plan_due and not apply:
                with self._state_lock:
                    self._status = {**self._status, "state": "idle", "error": ""}
                wait_seconds = RETENTION_CLEANUP_INTERVAL_SECONDS
                continue
            with self._state_lock:
                self._status = {
                    **self._status,
                    "state": "planning" if plan_due else "cleaning",
                    "error": "",
                }
            try:
                if plan_due:
                    plan = self.plan()
                    self._last_plan_monotonic = sampled_monotonic
                    self._capacity_reclaim_remaining = max(
                        int(plan["reclaim"]["quota_bytes"]),
                        int(plan["reclaim"]["free_space_bytes"]),
                    )
                    self._planned_reclaim_remaining = int(
                        plan["reclaim"]["planned_bytes"]
                    )
                else:
                    plan = cached_plan
                with self._state_lock:
                    self._status = {
                        **self._status,
                        "state": "cleaning" if apply else "idle",
                        "plan": plan,
                        "last_plan_at": (
                            plan["generated_at"]
                            if plan_due
                            else self._status.get("last_plan_at")
                        ),
                    }
                result = self._run_cached_plan(plan, apply=apply)
                if (
                    apply
                    and int(result["selected_files"]) == 0
                    and int(result["remaining_planned_bytes"]) > 0
                ):
                    # The local recording index can change between the daily plan
                    # and a cleanup pass. Refresh immediately rather than leaving
                    # already-pruned rows advertised as reclaimable for a day.
                    plan = self.plan()
                    self._last_plan_monotonic = time.monotonic()
                    self._capacity_reclaim_remaining = max(
                        int(plan["reclaim"]["quota_bytes"]),
                        int(plan["reclaim"]["free_space_bytes"]),
                    )
                    self._planned_reclaim_remaining = int(
                        plan["reclaim"]["planned_bytes"]
                    )
                    result["remaining_planned_bytes"] = self._planned_reclaim_remaining
                    result["plan_refreshed"] = True
                    result["refreshed_at"] = plan["generated_at"]
                should_continue = bool(
                    apply
                    and int(result["selected_files"]) >= self.config.cleanup_batch_files
                    and (result["deleted_files"] or result["missing_files"])
                )
                with self._state_lock:
                    self._cleanup_active = should_continue
                    self._status = {
                        **self._status,
                        "state": "waiting" if should_continue else "idle",
                        "enabled": self.config.enabled,
                        "automatic_cleanup": self.config.automatic_cleanup,
                        "last_run_at": result["finished_at"] if apply else self._status.get("last_run_at"),
                        "last_plan_at": (
                            plan["generated_at"]
                            if result.get("plan_refreshed")
                            else self._status.get("last_plan_at")
                        ),
                        "error": "",
                        "plan": plan,
                        "last_run": result if apply else self._status.get("last_run"),
                    }
                wait_seconds = (
                    RETENTION_RETRY_SECONDS
                    if should_continue
                    else RETENTION_CLEANUP_INTERVAL_SECONDS
                )
            except Exception as error:
                LOGGER.exception("Recording retention cycle failed")
                with self._state_lock:
                    self._cleanup_active = False
                    self._force_plan = True
                    self._status = {**self._status, "state": "error", "error": str(error)}
                wait_seconds = RETENTION_CLEANUP_INTERVAL_SECONDS

    def _run_cached_plan(
        self,
        plan: Mapping[str, Any],
        *,
        apply: bool,
    ) -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("recording retention is already running")
        try:
            result = self._apply_plan(
                plan,
                apply=apply,
                now_epoch=time.time(),
                capacity_reclaim_bytes=self._capacity_reclaim_remaining,
                planned_reclaim_bytes=self._planned_reclaim_remaining,
            )
            self._capacity_reclaim_remaining = max(
                0,
                self._capacity_reclaim_remaining - int(result["deleted_bytes"]),
            )
            self._planned_reclaim_remaining = int(
                result["remaining_planned_bytes"]
            )
            if isinstance(plan, dict):
                reclaim = dict(plan.get("reclaim") or {})
                reclaim["planned_bytes"] = self._planned_reclaim_remaining
                reclaim["quota_bytes"] = max(
                    0,
                    int(reclaim.get("quota_bytes") or 0)
                    - int(result["deleted_bytes"]),
                )
                reclaim["free_space_bytes"] = max(
                    0,
                    int(reclaim.get("free_space_bytes") or 0)
                    - int(result["deleted_bytes"]),
                )
                plan["reclaim"] = reclaim
            return result
        finally:
            self._run_lock.release()

    def _storage_below_minimum(self) -> bool:
        try:
            usage = shutil.disk_usage(self.storage_dir)
        except OSError:
            return False
        free_percent = usage.free / usage.total * 100 if usage.total else 0.0
        return free_percent < self.config.minimum_free_percent

    @staticmethod
    def _path_is_protected(raw_path: str, protected_paths: set[str]) -> bool:
        return str(Path(raw_path).resolve(strict=False)) in protected_paths

    def _retention_days(self, camera_id: str, source: str) -> int:
        camera = self._cameras.get(camera_id)
        override = None
        if camera is not None:
            override = camera.retention.main_days if source == "main" else camera.retention.live_days
        return int(override or (self.config.main_days if source == "main" else self.config.live_days))

    def _candidates(
        self,
        plan: Mapping[str, Any],
        *,
        now_epoch: float,
        capacity_reclaim_bytes: int | None = None,
    ) -> list[sqlite3.Row]:
        limit = self.config.cleanup_batch_files
        protected_cutoff = now_epoch - RECENT_RECORDING_PROTECTION_SECONDS
        protected_paths = self.protected_paths_provider()
        query_limit = limit + len(protected_paths)
        groups = plan.get("per_camera", [])
        candidates: dict[str, sqlite3.Row] = {}
        with self.connection_factory() as connection:
            expiring_groups = list(groups)
            if expiring_groups:
                predicates: list[str] = []
                parameters: list[object] = [protected_cutoff]
                for group in expiring_groups:
                    predicates.append("(camera_id = ? AND source = ? AND end_epoch < ?)")
                    parameters.extend((
                        group["camera_id"],
                        group["source"],
                        now_epoch - int(group["retention_days"]) * SECONDS_PER_DAY,
                    ))
                parameters.append(query_limit)
                rows = connection.execute(
                    f"""
                    SELECT path, size_bytes, start_epoch FROM recordings
                    WHERE end_epoch < ? AND ({' OR '.join(predicates)})
                    ORDER BY start_epoch ASC LIMIT ?
                    """,
                    parameters,
                ).fetchall()
                candidates.update(
                    (str(row["path"]), row)
                    for row in rows
                    if not self._path_is_protected(str(row["path"]), protected_paths)
                )
            capacity_reclaim = (
                max(
                    int(plan["reclaim"]["quota_bytes"]),
                    int(plan["reclaim"]["free_space_bytes"]),
                )
                if capacity_reclaim_bytes is None
                else max(0, int(capacity_reclaim_bytes))
            )
            selected_bytes = sum(int(row["size_bytes"] or 0) for row in candidates.values())
            if capacity_reclaim > selected_bytes and len(candidates) < limit:
                rows = connection.execute(
                    """
                    SELECT path, size_bytes, start_epoch FROM recordings
                    WHERE end_epoch < ? ORDER BY start_epoch ASC LIMIT ?
                    """,
                    (protected_cutoff, query_limit),
                ).fetchall()
                for row in rows:
                    if self._path_is_protected(str(row["path"]), protected_paths):
                        continue
                    path_key = str(row["path"])
                    if path_key not in candidates:
                        candidates[path_key] = row
                        selected_bytes += int(row["size_bytes"] or 0)
                    if len(candidates) >= limit or selected_bytes >= capacity_reclaim:
                        break
        return sorted(candidates.values(), key=lambda row: float(row["start_epoch"] or 0))[:limit]

    def _safe_recording_path(self, raw_path: str) -> Path:
        path = Path(raw_path).resolve(strict=False)
        path.relative_to(self.recordings_dir)
        if path.suffix.lower() != ".mp4":
            raise ValueError("retention only removes MP4 recordings")
        return path

    def _remove_empty_directories(self, directories: set[Path]) -> None:
        for directory in directories:
            current = directory
            while current != self.recordings_dir:
                try:
                    current.rmdir()
                except OSError:
                    break
                current = current.parent


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _iso_time(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _copy_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_payload(item) for item in value]
    return value
