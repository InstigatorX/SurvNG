from __future__ import annotations

import copy
import heapq
import logging
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from threading import Thread

from .config import CameraConfig, RecordingRetentionConfig
from .media_storage import MediaStorageRegistry


LOGGER = logging.getLogger(__name__)
GIB = 1024**3
TIB = 1024**4
SECONDS_PER_DAY = 24 * 60 * 60
RECENT_RECORDING_PROTECTION_SECONDS = 5 * 60
RETENTION_PLAN_INTERVAL_SECONDS = SECONDS_PER_DAY
RETENTION_CLEANUP_INTERVAL_SECONDS = 15 * 60
RETENTION_RETRY_SECONDS = 10
RETENTION_INITIAL_DELAY_SECONDS = 5
RETENTION_BATCH_TIME_BUDGET_SECONDS = 10.0
RETENTION_FAILURE_RETRY_MAX_SECONDS = 15 * 60


class RecordingRetentionService:
    """Index-driven continuous-recording retention with bounded NFS work."""

    def __init__(
        self,
        storage_dir: Path,
        recordings_dir: Path,
        connection_factory: Callable[[], sqlite3.Connection],
        config: RecordingRetentionConfig,
        protected_paths_provider: Callable[[], set[str]] | None = None,
        snapshot_plan_provider: Callable[[float], Mapping[str, Any]] | None = None,
        snapshot_cleanup_provider: Callable[[float, int], Mapping[str, Any]] | None = None,
        media_storage: MediaStorageRegistry | None = None,
        delete_recording_provider: Callable[[Path], bool] | None = None,
    ) -> None:
        self.storage_dir = storage_dir.resolve()
        self.recordings_dir = recordings_dir.resolve()
        self.connection_factory = connection_factory
        self.config = config
        self.protected_paths_provider = protected_paths_provider or set
        self.snapshot_plan_provider = snapshot_plan_provider
        self.snapshot_cleanup_provider = snapshot_cleanup_provider
        self.media_storage = media_storage
        self.delete_recording_provider = delete_recording_provider
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
        self._quota_reclaim_remaining = 0
        self._free_reclaim_remaining: dict[str, int] = {}
        self._planned_reclaim_remaining = 0
        self._cleanup_started_epoch: float | None = None
        self._cleanup_initial_bytes = 0
        self._cleanup_reclaimed_bytes = 0
        self._cleanup_batches_completed = 0
        self._delete_failure_attempts: dict[str, int] = {}
        self._delete_retry_after: dict[str, float] = {}
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
            payload = _copy_payload(self._status)
            if self._cleanup_started_epoch is not None:
                payload["progress"] = self._cleanup_progress(time.time())
            return payload

    def _start_cleanup_progress(self, *, planned_bytes: int, now_epoch: float) -> None:
        self._cleanup_started_epoch = float(now_epoch)
        self._cleanup_initial_bytes = max(0, int(planned_bytes))
        self._cleanup_reclaimed_bytes = 0
        self._cleanup_batches_completed = 0

    def _cleanup_progress(self, now_epoch: float) -> dict[str, Any]:
        started = self._cleanup_started_epoch
        initial = max(0, int(self._cleanup_initial_bytes))
        reclaimed = max(0, min(initial, int(self._cleanup_reclaimed_bytes)))
        remaining = max(0, initial - reclaimed)
        elapsed = max(0.0, float(now_epoch) - started) if started is not None else 0.0
        rate = reclaimed / elapsed if elapsed > 0 and reclaimed > 0 else 0.0
        eta = remaining / rate if rate > 0 and remaining > 0 else None
        percent = 100.0 if initial == 0 else reclaimed / initial * 100.0
        return {
            "active": bool(self._cleanup_active or self._status.get("state") in {"planning", "cleaning", "waiting", "queued"}),
            "started_at": _iso_time(started),
            "elapsed_seconds": round(elapsed, 1),
            "percent": round(min(100.0, percent), 2),
            "initial_bytes": initial,
            "reclaimed_bytes": reclaimed,
            "remaining_bytes": remaining,
            "average_bytes_per_second": round(rate),
            "eta_seconds": round(eta) if eta is not None else None,
            "batches_completed": self._cleanup_batches_completed,
        }

    def plan(self, *, now_epoch: float | None = None) -> dict[str, Any]:
        now = time.time() if now_epoch is None else float(now_epoch)
        with self._state_lock:
            config = self.config.model_copy(deep=True)
            cameras = dict(self._cameras)
        snapshot_cutoff = now - config.snapshot_days * SECONDS_PER_DAY
        snapshot_plan = (
            dict(self.snapshot_plan_provider(snapshot_cutoff))
            if self.snapshot_plan_provider is not None
            else {
                "file_count": 0,
                "bytes": 0,
                "unindexed_files": 0,
                "expired_files": 0,
                "expired_bytes": 0,
                "per_camera": [],
            }
        )
        location_usage = self._location_usage()
        capacity_groups = self._capacity_groups(location_usage)
        total_bytes = sum(item["total_bytes"] for item in capacity_groups)
        free_bytes = sum(item["free_bytes"] for item in capacity_groups)
        used_bytes = max(0, total_bytes - free_bytes)
        protected_paths = self._normalized_protected_paths()
        with self.connection_factory() as connection:
            connection.execute(
                "CREATE TEMP TABLE retention_policy ("
                "camera_id TEXT NOT NULL, source TEXT NOT NULL, cutoff REAL NOT NULL, "
                "PRIMARY KEY (camera_id, source)) WITHOUT ROWID"
            )
            group_keys = connection.execute(
                "SELECT DISTINCT camera_id, source FROM recordings"
            ).fetchall()
            policy_rows: list[tuple[str, str, float]] = []
            retention_days_by_group: dict[tuple[str, str], int] = {}
            for key in group_keys:
                camera_id = str(key["camera_id"])
                source = "main" if str(key["source"]) == "main" else "live"
                camera = cameras.get(camera_id)
                override = None
                if camera is not None:
                    override = (
                        camera.retention.main_days
                        if source == "main"
                        else camera.retention.live_days
                    )
                days = int(
                    override
                    or (config.main_days if source == "main" else config.live_days)
                )
                retention_days_by_group[(camera_id, source)] = days
                policy_rows.append((camera_id, source, now - days * SECONDS_PER_DAY))
            connection.executemany(
                "INSERT INTO retention_policy(camera_id, source, cutoff) VALUES (?, ?, ?)",
                policy_rows,
            )
            connection.execute(
                "CREATE TEMP TABLE retention_protected (path TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            connection.executemany(
                "INSERT OR IGNORE INTO retention_protected(path) VALUES (?)",
                ((path,) for path in protected_paths),
            )
            connection.set_progress_handler(
                lambda: int(self._stop.is_set()),
                10_000,
            )
            groups = connection.execute(
                """
                SELECT r.camera_id, r.source, count(*) AS file_count,
                       coalesce(sum(r.size_bytes), 0) AS bytes,
                       min(r.start_epoch) AS oldest, max(r.end_epoch) AS newest,
                       coalesce(sum(CASE
                           WHEN r.end_epoch < policy.cutoff
                            AND protected.path IS NULL THEN 1 ELSE 0 END), 0)
                           AS expired_file_count,
                       coalesce(sum(CASE
                           WHEN r.end_epoch < policy.cutoff
                            AND protected.path IS NULL THEN r.size_bytes ELSE 0 END), 0)
                           AS expired_bytes,
                       coalesce(sum(CASE
                           WHEN r.end_epoch < policy.cutoff
                            AND protected.path IS NOT NULL THEN 1 ELSE 0 END), 0)
                           AS protected_file_count,
                       coalesce(sum(CASE
                           WHEN r.end_epoch < policy.cutoff
                            AND protected.path IS NOT NULL THEN r.size_bytes ELSE 0 END), 0)
                           AS protected_bytes
                FROM recordings AS r
                JOIN retention_policy AS policy
                  ON policy.camera_id = r.camera_id AND policy.source = r.source
                LEFT JOIN retention_protected AS protected ON protected.path = r.path
                GROUP BY r.camera_id, r.source
                ORDER BY r.camera_id, r.source
                """
            ).fetchall()

            totals: dict[str, int | float | None] = {
                "file_count": 0,
                "bytes": 0,
                "oldest": None,
                "newest": None,
            }
            per_camera: list[dict[str, Any]] = []
            expired_count = 0
            expired_bytes = 0
            protected_expired_count = 0
            protected_expired_bytes = 0
            for row in groups:
                camera_id = str(row["camera_id"])
                source = "main" if str(row["source"]) == "main" else "live"
                retention_days = retention_days_by_group[(camera_id, source)]
                group_protected_expired_files = int(row["protected_file_count"] or 0)
                group_protected_expired_bytes = int(row["protected_bytes"] or 0)
                eligible_expired_files = int(row["expired_file_count"] or 0)
                eligible_expired_bytes = int(row["expired_bytes"] or 0)
                group_bytes = int(row["bytes"] or 0)
                oldest = _optional_float(row["oldest"])
                newest = _optional_float(row["newest"])
                totals["file_count"] = int(totals["file_count"] or 0) + int(
                    row["file_count"] or 0
                )
                totals["bytes"] = int(totals["bytes"] or 0) + group_bytes
                totals["oldest"] = (
                    oldest
                    if totals["oldest"] is None
                    else min(float(totals["oldest"]), oldest or float(totals["oldest"]))
                )
                totals["newest"] = (
                    newest
                    if totals["newest"] is None
                    else max(float(totals["newest"]), newest or float(totals["newest"]))
                )
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
        storage_limit_bytes = round(config.storage_limit_tb * TIB)
        quota_reclaim = max(0, indexed_bytes - storage_limit_bytes)
        free_percent = free_bytes / total_bytes * 100 if total_bytes else 0.0
        pressured_locations: list[str] = []
        pressure_location_ids: dict[str, list[str]] = {}
        free_reclaim_by_location: dict[str, int] = {}
        free_reclaim = 0
        locations_by_id = {str(item["id"]): item for item in location_usage}
        for item in capacity_groups:
            minimum_percent = max(
                float(config.minimum_free_percent),
                float(item.get("reserve_percent") or 0.0),
            )
            target_percent = max(
                float(config.target_free_percent),
                min(100.0, float(item.get("reserve_percent") or 0.0) + 1.0),
            )
            for location_id in item["location_ids"]:
                location = locations_by_id[location_id]
                location["effective_minimum_free_percent"] = minimum_percent
                location["effective_target_free_percent"] = target_percent
            if item["state"] == "full" or item["free_percent"] <= minimum_percent:
                pressured_locations.extend(item["location_ids"])
                target_free = round(item["total_bytes"] * target_percent / 100)
                item["reclaim_bytes"] = max(0, target_free - item["free_bytes"])
                pressure_key = str(item["id"])
                pressure_location_ids[pressure_key] = list(item["location_ids"])
                free_reclaim_by_location[pressure_key] = int(item["reclaim_bytes"])
                for location_id in item["location_ids"]:
                    locations_by_id[location_id]["reclaim_bytes"] = int(
                        item["reclaim_bytes"]
                    )
                free_reclaim += int(item["reclaim_bytes"])
        snapshot_expired_bytes = int(snapshot_plan.get("expired_bytes") or 0)
        # Policies may target disjoint files and filesystems. The executor
        # de-duplicates candidates; the planner must not collapse their budgets.
        planned_reclaim = max(expired_bytes, quota_reclaim, free_reclaim) + snapshot_expired_bytes
        total_span = max(
            1.0,
            (_optional_float(totals["newest"]) or now)
            - (_optional_float(totals["oldest"]) or now),
        )
        bytes_per_day = indexed_bytes / total_span * SECONDS_PER_DAY
        days_to_minimum = None
        minimum_free_bytes = round(total_bytes * config.minimum_free_percent / 100)
        if bytes_per_day > 0:
            days_to_minimum = max(0.0, (free_bytes - minimum_free_bytes) / bytes_per_day)

        reasons: list[str] = []
        if expired_bytes:
            reasons.append("age")
        if snapshot_expired_bytes:
            reasons.append("snapshot_age")
        if quota_reclaim:
            reasons.append("quota")
        if free_reclaim:
            reasons.append("free_space")
        return {
            "generated_at": _iso_time(now),
                "enabled": config.enabled,
                "automatic_cleanup": config.automatic_cleanup,
            "storage": {
                "total_bytes": total_bytes,
                "used_bytes": used_bytes,
                "free_bytes": free_bytes,
                "free_percent": round(free_percent, 1),
                "minimum_free_percent": config.minimum_free_percent,
                "target_free_percent": config.target_free_percent,
                "emergency_free_percent": config.emergency_free_percent,
                "emergency": any(
                    int(item["total_bytes"]) > 0
                    and item["free_percent"] < config.emergency_free_percent
                    for item in location_usage
                ),
                "degraded": any(
                    item["state"] not in {"online", "full"}
                    for item in location_usage
                ),
                "locations": location_usage,
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
                "main_days": config.main_days,
                "live_days": config.live_days,
                "snapshot_days": config.snapshot_days,
                "cleanup_batch_files": config.cleanup_batch_files,
                "protected_recent_seconds": RECENT_RECORDING_PROTECTION_SECONDS,
            },
            "reclaim": {
                "expired_files": expired_count,
                "expired_bytes": expired_bytes,
                "snapshot_expired_files": int(snapshot_plan.get("expired_files") or 0),
                "snapshot_expired_bytes": snapshot_expired_bytes,
                "protected_expired_files": protected_expired_count,
                "protected_expired_bytes": protected_expired_bytes,
                "quota_bytes": quota_reclaim,
                "free_space_bytes": free_reclaim,
                "free_space_by_location": free_reclaim_by_location,
                "pressure_location_ids": pressure_location_ids,
                "pressured_location_ids": pressured_locations,
                "planned_bytes": planned_reclaim,
                "reasons": reasons,
            },
            "snapshots": snapshot_plan,
            "per_camera": per_camera,
            "per_camera_storage": self._per_camera_storage(per_camera, snapshot_plan),
        }

    def run_once(self, *, apply: bool) -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("recording retention is already running")
        started = time.time()
        try:
            before = self.plan(now_epoch=started)
            result = self._apply_plan(
                before,
                apply=apply,
                now_epoch=started,
                capacity_reclaim_bytes=0,
                quota_reclaim_bytes=int(before["reclaim"]["quota_bytes"]),
                free_reclaim_by_location=dict(
                    before["reclaim"].get("free_space_by_location") or {}
                ),
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
        quota_reclaim_bytes: int | None = None,
        free_reclaim_by_location: Mapping[str, int] | None = None,
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
            "protected_files": 0,
            "recording_deleted_files": 0,
            "recording_deleted_bytes": 0,
            "recording_deleted_bytes_by_location": {},
            "snapshot_selected_files": 0,
            "snapshot_deleted_files": 0,
            "snapshot_missing_files": 0,
            "snapshot_deleted_bytes": 0,
            "snapshot_failed_files": 0,
            "batch_saturated": False,
            "cancelled": False,
            "remaining_planned_bytes": max(0, int(planned_reclaim_bytes)),
        }
        if apply:
            batch_deadline = time.monotonic() + RETENTION_BATCH_TIME_BUDGET_SECONDS
            candidates = self._candidates(
                plan,
                now_epoch=now_epoch,
                capacity_reclaim_bytes=max(0, int(capacity_reclaim_bytes)),
                quota_reclaim_bytes=quota_reclaim_bytes,
                free_reclaim_by_location=free_reclaim_by_location,
            )
            result["selected_files"] = len(candidates)
            removed_paths: list[str] = []
            empty_directory_candidates: set[Path] = set()
            for row in candidates:
                if self._stop.is_set():
                    result["cancelled"] = True
                    break
                if time.monotonic() >= batch_deadline:
                    result["batch_saturated"] = True
                    break
                raw_path = str(row["path"])
                try:
                    path = self._safe_recording_path(raw_path)
                except ValueError:
                    result["failed_files"] += 1
                    LOGGER.error("Retention rejected recording path outside storage: %s", raw_path)
                    continue
                try:
                    if self.delete_recording_provider is not None:
                        if not self.delete_recording_provider(path):
                            result["protected_files"] += 1
                            continue
                    else:
                        path.unlink()
                    result["deleted_files"] += 1
                    result["deleted_bytes"] += int(row["size_bytes"] or 0)
                    location_id = str(row["location_id"] or "default")
                    if location_id == "default" and self.media_storage is not None:
                        location_id = str(
                            self.media_storage.location_id_for(path, "recordings")
                            or "default"
                        )
                    deleted_by_location = result["recording_deleted_bytes_by_location"]
                    deleted_by_location[location_id] = (
                        int(deleted_by_location.get(location_id) or 0)
                        + int(row["size_bytes"] or 0)
                    )
                    empty_directory_candidates.add(path.parent)
                    removed_paths.append(raw_path)
                    self._clear_delete_failure(raw_path)
                except FileNotFoundError:
                    result["missing_files"] += 1
                    removed_paths.append(raw_path)
                    self._clear_delete_failure(raw_path)
                except OSError as error:
                    result["failed_files"] += 1
                    self._record_delete_failure(raw_path)
                    LOGGER.warning("Retention could not delete %s: %s", path, error)
            if removed_paths:
                for offset in range(0, len(removed_paths), 100):
                    with self.connection_factory() as connection:
                        connection.executemany(
                            "DELETE FROM recordings WHERE path = ?",
                            (
                                (path,)
                                for path in removed_paths[offset : offset + 100]
                            ),
                        )
            self._remove_empty_directories(empty_directory_candidates)
            result["recording_deleted_files"] = int(result["deleted_files"])
            result["recording_deleted_bytes"] = int(result["deleted_bytes"])
            if self.snapshot_cleanup_provider is not None and not self._stop.is_set():
                remaining_slots = max(
                    0,
                    min(
                        250,
                        int(plan.get("policy", {}).get("cleanup_batch_files") or 100)
                        - int(result["selected_files"]),
                    ),
                )
                snapshot_cutoff = now_epoch - int(
                    plan.get("policy", {}).get("snapshot_days") or 1095
                ) * SECONDS_PER_DAY
                snapshot_result = (
                    dict(
                        self.snapshot_cleanup_provider(
                            snapshot_cutoff, remaining_slots
                        )
                    )
                    if remaining_slots > 0
                    else {}
                )
                result["snapshot_selected_files"] = int(
                    snapshot_result.get("selected_files") or 0
                )
                result["snapshot_deleted_files"] = int(
                    snapshot_result.get("deleted_files") or 0
                )
                result["snapshot_missing_files"] = int(
                    snapshot_result.get("missing_files") or 0
                )
                result["snapshot_deleted_bytes"] = int(
                    snapshot_result.get("deleted_bytes") or 0
                )
                result["snapshot_failed_files"] = int(
                    snapshot_result.get("failed_files") or 0
                )
                result["selected_files"] += result["snapshot_selected_files"]
                result["deleted_files"] += result["snapshot_deleted_files"]
                result["missing_files"] += result["snapshot_missing_files"]
                result["deleted_bytes"] += result["snapshot_deleted_bytes"]
                result["failed_files"] += result["snapshot_failed_files"]
                result["batch_saturated"] = bool(
                    result["batch_saturated"]
                    or snapshot_result.get("batch_saturated")
                )
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
                    self._set_reclaim_budgets(plan)
                    self._planned_reclaim_remaining = int(
                        plan["reclaim"]["planned_bytes"]
                    )
                    if apply and (
                        self._cleanup_started_epoch is None
                        or requested_apply
                        or not cleanup_active
                    ):
                        self._start_cleanup_progress(
                            planned_bytes=self._planned_reclaim_remaining,
                            now_epoch=time.time(),
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
                if apply and self._cleanup_started_epoch is not None:
                    self._cleanup_reclaimed_bytes += int(result["deleted_bytes"])
                    self._cleanup_batches_completed += 1
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
                    self._set_reclaim_budgets(plan)
                    self._planned_reclaim_remaining = int(
                        plan["reclaim"]["planned_bytes"]
                    )
                    result["remaining_planned_bytes"] = self._planned_reclaim_remaining
                    result["plan_refreshed"] = True
                    result["refreshed_at"] = plan["generated_at"]
                should_continue = bool(
                    apply
                    and (
                        int(result["selected_files"])
                        >= int(plan.get("policy", {}).get("cleanup_batch_files") or 100)
                        or bool(result.get("batch_saturated"))
                    )
                    and (result["deleted_files"] or result["missing_files"])
                )
                if (
                    apply
                    and not should_continue
                    and (result["deleted_files"] or result["missing_files"])
                ):
                    # The plan shown to operators is otherwise the pre-cleanup
                    # snapshot for up to a day. Rebuild it once, after the final
                    # bounded batch, so free space, indexed usage, and eligible
                    # bytes all describe the completed cleanup.
                    plan = self.plan()
                    self._last_plan_monotonic = time.monotonic()
                    self._set_reclaim_budgets(plan)
                    self._planned_reclaim_remaining = int(
                        plan["reclaim"]["planned_bytes"]
                    )
                    result["remaining_planned_bytes"] = self._planned_reclaim_remaining
                    result["plan_refreshed"] = True
                    result["refreshed_at"] = plan["generated_at"]
                with self._state_lock:
                    self._cleanup_active = should_continue
                    progress = (
                        self._cleanup_progress(time.time())
                        if self._cleanup_started_epoch is not None
                        else None
                    )
                    if progress is not None and not should_continue:
                        progress["active"] = False
                        self._cleanup_started_epoch = None
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
                        "progress": progress,
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
                capacity_reclaim_bytes=0,
                quota_reclaim_bytes=self._quota_reclaim_remaining,
                free_reclaim_by_location=self._free_reclaim_remaining,
                planned_reclaim_bytes=self._planned_reclaim_remaining,
            )
            recording_deleted = int(result["recording_deleted_bytes"])
            self._quota_reclaim_remaining = max(
                0,
                self._quota_reclaim_remaining - recording_deleted,
            )
            for location_id, deleted_bytes in dict(
                result.get("recording_deleted_bytes_by_location") or {}
            ).items():
                for pressure_key, member_ids in dict(
                    plan.get("reclaim", {}).get("pressure_location_ids") or {}
                ).items():
                    if str(location_id) in member_ids:
                        self._free_reclaim_remaining[str(pressure_key)] = max(
                            0,
                            int(self._free_reclaim_remaining.get(str(pressure_key), 0))
                            - int(deleted_bytes),
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
                    - recording_deleted,
                )
                reclaim["free_space_by_location"] = dict(self._free_reclaim_remaining)
                reclaim["free_space_bytes"] = sum(self._free_reclaim_remaining.values())
                plan["reclaim"] = reclaim
            return result
        finally:
            self._run_lock.release()

    def _storage_below_minimum(self) -> bool:
        return any(
            int(item["total_bytes"]) > 0
            and (
                item["state"] == "full"
                or item["free_percent"]
                < max(
                    float(self.config.minimum_free_percent),
                    float(item.get("reserve_percent") or 0.0),
                )
            )
            for item in self._location_usage()
        )

    def _set_reclaim_budgets(self, plan: Mapping[str, Any]) -> None:
        reclaim = plan.get("reclaim") or {}
        self._quota_reclaim_remaining = max(0, int(reclaim.get("quota_bytes") or 0))
        self._free_reclaim_remaining = {
            str(location_id): max(0, int(value or 0))
            for location_id, value in dict(
                reclaim.get("free_space_by_location") or {}
            ).items()
        }

    def _record_delete_failure(self, path: str) -> None:
        if path not in self._delete_failure_attempts and len(
            self._delete_failure_attempts
        ) >= 5_000:
            oldest = next(iter(self._delete_failure_attempts))
            self._clear_delete_failure(oldest)
        attempts = min(10, self._delete_failure_attempts.get(path, 0) + 1)
        self._delete_failure_attempts[path] = attempts
        self._delete_retry_after[path] = time.monotonic() + min(
            RETENTION_FAILURE_RETRY_MAX_SECONDS,
            5 * (2 ** (attempts - 1)),
        )

    def _clear_delete_failure(self, path: str) -> None:
        self._delete_failure_attempts.pop(path, None)
        self._delete_retry_after.pop(path, None)

    def _candidate_retry_ready(self, path: str) -> bool:
        return self._delete_retry_after.get(path, 0.0) <= time.monotonic()

    def _location_usage(self) -> list[dict[str, Any]]:
        if self.media_storage is None:
            try:
                usage = shutil.disk_usage(self.storage_dir)
            except OSError:
                return []
            return [{
                "id": "default",
                "name": "Primary media",
                "path": str(self.storage_dir),
                "state": "online",
                "total_bytes": usage.total,
                "free_bytes": usage.free,
                "free_percent": round(usage.free / usage.total * 100, 1) if usage.total else 0.0,
                "reserve_percent": 0.0,
                "reclaim_bytes": 0,
            }]
        payload: list[dict[str, Any]] = []
        for status in self.media_storage.statuses():
            if "recordings" not in status.roles:
                continue
            payload.append({
                "id": status.id,
                "name": status.name,
                "path": str(status.path),
                "state": status.state,
                "total_bytes": status.total_bytes,
                "free_bytes": status.free_bytes,
                "free_percent": (
                    round(status.free_bytes / status.total_bytes * 100, 1)
                    if status.total_bytes > 0
                    else 0.0
                ),
                "reserve_percent": status.reserve_percent,
                "filesystem_id": status.filesystem_id,
                "error": status.error,
                "reclaim_bytes": 0,
            })
        return payload

    @staticmethod
    def _capacity_groups(
        location_usage: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Collapse logical roots that share one physical filesystem."""
        grouped: dict[str, dict[str, Any]] = {}
        for location in location_usage:
            if int(location.get("total_bytes") or 0) <= 0:
                continue
            filesystem_id = str(location.get("filesystem_id") or "")
            key = filesystem_id or str(location["id"])
            item = grouped.get(key)
            if item is None:
                grouped[key] = {
                    "id": key,
                    "location_ids": [str(location["id"])],
                    "state": str(location.get("state") or "unavailable"),
                    "total_bytes": int(location.get("total_bytes") or 0),
                    "free_bytes": int(location.get("free_bytes") or 0),
                    "free_percent": float(location.get("free_percent") or 0.0),
                    "reserve_percent": float(location.get("reserve_percent") or 0.0),
                    "reclaim_bytes": 0,
                }
                continue
            item["location_ids"].append(str(location["id"]))
            item["reserve_percent"] = max(
                float(item["reserve_percent"]),
                float(location.get("reserve_percent") or 0.0),
            )
            if str(location.get("state")) == "full":
                item["state"] = "full"
        return list(grouped.values())

    def _normalized_protected_paths(self) -> set[str]:
        """Normalize protection keys once without touching the media filesystem."""
        normalized: set[str] = set()
        for raw_path in self.protected_paths_provider():
            value = str(raw_path or "")
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.storage_dir / path
            normalized.add(os.path.normpath(str(path)))
        return normalized

    @classmethod
    def _register_protection_function(
        cls,
        connection: sqlite3.Connection,
        protected_paths: set[str],
    ) -> None:
        connection.create_function(
            "survng_path_protected",
            1,
            lambda raw_path: int(str(raw_path or "") in protected_paths),
            deterministic=True,
        )

    @staticmethod
    def _merge_oldest_rows(
        row_groups: list[list[sqlite3.Row]],
    ) -> Iterator[sqlite3.Row]:
        return heapq.merge(
            *row_groups,
            key=lambda row: float(row["start_epoch"] or 0),
        )

    def _oldest_group_rows(
        self,
        connection: sqlite3.Connection,
        groups: Sequence[Mapping[str, Any]],
        *,
        protected_cutoff: float,
        limit_per_group: int,
        age_limited: bool,
        now_epoch: float,
        location_ids: Sequence[str] = (),
    ) -> list[list[sqlite3.Row]]:
        row_groups: list[list[sqlite3.Row]] = []
        location_clause = ""
        location_parameters: tuple[object, ...] = ()
        if location_ids:
            placeholders = ",".join("?" for _ in location_ids)
            location_clause = f" AND location_id IN ({placeholders})"
            location_parameters = tuple(location_ids)
        # The existing range index keeps upgrades non-blocking. A location-first
        # index on millions of legacy rows would otherwise be built at startup.
        index_name = "recordings_range"
        for group in groups:
            cutoff = protected_cutoff
            if age_limited:
                cutoff = min(
                    cutoff,
                    now_epoch
                    - int(group["retention_days"]) * SECONDS_PER_DAY,
                )
            rows = connection.execute(
                f"""
                SELECT path, size_bytes, start_epoch, location_id FROM recordings
                INDEXED BY {index_name}
                WHERE camera_id = ? AND source = ? AND end_epoch < ?
                  AND survng_path_protected(path) = 0{location_clause}
                ORDER BY start_epoch ASC LIMIT ?
                """,
                (
                    group["camera_id"],
                    group["source"],
                    cutoff,
                    *location_parameters,
                    limit_per_group,
                ),
            ).fetchall()
            if rows:
                row_groups.append(rows)
        return row_groups

    @staticmethod
    def _per_camera_storage(
        recording_rows: Sequence[Mapping[str, Any]],
        snapshot_plan: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        by_camera: dict[str, dict[str, Any]] = {}
        for row in recording_rows:
            camera_id = str(row.get("camera_id") or "")
            item = by_camera.setdefault(
                camera_id,
                {
                    "camera_id": camera_id,
                    "recording_bytes": 0,
                    "recording_files": 0,
                    "snapshot_bytes": 0,
                    "snapshot_files": 0,
                },
            )
            item["recording_bytes"] += int(row.get("bytes") or 0)
            item["recording_files"] += int(row.get("file_count") or 0)
        for row in snapshot_plan.get("per_camera", []):
            if not isinstance(row, Mapping):
                continue
            camera_id = str(row.get("camera_id") or "")
            item = by_camera.setdefault(
                camera_id,
                {
                    "camera_id": camera_id,
                    "recording_bytes": 0,
                    "recording_files": 0,
                    "snapshot_bytes": 0,
                    "snapshot_files": 0,
                },
            )
            item["snapshot_bytes"] += int(row.get("bytes") or 0)
            item["snapshot_files"] += int(row.get("file_count") or 0)
        return [by_camera[camera_id] for camera_id in sorted(by_camera)]

    def _candidates(
        self,
        plan: Mapping[str, Any],
        *,
        now_epoch: float,
        capacity_reclaim_bytes: int | None = None,
        quota_reclaim_bytes: int | None = None,
        free_reclaim_by_location: Mapping[str, int] | None = None,
    ) -> list[sqlite3.Row]:
        limit = max(
            1,
            int(plan.get("policy", {}).get("cleanup_batch_files") or 100),
        )
        protected_cutoff = now_epoch - RECENT_RECORDING_PROTECTION_SECONDS
        protected_paths = self._normalized_protected_paths()
        groups = plan.get("per_camera", [])
        candidates: dict[str, sqlite3.Row] = {}
        with self.connection_factory() as connection:
            self._register_protection_function(connection, protected_paths)
            expiring_groups = list(groups)
            if expiring_groups:
                row_groups = self._oldest_group_rows(
                    connection,
                    expiring_groups,
                    protected_cutoff=protected_cutoff,
                    limit_per_group=limit * 4,
                    age_limited=True,
                    now_epoch=now_epoch,
                )
                for row in self._merge_oldest_rows(row_groups):
                    path_key = str(row["path"])
                    if not self._candidate_retry_ready(path_key):
                        continue
                    candidates[path_key] = row
                    if len(candidates) >= limit:
                        break
            legacy_capacity_reclaim = (
                max(
                    int(plan["reclaim"]["quota_bytes"]),
                    int(plan["reclaim"]["free_space_bytes"]),
                )
                if capacity_reclaim_bytes is None
                else max(0, int(capacity_reclaim_bytes))
            )
            quota_remaining = (
                legacy_capacity_reclaim
                if quota_reclaim_bytes is None and free_reclaim_by_location is None
                else max(0, int(quota_reclaim_bytes or 0))
            )
            location_budgets = {
                str(key): max(0, int(value or 0))
                for key, value in dict(
                    free_reclaim_by_location
                    if free_reclaim_by_location is not None
                    else plan.get("reclaim", {}).get("free_space_by_location", {})
                ).items()
            }
            selected_bytes = sum(int(row["size_bytes"] or 0) for row in candidates.values())
            selected_by_location: dict[str, int] = {}
            for row in candidates.values():
                location_id = str(row["location_id"] or "default")
                if location_id == "default" and self.media_storage is not None:
                    location_id = str(
                        self.media_storage.location_id_for(
                            Path(str(row["path"])), "recordings"
                        )
                        or "default"
                    )
                selected_by_location[location_id] = (
                    selected_by_location.get(location_id, 0)
                    + int(row["size_bytes"] or 0)
                )

            pressure_members = dict(
                plan.get("reclaim", {}).get("pressure_location_ids") or {}
            )
            for location_id, budget in location_budgets.items():
                if budget <= selected_by_location.get(location_id, 0) or len(candidates) >= limit:
                    continue
                member_ids = [
                    str(value)
                    for value in pressure_members.get(location_id, [location_id])
                ]
                already_selected = sum(
                    selected_by_location.get(member_id, 0) for member_id in member_ids
                )
                if budget <= already_selected:
                    continue
                row_groups = self._oldest_group_rows(
                    connection,
                    expiring_groups,
                    protected_cutoff=protected_cutoff,
                    limit_per_group=limit * 2,
                    age_limited=False,
                    now_epoch=now_epoch,
                    location_ids=tuple(dict.fromkeys([*member_ids, "default"])),
                )
                for row in self._merge_oldest_rows(row_groups):
                    path_key = str(row["path"])
                    row_location = str(row["location_id"] or "default")
                    resolved_location = row_location
                    if row_location == "default" and self.media_storage is not None:
                        resolved_location = str(
                            self.media_storage.location_id_for(
                                Path(path_key), "recordings"
                            )
                            or "default"
                        )
                    if (
                        row_location == "default"
                        and self.media_storage is not None
                        and resolved_location not in member_ids
                    ):
                        continue
                    if (
                        path_key not in candidates
                        and self._candidate_retry_ready(path_key)
                    ):
                        candidates[path_key] = row
                        row_bytes = int(row["size_bytes"] or 0)
                        selected_bytes += row_bytes
                        selected_by_location[resolved_location] = (
                            selected_by_location.get(resolved_location, 0) + row_bytes
                        )
                    group_selected = sum(
                        selected_by_location.get(member_id, 0)
                        for member_id in member_ids
                    )
                    if len(candidates) >= limit or group_selected >= budget:
                        break
            if quota_remaining > selected_bytes and len(candidates) < limit:
                row_groups = self._oldest_group_rows(
                    connection,
                    expiring_groups,
                    protected_cutoff=protected_cutoff,
                    limit_per_group=limit * 2,
                    age_limited=False,
                    now_epoch=now_epoch,
                )
                for row in self._merge_oldest_rows(row_groups):
                    path_key = str(row["path"])
                    if (
                        path_key not in candidates
                        and self._candidate_retry_ready(path_key)
                    ):
                        candidates[path_key] = row
                        selected_bytes += int(row["size_bytes"] or 0)
                    if len(candidates) >= limit or selected_bytes >= quota_remaining:
                        break
        return sorted(candidates.values(), key=lambda row: float(row["start_epoch"] or 0))[:limit]

    def _safe_recording_path(self, raw_path: str) -> Path:
        path = Path(raw_path).resolve(strict=False)
        if self.media_storage is not None:
            if not self.media_storage.contains(path, "recordings"):
                raise ValueError("recording is outside configured media storage")
        else:
            path.relative_to(self.recordings_dir)
        if path.suffix.lower() != ".mp4":
            raise ValueError("retention only removes MP4 recordings")
        return path

    def _remove_empty_directories(self, directories: set[Path]) -> None:
        roots = (
            {
                root.resolve(strict=False)
                for root in self.media_storage.configured_roots_for("recordings")
            }
            if self.media_storage is not None
            else {self.recordings_dir}
        )
        for directory in directories:
            current = directory
            while current not in roots:
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
