from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .recorder import Recorder


LOGGER = logging.getLogger(__name__)
MEDIA_SAMPLE_LIMIT = 20
RECENT_MEDIA_SECONDS = 60.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StorageReconciler:
    """Audit media references and repair indexes without deleting user media."""

    def __init__(self, storage_dir: Path, database_path: Path, recorder: Recorder) -> None:
        self.storage_dir = storage_dir.resolve()
        self.database_path = database_path
        self.recorder = recorder

    def run(self, *, apply: bool = False) -> dict[str, object]:
        repairs = {
            "stale_index_rows_removed": 0,
            "recordings_reindexed": 0,
            "event_media_references_cleared": 0,
            "motion_sample_references_cleared": 0,
            "face_media_references_cleared": 0,
        }
        if apply:
            repairs.update(self.recorder.reconcile_storage_index())
            repairs.update(self._clear_missing_references())
        summary = self._scan()
        return {
            "mode": "repair" if apply else "scan",
            "generated_at": _utc_now(),
            "summary": summary,
            "repairs": repairs,
            "note": (
                "Repairs completed. Incident, motion-audit, and face history was preserved."
                if apply
                else "No changes were made. Run Repair to reconcile the local databases."
            ),
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _media_path(self, path_value: object) -> Path:
        raw_path = str(path_value or "").strip()
        path = Path(raw_path)
        return path if path.is_absolute() else self.storage_dir / path

    def _missing(self, path_value: object) -> bool:
        raw_path = str(path_value or "").strip()
        return bool(raw_path) and not self._media_path(raw_path).is_file()

    def _display_path(self, path_value: object) -> str:
        path = self._media_path(path_value)
        try:
            return str(path.resolve().relative_to(self.storage_dir))
        except ValueError:
            return path.name

    def _path_key(self, path_value: object) -> str:
        return str(self._media_path(path_value).absolute())

    def _media_references(self) -> tuple[list[dict[str, object]], set[str]]:
        references: list[dict[str, object]] = []
        known_paths: set[str] = set()
        with self._connect() as connection:
            for row in connection.execute(
                "SELECT id, camera_id, snapshot_path, recording_path FROM events"
            ):
                for kind, column in (("event_snapshot", "snapshot_path"), ("event_recording", "recording_path")):
                    path = str(row[column] or "")
                    if path:
                        known_paths.add(self._path_key(path))
                    if path:
                        references.append({
                            "table": "events", "id": int(row["id"]), "camera_id": str(row["camera_id"]),
                            "kind": kind, "column": column, "path": path,
                        })
            for row in connection.execute(
                "SELECT id, camera_id, snapshot_path FROM motion_audits WHERE snapshot_path != ''"
            ):
                path = str(row["snapshot_path"] or "")
                known_paths.add(self._path_key(path))
                references.append({
                    "table": "motion_audits", "id": int(row["id"]), "camera_id": str(row["camera_id"]),
                    "kind": "motion_snapshot", "column": "snapshot_path", "path": path,
                })
            face_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'face_observations'"
            ).fetchone()
            if face_table:
                for row in connection.execute(
                    "SELECT id, camera_id, snapshot_path FROM face_observations WHERE snapshot_path != ''"
                ):
                    path = str(row["snapshot_path"] or "")
                    known_paths.add(self._path_key(path))
                    references.append({
                        "table": "face_observations", "id": int(row["id"]), "camera_id": str(row["camera_id"]),
                        "kind": "face_snapshot", "column": "snapshot_path", "path": path,
                    })
        return references, known_paths

    def _scan(self) -> dict[str, object]:
        all_references, known_paths = self._media_references()
        recording_health = self.recorder.storage_index_health()
        files_by_source = recording_health.pop("files_by_source", {})
        recording_paths = {
            self._path_key(path)
            for files in files_by_source.values()
            for path in files
        }
        missing_index_files = list(recording_health.pop("missing_index_files", []))
        unindexed_files = list(recording_health.pop("unindexed_files", []))
        media_files: list[Path] = []
        for directory_name in ("snapshots", "motion_samples"):
            directory = self.storage_dir / directory_name
            if directory.exists():
                media_files.extend(path for path in directory.rglob("*") if path.is_file())
        media_paths = {self._path_key(path) for path in media_files}
        references = [
            reference for reference in all_references
            if self._path_key(reference["path"]) not in (
                recording_paths if reference["kind"] == "event_recording" else media_paths
            )
        ]
        per_camera: dict[str, dict[str, int]] = {}
        counts = {
            "event_snapshot": 0,
            "event_recording": 0,
            "motion_snapshot": 0,
            "face_snapshot": 0,
        }
        for reference in references:
            kind = str(reference["kind"])
            counts[kind] += 1
            camera_counts = per_camera.setdefault(str(reference["camera_id"]), {})
            camera_counts[kind] = camera_counts.get(kind, 0) + 1
        for kind, paths in (("recording_index_missing", missing_index_files), ("recording_unindexed", unindexed_files)):
            for path in paths:
                camera_id = self._recording_camera_id(path)
                camera_counts = per_camera.setdefault(camera_id, {})
                camera_counts[kind] = camera_counts.get(kind, 0) + 1

        orphan_files: list[Path] = []
        now = datetime.now(timezone.utc).timestamp()
        for path in media_files:
            if self._path_key(path) in known_paths:
                continue
            try:
                if now - path.stat().st_mtime < RECENT_MEDIA_SECONDS:
                    continue
            except OSError:
                continue
            orphan_files.append(path)
        orphan_bytes = sum(self._file_size(path) for path in orphan_files)
        cache_bytes = sum(
            self._directory_size(self.storage_dir / name)
            for name in ("event_clips", "playback-cache", "hls")
        )
        usage = shutil.disk_usage(self.storage_dir)
        return {
            "storage_total_bytes": usage.total,
            "storage_used_bytes": usage.used,
            "storage_free_bytes": usage.free,
            **recording_health,
            "missing_index_rows": len(missing_index_files),
            "unindexed_recording_files": len(unindexed_files),
            "missing_event_snapshots": counts["event_snapshot"],
            "missing_event_recordings": counts["event_recording"],
            "missing_motion_snapshots": counts["motion_snapshot"],
            "missing_face_snapshots": counts["face_snapshot"],
            "orphan_media_files": len(orphan_files),
            "orphan_media_bytes": orphan_bytes,
            "regenerable_cache_bytes": cache_bytes,
            "per_camera": per_camera,
            "missing_reference_samples": [
                {
                    key: (self._display_path(value) if key == "path" else value)
                    for key, value in item.items() if key not in {"table", "column"}
                }
                for item in references[:MEDIA_SAMPLE_LIMIT]
            ],
            "orphan_media_samples": [self._display_path(path) for path in orphan_files[:MEDIA_SAMPLE_LIMIT]],
            "missing_index_samples": [self._display_path(path) for path in missing_index_files[:MEDIA_SAMPLE_LIMIT]],
            "unindexed_samples": [self._display_path(path) for path in unindexed_files[:MEDIA_SAMPLE_LIMIT]],
        }

    def _recording_camera_id(self, path_value: object) -> str:
        path = self._media_path(path_value)
        try:
            parts = path.resolve().relative_to(self.storage_dir / "recordings").parts
        except ValueError:
            return "Unknown camera"
        return parts[0] if parts else "Unknown camera"

    def _clear_missing_references(self) -> dict[str, int]:
        references, _ = self._media_references()
        references = [reference for reference in references if self._missing(reference["path"])]
        counts = {"events": 0, "motion_audits": 0, "face_observations": 0}
        with self._connect() as connection:
            for reference in references:
                table = str(reference["table"])
                column = str(reference["column"])
                if table not in counts or column not in {"snapshot_path", "recording_path"}:
                    continue
                cursor = connection.execute(
                    f"UPDATE {table} SET {column} = '' WHERE id = ? AND {column} = ?",
                    (int(reference["id"]), str(reference["path"])),
                )
                counts[table] += max(0, cursor.rowcount)
        return {
            "event_media_references_cleared": counts["events"],
            "motion_sample_references_cleared": counts["motion_audits"],
            "face_media_references_cleared": counts["face_observations"],
        }

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @classmethod
    def _directory_size(cls, directory: Path) -> int:
        if not directory.exists():
            return 0
        return sum(cls._file_size(path) for path in directory.rglob("*") if path.is_file())


class StorageMaintenanceRunner:
    """Run one storage reconciliation job at a time outside request threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, object] = {"status": "idle"}

    def status(self) -> dict[str, object]:
        with self._lock:
            return dict(self._state)

    def start(self, factory: Callable[[], StorageReconciler], *, apply: bool) -> dict[str, object]:
        with self._lock:
            if self._state.get("status") == "running":
                raise RuntimeError("storage maintenance is already running")
            self._state = {
                "status": "running",
                "mode": "repair" if apply else "scan",
                "started_at": _utc_now(),
            }
        thread = threading.Thread(
            target=self._run,
            args=(factory, apply),
            name="storage-maintenance",
            daemon=True,
        )
        thread.start()
        return self.status()

    def _run(self, factory: Callable[[], StorageReconciler], apply: bool) -> None:
        try:
            result = factory().run(apply=apply)
        except Exception as error:
            LOGGER.exception("Storage maintenance failed")
            state: dict[str, object] = {
                "status": "failed", "mode": "repair" if apply else "scan",
                "finished_at": _utc_now(), "error": str(error),
            }
        else:
            state = {
                "status": "complete", "mode": "repair" if apply else "scan",
                "finished_at": _utc_now(), "result": result,
            }
        with self._lock:
            self._state = state
