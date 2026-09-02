from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from ..config import CameraConfig, RecordingRetentionConfig
from ..media_storage import path_presence
from ..recording_media import mp4_stream_fingerprint


LOGGER = logging.getLogger("survng.app.recorder")
RECORDING_FINALIZE_GRACE_SECONDS = 2.0
RECORDING_PATH_REBASE_BATCH_SIZE = 1000
RECORDING_LOCATION_BACKFILL_BATCH_SIZE = 250
RECORDING_LOCATION_BACKFILL_VERSION = 1
RECORDING_LOCATION_BACKFILL_TOKEN_KEY = "recording_location_backfill_token"
RECORDING_LOCATION_BACKFILL_CURSOR_KEY = "recording_location_backfill_cursor"


class RecordingIndexMixin:
    def _index_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.index_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_recording_index(self) -> None:
        with self._index_connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recordings (
                    path TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    name TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_at REAL NOT NULL,
                    start_epoch REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    end_epoch REAL NOT NULL,
                    playable INTEGER NOT NULL DEFAULT 1,
                    health_error TEXT NOT NULL DEFAULT '',
                    validated INTEGER NOT NULL DEFAULT 0,
                    stream_fingerprint TEXT NOT NULL DEFAULT '',
                    fingerprint_checked INTEGER NOT NULL DEFAULT 0,
                    location_id TEXT NOT NULL DEFAULT 'default'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS recordings_range ON recordings(camera_id, source, start_epoch, end_epoch)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS recordings_retention_expiry ON recordings(camera_id, source, end_epoch)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recording_index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(recordings)")}
            if "playable" not in columns:
                connection.execute("ALTER TABLE recordings ADD COLUMN playable INTEGER NOT NULL DEFAULT 1")
            if "health_error" not in columns:
                connection.execute("ALTER TABLE recordings ADD COLUMN health_error TEXT NOT NULL DEFAULT ''")
            if "validated" not in columns:
                connection.execute("ALTER TABLE recordings ADD COLUMN validated INTEGER NOT NULL DEFAULT 0")
            if "stream_fingerprint" not in columns:
                connection.execute("ALTER TABLE recordings ADD COLUMN stream_fingerprint TEXT NOT NULL DEFAULT ''")
            if "fingerprint_checked" not in columns:
                connection.execute("ALTER TABLE recordings ADD COLUMN fingerprint_checked INTEGER NOT NULL DEFAULT 0")
                connection.execute(
                    """
                    UPDATE recordings
                    SET fingerprint_checked = 1
                    WHERE stream_fingerprint != ''
                    """
                )
            if "location_id" not in columns:
                connection.execute(
                    "ALTER TABLE recordings ADD COLUMN location_id TEXT NOT NULL DEFAULT 'default'"
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS recordings_fingerprint_pending
                ON recordings(start_epoch DESC)
                WHERE fingerprint_checked = 0
                """
            )

    def _rebase_recording_index_paths(self) -> None:
        """Rebase absolute media paths when the same index is used under a new mount."""
        if len(self.recording_roots) > 1:
            # In a storage pool, absolute legacy paths remain valid and a root
            # change must be handled by an explicit drain/migration operation.
            # Rebasing every row onto the first root would corrupt placement.
            return
        recordings_root = self.recordings_dir.resolve()
        root_value = str(recordings_root)
        root_prefix = f"{root_value}{os.sep}"
        root_upper_bound = f"{root_prefix}\U0010ffff"
        with self._index_connection() as connection:
            metadata = connection.execute(
                "SELECT value FROM recording_index_metadata WHERE key = 'recordings_root'"
            ).fetchone()
            if metadata is not None and str(metadata["value"]) == root_value:
                return
            # Older indexes predate the metadata marker. Check the primary-key
            # range in SQLite rather than materializing millions of paths in
            # Python on every service start.
            mismatch = connection.execute(
                """
                SELECT path FROM recordings
                WHERE path < ? OR path >= ?
                LIMIT 1
                """,
                (root_prefix, root_upper_bound),
            ).fetchone()
            if mismatch is None:
                connection.execute(
                    """
                    INSERT INTO recording_index_metadata(key, value)
                    VALUES ('recordings_root', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (root_value,),
                )
                return
            previous_root = (
                Path(str(metadata["value"])) if metadata is not None else None
            )
            changed = 0
            last_rowid = 0
            while True:
                rows = connection.execute(
                    """
                    SELECT rowid, path FROM recordings
                    WHERE rowid > ?
                    ORDER BY rowid
                    LIMIT ?
                    """,
                    (last_rowid, RECORDING_PATH_REBASE_BATCH_SIZE),
                ).fetchall()
                if not rows:
                    break
                last_rowid = int(rows[-1]["rowid"])
                updates: list[tuple[str, str]] = []
                for row in rows:
                    old_path = str(row["path"])
                    if root_prefix <= old_path < root_upper_bound:
                        continue
                    candidate = self._rebased_recording_path(
                        old_path,
                        recordings_root,
                        previous_root,
                    )
                    if candidate is not None and str(candidate) != old_path:
                        updates.append((str(candidate), old_path))
                if not updates:
                    continue
                before = connection.total_changes
                # A target may already exist when two old roots indexed the
                # same segment. Preserve that row, then discard the duplicate
                # source path. Processing bounded batches avoids a multi-
                # million-entry Python set during a real mount migration.
                connection.executemany(
                    "UPDATE OR IGNORE recordings SET path = ? WHERE path = ?",
                    updates,
                )
                connection.executemany(
                    "DELETE FROM recordings WHERE path = ?",
                    [(old_path,) for _new_path, old_path in updates],
                )
                changed += connection.total_changes - before
                connection.commit()
            connection.execute(
                """
                INSERT INTO recording_index_metadata(key, value)
                VALUES ('recordings_root', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (root_value,),
            )
        if changed:
            LOGGER.info(
                "Rebased %d recording index paths for storage root %s",
                changed,
                self.storage_dir,
            )

    def _recording_location_backfill_token(self) -> str:
        if self.media_storage is None:
            return ""
        topology = sorted(
            (
                str(location.id),
                str(Path(location.path).expanduser().resolve(strict=False)),
            )
            for location in self.media_storage.config.locations
            if "recordings" in location.roles
        )
        serialized = json.dumps(topology, separators=(",", ":"), ensure_ascii=True)
        fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"v{RECORDING_LOCATION_BACKFILL_VERSION}:{fingerprint}"

    def _backfill_recording_location_id_batch(
        self,
        *,
        limit: int = RECORDING_LOCATION_BACKFILL_BATCH_SIZE,
    ) -> int:
        """Classify one bounded legacy recording cohort by configured storage root."""
        if self.media_storage is None or not self.media_storage.config.locations:
            return 0
        bounded_limit = max(1, min(5000, int(limit)))
        expected_token = self._recording_location_backfill_token()
        with self._location_backfill_lock:
            with self._index_connection() as connection:
                stored_token = connection.execute(
                    "SELECT value FROM recording_index_metadata WHERE key = ?",
                    (RECORDING_LOCATION_BACKFILL_TOKEN_KEY,),
                ).fetchone()
                cursor_row = connection.execute(
                    "SELECT value FROM recording_index_metadata WHERE key = ?",
                    (RECORDING_LOCATION_BACKFILL_CURSOR_KEY,),
                ).fetchone()
                token_matches = (
                    stored_token is not None
                    and str(stored_token["value"]) == expected_token
                )
                try:
                    cursor = int(cursor_row["value"]) if token_matches and cursor_row is not None else 0
                except (TypeError, ValueError):
                    cursor = 0
                if token_matches and cursor < 0:
                    return 0
                if not token_matches:
                    connection.execute(
                        """
                        INSERT INTO recording_index_metadata(key, value) VALUES (?, ?)
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value
                        """,
                        (RECORDING_LOCATION_BACKFILL_TOKEN_KEY, expected_token),
                    )
                    connection.execute(
                        """
                        INSERT INTO recording_index_metadata(key, value) VALUES (?, '0')
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value
                        """,
                        (RECORDING_LOCATION_BACKFILL_CURSOR_KEY,),
                    )
                rows = connection.execute(
                    """
                    SELECT rowid, path FROM recordings
                    WHERE rowid > ? AND location_id = 'default'
                    ORDER BY rowid LIMIT ?
                    """,
                    (cursor, bounded_limit),
                ).fetchall()

            updates: list[tuple[str, int]] = []
            for row in rows:
                location_id = self.media_storage.location_id_for(
                    Path(str(row["path"])),
                    "recordings",
                )
                if location_id is not None and location_id != "default":
                    updates.append((location_id, int(row["rowid"])))
            next_cursor = -1 if len(rows) < bounded_limit else int(rows[-1]["rowid"])
            with self._index_connection() as connection:
                if updates:
                    connection.executemany(
                        "UPDATE recordings SET location_id = ? "
                        "WHERE rowid = ? AND location_id = 'default'",
                        updates,
                    )
                connection.execute(
                    """
                    INSERT INTO recording_index_metadata(key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (RECORDING_LOCATION_BACKFILL_CURSOR_KEY, str(next_cursor)),
                )
            return len(rows)

    @staticmethod
    def _rebased_recording_path(
        old_path: str,
        recordings_root: Path,
        previous_root: Path | None,
    ) -> Path | None:
        path = Path(old_path)
        if previous_root is not None:
            try:
                relative = path.relative_to(previous_root)
            except ValueError:
                pass
            else:
                if relative.parts and all(
                    part not in {"", ".", ".."} for part in relative.parts
                ):
                    return recordings_root.joinpath(*relative.parts)

        # Legacy indexes have no root marker. Prefer the component whose
        # suffix matches the current camera/source/date layout. This remains
        # correct when an ancestor directory is itself named "recordings".
        parts = path.parts
        fallback: tuple[str, ...] | None = None
        for index, part in enumerate(parts):
            if part != "recordings":
                continue
            relative_parts = parts[index + 1 :]
            if len(relative_parts) < 4 or any(
                value in {"", ".", ".."} for value in relative_parts
            ):
                continue
            if fallback is None:
                fallback = relative_parts
            if len(relative_parts) >= 5 and relative_parts[1] in {"main", "live"}:
                return recordings_root.joinpath(*relative_parts)
        if fallback is None:
            return None
        return recordings_root.joinpath(*fallback)

    def recent_files(self, camera_id: str, limit: int = 20, source: str = "main") -> list[str]:
        source = "main" if source == "main" else "live"
        search_dirs = self._recording_search_dirs(camera_id, source)
        files = []
        for camera_dir in search_dirs:
            if camera_dir.exists():
                files.extend(camera_dir.glob("????-??-??/??/*.mp4"))
        files = sorted(
            set(files),
            key=lambda path: self.recording_start_epoch(path) or path.stat().st_mtime,
        )
        return [str(path) for path in files[-limit:]][::-1]

    def recording_rows(self, camera_id: str, limit: int = 1000, source: str = "main") -> list[dict]:
        """Return recent playable recordings from the local index only."""
        source = "main" if source == "main" else "live"
        with self._index_connection() as connection:
            rows = connection.execute(
                """
                SELECT path, name, size_bytes, modified_at, start_epoch, duration_seconds,
                       end_epoch, source, playable, health_error, validated,
                       stream_fingerprint, fingerprint_checked, location_id
                FROM recordings
                WHERE camera_id = ? AND source = ? AND playable = 1
                ORDER BY start_epoch DESC LIMIT ?
                """,
                (camera_id, source, max(1, int(limit))),
            ).fetchall()
        payloads = [dict(row) for row in reversed(rows)]
        for row in payloads:
            row["start_at"] = datetime.fromtimestamp(
                float(row["start_epoch"]), timezone.utc
            ).isoformat()
        return payloads

    def recording_rows_between(
        self,
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
        *,
        discover_missing: bool = True,
    ) -> list[dict]:
        source = "main" if source == "main" else "live"
        def indexed_rows() -> list[dict]:
            with self._index_connection() as connection:
                indexed = connection.execute(
                    """
                    SELECT path, name, size_bytes, modified_at, start_epoch, duration_seconds, end_epoch, source,
                           playable, health_error, stream_fingerprint, fingerprint_checked,
                           location_id
                    FROM recordings
                    WHERE camera_id = ? AND source = ? AND playable = 1 AND end_epoch > ? AND start_epoch < ?
                    ORDER BY start_epoch
                    """,
                    (camera_id, source, start_epoch, end_epoch),
                ).fetchall()
            return [dict(row) for row in indexed]

        rows = indexed_rows()
        if not discover_missing:
            for row in rows:
                row["start_at"] = datetime.fromtimestamp(float(row["start_epoch"]), timezone.utc).isoformat()
            return rows

        stale_paths = {str(row["path"]) for row in rows if not Path(str(row["path"])).is_file()}
        if stale_paths:
            self._delete_index_paths(list(stale_paths))
            rows = [row for row in rows if str(row["path"]) not in stale_paths]

        start_date = datetime.fromtimestamp(start_epoch).date() - timedelta(days=1)
        end_date = datetime.fromtimestamp(end_epoch).date() + timedelta(days=1)
        files: list[Path] = []
        current_date = start_date
        while current_date <= end_date:
            for camera_dir in self._recording_search_dirs(camera_id, source):
                day_dir = camera_dir / current_date.isoformat()
                files.extend(self._glob_mp4s(day_dir, "??/*.mp4"))
            current_date += timedelta(days=1)
        with self._index_connection() as connection:
            indexed_paths = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT path FROM recordings
                    WHERE camera_id = ? AND source = ? AND end_epoch > ? AND start_epoch < ?
                    """,
                    (camera_id, source, start_epoch, end_epoch),
                )
            }
        relevant_files = [
            path
            for path in files
            if (file_start := self.recording_start_epoch(path)) is not None
            and file_start < end_epoch
            and file_start + self.segment_seconds > start_epoch
        ]
        with self._lock:
            active_item = self.processes.get((camera_id, source))
            recorder_active = active_item is not None and active_item[0].poll() is None
        if recorder_active and relevant_files:
            now = datetime.now()
            current_hours = set(self._recording_hour_dirs(camera_id, source, now))
            active_tail = max(relevant_files, key=lambda path: self.recording_start_epoch(path) or 0.0)
            if active_tail.parent in current_hours:
                relevant_files.remove(active_tail)
        if any(str(path) not in indexed_paths for path in relevant_files):
            discovered = [
                row for row in self._recording_rows_for_files(camera_id, source, files)
                if row.get("start_epoch") is not None
                and row.get("end_epoch") is not None
                and float(row["end_epoch"]) > start_epoch
                and float(row["start_epoch"]) < end_epoch
            ]
            self._store_recording_rows(camera_id, source, discovered)
            self.queue_recording_validation(discovered)
            rows = indexed_rows()
        for row in rows:
            row["start_at"] = datetime.fromtimestamp(float(row["start_epoch"]), timezone.utc).isoformat()
        return rows

    def lease_recordings_for_playback(
        self,
        rows: list[dict],
        *,
        ttl_seconds: float = 90.0,
    ) -> None:
        """Keep manifest segments out of retention while clients fetch them."""
        expires_at = time.monotonic() + max(10.0, float(ttl_seconds))
        leased: list[str] = []
        for row in rows:
            raw_path = str(row.get("path") or "")
            if not raw_path:
                continue
            resolved = Path(raw_path).resolve(strict=False)
            if self.media_storage is not None:
                if not self.media_storage.contains(resolved, "recordings"):
                    continue
            else:
                try:
                    resolved.relative_to(self.recordings_dir.resolve())
                except ValueError:
                    continue
            leased.append(str(resolved))
        if not leased:
            return
        with self._playback_lease_lock:
            self._discard_expired_playback_leases_locked()
            for path in leased:
                if path in self._retention_deletions:
                    continue
                self._playback_leases[path] = max(
                    expires_at,
                    self._playback_leases.get(path, 0.0),
                )

    def discard_missing_recording_rows(self, rows: list[dict]) -> list[dict]:
        """Remove missing indexed files before a playback manifest advertises them."""
        existing: list[dict] = []
        stale_paths: list[str] = []
        for row in rows:
            path = str(row.get("path") or "")
            if path and Path(path).is_file():
                existing.append(row)
            elif path:
                stale_paths.append(path)
        self._delete_index_paths(stale_paths)
        return existing

    def _protected_recording_paths(self) -> set[str]:
        protected = set(self._external_protected_recording_paths())
        with self._playback_lease_lock:
            self._discard_expired_playback_leases_locked()
            protected.update(self._playback_leases)
        return protected

    def _discard_expired_playback_leases_locked(self) -> None:
        now = time.monotonic()
        expired = [
            path for path, expires_at in self._playback_leases.items()
            if expires_at <= now
        ]
        for path in expired:
            self._playback_leases.pop(path, None)

    def _delete_recording_for_retention(self, path: Path) -> bool:
        """Atomically recheck playback ownership immediately before unlink."""
        resolved = str(path.resolve(strict=False))
        with self._playback_lease_lock:
            self._discard_expired_playback_leases_locked()
            if resolved in self._playback_leases or resolved in self._retention_deletions:
                return False
            self._retention_deletions.add(resolved)
        try:
            path.unlink()
            return True
        finally:
            with self._playback_lease_lock:
                self._retention_deletions.discard(resolved)

    def recording_availability_between(
        self,
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
        *,
        discover_missing: bool = True,
    ) -> dict:
        source = "main" if source == "main" else "live"
        rows = [
            {"start_epoch": row["start_epoch"], "end_epoch": row["end_epoch"]}
            for row in self.recording_rows_between(
                camera_id,
                start_epoch,
                end_epoch,
                source,
                discover_missing=discover_missing,
            )
            if int(row.get("size_bytes") or 0) > 1024
        ]
        return {
            "ranges": self._merge_availability_rows(
                rows,
                start_epoch,
                end_epoch,
                gap_tolerance=min(5.0, max(0.25, self.segment_seconds / 2)),
            ),
            "segment_count": len(rows),
        }

    def recording_grid_availability_between(
        self,
        camera_ids: list[str],
        start_epoch: float,
        end_epoch: float,
    ) -> dict[str, dict[str, dict]]:
        """Read compact Main/Sub availability for multiple cameras in one index query."""
        unique_camera_ids = list(dict.fromkeys(str(camera_id) for camera_id in camera_ids if camera_id))
        result = {
            camera_id: {
                source: {"ranges": [], "segment_count": 0}
                for source in ("main", "live")
            }
            for camera_id in unique_camera_ids
        }
        if not unique_camera_ids:
            return result
        placeholders = ",".join("?" for _ in unique_camera_ids)
        with self._index_connection() as connection:
            indexed = connection.execute(
                f"""
                SELECT camera_id, source, start_epoch, end_epoch
                FROM recordings
                WHERE camera_id IN ({placeholders})
                    AND source IN ('main', 'live')
                    AND playable = 1
                    AND size_bytes > 1024
                    AND end_epoch > ?
                    AND start_epoch < ?
                ORDER BY camera_id, source, start_epoch
                """,
                (*unique_camera_ids, start_epoch, end_epoch),
            ).fetchall()
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in indexed:
            key = (str(row["camera_id"]), str(row["source"]))
            grouped.setdefault(key, []).append(dict(row))
        gap_tolerance = min(5.0, max(0.25, self.segment_seconds / 2))
        for (camera_id, source), rows in grouped.items():
            result[camera_id][source] = {
                "ranges": self._merge_availability_rows(
                    rows,
                    start_epoch,
                    end_epoch,
                    gap_tolerance=gap_tolerance,
                ),
                "segment_count": len(rows),
            }
        return result

    def _log_index_exception(self, key: str, message: str, *args: object) -> None:
        now = time.monotonic()
        last = getattr(self, "_index_error_log_at", {}).get(key, 0.0)
        if now - last < 60.0:
            LOGGER.debug(message, *args, exc_info=True)
            return
        log_at = getattr(self, "_index_error_log_at", None)
        if isinstance(log_at, dict):
            log_at[key] = now
        LOGGER.exception(message, *args)

    @staticmethod
    def _glob_mp4s(directory: Path, pattern: str) -> list[Path]:
        try:
            return list(directory.glob(pattern))
        except OSError:
            return []

    def _recording_hour_dirs(
        self,
        camera_id: str,
        source: str,
        when: datetime,
    ) -> list[Path]:
        relative = Path(when.strftime("%Y-%m-%d")) / when.strftime("%H")
        return [
            search_dir / relative
            for search_dir in self._recording_search_dirs(camera_id, source)
        ]

    def _recent_hour_recording_files(
        self,
        camera_id: str,
        source: str,
        *,
        after_epoch: float | None = None,
        hours_back: tuple[int, ...] = (1, 0),
    ) -> list[Path]:
        now = datetime.now()
        files: list[Path] = []
        seen: set[Path] = set()
        for offset in hours_back:
            for hour_dir in self._recording_hour_dirs(
                camera_id,
                source,
                now - timedelta(hours=offset),
            ):
                for path in self._glob_mp4s(hour_dir, "*.mp4"):
                    if path in seen:
                        continue
                    if after_epoch is not None and (
                        self.recording_start_epoch(path) or 0.0
                    ) < after_epoch:
                        continue
                    seen.add(path)
                    files.append(path)
        return files

    def refresh_recording_edge(
        self,
        camera_id: str,
        source: str,
        after_epoch: float,
    ) -> int:
        """Index completed segments near a live playback edge without probing them."""
        source = "main" if source == "main" else "live"
        cutoff = after_epoch - max(5.0, self.segment_seconds * 2)
        files = self._recent_hour_recording_files(
            camera_id,
            source,
            after_epoch=cutoff,
        )
        rows = self._recording_rows_for_files(camera_id, source, files)
        self._store_recording_rows(camera_id, source, rows)
        return len(rows)

    @staticmethod
    def _merge_availability_rows(
        rows: list[dict],
        start_epoch: float,
        end_epoch: float,
        gap_tolerance: float = 0.25,
    ) -> list[dict]:
        ranges: list[dict] = []
        for row in rows:
            try:
                row_start = max(start_epoch, float(row["start_epoch"]))
                row_end = min(end_epoch, float(row["end_epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
            if row_end <= row_start:
                continue
            if ranges and row_start <= float(ranges[-1]["end_epoch"]) + gap_tolerance:
                current = ranges[-1]
                current["end_epoch"] = max(float(current["end_epoch"]), row_end)
                current["duration_seconds"] = float(current["end_epoch"]) - float(current["start_epoch"])
                current["segment_count"] = int(current["segment_count"]) + 1
                continue
            ranges.append({
                "start_epoch": row_start,
                "end_epoch": row_end,
                "duration_seconds": row_end - row_start,
                "segment_count": 1,
            })
        return ranges

    def _recording_rows_for_files(self, camera_id: str, source: str, files: list[Path]) -> list[dict]:
        files = list({path for path in files if path.is_file()})
        def file_order(path: Path) -> float:
            start_epoch = self.recording_start_epoch(path)
            if start_epoch is not None:
                return start_epoch
            try:
                return path.stat().st_mtime
            except OSError:
                return float("inf")

        files.sort(key=file_order)
        all_files = files
        with self._lock:
            item = self.processes.get((camera_id, source))
            recorder_active = item is not None and item[0].poll() is None
        if recorder_active and files and self._recording_file_may_be_active(files[-1]):
            files = files[:-1]
        rows: list[dict] = []
        for index, file_path in enumerate(files):
            try:
                stat = file_path.stat()
            except OSError:
                continue
            start_epoch = self.recording_start_epoch(file_path)
            if start_epoch is None:
                continue
            next_start_epoch = self.recording_start_epoch(all_files[index + 1]) if index + 1 < len(all_files) else None
            duration_seconds = self.segment_seconds
            if start_epoch is not None and next_start_epoch is not None:
                duration_seconds = max(1.0, min(self.segment_seconds, next_start_epoch - start_epoch))
            rows.append(
                {
                    "path": str(file_path),
                    "name": file_path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "start_epoch": start_epoch,
                    "start_at": (
                        datetime.fromtimestamp(start_epoch, timezone.utc).isoformat()
                        if start_epoch is not None
                        else ""
                    ),
                    "duration_seconds": duration_seconds,
                    "end_epoch": start_epoch + duration_seconds,
                    "source": source,
                    "location_id": (
                        self.media_storage.location_id_for(file_path, "recordings")
                        if self.media_storage is not None
                        else "default"
                    ) or "default",
                }
            )
        return rows

    def _recording_file_may_be_active(self, path: Path, now_epoch: float | None = None) -> bool:
        now_epoch = time.time() if now_epoch is None else now_epoch
        start_epoch = self.recording_start_epoch(path)
        if start_epoch is not None:
            return now_epoch < start_epoch + self.segment_seconds + RECORDING_FINALIZE_GRACE_SECONDS
        try:
            return now_epoch - path.stat().st_mtime < self.segment_seconds + RECORDING_FINALIZE_GRACE_SECONDS
        except OSError:
            return False

    def _recording_file_is_stable(self, path: Path, now_epoch: float | None = None) -> bool:
        """Return whether FFmpeg has had enough time to finalize a segment."""
        now_epoch = time.time() if now_epoch is None else now_epoch
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            return False
        start_epoch = self.recording_start_epoch(path)
        if (
            start_epoch is not None
            and now_epoch < start_epoch + self.segment_seconds + RECORDING_FINALIZE_GRACE_SECONDS
        ):
            return False
        return now_epoch - modified_at >= RECORDING_FINALIZE_GRACE_SECONDS

    def _store_recording_rows(self, camera_id: str, source: str, rows: list[dict], validate_new: bool = False) -> None:
        if not rows:
            return
        if validate_new:
            with self._index_connection() as connection:
                existing = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT path FROM recordings WHERE path IN ({})".format(",".join("?" for _ in rows)),
                        [row["path"] for row in rows],
                    )
                }
            validated_rows: list[dict] = []
            for row in rows:
                if row["path"] not in existing:
                    duration, error = self._probe_recording(Path(row["path"]))
                    if error.startswith("probe failed") or error.startswith("probe metadata"):
                        continue
                    row["playable"] = not error
                    row["health_error"] = error
                    row["validated"] = True
                    row["stream_fingerprint"] = mp4_stream_fingerprint(Path(row["path"]))
                    row["fingerprint_checked"] = True
                    if duration is not None and row.get("start_epoch") is not None:
                        row["duration_seconds"] = duration
                        row["end_epoch"] = float(row["start_epoch"]) + duration
                validated_rows.append(row)
            rows = validated_rows
            if not rows:
                return
        values = [
            (
                row["path"], camera_id, source, row["name"], row["size_bytes"], row["modified_at"],
                row["start_epoch"], row["duration_seconds"], row["end_epoch"],
                1 if row.get("playable", True) else 0, str(row.get("health_error") or ""),
                1 if row.get("validated", False) else 0,
                str(row.get("stream_fingerprint") or ""),
                1 if row.get("fingerprint_checked", False) else 0,
                str(row.get("location_id") or "default"),
            )
            for row in rows
        ]
        with self._index_connection() as connection:
            connection.executemany(
                """
                INSERT INTO recordings(path, camera_id, source, name, size_bytes, modified_at, start_epoch,
                                       duration_seconds, end_epoch, playable, health_error, validated,
                                       stream_fingerprint, fingerprint_checked, location_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size_bytes=excluded.size_bytes,
                    modified_at=excluded.modified_at,
                    duration_seconds=excluded.duration_seconds,
                    end_epoch=excluded.end_epoch,
                    stream_fingerprint=CASE
                        WHEN excluded.fingerprint_checked = 1 THEN excluded.stream_fingerprint
                        ELSE recordings.stream_fingerprint
                    END,
                    fingerprint_checked=MAX(recordings.fingerprint_checked, excluded.fingerprint_checked),
                    location_id=excluded.location_id
                """,
                values,
            )

    def start_indexer(self, cameras: list[CameraConfig]) -> None:
        self._rebase_recording_index_paths()
        camera_map = {camera.id: camera for camera in cameras if camera.record or camera.record_sub}
        self._index_stop.clear()
        self._index_wake.clear()
        self.retention.start(cameras)
        if self._index_thread is None or not self._index_thread.is_alive():
            self._index_thread = threading.Thread(
                target=self._recording_index_loop,
                args=(camera_map,),
                name="recording-indexer",
                daemon=True,
            )
            self._index_thread.start()
        if (
            self._index_maintenance_thread is None
            or not self._index_maintenance_thread.is_alive()
        ):
            self._index_maintenance_thread = threading.Thread(
                target=self._recording_index_maintenance_loop,
                args=(camera_map,),
                name="recording-index-maintenance",
                daemon=True,
            )
            self._index_maintenance_thread.start()

    def retention_status(self) -> dict[str, object]:
        return self.retention.status()

    def request_retention_run(self, *, apply: bool = False) -> dict[str, object]:
        return self.retention.request_run(apply=apply)

    def reconfigure_retention(
        self,
        config: RecordingRetentionConfig,
        cameras: list[CameraConfig],
    ) -> None:
        self.retention.reconfigure(config, cameras)

    def _recording_index_loop(self, camera_map: dict[str, CameraConfig]) -> None:
        try:
            self.refresh_recording_index(camera_map, full=False, run_maintenance=False)
        except Exception:
            LOGGER.exception("Initial recording index discovery failed")
        next_discovery = time.monotonic() + 10.0
        while not self._index_stop.is_set():
            self._index_wake.wait(max(0.0, next_discovery - time.monotonic()))
            self._index_wake.clear()
            if self._index_stop.is_set():
                return
            for (camera_id, source), after_epoch in self._take_recording_edge_refreshes().items():
                try:
                    self.refresh_recording_edge(camera_id, source, after_epoch)
                except Exception:
                    self._log_index_exception(
                        f"edge:{camera_id}:{source}",
                        "Near-live recording index discovery failed for %s/%s",
                        camera_id,
                        source,
                    )
            if time.monotonic() < next_discovery:
                continue
            try:
                self.refresh_recording_index(camera_map, full=False, run_maintenance=False)
            except Exception:
                self._log_index_exception(
                    "index",
                    "Recording index discovery failed",
                )
            finally:
                next_discovery = time.monotonic() + 10.0

    def _recording_index_maintenance_loop(self, camera_map: dict[str, CameraConfig]) -> None:
        if self._index_stop.wait(30):
            return
        next_snapshot_size_migration = 0.0
        while not self._index_stop.is_set():
            try:
                self._backfill_recording_location_id_batch()
                now = time.monotonic()
                if (
                    self._migrate_snapshot_sizes is not None
                    and now >= next_snapshot_size_migration
                ):
                    self._migrate_snapshot_sizes(limit=50, write_batch_size=25)
                    next_snapshot_size_migration = now + 30.0
                self._validate_index_batch(limit=20)
                self._backfill_stream_fingerprints(limit=20)
            except Exception:
                LOGGER.exception("Recording index maintenance failed")
            if self._index_stop.wait(5):
                return

    def _reconcile_recording_source(self, camera_id: str, source: str) -> None:
        files = [
            path
            for camera_dir in self._recording_search_dirs(camera_id, source)
            if camera_dir.exists()
            for path in camera_dir.glob("????-??-??/??/*.mp4")
        ]
        rows = self._recording_rows_for_files(camera_id, source, files)
        self._store_recording_rows(camera_id, source, rows)
        self._prune_recording_index(camera_id, source, files)

    def _unavailable_recording_roots(self) -> list[Path]:
        unavailable: list[Path] = []
        for root in self.recording_roots:
            try:
                if not root.is_dir():
                    unavailable.append(root)
            except OSError:
                unavailable.append(root)
        return unavailable

    @staticmethod
    def _path_under_roots(path_value: str, roots: list[Path]) -> bool:
        path = Path(path_value)
        for root in roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                try:
                    path.resolve(strict=False).relative_to(root.resolve(strict=False))
                    return True
                except ValueError:
                    continue
        return False

    def _confirm_missing_index_paths(
        self,
        candidates: set[str],
        *,
        discovered_paths: set[str],
        unavailable_roots: list[Path],
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Split undiscovered index paths into missing, present, and unknown."""
        stale: list[str] = []
        retained_present: list[str] = []
        unknown: list[str] = []
        ordered = sorted(candidates - discovered_paths)
        for index, path in enumerate(ordered, start=1):
            self._raise_if_cancelled(cancel_event)
            if self._path_under_roots(path, unavailable_roots):
                unknown.append(path)
            else:
                presence = path_presence(path)
                if presence == "missing":
                    stale.append(path)
                elif presence == "present":
                    retained_present.append(path)
                else:
                    unknown.append(path)
            if progress and index % 100 == 0:
                progress("Confirming indexed recordings", index, len(ordered))
        return stale, retained_present, unknown

    def storage_index_health(
        self,
        *,
        full: bool = False,
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
        quick_index_limit: int = 1000,
        quick_hours: int = 6,
    ) -> dict[str, object]:
        """Compare recording files with the index using bounded or exhaustive discovery."""
        unavailable_roots = self._unavailable_recording_roots()
        if full:
            # Freeze the index side before walking remote storage. A full NFS walk
            # can take hours while the live index keeps receiving new segments.
            # Reading it afterward would misclassify those new rows as files that
            # disappeared merely because their directory was visited earlier.
            with self._index_connection() as connection:
                indexed_total = int(connection.execute("SELECT count(*) FROM recordings").fetchone()[0])
                indexed_rows = connection.execute(
                    "SELECT path, camera_id, source FROM recordings"
                ).fetchall()
            files_by_source = self._recording_files_by_source(cancel_event=cancel_event, progress=progress)
        else:
            files_by_source = self._recent_recording_files_by_source(
                hours=quick_hours,
                cancel_event=cancel_event,
                progress=progress,
            )
            with self._index_connection() as connection:
                indexed_total = int(connection.execute("SELECT count(*) FROM recordings").fetchone()[0])
                indexed_rows = connection.execute(
                    "SELECT path, camera_id, source FROM recordings ORDER BY start_epoch DESC LIMIT ?",
                    (max(1, quick_index_limit),),
                ).fetchall()
        all_disk_paths = {
            str(path)
            for files in files_by_source.values()
            for path in files
        }
        stable_disk_paths = {
            str(path)
            for files in files_by_source.values()
            for path in files
            if not self._recording_file_may_be_active(path)
        }
        indexed_sample = {str(row["path"]) for row in indexed_rows}
        # Never treat a path discovered in this same snapshot as stale, even when a
        # later independent is_file()/stat probe fails on flaky network storage.
        stale, retained_present, unknown = self._confirm_missing_index_paths(
            indexed_sample,
            discovered_paths=all_disk_paths,
            unavailable_roots=unavailable_roots,
            cancel_event=cancel_event,
            progress=progress,
        )
        if full:
            discovery_candidates = stable_disk_paths - indexed_sample
            indexed_for_discovery = indexed_sample | self.indexed_path_subset(discovery_candidates)
        else:
            indexed_for_discovery = self.indexed_path_subset(stable_disk_paths)
        unindexed = sorted(stable_disk_paths - indexed_for_discovery)
        return {
            "recording_files": len(all_disk_paths),
            "recent_recording_files": len(all_disk_paths - stable_disk_paths),
            "indexed_recordings": indexed_total,
            "index_rows_scanned": len(indexed_sample),
            "recording_hours_scanned": None if full else max(1, quick_hours),
            "scan_complete": full and not unavailable_roots and not unknown,
            "missing_index_files": stale,
            "unindexed_files": unindexed,
            "index_rows_retained_present": len(retained_present),
            "index_rows_presence_unknown": len(unknown),
            "unavailable_recording_roots": [str(root) for root in unavailable_roots],
            "files_by_source": files_by_source,
        }

    def reconcile_storage_index(
        self,
        *,
        full: bool = False,
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
        health: dict[str, object] | None = None,
    ) -> dict[str, int]:
        """Synchronize the local index with a recording-storage snapshot.

        Callers that already collected health may supply it so a full repair does
        not walk a large remote recording library a second time.
        """
        if health is None:
            health = self.storage_index_health(full=full, cancel_event=cancel_event, progress=progress)
        files_by_source = health.get("files_by_source", {})
        if not isinstance(files_by_source, dict):
            raise ValueError("recording health is missing its file snapshot")
        snapshot_paths = {
            str(path)
            for files in files_by_source.values()
            for path in files
        }
        added = 0
        unindexed = set(health.get("unindexed_files", []))
        for (camera_id, source), files in files_by_source.items():
            rows = self._recording_rows_for_files(camera_id, source, files)
            if rows:
                self._store_recording_rows(camera_id, source, rows)
                added += sum(1 for row in rows if str(row["path"]) in unindexed)
        stale_candidates = [
            str(path)
            for path in health.get("missing_index_files", [])
            if str(path) not in snapshot_paths
        ]
        confirmed_missing = [
            path for path in stale_candidates if path_presence(path) == "missing"
        ]
        self._delete_index_paths(confirmed_missing)
        return {
            "recordings_reindexed": added,
            "stale_index_rows_removed": len(confirmed_missing),
            "stale_index_rows_skipped": len(stale_candidates) - len(confirmed_missing),
        }

    def _recording_files_by_source(
        self,
        *,
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
    ) -> dict[RecorderKey, list[Path]]:
        """Discover current and legacy recording layouts without relying on config state."""
        grouped: dict[RecorderKey, list[Path]] = {}
        index = 0
        for recording_root in self.recording_roots:
            if not recording_root.exists():
                continue
            for path in recording_root.glob("*/**/*.mp4"):
                index += 1
                self._raise_if_cancelled(cancel_event)
                try:
                    parts = path.relative_to(recording_root).parts
                except ValueError:
                    continue
                if len(parts) == 5 and parts[1] in {"main", "live"}:
                    camera_id, source = parts[0], parts[1]
                elif len(parts) == 4:
                    camera_id, source = parts[0], "main"
                else:
                    continue
                grouped.setdefault((camera_id, source), []).append(path)
                if progress and index % 1000 == 0:
                    progress("Scanning all recording files", index, None)
        return grouped

    def _recent_recording_files_by_source(
        self,
        *,
        hours: int,
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
    ) -> dict[RecorderKey, list[Path]]:
        grouped: dict[RecorderKey, list[Path]] = {}
        camera_dirs = [
            path
            for root in self.recording_roots if root.exists()
            for path in root.iterdir() if path.is_dir()
        ]
        targets = [datetime.now() - timedelta(hours=offset) for offset in range(max(1, hours) + 1)]
        total = max(1, len(camera_dirs) * len(targets) * 2)
        checked = 0
        for camera_dir in camera_dirs:
            self._raise_if_cancelled(cancel_event)
            for source in ("main", "live"):
                search_roots = [camera_dir / source]
                if source == "main":
                    search_roots.append(camera_dir)
                files: list[Path] = []
                for target in targets:
                    for root in search_roots:
                        self._raise_if_cancelled(cancel_event)
                        hour_dir = root / target.strftime("%Y-%m-%d") / target.strftime("%H")
                        if hour_dir.exists():
                            files.extend(hour_dir.glob("*.mp4"))
                    checked += 1
                    if progress and checked % 10 == 0:
                        progress("Scanning recent recording folders", min(checked, total), total)
                if files:
                    grouped[(camera_dir.name, source)] = list(set(files))
        return grouped

    def indexed_path_subset(self, paths: set[str]) -> set[str]:
        """Return paths present in the local index without touching media storage."""
        if not paths:
            return set()
        found: set[str] = set()
        values = list(paths)
        with self._index_connection() as connection:
            for offset in range(0, len(values), 500):
                batch = values[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                found.update(
                    str(row[0])
                    for row in connection.execute(
                        f"SELECT path FROM recordings WHERE path IN ({placeholders})",
                        batch,
                    )
                )
        return found

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("storage maintenance was cancelled")

    def refresh_recording_index(
        self,
        camera_map: dict[str, CameraConfig],
        full: bool = False,
        run_maintenance: bool = True,
    ) -> None:
        wanted = self._wanted_keys(camera_map)
        for camera_id, source in wanted:
            try:
                if full:
                    files = [
                        path
                        for search_dir in self._recording_search_dirs(camera_id, source)
                        for path in self._glob_mp4s(search_dir, "????-??-??/??/*.mp4")
                    ]
                else:
                    files = self._recent_hour_recording_files(camera_id, source)
                rows = self._recording_rows_for_files(camera_id, source, files)
                self._store_recording_rows(camera_id, source, rows)
                if full:
                    self._prune_recording_index(camera_id, source, files)
            except Exception:
                self._log_index_exception(
                    f"discover:{camera_id}:{source}",
                    "Recording index discovery failed for %s/%s",
                    camera_id,
                    source,
                )
        if run_maintenance:
            self._prune_missing_index_rows()
            self._validate_index_batch()
            self._backfill_stream_fingerprints()

    def queue_stream_fingerprints(self, rows: list[dict]) -> None:
        with self._fingerprint_lock:
            for row in rows:
                if int(row.get("fingerprint_checked") or 0):
                    continue
                path = str(row.get("path") or "")
                if path and path not in self._fingerprint_pending_set:
                    self._fingerprint_pending.append(path)
                    self._fingerprint_pending_set.add(path)

    def queue_recording_validation(self, rows: list[dict]) -> None:
        candidates = [str(row.get("path") or "") for row in rows if not int(row.get("validated") or 0)]
        candidates = [path for path in candidates if path]
        if not candidates:
            return
        eligible: set[str] = set()
        with self._index_connection() as connection:
            for offset in range(0, len(candidates), 500):
                batch = candidates[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                eligible.update(
                    str(row[0])
                    for row in connection.execute(
                        f"SELECT path FROM recordings WHERE validated = 0 AND path IN ({placeholders})",
                        batch,
                    )
                )
        with self._validation_lock:
            for path in candidates:
                if path in eligible and path not in self._validation_pending_set:
                    self._validation_pending.append(path)
                    self._validation_pending_set.add(path)

    def _backfill_stream_fingerprints(
        self,
        limit: int = 20,
        *,
        discover_unqueued: bool = False,
    ) -> int:
        batch_limit = max(1, limit)
        paths: list[str] = []
        with self._fingerprint_lock:
            while self._fingerprint_pending and len(paths) < batch_limit:
                path = self._fingerprint_pending.popleft()
                self._fingerprint_pending_set.discard(path)
                paths.append(path)
        if discover_unqueued and len(paths) < batch_limit:
            with self._index_connection() as connection:
                paths.extend(
                    str(row[0])
                    for row in connection.execute(
                        """
                        SELECT path FROM recordings
                        WHERE fingerprint_checked = 0
                        ORDER BY start_epoch DESC LIMIT ?
                        """,
                        (batch_limit - len(paths),),
                    )
                    if str(row[0]) not in paths
                )
        updated = 0
        for path_value in paths:
            if self._index_stop.is_set():
                break
            path = Path(path_value)
            if not path.is_file():
                self._delete_index_paths([path_value])
                continue
            fingerprint = mp4_stream_fingerprint(path)
            with self._index_connection() as connection:
                connection.execute(
                    """
                    UPDATE recordings
                    SET stream_fingerprint = ?, fingerprint_checked = 1
                    WHERE path = ?
                    """,
                    (fingerprint, path_value),
                )
            updated += 1
        return updated

    def _prune_recording_index(self, camera_id: str, source: str, files: list[Path]) -> None:
        existing_paths = {str(path) for path in files}
        with self._index_connection() as connection:
            indexed_paths = [
                str(row[0])
                for row in connection.execute(
                    "SELECT path FROM recordings WHERE camera_id = ? AND source = ?",
                    (camera_id, source),
                )
            ]
            stale = [(path,) for path in indexed_paths if path not in existing_paths]
            if stale:
                connection.executemany("DELETE FROM recordings WHERE path = ?", stale)

    def _delete_index_paths(self, paths: list[str]) -> None:
        if not paths:
            return
        with self._index_connection() as connection:
            connection.executemany("DELETE FROM recordings WHERE path = ?", [(path,) for path in paths])

    def _prune_missing_index_rows(self, limit: int = 500) -> int:
        with self._index_connection() as connection:
            rows = connection.execute(
                "SELECT rowid, path FROM recordings WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (self._prune_cursor, max(1, limit)),
            ).fetchall()
            if not rows and self._prune_cursor:
                self._prune_cursor = 0
                rows = connection.execute(
                    "SELECT rowid, path FROM recordings ORDER BY rowid LIMIT ?",
                    (max(1, limit),),
                ).fetchall()
        if not rows:
            return 0
        self._prune_cursor = int(rows[-1]["rowid"])
        stale_paths = [
            str(row["path"])
            for row in rows
            if path_presence(str(row["path"])) == "missing"
        ]
        self._delete_index_paths(stale_paths)
        return len(stale_paths)

    def _validate_index_batch(
        self,
        limit: int = 20,
        *,
        discover_unqueued: bool = False,
    ) -> int:
        batch_limit = max(1, limit)
        paths: list[str] = []
        with self._validation_lock:
            while self._validation_pending and len(paths) < batch_limit:
                path = self._validation_pending.popleft()
                self._validation_pending_set.discard(path)
                paths.append(path)
        if discover_unqueued and len(paths) < batch_limit:
            with self._index_connection() as connection:
                paths.extend(
                    str(row[0])
                    for row in connection.execute(
                        """
                        SELECT path FROM recordings
                        WHERE validated = 0
                        ORDER BY start_epoch DESC LIMIT ?
                        """,
                        (batch_limit - len(paths),),
                    )
                    if str(row[0]) not in paths
                )
        validated = 0
        for path_value in paths:
            if self._index_stop.is_set():
                break
            path = Path(path_value)
            if not path.is_file():
                self._delete_index_paths([str(path)])
                continue
            if not self._recording_file_is_stable(path):
                # Keep the row unvalidated. The maintenance loop will retry it
                # after FFmpeg has closed the file and written the MP4 trailer.
                with self._validation_lock:
                    if path_value not in self._validation_pending_set:
                        self._validation_pending.append(path_value)
                        self._validation_pending_set.add(path_value)
                continue
            duration, error = self._probe_recording(path)
            with self._index_connection() as connection:
                if duration is not None:
                    connection.execute(
                        """
                        UPDATE recordings
                        SET duration_seconds = ?, end_epoch = start_epoch + ?, playable = ?, health_error = ?,
                            validated = 1, stream_fingerprint = ?, fingerprint_checked = 1
                        WHERE path = ?
                        """,
                        (
                            duration, duration, 0 if error else 1, error,
                            mp4_stream_fingerprint(path), str(path),
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE recordings SET playable = 0, health_error = ?, validated = 1 WHERE path = ?",
                        (error or "recording validation failed", str(path)),
                    )
            validated += 1
        return validated

    def maintain_historical_metadata(self, limit: int = 20) -> dict[str, int]:
        """Advance legacy recording metadata only during an explicit repair.

        Normal background work drains queues populated by recent indexing,
        playback, and playback-failure recovery. This bounded opt-in path keeps
        old validation and codec fingerprinting available without continuously
        reading the historical recording library.
        """
        safe_limit = max(1, min(100, int(limit)))
        validated = self._validate_index_batch(
            limit=safe_limit,
            discover_unqueued=True,
        )
        fingerprints = self._backfill_stream_fingerprints(
            limit=safe_limit,
            discover_unqueued=True,
        )
        return {
            "recordings_validated": validated,
            "recording_fingerprints_added": fingerprints,
        }

    def _probe_recording(self, path: Path) -> tuple[float | None, str]:
        ffprobe = str(Path(self.ffmpeg_path).with_name("ffprobe")) if Path(self.ffmpeg_path).name == "ffmpeg" else "ffprobe"
        try:
            result = subprocess.run(
                [
                    ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
                    "-show_entries", "format=duration:packet=flags", "-of", "json", str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"probe failed: {exc}"
        if result.returncode != 0:
            return None, f"probe failed: {(result.stderr or 'invalid media').strip()[-240:]}"
        try:
            payload = json.loads(result.stdout or "{}")
            duration = float((payload.get("format") or {}).get("duration") or 0)
            flags = str(((payload.get("packets") or [{}])[0]).get("flags") or "")
        except (ValueError, TypeError, IndexError) as exc:
            return None, f"probe metadata invalid: {exc}"
        if not math.isfinite(duration) or duration <= 0 or duration > max(60.0, self.segment_seconds * 3):
            return None, "recording duration is invalid"
        if "K" not in flags:
            return duration, "recording does not start on a video keyframe"
        return duration, ""

    def schedule_revalidation(self, path: Path, error: str) -> None:
        path_value = str(path)
        with self._index_connection() as connection:
            connection.execute(
                "UPDATE recordings SET playable = 0, health_error = ?, validated = 0 WHERE path = ?",
                (str(error)[-500:], path_value),
            )
        with self._validation_lock:
            if path_value not in self._validation_pending_set:
                self._validation_pending.append(path_value)
                self._validation_pending_set.add(path_value)

    def latest_indexed_row(self, camera_id: str, source: str) -> dict | None:
        source = "main" if source == "main" else "live"
        with self._index_connection() as connection:
            row = connection.execute(
                """
                SELECT path, name, size_bytes, modified_at, start_epoch, duration_seconds, end_epoch, source
                FROM recordings
                WHERE camera_id = ? AND source = ? AND playable = 1
                ORDER BY start_epoch DESC LIMIT 1
                """,
                (camera_id, source),
            ).fetchone()
        return dict(row) if row is not None else None

    def recording_at(self, camera_id: str, epoch: float, limit: int = 1000, source: str = "main") -> dict | None:
        """Return an indexed recording without scanning media storage on the request path."""
        del limit  # Retained for API compatibility with older callers.
        source = "main" if source == "main" else "live"
        with self._index_connection() as connection:
            row = connection.execute(
                """
                SELECT path, name, size_bytes, modified_at, start_epoch, duration_seconds,
                       end_epoch, source, playable, health_error, validated,
                       stream_fingerprint, fingerprint_checked
                FROM recordings
                WHERE camera_id = ? AND source = ? AND playable = 1
                  AND start_epoch <= ? AND end_epoch >= ?
                ORDER BY start_epoch DESC LIMIT 1
                """,
                (camera_id, source, epoch, epoch),
            ).fetchone()
        if row is not None:
            payload = dict(row)
            if Path(str(payload["path"])).is_file():
                return payload
            self._delete_index_paths([str(payload["path"])])
        if abs(time.time() - epoch) <= max(60.0, self.segment_seconds * 6):
            self.request_recording_edge_refresh(camera_id, source, epoch)
        return None

    def request_recording_edge_refresh(self, camera_id: str, source: str, after_epoch: float) -> None:
        """Ask the background indexer to discover a newly finalized near-live segment."""
        key = (camera_id, "main" if source == "main" else "live")
        with self._edge_refresh_lock:
            previous = self._edge_refresh_requests.get(key)
            self._edge_refresh_requests[key] = (
                min(previous, after_epoch) if previous is not None else after_epoch
            )
        self._index_wake.set()

    def _take_recording_edge_refreshes(self) -> dict[RecorderKey, float]:
        with self._edge_refresh_lock:
            requests = self._edge_refresh_requests
            self._edge_refresh_requests = {}
        return requests

    def recording_start_epoch(self, path: Path) -> float | None:
        try:
            value = datetime.strptime(path.stem, "%Y%m%d-%H%M%S%z")
            return value.timestamp()
        except ValueError:
            try:
                value = datetime.strptime(path.stem, "%Y%m%d-%H%M%S")
            except ValueError:
                return None
        return value.astimezone(timezone.utc).timestamp()
