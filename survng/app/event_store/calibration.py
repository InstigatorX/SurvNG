from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


class EventStoreCalibrationMixin:
    @staticmethod
    def _calibration_run_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        for column, target, fallback in (
            ("camera_ids_json", "camera_ids", []),
            ("result_json", "result", {}),
        ):
            if column not in payload:
                payload[target] = fallback
                continue
            try:
                payload[target] = json.loads(str(payload.pop(column) or ""))
            except (json.JSONDecodeError, TypeError):
                payload[target] = fallback
        return payload

    def create_calibration_run(
        self,
        *,
        mode: str,
        camera_ids: list[str],
        configuration_fingerprint: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into calibration_runs (
                    status, mode, camera_ids_json, configuration_fingerprint,
                    created_at, updated_at
                ) values ('queued', ?, ?, ?, ?, ?)
                """,
                (
                    mode,
                    json.dumps(camera_ids, separators=(",", ":")),
                    configuration_fingerprint,
                    now,
                    now,
                ),
            )
            run_id = int(cursor.lastrowid)
        return self.get_calibration_run(run_id) or {}

    def update_calibration_run(
        self,
        run_id: int,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        completed_at = now if status in {"completed", "failed", "interrupted", "cancelled"} else None
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                update calibration_runs
                set status = ?, result_json = coalesce(?, result_json), error = ?,
                    updated_at = ?, completed_at = coalesce(?, completed_at)
                where id = ?
                """,
                (
                    status,
                    (
                        json.dumps(result, separators=(",", ":"), allow_nan=False)
                        if result is not None
                        else None
                    ),
                    error,
                    now,
                    completed_at,
                    int(run_id),
                ),
            )
        return self.get_calibration_run(run_id) or {}

    def get_calibration_run(self, run_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from calibration_runs where id = ?",
                (int(run_id),),
            ).fetchone()
        return self._calibration_run_row(row)

    def calibration_runs(
        self,
        limit: int = 20,
        *,
        include_result: bool = False,
    ) -> list[dict[str, Any]]:
        columns = "*" if include_result else """
            id, status, mode, camera_ids_json, configuration_fingerprint,
            error, created_at, updated_at, completed_at
        """
        with self._connect() as conn:
            rows = conn.execute(
                f"select {columns} from calibration_runs order by created_at desc, id desc limit ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [item for row in rows if (item := self._calibration_run_row(row))]

    def calibration_rollback_change_ids(self, parent_change_set_id: int) -> set[str]:
        """Return source change IDs already reversed by child rollback entries."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                select changes_json from calibration_change_sets
                where parent_change_set_id = ? and action = 'rollback'
                """,
                (int(parent_change_set_id),),
            ).fetchall()
        change_ids: set[str] = set()
        for row in rows:
            try:
                changes = json.loads(str(row["changes_json"] or "[]"))
            except (json.JSONDecodeError, TypeError):
                continue
            for change in changes:
                source_id = str(change.get("source_change_id") or "")
                if source_id:
                    change_ids.add(source_id)
        return change_ids

    @staticmethod
    def _calibration_change_set_row(
        row: sqlite3.Row | None,
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        for column, target, fallback in (
            ("changes_json", "changes", []),
            ("apply_result_json", "apply_result", {}),
            ("evaluation_json", "evaluation", {}),
        ):
            try:
                payload[target] = json.loads(str(payload.pop(column) or ""))
            except (json.JSONDecodeError, TypeError):
                payload[target] = fallback
        return payload

    def create_calibration_change_set(
        self,
        *,
        run_id: int | None,
        parent_change_set_id: int | None,
        action: str,
        status: str,
        evaluation_hours: float,
        configuration_fingerprint_before: str,
        configuration_fingerprint_after: str,
        changes: list[dict[str, Any]],
        apply_result: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into calibration_change_sets (
                    run_id, parent_change_set_id, action, status,
                    evaluation_hours, configuration_fingerprint_before,
                    configuration_fingerprint_after, changes_json,
                    apply_result_json, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    parent_change_set_id,
                    action,
                    status,
                    max(24.0, min(float(evaluation_hours), 168.0)),
                    configuration_fingerprint_before,
                    configuration_fingerprint_after,
                    json.dumps(changes, separators=(",", ":"), allow_nan=False),
                    json.dumps(apply_result, separators=(",", ":"), allow_nan=False),
                    now,
                    now,
                ),
            )
            change_set_id = int(cursor.lastrowid)
        return self.get_calibration_change_set(change_set_id) or {}

    def get_calibration_change_set(
        self,
        change_set_id: int,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from calibration_change_sets where id = ?",
                (int(change_set_id),),
            ).fetchone()
        return self._calibration_change_set_row(row)

    def calibration_change_sets(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from calibration_change_sets order by created_at desc, id desc limit ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [
            item for row in rows
            if (item := self._calibration_change_set_row(row))
        ]

    def update_calibration_evaluation(
        self,
        change_set_id: int,
        evaluation: dict[str, Any],
        *,
        status: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                update calibration_change_sets
                set status = ?, evaluation_json = ?, updated_at = ? where id = ?
                """,
                (
                    status,
                    json.dumps(evaluation, separators=(",", ":"), allow_nan=False),
                    now,
                    int(change_set_id),
                ),
            )
        return self.get_calibration_change_set(change_set_id) or {}

    def update_calibration_change_set_status(
        self,
        change_set_id: int,
        status: str,
    ) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute(
                "update calibration_change_sets set status = ?, updated_at = ? where id = ?",
                (status, datetime.now(timezone.utc).isoformat(), int(change_set_id)),
            )
        return self.get_calibration_change_set(change_set_id) or {}
