from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


class EventStoreTrackingMixin:
    @staticmethod
    def _result_json_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        try:
            result = json.loads(str(payload.pop("result_json") or "{}"))
        except (json.JSONDecodeError, TypeError):
            result = {}
        payload["result"] = result if isinstance(result, dict) else {}
        return payload

    def save_tracking_comparison(
        self,
        *,
        event_id: int,
        camera_id: str,
        event_created_at: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_event_id = int(event_id)
        if normalized_event_id <= 0:
            raise ValueError("tracking comparison event id must be positive")
        normalized_camera_id = str(camera_id or "").strip()
        if not normalized_camera_id:
            raise ValueError("tracking comparison camera id is required")
        result_json = json.dumps(result, separators=(",", ":"), allow_nan=False)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                insert into tracking_comparisons (
                    event_id, camera_id, event_created_at, result_json, created_at
                ) values (?, ?, ?, ?, ?)
                on conflict(event_id) do update set
                    camera_id = excluded.camera_id,
                    event_created_at = excluded.event_created_at,
                    result_json = excluded.result_json,
                    verdict = '',
                    reviewed_at = null,
                    created_at = excluded.created_at
                """,
                (
                    normalized_event_id,
                    normalized_camera_id,
                    str(event_created_at or ""),
                    result_json,
                    now,
                ),
            )
            row = conn.execute(
                "select * from tracking_comparisons where event_id = ?",
                (normalized_event_id,),
            ).fetchone()
            conn.execute(
                """
                delete from tracking_comparisons
                where camera_id = ? and id not in (
                    select id from tracking_comparisons
                    where camera_id = ?
                    order by created_at desc, id desc
                    limit ?
                )
                """,
                (
                    normalized_camera_id,
                    normalized_camera_id,
                    self.TRACKING_COMPARISON_HISTORY_PER_CAMERA,
                ),
            )
        comparison = self._result_json_row(row)
        if comparison is None:
            raise RuntimeError("tracking comparison could not be persisted")
        return comparison

    def set_tracking_comparison_verdict(
        self,
        comparison_id: int,
        verdict: str,
    ) -> dict[str, Any] | None:
        normalized_verdict = str(verdict or "").strip()
        if normalized_verdict not in self.TRACKING_COMPARISON_VERDICTS:
            raise ValueError("invalid tracking comparison verdict")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "update tracking_comparisons set verdict = ?, reviewed_at = ? where id = ?",
                (normalized_verdict, now, int(comparison_id)),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "select * from tracking_comparisons where id = ?",
                (int(comparison_id),),
            ).fetchone()
        return self._result_json_row(row)

    def tracking_comparison_history(
        self,
        *,
        camera_id: str = "",
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 100))
        normalized_camera_id = str(camera_id or "").strip()
        where = "where camera_id = ?" if normalized_camera_id else ""
        values: list[Any] = [normalized_camera_id] if normalized_camera_id else []
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select * from tracking_comparisons
                {where}
                order by created_at desc, id desc
                limit ?
                """,
                [*values, bounded_limit],
            ).fetchall()
        return [self._result_json_row(row) or {} for row in rows]

    def tracking_comparison_summary(self, *, camera_id: str = "") -> dict[str, Any]:
        normalized_camera_id = str(camera_id or "").strip()
        where = "where camera_id = ?" if normalized_camera_id else ""
        values: tuple[Any, ...] = (normalized_camera_id,) if normalized_camera_id else ()
        with self._connect() as conn:
            rows = conn.execute(
                f"select verdict, count(*) as count from tracking_comparisons {where} group by verdict",
                values,
            ).fetchall()
        counts = {"unreviewed": 0, **{value: 0 for value in self.TRACKING_COMPARISON_VERDICTS}}
        for row in rows:
            key = str(row["verdict"] or "unreviewed")
            if key in counts:
                counts[key] = int(row["count"])
        return {
            "camera_id": normalized_camera_id,
            "total": sum(counts.values()),
            "reviewed": sum(counts[value] for value in self.TRACKING_COMPARISON_VERDICTS),
            "verdicts": counts,
        }
