"""Durable per-camera ledger of stationary scene-context subjects."""

from __future__ import annotations

import json
import threading
from typing import Any


class EventStoreSceneContextMixin:
    def _init_scene_context_db(self) -> None:
        self._scene_context_cache_lock = threading.Lock()
        self._scene_context_memories: dict[str, Any] = {}
        with self._lock, self._connect() as conn:
            conn.execute(
                """create table if not exists scene_context_subjects (
                    id integer primary key,
                    camera_id text not null,
                    label text not null,
                    box_x1 real not null,
                    box_y1 real not null,
                    box_x2 real not null,
                    box_y2 real not null,
                    first_seen_epoch real not null,
                    last_seen_epoch real not null,
                    sightings_json text not null,
                    last_moved_epoch real
                )"""
            )
            conn.execute(
                "create index if not exists scene_context_subjects_camera_label "
                "on scene_context_subjects(camera_id, label)"
            )

    def scene_context_memory(self, camera_id: str):
        """Return the shared working set for one camera, hydrated once."""
        from ..scene_context_memory import DurableSceneContextMemory

        with self._scene_context_cache_lock:
            memory = self._scene_context_memories.get(camera_id)
            if memory is None:
                memory = DurableSceneContextMemory(self, camera_id)
                self._scene_context_memories[camera_id] = memory
            return memory

    def scene_context_snapshot(self, conn, camera_id: str):
        """Subjects visible to the caller's open transaction, without taking the events lock."""
        from ..scene_context_memory import SceneContextSnapshot, SceneContextSubject

        rows = conn.execute(
            "select id, label, box_x1, box_y1, box_x2, box_y2, first_seen_epoch, "
            "last_seen_epoch, sightings_json, last_moved_epoch "
            "from scene_context_subjects where camera_id=? order by id",
            (camera_id,),
        ).fetchall()
        subjects = []
        for row in rows:
            try:
                sightings = json.loads(row["sightings_json"] or "[]")
            except json.JSONDecodeError:
                sightings = []
            if not isinstance(sightings, list):
                sightings = []
            subjects.append(SceneContextSubject(
                label=str(row["label"]),
                box=(float(row["box_x1"]), float(row["box_y1"]), float(row["box_x2"]), float(row["box_y2"])),
                last_seen_epoch=float(row["last_seen_epoch"]),
                first_seen_epoch=float(row["first_seen_epoch"]),
                stable_event_keys=[str(item) for item in sightings],
                row_id=int(row["id"]) if row["id"] is not None else None,
                last_moved_epoch=row["last_moved_epoch"],
            ))
        return SceneContextSnapshot(subjects)

    def load_scene_context(self, camera_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "select id, label, box_x1, box_y1, box_x2, box_y2, first_seen_epoch, "
                "last_seen_epoch, sightings_json, last_moved_epoch "
                "from scene_context_subjects where camera_id=? order by id",
                (camera_id,),
            ).fetchall()
        loaded = []
        for row in rows:
            try:
                sightings = json.loads(row["sightings_json"] or "[]")
            except json.JSONDecodeError:
                sightings = []
            if not isinstance(sightings, list):
                sightings = []
            loaded.append({
                "id": int(row["id"]),
                "label": row["label"],
                "box_x1": row["box_x1"],
                "box_y1": row["box_y1"],
                "box_x2": row["box_x2"],
                "box_y2": row["box_y2"],
                "first_seen_epoch": row["first_seen_epoch"],
                "last_seen_epoch": row["last_seen_epoch"],
                "sightings": [str(item) for item in sightings],
                "last_moved_epoch": row["last_moved_epoch"],
            })
        return loaded

    def save_scene_context(self, camera_id: str, subject: dict[str, Any]) -> int:
        box = subject["box"]
        payload = (
            subject["label"],
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3]),
            float(subject["first_seen_epoch"]),
            float(subject["last_seen_epoch"]),
            json.dumps(list(subject.get("sightings") or []), separators=(",", ":")),
            subject.get("last_moved_epoch"),
        )
        with self._lock, self._connect() as conn:
            row_id = subject.get("id")
            if row_id:
                conn.execute(
                    "update scene_context_subjects set label=?, box_x1=?, box_y1=?, box_x2=?, "
                    "box_y2=?, first_seen_epoch=?, last_seen_epoch=?, sightings_json=?, "
                    "last_moved_epoch=? where id=? and camera_id=?",
                    (*payload, int(row_id), camera_id),
                )
                return int(row_id)
            cursor = conn.execute(
                "insert into scene_context_subjects(camera_id, label, box_x1, box_y1, box_x2, "
                "box_y2, first_seen_epoch, last_seen_epoch, sightings_json, last_moved_epoch) "
                "values(?,?,?,?,?,?,?,?,?,?)",
                (camera_id, *payload),
            )
            return int(cursor.lastrowid)

    def delete_scene_context(self, camera_id: str, row_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "delete from scene_context_subjects where id=? and camera_id=?",
                (int(row_id), camera_id),
            )

    def clear_scene_context(self, camera_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "delete from scene_context_subjects where camera_id=?",
                (camera_id,),
            )

    def scene_context_status(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "select camera_id, count(*) from scene_context_subjects group by camera_id"
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}
