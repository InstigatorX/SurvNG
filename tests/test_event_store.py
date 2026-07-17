from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from survng.app.events import EventStore


class EventStoreTest(unittest.TestCase):
    def test_compact_queries_omit_media_fields_and_support_keyset_paging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            events = [
                store.add_event(
                    camera_id="front-door",
                    kind="motion",
                    message=f"message-{index}",
                    snapshot_path=f"/snapshots/{index}.jpg",
                    recording_path=f"/recordings/{index}.mp4",
                    created_at="2026-07-15T12:00:00+00:00",
                )
                for index in range(7)
            ]

            first_page = store.recent_compact(3)
            cursor = first_page[-1]
            second_page = store.recent_compact(3, cursor["created_at"], int(cursor["id"]))

            self.assertEqual([row["id"] for row in first_page], [events[6]["id"], events[5]["id"], events[4]["id"]])
            self.assertEqual([row["id"] for row in second_page], [events[3]["id"], events[2]["id"], events[1]["id"]])
            self.assertEqual(
                set(first_page[0]),
                {"id", "camera_id", "kind", "objects_json", "created_at"},
            )

    def test_between_compact_returns_complete_range_without_media_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            for index in range(25):
                store.add_event(
                    camera_id="back-middle",
                    kind="motion",
                    message="large payload" * 100,
                    snapshot_path=f"/snapshots/{index}.jpg",
                    created_at=f"2026-07-15T12:00:{index:02d}+00:00",
                )

            rows = store.between_compact(
                "2026-07-15T12:00:00+00:00",
                "2026-07-15T12:01:00+00:00",
            )

            self.assertEqual(len(rows), 25)
            self.assertNotIn("message", rows[0])
            self.assertNotIn("snapshot_path", rows[0])

    def test_get_many_hydrates_only_requested_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            first = store.add_event(camera_id="gate", kind="motion", message="first")
            second = store.add_event(camera_id="gate", kind="motion", message="second")
            store.add_event(camera_id="gate", kind="motion", message="third")

            rows = store.get_many([int(second["id"]), int(first["id"])])

            self.assertEqual({row["message"] for row in rows}, {"first", "second"})
            self.assertEqual(len(rows), 2)

    def test_update_objects_persists_manual_detection_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event(
                camera_id="back-middle",
                kind="motion",
                snapshot_path="/tmp/snapshot.jpg",
                objects_json=json.dumps([{"status": "no_recorded_frame"}]),
                created_at="2026-07-11T15:36:57+00:00",
            )

            updated = store.update_objects(
                int(event["id"]),
                json.dumps([{"label": "car", "confidence": 0.8, "detection_source": "manual_openvino"}]),
            )

            self.assertIsNotNone(updated)
            loaded = store.get(int(event["id"]))
            self.assertEqual(updated, loaded)
            objects = json.loads(loaded["objects_json"])
            self.assertEqual(objects[0]["label"], "car")
            self.assertEqual(objects[0]["detection_source"], "manual_openvino")

    def test_motion_audits_filter_and_backfill_rejected_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event(
                camera_id="gate",
                kind="motion",
                snapshot_path=f"{tmpdir}/snapshots/gate/rejected.jpg",
                objects_json=json.dumps([
                    {"label": "car", "confidence": 0.82, "incident_eligible": True},
                    {
                        "status": "motion_qualification",
                        "motion_qualification": {
                            "mode": "audit",
                            "sensitivity": "balanced",
                            "score": 0.41,
                            "threshold": 0.48,
                            "reason": "edge_motion",
                            "trigger_count": 3,
                            "would_suppress": True,
                            "features": {"persistence": 0.75, "interior": 0.0},
                        },
                    },
                ], separators=(",", ":")),
                created_at="2026-07-16T12:00:00+00:00",
            )

            reloaded = EventStore(Path(tmpdir))
            object_rows, object_total = reloaded.motion_audits(outcome="object")
            clear_rows, clear_total = reloaded.motion_audits(outcome="clear")

            self.assertEqual(object_total, 1)
            self.assertEqual(clear_total, 0)
            self.assertEqual(object_rows[0]["event_id"], event["id"])
            self.assertEqual(object_rows[0]["reason"], "edge_motion")
            self.assertEqual(json.loads(object_rows[0]["features_json"])["persistence"], 0.75)
            self.assertEqual(clear_rows, [])

    def test_motion_audits_preserve_skipped_detector_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            audit = store.add_motion_audit(
                camera_id="front-door",
                snapshot_path="",
                created_at="2026-07-16T12:00:00+00:00",
                mode="enforce",
                sensitivity="balanced",
                score=0.22,
                threshold=0.48,
                reason="low_persistence",
                object_detected=None,
                trigger_count=2,
                features={"persistence": 0.2},
            )

            rows, total = store.motion_audits(camera_id="front-door", outcome="not_run")

            self.assertEqual(total, 1)
            self.assertEqual(rows[0]["id"], audit["id"])
            self.assertIsNone(rows[0]["object_detected"])


if __name__ == "__main__":
    unittest.main()
