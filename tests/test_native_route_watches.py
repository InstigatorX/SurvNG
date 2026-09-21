"""Native-first advisory camera transition route watches."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from survng.app.config import AppConfig, CameraConfig, CameraTransitionRoute
from survng.app.cross_camera_trace import build_cross_camera_trace
from survng.app.detection_watch import RouteDetectionWatch
from survng.app.event_store import EventStore
from survng.app.manager import (
    _route_eligible_objects,
    _route_provenance_from_event,
)
from survng.app.native_activity import NativeActivity


def _activity(tmp_path: Path, camera_id: str = "gate") -> NativeActivity:
    camera = CameraConfig(
        id=camera_id,
        name=camera_id,
        stream_url="rtsp://example.invalid/live",
    )
    config = AppConfig(cameras=[camera]).detector
    events = EventStore(tmp_path)
    return NativeActivity(camera, config, events, Mock(), lambda *_args: "")


class NativeRouteWatchTests(unittest.TestCase):
    def test_observe_opens_downstream_watch_for_enabled_route(self) -> None:
        watches = RouteDetectionWatch(
            [
                CameraTransitionRoute(
                    from_camera="gate",
                    to_camera="lower-garage",
                    min_seconds=1,
                    max_seconds=20,
                )
            ]
        )
        opened = watches.observe_incident(
            camera_id="gate",
            event_id=7,
            event_at=100.0,
            objects=[{"label": "car", "incident_eligible": True}],
        )
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0].target_camera_id, "lower-garage")
        matched = watches.match("lower-garage", 105.0)
        self.assertIsNotNone(matched)
        self.assertEqual(matched.source_event_id, 7)

    def test_native_activity_stamps_handoff_without_softening_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            activity = _activity(Path(tmp), "lower-garage")
            watch = RouteDetectionWatch(
                [
                    CameraTransitionRoute(
                        from_camera="gate",
                        to_camera="lower-garage",
                        min_seconds=0,
                        max_seconds=30,
                    )
                ]
            ).observe_incident(
                camera_id="gate",
                event_id=11,
                event_at=50.0,
                objects=[{"label": "car", "incident_eligible": True}],
                incident_id=101,
            )[0]
            consumed = []
            activity.consume_route_watch = (
                lambda target, source: consumed.append((target, source)) or True
            )
            activity.route_watch_match = lambda epoch: watch
            activity.note_expected_handoff(watch)

            stored = [
                {
                    "label": "car",
                    "confidence": 0.9,
                    "incident_eligible": True,
                    "track_id": 1,
                    "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                }
            ]
            matched = activity._matching_route_watch(55.0, stored)
            self.assertIsNotNone(matched)
            self.assertEqual(matched.source_event_id, 11)
            self.assertEqual(matched.source_incident_id, 101)

            # Outside-zone-only inventory must not consume a watch.
            self.assertIsNone(
                activity._matching_route_watch(
                    55.0,
                    [{"label": "car", "incident_eligible": False}],
                )
            )

            handoff = {
                "status": "native_route_handoff",
                "route_detection_watch": watch.as_dict(),
            }
            path, origin_camera, origin_event = _route_provenance_from_event(
                {
                    "objects_json": json.dumps([*stored, handoff]),
                }
            )
            self.assertEqual(path[-1], "lower-garage")
            self.assertEqual(origin_camera, "gate")
            self.assertEqual(origin_event, 11)
            self.assertEqual(
                _route_eligible_objects([*stored, handoff])[0]["label"],
                "car",
            )

    def test_route_handoff_persists_incident_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from survng.app.config import DetectorConfig
            from survng.app.live_detections import DetectionSnapshot
            from survng.native_spatial import spatial_plan

            camera = CameraConfig(
                id="lower-garage",
                name="lower-garage",
                stream_url="rtsp://example.invalid/live",
            )
            revision = spatial_plan(camera)["revision"]
            config = DetectorConfig(
                enabled=True,
                event_confirmation_frames=1,
                native={"stationary": {"labels": []}},
            )
            events = EventStore(root)
            upstream_event = events.add_event(
                camera_id="gate",
                kind="motion",
                topic="native/object-presence",
                message="upstream",
                created_at=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc).isoformat(),
                objects_json=json.dumps(
                    [{"label": "car", "confidence": 0.9, "incident_eligible": True}]
                ),
            )
            upstream_incident = events.open_incident(
                camera_id="gate",
                start_at=upstream_event["created_at"],
                participants=[{"label": "car", "confidence": 0.9}],
                observation_objects=[{"label": "car", "confidence": 0.9}],
                seed_event_id=int(upstream_event["id"]),
            )
            watch = RouteDetectionWatch(
                [
                    CameraTransitionRoute(
                        from_camera="gate",
                        to_camera="lower-garage",
                        min_seconds=0,
                        max_seconds=30,
                    )
                ]
            ).observe_incident(
                camera_id="gate",
                event_id=int(upstream_event["id"]),
                event_at=50.0,
                objects=[{"label": "car", "incident_eligible": True}],
                incident_id=int(upstream_incident["id"]),
            )[0]
            downstream = NativeActivity(
                camera, config, events, Mock(), lambda *_args: "snap.webp"
            )
            downstream.route_watch_match = lambda epoch: watch
            downstream.consume_route_watch = lambda *_args: True
            obj = {
                "label": "car",
                "confidence": 0.95,
                "incident_eligible": True,
                "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 50},
                "detection_provenance": "native_fresh_detection",
            }
            observation = DetectionSnapshot(
                1.0,
                1,
                100,
                100,
                (obj,),
                "s1",
                "native_fresh_detection",
                100.0,
                zone_revision=revision,
            )
            downstream.consume(observation, now=55.0, epoch=55.0)
            self.assertIsNotNone(downstream.incident_id)
            links = events.list_incident_links(int(upstream_incident["id"]))
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0]["relation_type"], "route_handoff")
            self.assertEqual(links[0]["from_incident_id"], int(upstream_incident["id"]))
            self.assertEqual(links[0]["to_incident_id"], int(downstream.incident_id))
            self.assertEqual(links[0]["from_camera_id"], "gate")
            self.assertEqual(links[0]["to_camera_id"], "lower-garage")

    def test_cross_camera_trace_biases_configured_route(self) -> None:
        anchor = {
            "id": "incident-gate-42",
            "representative_event_id": 42,
            "camera_id": "gate",
            "start_at": "2026-08-01T12:00:00+00:00",
            "end_at": "2026-08-01T12:00:00+00:00",
            "duration_seconds": 1,
            "event_count": 1,
            "trigger_source": "camera",
            "labels": ["car"],
            "zones": [],
            "motion_observations": [],
            "faces": [],
            "events": [{"id": 42, "kind": "motion", "objects": [], "faces": []}],
        }
        match = {
            "id": "incident-lower-43",
            "representative_event_id": 43,
            "camera_id": "lower-garage",
            "start_at": "2026-08-01T12:00:08+00:00",
            "end_at": "2026-08-01T12:00:08+00:00",
            "duration_seconds": 1,
            "event_count": 1,
            "trigger_source": "camera",
            "labels": ["car"],
            "zones": [],
            "motion_observations": [],
            "faces": [],
            "events": [{"id": 43, "kind": "motion", "objects": [], "faces": []}],
        }
        manager = SimpleNamespace(
            events=SimpleNamespace(between_compact=lambda *_args: [{"id": 43}]),
            appearance_index=None,
            config=SimpleNamespace(
                detector=SimpleNamespace(
                    tracking=SimpleNamespace(
                        camera_transition_routes=[
                            CameraTransitionRoute(
                                from_camera="gate",
                                to_camera="lower-garage",
                                min_seconds=1,
                                max_seconds=30,
                            )
                        ]
                    )
                )
            ),
        )

        with patch("survng.app.cross_camera_trace._incident_rows", return_value=[match]):
            with patch(
                "survng.app.cross_camera_trace.correlate_incident_timeline",
                return_value=[],
            ):
                payload = build_cross_camera_trace(
                    manager,
                    resolve_event=lambda _manager, event_id: (
                        anchor if event_id == 42 else match
                    ),
                    hydrate=lambda _manager, summaries: summaries,
                    with_faces=lambda _manager, summaries: summaries,
                    event_id=42,
                    start_at="2026-08-01T11:45:00+00:00",
                    end_at="2026-08-01T12:15:00+00:00",
                )

        self.assertEqual(payload["matches"][0]["event_id"], 43)
        self.assertEqual(payload["matches"][0]["match_strength"], "context_candidate")
        self.assertTrue(
            any("Expected camera transition" in reason for reason in payload["matches"][0]["reasons"])
        )
        self.assertEqual(payload["matches"][0]["route_to_camera"], "lower-garage")


if __name__ == "__main__":
    unittest.main()
