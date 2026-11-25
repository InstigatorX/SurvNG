"""Per-object visible intervals that drive the incident storyline lanes."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from survng.app.event_store.scenes import _scene_visible_intervals
from survng.app.events import EventStore

BASE = 1780000000


def at(seconds):
    return datetime.fromtimestamp(BASE + seconds, timezone.utc).isoformat()


def seen(identifier, seconds, camera="driveway", x=10, **extra):
    return {"id": identifier, "camera_id": camera, "captured_at": at(seconds),
            "box": {"x1": x, "y1": 10, "x2": x + 20, "y2": 40},
            "detection_frame_width": 100, "detection_frame_height": 100, **extra}


class VisibleIntervalTest(unittest.TestCase):
    def test_single_sighting_is_an_observed_point(self):
        [interval] = _scene_visible_intervals([("ep-1", seen("a", 0, zones=["Driveway"]))])
        self.assertEqual(interval["state"], "observed")
        self.assertEqual(interval["start_at"], interval["end_at"])
        self.assertEqual((interval["camera_id"], interval["episode_id"]), ("driveway", "ep-1"))
        self.assertEqual(interval["zones"], ["Driveway"])

    def test_arrival_then_stationary_continues_without_a_hole(self):
        evidence = [("ep-1", seen("a", 0, x=0)), ("ep-1", seen("b", 5, x=30)),
                    ("ep-1", seen("c", 10, x=31)), ("ep-1", seen("d", 40, x=31))]
        moving, stationary = _scene_visible_intervals(evidence)
        self.assertEqual((moving["state"], moving["start_at"], moving["end_at"]), ("moving", at(0), at(5)))
        self.assertEqual((stationary["state"], stationary["start_at"], stationary["end_at"]), ("stationary", at(5), at(40)))
        self.assertEqual((stationary["observation_id"], stationary["last_observation_id"]), ("b", "d"))

    def test_activity_attribution_outranks_box_geometry(self):
        # Same box, but the attribution judged the sighting active (moving in place).
        evidence = [("ep-1", seen("a", 0)), ("ep-1", seen("b", 5, activity_role="active")),
                    ("ep-1", seen("c", 9, x=60, activity_role="scene_context"))]
        states = [interval["state"] for interval in _scene_visible_intervals(evidence)]
        self.assertEqual(states, ["moving", "stationary"])

    def test_long_silence_and_other_cameras_start_new_intervals(self):
        evidence = [("ep-1", seen("a", 0)), ("ep-1", seen("b", 200)),
                    ("ep-2", seen("c", 50, camera="front-door", zones=["Entry"])),
                    ("ep-2", seen("d", 60, camera="front-door", x=60, zones=["Porch"]))]
        intervals = _scene_visible_intervals(evidence)
        self.assertEqual([(i["camera_id"], i["start_at"]) for i in intervals],
                         [("driveway", at(0)), ("front-door", at(50)), ("driveway", at(200))])
        self.assertEqual(intervals[1]["zones"], ["Entry", "Porch"])
        self.assertEqual(intervals[1]["observation_count"], 2)


class VisibleIntervalPayloadTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = EventStore(Path(tmp.name))

    def test_notification_payload_keeps_intervals_after_trimming_observations(self):
        def dog(x, offset):
            return {"label": "dog", "confidence": .8, "temporal_observations": 5, "scene_track_key": "t2",
                    "offset_seconds": offset, "box": {"x1": x, "y1": 10, "x2": x + 10, "y2": 40}}
        batch = {"status": "scene_observations", "observations": [dog(10, 0), dog(40, 3), dog(70, 6)]}
        self.store.add_event("driveway", "motion", created_at=at(0), objects_json=json.dumps([batch]))
        notification = self.store.list_scene_incident_notifications()[0]
        subject = next(item for item in notification["scene_objects"] if item["observation_count"] > 1)
        self.assertEqual(len(subject["observations"]), 1)
        self.assertTrue(subject["visible_intervals"])
        first, last = subject["visible_intervals"][0], subject["visible_intervals"][-1]
        self.assertEqual(first["camera_id"], "driveway")
        self.assertIn(first["episode_id"], {episode["id"] for episode in notification["episodes"]})
        self.assertEqual(first["start_at"], subject["first_seen_at"])
        self.assertEqual(last["end_at"], subject["last_seen_at"])


if __name__ == "__main__":
    unittest.main()
