from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from survng.app.config import (
    AppConfig,
    CameraConfig,
    MqttConfig,
    ObjectTrackingConfig,
    load_config,
    normalize_config,
    save_config,
)


class AppConfigTest(unittest.TestCase):
    def test_person_reid_uses_conservative_similarity_default(self) -> None:
        self.assertEqual(ObjectTrackingConfig().reid_match_threshold, 0.70)

    def test_person_reid_requires_a_model_when_enabled(self) -> None:
        with self.assertRaisesRegex(ValidationError, "reid_model_path"):
            ObjectTrackingConfig(reid_enabled=True)

        tracking = ObjectTrackingConfig(
            reid_enabled=True,
            reid_model_path="person-reid.xml",
        )
        self.assertEqual(tracking.reid_max_embeddings_per_frame, 8)
        self.assertEqual(tracking.reid_refresh_interval_frames, 8)

    def test_vehicle_reid_requires_model_and_normalizes_labels(self) -> None:
        with self.assertRaisesRegex(ValidationError, "vehicle_reid_model_path"):
            ObjectTrackingConfig(vehicle_reid_enabled=True)

        tracking = ObjectTrackingConfig(
            vehicle_reid_enabled=True,
            vehicle_reid_model_path="vehicle-reid.xml",
            vehicle_reid_labels=["Car", " truck ", "car", ""],
        )

        self.assertTrue(tracking.appearance_reid_enabled)
        self.assertEqual(tracking.vehicle_reid_labels, ["car", "truck"])
        self.assertTrue(tracking.reid_enabled_for_label("CAR"))
        self.assertFalse(tracking.reid_enabled_for_label("person"))

    def test_legacy_bytetrack_name_migrates_to_survng_hybrid(self) -> None:
        self.assertEqual(ObjectTrackingConfig().implementation, "survng_hybrid")
        self.assertEqual(
            ObjectTrackingConfig(implementation="ByteTrack").implementation,
            "survng_hybrid",
        )
    def test_base_path_defaults_to_survng(self) -> None:
        self.assertEqual(AppConfig().base_path, "/survng")
        self.assertEqual(AppConfig().database_dir, "")
        self.assertEqual(AppConfig().recording_index_dir, "")

    def test_base_path_is_normalized(self) -> None:
        self.assertEqual(AppConfig(base_path=" cameras/ ").base_path, "/cameras")
        self.assertEqual(AppConfig(base_path="/").base_path, "")
        self.assertEqual(AppConfig(base_path="").base_path, "")

    def test_base_path_rejects_query_or_fragment(self) -> None:
        with self.assertRaises(ValidationError):
            AppConfig(base_path="/survng?mode=remote")
        with self.assertRaises(ValidationError):
            AppConfig(base_path="/survng#remote")
        with self.assertRaises(ValidationError):
            AppConfig(base_path="/../survng")
        with self.assertRaises(ValidationError):
            AppConfig(base_path="/survng/<script>")

    def test_event_clip_window_requires_a_bounded_nonempty_duration(self) -> None:
        self.assertEqual(AppConfig(event_clip_before_seconds=0, event_clip_after_seconds=2).event_clip_before_seconds, 0)
        with self.assertRaises(ValidationError):
            AppConfig(event_clip_before_seconds=0, event_clip_after_seconds=0)
        with self.assertRaises(ValidationError):
            AppConfig(event_clip_before_seconds=-1)
        with self.assertRaises(ValidationError):
            AppConfig(event_clip_after_seconds=3601)

    def test_motion_qualification_width_defaults_and_camera_override(self) -> None:
        config = AppConfig.model_validate({
            "cameras": [{
                "id": "back-middle",
                "name": "Back Middle",
                "stream_url": "rtsp://example.invalid/main",
                "motion_qualification": {"frame_width": 480},
            }],
        })

        self.assertEqual(config.motion_qualification.frame_width, 320)
        self.assertEqual(config.motion_qualification.camera_mode_background_fps, 2.0)
        self.assertEqual(config.cameras[0].motion_qualification.frame_width, 480)
        self.assertTrue(config.motion_qualification.borderline_rescue_enabled)
        self.assertEqual(config.motion_qualification.borderline_margin, 0.03)
        self.assertFalse(config.motion_qualification.mog2_audit_enabled)
        self.assertEqual(config.motion_qualification.mog2_history_seconds, 30.0)
        self.assertEqual(config.motion_qualification.rejected_sample_rate, 1.0)
        self.assertEqual(config.motion_qualification.suppression_verification_rate, 0.05)
        self.assertIsNone(
            config.cameras[0].motion_qualification.suppression_verification_rate
        )
        self.assertIsNone(config.cameras[0].motion_qualification.mog2_audit_enabled)
        self.assertTrue(config.detector.require_incident_zone)
        self.assertEqual(config.detector.event_confirmation_frames, 2)
        self.assertEqual(config.detector.event_class_confirmation_frames, {})
        self.assertIsNone(config.cameras[0].require_incident_zone)
        self.assertEqual(config.motion_qualification.pipeline.qualification, [])
        self.assertIsNone(
            config.cameras[0].motion_qualification.pipeline.qualification
        )

    def test_camera_incident_zone_policy_can_override_global_default(self) -> None:
        config = AppConfig.model_validate({
            "detector": {"require_incident_zone": False},
            "cameras": [{
                "id": "gate",
                "name": "Gate",
                "stream_url": "rtsp://example.invalid/main",
                "require_incident_zone": True,
            }],
        })

        self.assertFalse(config.detector.require_incident_zone)
        self.assertTrue(config.cameras[0].require_incident_zone)

    def test_detector_normalizes_class_specific_event_confirmations(self) -> None:
        config = AppConfig.model_validate({
            "detector": {
                "event_confirmation_frames": 2,
                "event_class_confirmation_frames": {
                    " Robot_Lawnmower ": 3,
                    "PERSON": 1,
                },
            },
        })

        self.assertEqual(config.detector.event_class_confirmation_frames, {
            "robot_lawnmower": 3,
            "person": 1,
        })

        with self.assertRaises(ValidationError):
            AppConfig.model_validate({
                "detector": {"event_class_confirmation_frames": {"person": 6}},
            })
        for invalid in (True, 2.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                AppConfig.model_validate({
                    "detector": {
                        "event_class_confirmation_frames": {"person": invalid},
                    },
                })

    def test_camera_identity_and_stream_urls_are_safe_for_runtime_paths(self) -> None:
        with self.assertRaises(ValidationError):
            CameraConfig(id="../gate", name="Gate", stream_url="rtsp://camera/main")
        with self.assertRaises(ValidationError):
            CameraConfig(id="gate", name="Gate", stream_url="not-a-camera-url")
        with self.assertRaises(ValidationError):
            CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://camera/main",
                live_stream_url="file:///tmp/video.mp4",
            )
        with self.assertRaises(ValidationError):
            CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://camera/main",
                onvif={"enabled": True, "host": "camera", "port": 70000},
            )

    def test_duplicate_camera_ids_are_rejected(self) -> None:
        camera = {
            "id": "gate",
            "name": "Gate",
            "stream_url": "rtsp://camera/main",
        }
        with self.assertRaises(ValidationError):
            AppConfig.model_validate({"cameras": [camera, {**camera, "name": "Other"}]})

    def test_mqtt_topic_prefixes_are_normalized_and_reject_wildcards(self) -> None:
        config = MqttConfig(topic_prefix=" /house/survng/ ", discovery_prefix=" /ha/ ")

        self.assertEqual(config.topic_prefix, "house/survng")
        self.assertEqual(config.discovery_prefix, "ha")
        self.assertEqual(MqttConfig(topic_prefix="").topic_prefix, "survng")
        self.assertEqual(MqttConfig(discovery_prefix="/").discovery_prefix, "homeassistant")
        with self.assertRaises(ValidationError):
            MqttConfig(topic_prefix="survng/#")
        with self.assertRaises(ValidationError):
            MqttConfig(discovery_prefix="home/+/assistant")

    def test_save_config_is_atomic_and_does_not_mutate_assigned_ids(self) -> None:
        config = AppConfig(
            cameras=[CameraConfig(id="old", name="Front Door", stream_url="rtsp://camera/main")]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "config.json"
            save_config(config, str(path), assign_ids=True)
            saved = load_config(str(path))

        self.assertEqual(config.cameras[0].id, "old")
        self.assertEqual(saved.cameras[0].id, "front-door")

    def test_assigned_camera_ids_remain_ascii_path_safe_and_bounded(self) -> None:
        long_name = "A" * 128
        config = AppConfig(cameras=[
            CameraConfig(id="one", name="Cámara 🚪", stream_url="rtsp://camera/one"),
            CameraConfig(id="two", name=long_name, stream_url="rtsp://camera/two"),
            CameraConfig(id="three", name=long_name, stream_url="rtsp://camera/three"),
        ])

        normalized = normalize_config(config, assign_ids=True)

        self.assertEqual(normalized.cameras[0].id, "c-mara")
        self.assertEqual(len(normalized.cameras[1].id), 128)
        self.assertEqual(len(normalized.cameras[2].id), 128)
        self.assertTrue(normalized.cameras[2].id.endswith("-2"))
        AppConfig.model_validate(normalized.model_dump(mode="json"))

    def test_reolink_url_preserves_explicit_credentials_and_bounds_channel(self) -> None:
        camera = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="reolink://camera.local/stream?channel=3",
            baichuan={"username": "configured", "password": "secret", "channel": 1},
        )

        self.assertEqual(camera.baichuan.username, "configured")
        self.assertEqual(camera.baichuan.password, "secret")
        self.assertEqual(camera.baichuan.channel, 3)
        with self.assertRaisesRegex(ValidationError, "between 0 and 255"):
            CameraConfig(
                id="invalid",
                name="Invalid",
                stream_url="reolink://camera.local/stream?channel=999",
            )

    def test_url_derived_credentials_cannot_bypass_nested_field_limits(self) -> None:
        oversized_username = "u" * 257
        with self.assertRaises(ValidationError):
            CameraConfig(
                id="rtsp",
                name="RTSP",
                stream_url=f"rtsp://{oversized_username}:secret@camera.local/main",
            )

        with self.assertRaises(ValidationError):
            CameraConfig(
                id="reolink",
                name="Reolink",
                stream_url=f"reolink://{oversized_username}:secret@camera.local/main",
                onvif={"username": "configured"},
            )

    def test_failed_config_serialization_preserves_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            original = AppConfig(base_path="/original")
            save_config(original, str(path), assign_ids=False)
            with patch("survng.app.config.json.dump", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_config(AppConfig(base_path="/replacement"), str(path), assign_ids=False)

            payload = json.loads(path.read_text(encoding="utf-8"))
            temporary_files = list(path.parent.glob(f".{path.name}.*.tmp"))

        self.assertEqual(payload["base_path"], "/original")
        self.assertEqual(temporary_files, [])


if __name__ == "__main__":
    unittest.main()
