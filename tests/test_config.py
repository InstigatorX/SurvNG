from __future__ import annotations

import unittest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from survng.app.config import (
    ApiAuthConfig,
    ApiTokenConfig,
    AppConfig,
    CameraConfig,
    CameraTransitionRoute,
    DetectionZone,
    MqttConfig,
    ObjectTrackingConfig,
    load_config,
    normalize_config,
    save_config,
)


class AppConfigTest(unittest.TestCase):
    def test_detection_zone_can_exclude_ema_without_ignoring_objects(self) -> None:
        zone = DetectionZone(name="Road", exclude_from_ema=True, behavior="none")

        self.assertTrue(zone.exclude_from_ema)
        self.assertEqual(zone.behavior, "none")

    def test_camera_transition_routes_validate_topology_and_timing(self) -> None:
        config = AppConfig.model_validate({
            "cameras": [
                {"id": "back-left", "name": "Back Left", "stream_url": "rtsp://example.invalid/a"},
                {"id": "gate", "name": "Gate", "stream_url": "rtsp://example.invalid/b"},
            ],
            "detector": {"tracking": {"camera_transition_routes": [{
                "from_camera": "back-left",
                "to_camera": "gate",
                "min_seconds": 2,
                "max_seconds": 12,
            }]}},
        })
        self.assertEqual(config.detector.tracking.camera_transition_routes[0].to_camera, "gate")
        with self.assertRaisesRegex(ValidationError, "unknown camera"):
            AppConfig.model_validate({
                "cameras": [{"id": "gate", "name": "Gate", "stream_url": "rtsp://example.invalid/a"}],
                "detector": {"tracking": {"camera_transition_routes": [{
                    "from_camera": "missing",
                    "to_camera": "gate",
                }]}},
            })
        with self.assertRaisesRegex(ValidationError, "maximum time"):
            CameraTransitionRoute(
                from_camera="back-left",
                to_camera="gate",
                min_seconds=10,
                max_seconds=10,
            )

    def test_durable_image_storage_defaults_to_high_quality_webp(self) -> None:
        config = AppConfig()

        self.assertEqual(config.image_storage.format, "webp")
        self.assertEqual(config.image_storage.quality, 95)

    def test_object_tracking_excludes_faces_by_default_and_normalizes_overrides(self) -> None:
        defaults = ObjectTrackingConfig()
        self.assertFalse(defaults.tracks_label("face"))
        self.assertTrue(defaults.tracks_label("person"))

        tracking = ObjectTrackingConfig(excluded_labels=[" Face ", "FACE", "bird", ""])
        self.assertEqual(tracking.excluded_labels, ["face", "bird"])
        self.assertFalse(tracking.tracks_label("BIRD"))

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
        self.assertEqual(
            ObjectTrackingConfig(implementation="ultralytics_botsort").implementation,
            "survng_hybrid",
        )
        self.assertEqual(
            ObjectTrackingConfig(implementation="ultralytics_deepocsort").implementation,
            "survng_hybrid",
        )
    def test_base_path_defaults_to_survng(self) -> None:
        self.assertEqual(AppConfig().base_path, "/survng")
        self.assertEqual(AppConfig().database_dir, "")
        self.assertEqual(AppConfig().recording_index_dir, "")
        self.assertFalse(AppConfig().incident_thumbnail_annotations)
        self.assertFalse(AppConfig().api_auth.enabled)

    def test_api_auth_requires_tokens_and_unique_ids(self) -> None:
        digest = "a" * 64
        with self.assertRaises(ValidationError):
            ApiAuthConfig(enabled=True)
        with self.assertRaises(ValidationError):
            ApiAuthConfig(tokens=[
                ApiTokenConfig(id="ha", name="HA", token_hash=digest),
                ApiTokenConfig(id="ha", name="Other", token_hash="b" * 64),
            ])

    def test_base_path_is_normalized(self) -> None:
        self.assertEqual(AppConfig(base_path=" cameras/ ").base_path, "/cameras")
        self.assertEqual(AppConfig(base_path="/").base_path, "")
        self.assertEqual(AppConfig(base_path="").base_path, "")

    def test_legacy_camera_startup_tuning_is_ignored(self) -> None:
        config = AppConfig.model_validate({
            "camera_startup": {
                "max_concurrent_cameras": 8,
                "first_frame_timeout_seconds": 30,
                "recorder_settle_seconds": 5,
            }
        })

        self.assertNotIn("camera_startup", config.model_dump())

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
                "motion_qualification": {
                    "frame_width": 480,
                    "visual_backup_grace_seconds": 2.0,
                    "visual_backup_min_score": 0.74,
                    "visual_backup_min_consecutive": 4,
                    "visual_backup_cooldown_seconds": 30.0,
                    "visual_backup_max_triggers_5m": 2,
                },
            }],
        })

        self.assertEqual(config.motion_qualification.frame_width, 320)
        self.assertEqual(config.motion_qualification.mode, "camera_rescue")
        self.assertEqual(config.motion_qualification.stationary_object_tolerance, "balanced")
        self.assertEqual(config.motion_qualification.camera_mode_background_fps, 2.0)
        self.assertEqual(config.motion_qualification.visual_backup_warmup_seconds, 10.0)
        self.assertEqual(config.motion_qualification.visual_backup_grace_seconds, 1.5)
        self.assertEqual(config.motion_qualification.visual_backup_min_score, 0.70)
        self.assertEqual(config.motion_qualification.visual_backup_min_consecutive, 3)
        self.assertEqual(config.cameras[0].motion_qualification.frame_width, 480)
        self.assertEqual(
            config.cameras[0].motion_qualification.visual_backup_grace_seconds,
            2.0,
        )
        self.assertEqual(
            config.cameras[0].motion_qualification.visual_backup_min_score,
            0.74,
        )
        self.assertEqual(
            config.cameras[0].motion_qualification.visual_backup_min_consecutive,
            4,
        )
        self.assertEqual(
            config.cameras[0].motion_qualification.visual_backup_cooldown_seconds,
            30.0,
        )
        self.assertEqual(
            config.cameras[0].motion_qualification.visual_backup_max_triggers_5m,
            2,
        )
        self.assertEqual(
            config.cameras[0].motion_qualification.stationary_object_tolerance,
            "inherit",
        )
        self.assertTrue(config.motion_qualification.borderline_rescue_enabled)
        self.assertEqual(config.motion_qualification.borderline_margin, 0.03)
        self.assertEqual(config.motion_qualification.rejected_sample_rate, 1.0)
        self.assertEqual(config.motion_qualification.suppression_verification_rate, 0.05)
        self.assertIsNone(
            config.cameras[0].motion_qualification.suppression_verification_rate
        )
        self.assertTrue(config.detector.require_incident_zone)
        self.assertEqual(config.cameras[0].object_activity_attribution, "inherit")
        self.assertEqual(config.detector.event_confirmation_frames, 2)
        self.assertEqual(config.detector.event_class_confirmation_frames, {})
        self.assertEqual(config.detector.event_class_confidence_thresholds, {})
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

    def test_detector_normalizes_class_specific_confidence_thresholds(self) -> None:
        config = AppConfig.model_validate({
            "detector": {
                "confidence_threshold": 0.45,
                "event_class_confidence_thresholds": {
                    " Robot_Lawnmower ": "0.75",
                    "PERSON": 0.35,
                },
            },
        })

        self.assertEqual(config.detector.event_class_confidence_thresholds, {
            "robot_lawnmower": 0.75,
            "person": 0.35,
        })

        for invalid in (True, float("nan"), 0.0, 1.0, "not-a-number"):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                AppConfig.model_validate({
                    "detector": {
                        "event_class_confidence_thresholds": {"person": invalid},
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

    def test_mqtt_server_metrics_interval_is_bounded(self) -> None:
        self.assertEqual(MqttConfig().server_metrics_interval_seconds, 30)
        with self.assertRaises(ValidationError):
            MqttConfig(server_metrics_interval_seconds=9)
        with self.assertRaises(ValidationError):
            MqttConfig(server_metrics_interval_seconds=3601)

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

    def test_environment_config_path_is_used_for_load_and_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "private" / "config.json"
            with patch.dict(os.environ, {"SURVNG_CONFIG_PATH": str(path)}):
                save_config(AppConfig(base_path="/docker"), assign_ids=False)
                saved = load_config()

            self.assertEqual(saved.base_path, "/docker")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_missing_environment_config_path_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.json"
            with patch.dict(os.environ, {"SURVNG_CONFIG_PATH": str(path)}):
                with self.assertRaisesRegex(FileNotFoundError, "configured SurvNG config"):
                    load_config()

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

    def test_non_reolink_urls_preserve_explicit_native_backend_selection(self) -> None:
        for scheme in ("rtsp", "rtsps", "https"):
            with self.subTest(scheme=scheme):
                camera = CameraConfig(
                    id="gate",
                    name="Gate",
                    stream_url=f"{scheme}://camera.local/main",
                    video_backend="baichuan_native",
                    baichuan={"enabled": True, "host": "camera.local"},
                )

                self.assertEqual(camera.video_backend, "baichuan_native")
                self.assertTrue(camera.baichuan.enabled)

    def test_non_reolink_urls_keep_default_url_backend(self) -> None:
        camera = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsps://camera.local/main",
            baichuan={"enabled": True, "host": "camera.local"},
        )

        self.assertEqual(camera.video_backend, "url")
        self.assertFalse(camera.baichuan.enabled)

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
