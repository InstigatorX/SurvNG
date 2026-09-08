from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from survng.app.config import (
    ApiTokenConfig,
    AppConfig,
    CameraConfig,
    DetectionZone,
    OnvifConfig,
)
from survng.app.config_routes import (
    ApiTokenCreateRequest,
    ConfigProbeRequest,
    ConfigRouteDependencies,
    SECRET_PLACEHOLDER,
    create_config_router,
    redacted_config_payload,
    restore_config_secrets,
)


class ConfigRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            mqtt={"password": "broker-secret"},
            cameras=[CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://admin:camera-secret@gate/main",
            )],
        )
        self.manager = Mock()
        self.manager.workers = {"gate": Mock()}
        self.save = Mock()
        self.apply = Mock(return_value=(self.config, {
            "apply_mode": "hot",
            "camera_workers_restarted": False,
            "subsystems_restarted": [],
            "hot_updated": [],
        }))
        self.publish = Mock(side_effect=self._publish)
        self.limiter = threading.BoundedSemaphore(1)
        dependencies = ConfigRouteDependencies(
            get_config=lambda: self.config,
            get_manager=lambda: self.manager,
            publish_config=self.publish,
            apply_config=self.apply,
            reload_manager=Mock(return_value=self.config),
            save_config=self.save,
            validate_config=Mock(),
            lock=threading.RLock(),
            probe_limiter=self.limiter,
        )
        self.router = create_config_router(dependencies)

    def _publish(self, config: AppConfig) -> None:
        self.config = config
        self.manager.config = config

    def endpoint(self, path: str, method: str):
        return next(
            route.endpoint
            for route in self.router.routes
            if route.path == path and method in route.methods
        )

    def test_config_read_masks_every_secret(self) -> None:
        payload = self.endpoint("/api/config", "GET")()

        self.assertEqual(payload["mqtt"]["password"], SECRET_PLACEHOLDER)
        self.assertIn(SECRET_PLACEHOLDER, payload["cameras"][0]["stream_url"])
        self.assertNotIn("camera-secret", str(payload))
        self.assertNotIn("broker-secret", str(payload))
        self.assertEqual(payload["cameras"][0]["live_view"], {
            "main": {"fit": "cover", "focal_x": 50.0, "focal_y": 50.0, "zoom": 1.0},
            "live": {"fit": "cover", "focal_x": 50.0, "focal_y": 50.0, "zoom": 1.0},
        })

    def test_multiple_masked_api_tokens_round_trip_through_config_put(self) -> None:
        self.config.api_auth.tokens = [
            ApiTokenConfig(id="one", name="One", token_hash="a" * 64),
            ApiTokenConfig(id="two", name="Two", token_hash="b" * 64),
        ]
        payload = self.endpoint("/api/config", "GET")()
        payload["base_path"] = "/updated"
        self.endpoint("/api/config", "PUT")(AppConfig.model_validate(payload))
        restored = self.apply.call_args.args[0]
        self.assertEqual([token.token_hash for token in restored.api_auth.tokens], ["a" * 64, "b" * 64])
        self.assertEqual(restored.base_path, "/updated")
        payload["api_auth"]["tokens"][1]["id"] = "unknown"
        with self.assertRaises(HTTPException) as error:
            self.endpoint("/api/config", "PUT")(AppConfig.model_validate(payload))
        self.assertEqual(error.exception.status_code, 422)

    def test_url_query_secrets_redact_and_restore_without_reencoding(self) -> None:
        original = "http://admin:user%40pass@gate/live?channel=0&password=a%26b&token=x+y&token=z%2Bv&API%5FKEY=secret&empty=&flag#view"
        self.config.cameras[0].stream_url = original
        self.config.cameras[0].live_stream_url = "http://gate/sub?user=admin&pass=subsecret&quality=high"
        payload = self.endpoint("/api/config", "GET")()
        camera = payload["cameras"][0]
        for secret in ("user%40pass", "a%26b", "x+y", "z%2Bv", "subsecret"):
            self.assertNotIn(secret, str(camera))
        self.assertEqual(camera["stream_url"].count(SECRET_PLACEHOLDER), 5)
        self.endpoint("/api/config", "PUT")(AppConfig.model_validate(payload))
        restored = self.apply.call_args.args[0].cameras[0]
        self.assertEqual(restored.stream_url, original)
        self.assertEqual(restored.live_stream_url, self.config.cameras[0].live_stream_url)
        camera["stream_url"] = camera["stream_url"].replace("channel=0", "channel=1").replace("token=" + SECRET_PLACEHOLDER, "token=new%2Bsecret", 1)
        self.endpoint("/api/config/cameras/{camera_id}", "PUT")("gate", CameraConfig.model_validate(camera))
        updated = self.apply.call_args.args[0].cameras[0].stream_url
        self.assertIn("channel=1", updated)
        self.assertIn("token=new%2Bsecret&token=z%2Bv", updated)

    def test_query_placeholders_need_existing_credentials(self) -> None:
        body = CameraConfig(id="new", name="New", stream_url=f"http://new/live?password={SECRET_PLACEHOLDER}")
        with self.assertRaises(HTTPException) as error:
            self.endpoint("/api/config/cameras/{camera_id}", "PUT")("new", body)
        self.assertEqual(error.exception.status_code, 422)
        self.config.cameras[0].stream_url = "http://gate/live"
        body.id = "gate"
        with self.assertRaises(HTTPException):
            self.endpoint("/api/config/cameras/{camera_id}", "PUT")("gate", body)

    def test_config_cannot_expose_or_clear_persisted_revocations(self) -> None:
        self.config.web_auth.revoked_sessions = {"a" * 64: 2_000_000_000}
        payload = redacted_config_payload(self.config)
        self.assertNotIn("revoked_sessions", payload["web_auth"])
        payload["web_auth"]["revoked_sessions"] = {"b" * 64: 2_000_000_000}
        restored = restore_config_secrets(AppConfig.model_validate(payload), self.config)
        self.assertEqual(restored.web_auth.revoked_sessions, self.config.web_auth.revoked_sessions)

    def test_config_update_reports_runtime_storage_validation_as_422(self) -> None:
        self.apply.side_effect = OSError(
            "no writable media location supports snapshots (Media 1: directory is not writable)"
        )

        with self.assertRaises(HTTPException) as raised:
            self.endpoint("/api/config", "PUT")(self.config)

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("Media 1: directory is not writable", raised.exception.detail)

    def test_api_token_create_lists_metadata_and_returns_secret_once(self) -> None:
        self.apply.side_effect = lambda next_config, **_kwargs: (next_config, {
            "apply_mode": "hot", "camera_workers_restarted": False,
            "subsystems_restarted": [], "hot_updated": ["api_auth"],
        })
        created = self.endpoint("/api/config/api-tokens", "POST")(
            ApiTokenCreateRequest(id="ha", name="Home Assistant", scopes=["read", "camera:control"])
        )
        # FastAPI normally validates the request model before invoking the handler.
        self.assertTrue(created["token"].startswith("survng_"))
        self.assertNotIn("token_hash", created["credential"])

    def test_api_token_list_never_exposes_hashes(self) -> None:
        self.config.api_auth.tokens = [ApiTokenConfig(
            id="ha", name="Home Assistant", token_hash="a" * 64, scopes=["read"],
        )]

        result = self.endpoint("/api/config/api-tokens", "GET")()

        self.assertEqual(result["tokens"], [{
            "id": "ha", "name": "Home Assistant", "scopes": ["read"],
        }])
        self.assertNotIn("token_hash", str(result))

    def test_api_token_delete_disables_auth_when_last_token_is_removed(self) -> None:
        self.config.api_auth.enabled = True
        self.config.api_auth.tokens = [ApiTokenConfig(
            id="ha", name="Home Assistant", token_hash="a" * 64, scopes=["read"],
        )]
        self.apply.side_effect = lambda next_config, **_kwargs: (next_config, {
            "apply_mode": "hot", "camera_workers_restarted": False,
            "subsystems_restarted": [], "hot_updated": ["api_auth"],
        })
        result = self.endpoint("/api/config/api-tokens/{token_id}", "DELETE")("ha")

        applied = self.apply.call_args.args[0]
        self.assertEqual(applied.api_auth.tokens, [])
        self.assertFalse(applied.api_auth.enabled)
        self.assertFalse(result["enabled"])

    def test_order_rejects_missing_runtime_worker_before_persistence(self) -> None:
        self.manager.workers = {}

        endpoint = self.endpoint("/api/config/cameras/order", "PUT")
        with self.assertRaises(HTTPException) as raised:
            endpoint(["gate"])

        self.assertEqual(raised.exception.status_code, 409)
        self.save.assert_not_called()
        self.publish.assert_not_called()

    def test_display_name_edit_preserves_camera_identity(self) -> None:
        endpoint = self.endpoint("/api/config/cameras/{camera_id}", "PUT")
        result = endpoint("gate", CameraConfig(id="front-gate", name="Front Gate", stream_url="rtsp://gate/main"))
        self.assertEqual(result["camera"]["id"], "gate")
        self.assertEqual(self.apply.call_args.args[0].cameras[0].id, "gate")

    def test_full_config_save_preserves_existing_camera_ids(self) -> None:
        edited = self.config.model_copy(deep=True)
        edited.cameras[0].name = "Front Gate"
        self.endpoint("/api/config", "PUT")(edited)
        self.assertFalse(self.apply.call_args.kwargs["assign_ids"])
        self.assertEqual(self.apply.call_args.args[0].cameras[0].id, "gate")

    def test_probe_capacity_is_bounded_and_released(self) -> None:
        self.assertTrue(self.limiter.acquire(blocking=False))
        endpoint = self.endpoint("/api/config/probe", "POST")
        request = ConfigProbeRequest(host="camera.local")
        with self.assertRaises(HTTPException) as raised:
            endpoint(request)
        self.limiter.release()
        with patch("survng.app.config_routes._tcp_reachable", return_value=False):
            available = endpoint(request)

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.headers["Retry-After"], "2")
        self.assertFalse(available["onvif"]["reachable"])
        self.assertTrue(self.limiter.acquire(blocking=False))
        self.limiter.release()

    def test_masked_probe_credentials_cannot_be_forwarded_to_another_host(self) -> None:
        self.config.cameras[0].onvif = OnvifConfig(
            enabled=True,
            host="gate.local",
            port=8000,
            username="viewer",
            password="camera-secret",
        )
        endpoint = self.endpoint("/api/config/probe", "POST")

        with self.assertRaises(HTTPException) as raised:
            endpoint(ConfigProbeRequest(
                camera_id="gate",
                host="attacker.example",
                username="viewer",
                password=SECRET_PLACEHOLDER,
            ))

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("configured camera endpoint", raised.exception.detail)

    def test_masked_probe_credentials_allow_the_configured_camera_endpoint(self) -> None:
        self.config.cameras[0].onvif = OnvifConfig(
            enabled=True,
            host="gate.local",
            port=8000,
            username="viewer",
            password="camera-secret",
        )
        endpoint = self.endpoint("/api/config/probe", "POST")
        with patch("survng.app.config_routes._tcp_reachable", return_value=False):
            result = endpoint(ConfigProbeRequest(
                camera_id="gate",
                host="gate.local",
                username="viewer",
                password=SECRET_PLACEHOLDER,
            ))

        self.assertFalse(result["onvif"]["reachable"])

    def test_zone_persistence_failure_restores_runtime_zones(self) -> None:
        self.config.cameras[0].zones = [
            DetectionZone(
                name="driveway",
                points=[
                    {"x": 0.0, "y": 0.0},
                    {"x": 1.0, "y": 0.0},
                    {"x": 1.0, "y": 1.0},
                ],
            )
        ]
        self.save.side_effect = OSError("disk unavailable")
        endpoint = self.endpoint("/api/config/cameras/{camera_id}/zones", "PUT")
        replacement = [
            DetectionZone(
                name="porch",
                points=[
                    {"x": 0.0, "y": 0.0},
                    {"x": 0.5, "y": 0.0},
                    {"x": 0.5, "y": 0.5},
                ],
            )
        ]

        with self.assertRaises(OSError):
            endpoint("gate", replacement)

        self.assertEqual(self.manager.update_camera_zones.call_count, 2)
        first, rollback = self.manager.update_camera_zones.call_args_list
        self.assertEqual(first.args[0], "gate")
        self.assertEqual(first.args[1][0].name, "porch")
        self.assertEqual(rollback.args[1][0].name, "driveway")
        self.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
