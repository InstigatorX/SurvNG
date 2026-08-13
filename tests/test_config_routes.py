from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from survng.app.config import ApiTokenConfig, AppConfig, CameraConfig, DetectionZone
from survng.app.config_routes import (
    ApiTokenCreateRequest,
    ConfigProbeRequest,
    ConfigRouteDependencies,
    SECRET_PLACEHOLDER,
    create_config_router,
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
