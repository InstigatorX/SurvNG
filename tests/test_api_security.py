from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from survng.app import main
from survng.app.config import (
    AppConfig,
    AuditAiConfig,
    BaichuanConfig,
    CameraConfig,
    MqttConfig,
    OnvifConfig,
)
from survng.app.security import redact_secret_text


def camera(camera_id: str = "gate", name: str = "Gate") -> CameraConfig:
    return CameraConfig(
        id=camera_id,
        name=name,
        stream_url="rtsp://viewer:main-secret@camera.local/main",
        live_stream_url="rtsp://viewer:live-secret@camera.local/sub",
        onvif=OnvifConfig(
            enabled=True,
            host="camera.local",
            username="viewer",
            password="onvif-secret",
        ),
        baichuan=BaichuanConfig(
            enabled=True,
            host="camera.local",
            username="viewer",
            password="baichuan-secret",
        ),
    )


class ApiSecretBoundaryTest(unittest.TestCase):
    def test_health_response_is_minimal_and_does_not_probe_runtime(self) -> None:
        self.assertEqual(main.health(), {"status": "ok"})

    def test_config_payload_masks_and_round_trips_every_secret(self) -> None:
        current = AppConfig(
            cameras=[camera()],
            mqtt=MqttConfig(password="mqtt-secret"),
            audit_ai=AuditAiConfig(api_key="ai-secret"),
        )

        payload = main._redacted_config_payload(current)
        serialized = json.dumps(payload)

        for secret in (
            "main-secret",
            "live-secret",
            "onvif-secret",
            "baichuan-secret",
            "mqtt-secret",
            "ai-secret",
        ):
            self.assertNotIn(secret, serialized)
        restored = main._restore_config_secrets(AppConfig.model_validate(payload), current)
        self.assertEqual(restored, current)

    def test_masked_secrets_follow_a_renamed_camera_only_when_identity_matches(self) -> None:
        current = AppConfig(cameras=[camera(camera_id="legacy", name="Old name")])
        payload = main._redacted_config_payload(current)
        payload["cameras"][0]["id"] = "new-name"
        payload["cameras"][0]["name"] = "New name"

        restored = main._restore_config_secrets(AppConfig.model_validate(payload), current)

        self.assertEqual(restored.cameras[0].onvif.password, "onvif-secret")

    def test_masked_secrets_are_not_copied_to_an_unrelated_replacement_camera(self) -> None:
        current = AppConfig(cameras=[camera(camera_id="legacy")])
        replacement = camera(camera_id="replacement", name="Replacement")
        replacement.stream_url = "rtsp://other:__SURVNG_SECRET_SET__@other.local/main"
        replacement.live_stream_url = None
        replacement.onvif.host = "other.local"
        replacement.onvif.username = "other"
        replacement.baichuan.host = "other.local"
        replacement.baichuan.username = "other"

        with self.assertRaisesRegex(ValueError, "new cameras"):
            main._restore_config_secrets(AppConfig(cameras=[replacement]), current)

    def test_placeholder_text_outside_a_url_password_is_not_replaced(self) -> None:
        current = camera()
        incoming = current.model_copy(deep=True)
        incoming.stream_url = "rtsp://viewer:new-secret@camera.local/__SURVNG_SECRET_SET__"

        restored = main._restore_camera_secrets(incoming, current)

        self.assertEqual(restored.stream_url, incoming.stream_url)

    def test_secret_redaction_handles_urls_structured_fields_and_authorization(self) -> None:
        message = (
            'rtsp://user:camera-pass@host/live password="secret value" '
            "rtsps://secure:tls-pass@host/live "
            "api_key=abc Authorization: Bearer token-value"
        )

        redacted = redact_secret_text(message)

        for secret in ("camera-pass", "tls-pass", "secret value", "abc", "token-value"):
            self.assertNotIn(secret, redacted)
        self.assertIn("rtsp://user:***@host/live", redacted)
        self.assertIn("rtsps://secure:***@host/live", redacted)

    def test_public_rows_do_not_expose_filesystem_paths(self) -> None:
        event = main._event_row(
            {
                "id": 1,
                "snapshot_path": "/srv/private/snapshot.jpg",
                "recording_path": "/srv/private/recording.mp4",
                "objects_json": "[]",
            }
        )
        recording = main._public_recording_row({"path": "/srv/private/clip.mp4", "name": "clip.mp4"})
        face = main._public_face_observation({"snapshot_path": "/srv/private/face.jpg", "id": 3})

        self.assertEqual(event["snapshot_path"], "available")
        self.assertEqual(event["recording_path"], "available")
        self.assertNotIn("path", recording)
        self.assertNotIn("snapshot_path", face)

    def test_recording_paths_must_resolve_inside_recording_storage(self) -> None:
        with tempfile.TemporaryDirectory() as storage, tempfile.TemporaryDirectory() as outside:
            recordings = Path(storage) / "recordings"
            recordings.mkdir()
            allowed = recordings / "allowed.mp4"
            allowed.write_bytes(b"video")
            denied = Path(outside) / "private.mp4"
            denied.write_bytes(b"private")
            fake_manager = SimpleNamespace(
                recorder=SimpleNamespace(recordings_dir=recordings),
            )
            with patch.object(main, "manager", fake_manager):
                self.assertEqual(main._recording_storage_path(allowed), allowed.resolve())
                with self.assertRaisesRegex(Exception, "outside storage"):
                    main._recording_storage_path(denied)

    def test_probe_request_rejects_invalid_network_targets(self) -> None:
        with self.assertRaises(ValidationError):
            main.ConfigProbeRequest(host="camera.local/path")
        with self.assertRaises(ValidationError):
            main.ConfigProbeRequest(host="camera.local", onvif_port=70000)


class SameOriginMiddlewareTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _collector(messages: list[dict]):
        async def collect(message: dict) -> None:
            messages.append(message)

        return collect

    async def test_cross_origin_state_change_is_rejected(self) -> None:
        called = False

        async def inner(_scope, _receive, _send) -> None:
            nonlocal called
            called = True

        middleware = main.SecurityBoundaryMiddleware(inner)
        messages: list[dict] = []
        scope = {
            "type": "http",
            "scheme": "https",
            "method": "PUT",
            "path": "/api/config",
            "headers": [(b"host", b"survng.local"), (b"origin", b"https://evil.example")],
        }

        await middleware(scope, self._receive, self._collector(messages))

        self.assertFalse(called)
        self.assertEqual(messages[0]["status"], 403)

    async def test_cross_site_fetch_without_origin_is_rejected(self) -> None:
        async def inner(_scope, _receive, _send) -> None:
            self.fail("cross-site request reached the application")

        middleware = main.SecurityBoundaryMiddleware(inner)
        messages: list[dict] = []
        scope = {
            "type": "http",
            "scheme": "https",
            "method": "POST",
            "path": "/api/cameras/gate/camera/stop",
            "headers": [(b"host", b"survng.local"), (b"sec-fetch-site", b"cross-site")],
        }

        await middleware(scope, self._receive, self._collector(messages))

        self.assertEqual(messages[0]["status"], 403)

    async def test_cross_origin_api_read_is_rejected(self) -> None:
        async def inner(_scope, _receive, _send) -> None:
            self.fail("cross-origin API read reached the application")

        middleware = main.SecurityBoundaryMiddleware(inner)
        messages: list[dict] = []
        scope = {
            "type": "http",
            "scheme": "https",
            "method": "GET",
            "path": "/survng/api/accelerator",
            "headers": [(b"host", b"survng.local"), (b"origin", b"https://evil.example")],
        }

        await middleware(scope, self._receive, self._collector(messages))

        self.assertEqual(messages[0]["status"], 403)

    async def test_origin_with_a_path_is_rejected_as_malformed(self) -> None:
        scope = {
            "type": "http",
            "scheme": "https",
            "headers": [(b"host", b"survng.local"), (b"origin", b"https://survng.local/path")],
        }

        self.assertFalse(main._same_origin_request(scope))

    async def test_api_response_receives_security_and_no_store_headers(self) -> None:
        async def inner(_scope, _receive, send) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        middleware = main.SecurityBoundaryMiddleware(inner)
        messages: list[dict] = []
        scope = {
            "type": "http",
            "scheme": "https",
            "method": "GET",
            "path": "/api/config",
            "headers": [(b"host", b"survng.local"), (b"origin", b"https://survng.local")],
        }

        await middleware(scope, self._receive, self._collector(messages))

        headers = dict(messages[0]["headers"])
        self.assertEqual(headers[b"cache-control"], b"no-store")
        self.assertEqual(headers[b"x-content-type-options"], b"nosniff")
        self.assertEqual(headers[b"x-frame-options"], b"SAMEORIGIN")

    async def test_cross_origin_websocket_is_closed_before_accept(self) -> None:
        async def inner(_scope, _receive, _send) -> None:
            self.fail("cross-origin WebSocket reached the application")

        middleware = main.SecurityBoundaryMiddleware(inner)
        messages: list[dict] = []
        scope = {
            "type": "websocket",
            "scheme": "wss",
            "path": "/api/cameras/gate/webrtc",
            "headers": [(b"host", b"survng.local"), (b"origin", b"https://evil.example")],
        }

        await middleware(scope, self._receive, self._collector(messages))

        self.assertEqual(messages, [{"type": "websocket.close", "code": 1008}])

    async def test_same_origin_websocket_reaches_application(self) -> None:
        called = False

        async def inner(_scope, _receive, _send) -> None:
            nonlocal called
            called = True

        middleware = main.SecurityBoundaryMiddleware(inner)
        scope = {
            "type": "websocket",
            "scheme": "wss",
            "path": "/api/cameras/gate/webrtc",
            "headers": [(b"host", b"survng.local"), (b"origin", b"https://survng.local")],
        }

        await middleware(scope, self._receive, self._collector([]))

        self.assertTrue(called)

    @staticmethod
    async def _receive() -> dict:
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}


if __name__ == "__main__":
    unittest.main()
