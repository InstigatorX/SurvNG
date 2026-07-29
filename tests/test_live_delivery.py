from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from starlette.responses import Response

from survng.app.main import live_info, relay_go2rtc_websocket, stream


class LiveDeliveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_live_metadata_and_mjpeg_disable_proxy_buffering_and_caching(self) -> None:
        response = Response()
        camera = Mock()
        worker = Mock()
        worker.status.return_value = {"running": True}
        with (
            patch("survng.app.main.manager.camera", return_value=camera),
            patch("survng.app.main.manager.workers", {"gate": worker}),
            patch(
                "survng.app.main.manager.go2rtc.stream_info",
                return_value={"available": True},
            ),
        ):
            self.assertEqual(live_info("gate", response), {"available": True})

        worker = Mock()
        with patch("survng.app.main.manager.workers", {"gate": worker}):
            mjpeg = await stream("gate", Mock())

        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(mjpeg.headers["cache-control"], "no-store")
        self.assertEqual(mjpeg.headers["x-accel-buffering"], "no")

    async def test_relay_resolves_blocking_go2rtc_url_off_the_event_loop(self) -> None:
        websocket = SimpleNamespace(
            query_params={"source": "main", "compat": "h264"},
            accept=AsyncMock(),
            close=AsyncMock(),
        )
        camera = Mock()
        worker = Mock()
        worker.status.return_value = {"running": True}
        with (
            patch("survng.app.main.manager.camera", return_value=camera),
            patch("survng.app.main.manager.workers", {"gate": worker}),
            patch("survng.app.main.manager.go2rtc.websocket_url") as websocket_url,
            patch(
                "survng.app.main.asyncio.to_thread",
                new=AsyncMock(return_value="ws://go2rtc.invalid/api/ws"),
            ) as to_thread,
            patch("survng.app.main.websockets.connect", side_effect=OSError("offline")) as connect,
        ):
            await relay_go2rtc_websocket(websocket, "gate", "MSE stream")

        # Legacy compat parameters are deliberately ignored; relay ownership
        # stops at selecting the configured native go2rtc source.
        to_thread.assert_awaited_once_with(websocket_url, camera, "main")
        websocket.accept.assert_not_awaited()
        self.assertEqual(connect.call_args.kwargs["max_size"], 8 * 1024 * 1024)
        self.assertEqual(connect.call_args.kwargs["max_queue"], 4)
        self.assertIsNone(connect.call_args.kwargs["compression"])
        websocket.close.assert_awaited_once_with(code=1013)

    async def test_relay_rejects_a_powered_off_camera_before_upstream_connect(self) -> None:
        websocket = SimpleNamespace(
            query_params={"source": "live"},
            accept=AsyncMock(),
            close=AsyncMock(),
        )
        worker = Mock()
        worker.status.return_value = {"running": False}
        with (
            patch("survng.app.main.manager.camera", return_value=Mock()),
            patch("survng.app.main.manager.workers", {"gate": worker}),
            patch("survng.app.main.websockets.connect") as connect,
        ):
            await relay_go2rtc_websocket(websocket, "gate", "MSE stream")

        connect.assert_not_called()
        websocket.accept.assert_not_awaited()
        websocket.close.assert_awaited_once_with(code=1008)

    async def test_relay_accepts_only_after_upstream_connection_succeeds(self) -> None:
        class Upstream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def send(self, _message):
                return None

        class Connection:
            async def __aenter__(self):
                return Upstream()

            async def __aexit__(self, _exc_type, _exc, _traceback):
                return False

        websocket = SimpleNamespace(
            query_params={"source": "live"},
            accept=AsyncMock(),
            close=AsyncMock(),
            receive=AsyncMock(return_value={"type": "websocket.disconnect"}),
            send_bytes=AsyncMock(),
            send_text=AsyncMock(),
        )
        worker = Mock()
        worker.status.return_value = {"running": True}
        with (
            patch("survng.app.main.manager.camera", return_value=Mock()),
            patch("survng.app.main.manager.workers", {"gate": worker}),
            patch(
                "survng.app.main.asyncio.to_thread",
                new=AsyncMock(return_value="ws://go2rtc.invalid/api/ws"),
            ),
            patch("survng.app.main.websockets.connect", return_value=Connection()),
        ):
            await relay_go2rtc_websocket(websocket, "gate", "MSE stream")

        websocket.accept.assert_awaited_once_with()
        websocket.close.assert_awaited_once_with(code=1000)

    async def test_mjpeg_rejects_a_powered_off_camera_immediately(self) -> None:
        worker = Mock()
        worker.status.return_value = {"running": False}
        with patch("survng.app.main.manager.workers", {"gate": worker}):
            with self.assertRaisesRegex(HTTPException, "powered off"):
                await stream("gate", Mock())


if __name__ == "__main__":
    unittest.main()
