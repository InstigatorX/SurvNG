from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from starlette.responses import Response

from survng.app.main import live_info, relay_go2rtc_websocket, stream


class LiveDeliveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_live_metadata_and_mjpeg_disable_proxy_buffering_and_caching(self) -> None:
        response = Response()
        camera = Mock()
        with (
            patch("survng.app.main.manager.camera", return_value=camera),
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
        with (
            patch("survng.app.main.manager.camera", return_value=camera),
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
        websocket.accept.assert_awaited_once_with()
        self.assertEqual(connect.call_args.kwargs["max_size"], 8 * 1024 * 1024)
        self.assertEqual(connect.call_args.kwargs["max_queue"], 4)
        self.assertIsNone(connect.call_args.kwargs["compression"])
        websocket.close.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
