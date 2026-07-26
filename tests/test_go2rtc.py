from __future__ import annotations

import hashlib
import threading
import unittest
from unittest.mock import patch

from survng.app.config import CameraConfig
from survng.app.go2rtc import Go2RtcAdapter, Go2RtcError, Go2RtcStream


class FakeResponse:
    def __init__(self, body: bytes = b"", status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, _size: int = -1) -> bytes:
        return self.body


class Go2RtcAdapterTest(unittest.TestCase):
    def camera(self) -> CameraConfig:
        return CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://10.1.1.1:8554/back_middle",
            live_stream_url="rtsp://10.1.1.1:8554/back_middle_ext",
        )

    def test_stream_info_selects_h264_compatibility_for_h265(self) -> None:
        adapter = Go2RtcAdapter()
        payload = {
            "back_middle": {
                "producers": [{"medias": ["video, recvonly, H265", "audio, recvonly, PCMA/8000"]}],
            },
        }
        with patch.object(adapter, "_streams", return_value=payload):
            info = adapter.stream_info(self.camera(), "main")

        self.assertEqual(info["video_codec"], "H265")
        self.assertEqual(info["compatibility"], "h264")

    def test_stream_info_keeps_native_mode_when_h264_is_available(self) -> None:
        adapter = Go2RtcAdapter()
        payload = {
            "back_middle": {
                "producers": [{"medias": ["video, recvonly, H265, H264"]}],
            },
        }
        with patch.object(adapter, "_streams", return_value=payload):
            info = adapter.stream_info(self.camera(), "main")

        self.assertEqual(info["compatibility"], "native")

    def test_compatibility_name_is_scoped_by_host(self) -> None:
        adapter = Go2RtcAdapter()
        with (
            patch.object(adapter, "_streams", return_value={}),
            patch("survng.app.go2rtc.urlopen", return_value=FakeResponse()),
        ):
            first = adapter._ensure_h264(Go2RtcStream("10.1.1.1", "camera"))
            second = adapter._ensure_h264(Go2RtcStream("10.1.1.2", "camera"))

        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith("_h264"))
        self.assertTrue(second.endswith("_h264"))

    def test_ipv6_go2rtc_urls_use_bracketed_authorities(self) -> None:
        adapter = Go2RtcAdapter()
        camera = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://[fd00::10]:8554/gate",
        )

        self.assertEqual(
            adapter.websocket_url(camera, "main"),
            "ws://[fd00::10]:1984/api/ws?src=gate",
        )
        self.assertEqual(adapter._base_url("fd00::10"), "http://[fd00::10]:1984")

    def test_existing_compatibility_stream_is_reused(self) -> None:
        adapter = Go2RtcAdapter()
        stream = Go2RtcStream("10.1.1.1", "camera")
        expected = "survng_camera_" + hashlib.sha1(b"10.1.1.1/camera").hexdigest()[:8] + "_h264"
        with (
            patch.object(adapter, "_streams", return_value={expected: {}}),
            patch("survng.app.go2rtc.urlopen") as request,
        ):
            actual = adapter._ensure_h264(stream)

        self.assertEqual(actual, expected)
        request.assert_not_called()

    def test_compatibility_creation_does_not_hold_the_global_cache_lock(self) -> None:
        adapter = Go2RtcAdapter()
        put_started = threading.Event()
        release_put = threading.Event()

        def request(url, **_kwargs):
            if not isinstance(url, str):
                put_started.set()
                release_put.wait(1)
                return FakeResponse()
            return FakeResponse(b"{}")

        with patch("survng.app.go2rtc.urlopen", side_effect=request):
            thread = threading.Thread(
                target=adapter._ensure_h264,
                args=(Go2RtcStream("10.1.1.1", "camera"),),
            )
            thread.start()
            self.assertTrue(put_started.wait(1))
            acquired = adapter._lock.acquire(blocking=False)
            if acquired:
                adapter._lock.release()
            release_put.set()
            thread.join(timeout=1)

        self.assertTrue(acquired)
        self.assertFalse(thread.is_alive())

    def test_invalid_stream_response_clears_an_existing_cache_entry(self) -> None:
        adapter = Go2RtcAdapter()
        adapter._streams_cache["10.1.1.1"] = (0.0, {"stale": {}})
        with patch("survng.app.go2rtc.urlopen", return_value=FakeResponse(b"[]")):
            with self.assertRaisesRegex(Go2RtcError, "invalid"):
                adapter._streams("10.1.1.1", force=True)

        self.assertNotIn("10.1.1.1", adapter._streams_cache)

    def test_concurrent_stream_reads_share_one_go2rtc_request(self) -> None:
        adapter = Go2RtcAdapter()
        request_started = threading.Event()
        release_request = threading.Event()
        calls = 0

        def request(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            request_started.set()
            release_request.wait(1)
            return FakeResponse(b'{"camera": {}}')

        results: list[dict] = []
        with patch("survng.app.go2rtc.urlopen", side_effect=request):
            threads = [
                threading.Thread(target=lambda: results.append(adapter._streams("10.1.1.1")))
                for _ in range(4)
            ]
            for thread in threads:
                thread.start()
            self.assertTrue(request_started.wait(1))
            release_request.set()
            for thread in threads:
                thread.join(timeout=1)

        self.assertEqual(calls, 1)
        self.assertEqual(len(results), 4)

    def test_snapshot_rejects_an_unbounded_response(self) -> None:
        adapter = Go2RtcAdapter()
        with patch(
            "survng.app.go2rtc.urlopen",
            return_value=FakeResponse(b"x" * (32 * 1024 * 1024 + 1)),
        ):
            with self.assertRaisesRegex(Go2RtcError, "size limit"):
                adapter._snapshot_bytes("10.1.1.1", "camera")

    def test_snapshot_uses_go2rtc_jpeg(self) -> None:
        adapter = Go2RtcAdapter()
        image = b"\xff\xd8jpeg\xff\xd9"
        with (
            patch.object(adapter, "_streams", return_value={"back_middle_ext": {"producers": [{"medias": ["video, recvonly, H264"]}]}}),
            patch("survng.app.go2rtc.urlopen", return_value=FakeResponse(image)),
        ):
            result = adapter.snapshot(self.camera(), "live")

        self.assertEqual(result, image)

    def test_h265_snapshot_defers_to_camera_worker(self) -> None:
        adapter = Go2RtcAdapter()
        payload = {"back_middle": {"producers": [{"medias": ["video, recvonly, H265"]}]}}
        with (
            patch.object(adapter, "_streams", return_value=payload),
            patch.object(adapter, "_ensure_h264") as ensure,
            patch.object(adapter, "_snapshot_bytes") as snapshot,
        ):
            with self.assertRaises(Go2RtcError):
                adapter.snapshot(self.camera(), "main")

        ensure.assert_not_called()
        snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
