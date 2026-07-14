from __future__ import annotations

import hashlib
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

    def read(self) -> bytes:
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
