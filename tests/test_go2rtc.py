from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from survng.app.config import CameraConfig
from survng.app.go2rtc import Go2RtcAdapter, Go2RtcError


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

    def test_stream_info_reports_h265_but_keeps_native_delivery(self) -> None:
        adapter = Go2RtcAdapter()
        payload = {
            "back_middle": {
                "producers": [{"medias": ["video, recvonly, H265", "audio, recvonly, PCMA/8000"]}],
            },
        }
        with patch.object(adapter, "_streams", return_value=payload):
            info = adapter.stream_info(self.camera(), "main")

        self.assertEqual(info["video_codec"], "H265")
        self.assertEqual(info["compatibility"], "native")
        self.assertEqual(info["delivery"], "native")
        self.assertFalse(info["transcoding"])
        self.assertEqual(
            adapter.websocket_url(self.camera(), "main"),
            "ws://10.1.1.1:1984/api/ws?src=back_middle",
        )

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

    def test_audio_stream_info_uses_receiver_codec_metadata(self) -> None:
        adapter = Go2RtcAdapter()
        payload = {
            "producers": [{
                "receivers": [{
                    "codec": {
                        "codec_name": "aac",
                        "codec_type": "audio",
                        "sample_rate": 8000,
                    },
                }],
            }],
        }
        with patch.object(adapter, "_stream_details", return_value=payload):
            info = adapter.audio_stream_info(self.camera(), "main")

        self.assertEqual(info, {
            "available": True,
            "codec": "aac",
            "sample_rate": 8000,
        })

    def test_audio_stream_info_reports_active_video_only_source(self) -> None:
        adapter = Go2RtcAdapter()
        payload = {
            "producers": [{
                "receivers": [{
                    "codec": {"codec_name": "hevc", "codec_type": "video"},
                }],
            }],
        }
        with patch.object(adapter, "_stream_details", return_value=payload):
            info = adapter.audio_stream_info(self.camera(), "live")

        self.assertEqual(info, {"available": True, "codec": "", "sample_rate": 0})

    def test_audio_stream_info_uses_sdp_media_while_receivers_initialize(self) -> None:
        adapter = Go2RtcAdapter()
        payload = {
            "producers": [{
                "medias": [
                    "video, recvonly, H265",
                    "audio, recvonly, MPEG4-GENERIC/8000",
                ],
                "receivers": [],
            }],
        }
        with patch.object(adapter, "_stream_details", return_value=payload):
            info = adapter.audio_stream_info(self.camera(), "main")

        self.assertEqual(info, {"available": True, "codec": "aac", "sample_rate": 8000})

    def test_audio_stream_info_does_not_call_initializing_producer_silent(self) -> None:
        adapter = Go2RtcAdapter()
        with patch.object(adapter, "_stream_details", return_value={"producers": [{}]}):
            info = adapter.audio_stream_info(self.camera(), "main")

        self.assertEqual(info, {"available": False, "codec": "", "sample_rate": 0})

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
            patch.object(adapter, "_snapshot_bytes") as snapshot,
        ):
            with self.assertRaises(Go2RtcError):
                adapter.snapshot(self.camera(), "main")

        snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
