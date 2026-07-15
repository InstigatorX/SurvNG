from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from survng.app.recording_media import (
    hls_map_transition,
    mp4_stream_fingerprint,
    playback_segment_duration,
    resolve_stream_fingerprints,
)


def box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def video_entry(codec: bytes, config: bytes, dimensions: bytes = b"1280x720", bitrate: int = 1) -> bytes:
    fixed = bytearray(78)
    fixed[24:28] = dimensions[:4].ljust(4, b" ")
    children = box(b"avcC" if codec == b"avc1" else b"hvcC", config)
    children += box(b"btrt", struct.pack(">III", 0, bitrate, bitrate))
    return box(codec, bytes(fixed) + children)


def audio_entry(config: bytes) -> bytes:
    fixed = bytearray(28)
    fixed[16:20] = struct.pack(">HH", 2, 16)
    fixed[24:28] = struct.pack(">I", 48_000 << 16)
    return box(b"mp4a", bytes(fixed) + box(b"dOps", config))


def stream_track(handler: bytes, sample_entry: bytes, timescale: int = 90_000) -> bytes:
    mdhd = box(
        b"mdhd",
        bytes(12) + struct.pack(">I", timescale) + bytes(8),
    )
    hdlr = box(b"hdlr", bytes(8) + handler + bytes(12))
    stsd = box(b"stsd", bytes(4) + struct.pack(">I", 1) + sample_entry)
    return box(b"trak", box(b"mdia", mdhd + hdlr + box(b"minf", box(b"stbl", stsd))))


def recording_file(video_entry: bytes, audio_entry: bytes | None = None, noise: bytes = b"") -> bytes:
    tracks = stream_track(b"vide", video_entry)
    if audio_entry is not None:
        tracks += stream_track(b"soun", audio_entry, timescale=48_000)
    return (
        box(b"ftyp", b"isom" + bytes(12))
        + box(b"free", noise)
        + box(b"moov", tracks)
        + box(b"mdat", b"frame")
    )


class RecordingMediaTest(unittest.TestCase):
    def test_event_fragment_duration_trims_only_at_window_end(self) -> None:
        self.assertEqual(playback_segment_duration(100.0, 10.0, 107.25, True), 7.25)
        self.assertEqual(playback_segment_duration(100.0, 10.0, 115.0, True), 10.0)

    def test_recording_fragment_duration_remains_full_without_trim_flag(self) -> None:
        self.assertEqual(playback_segment_duration(100.0, 10.0, 104.0, False), 10.0)

    def test_unknown_fingerprints_do_not_create_false_transitions(self) -> None:
        self.assertEqual(
            resolve_stream_fingerprints(["", "h265", "", "h264", "", ""]),
            ["h265", "h265", "h265", "h264", "h264", "h264"],
        )

    def test_all_unknown_fingerprints_remain_unknown(self) -> None:
        self.assertEqual(resolve_stream_fingerprints(["", None, ""]), ["", "", ""])

    def test_fingerprint_uses_stream_metadata_not_unrelated_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.mp4"
            second = Path(tmpdir) / "second.mp4"
            first.write_bytes(recording_file(video_entry(b"avc1", b"same", bitrate=1), noise=b"one"))
            second.write_bytes(recording_file(video_entry(b"avc1", b"same", bitrate=2), noise=b"different"))

            self.assertEqual(mp4_stream_fingerprint(first), mp4_stream_fingerprint(second))

    def test_fingerprint_changes_with_video_or_audio_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base.mp4"
            resized = Path(tmpdir) / "resized.mp4"
            audio = Path(tmpdir) / "audio.mp4"
            base.write_bytes(recording_file(video_entry(b"avc1", b"config"), noise=b"base"))
            resized.write_bytes(recording_file(video_entry(b"avc1", b"config", dimensions=b"1920"), noise=b"base"))
            audio.write_bytes(recording_file(video_entry(b"avc1", b"config"), audio_entry(b"aac")))

            base_fingerprint = mp4_stream_fingerprint(base)
            self.assertTrue(base_fingerprint)
            self.assertNotEqual(base_fingerprint, mp4_stream_fingerprint(resized))
            self.assertNotEqual(base_fingerprint, mp4_stream_fingerprint(audio))

    def test_map_transition_only_emits_discontinuity_after_first_map(self) -> None:
        first_lines, previous = hls_map_transition(None, "video-a", "day/0/init.mp4")
        same_lines, previous = hls_map_transition(previous, "video-a", "day/1/init.mp4")
        changed_lines, previous = hls_map_transition(previous, "video-b", "day/2/init.mp4")

        self.assertEqual(first_lines, ['#EXT-X-MAP:URI="day/0/init.mp4"'])
        self.assertEqual(same_lines, [])
        self.assertEqual(changed_lines, [
            "#EXT-X-DISCONTINUITY",
            '#EXT-X-MAP:URI="day/2/init.mp4"',
        ])
        self.assertEqual(previous, "video-b")


if __name__ == "__main__":
    unittest.main()
