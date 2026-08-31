from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.object_tracking import ByteTrackObjectTracker, ObjectTrackerRegistry
from survng.app.tracking_comparison import (
    TrackingComparisonRunner,
    sampled_video_frames,
    video_frame_at_reference,
)
from survng.app.video_frames import VideoFrameReference


class Detector:
    config = SimpleNamespace(confidence_threshold=0.45)

    def detect(self, frame, confidence_threshold=None):
        offset = int(frame[0, 0, 0])
        return [{
            "label": "person",
            "confidence": 0.9,
            "box": {"x1": 10 + offset, "y1": 10, "x2": 40 + offset, "y2": 80},
        }]


class TrackingComparisonRunnerTest(unittest.TestCase):
    def test_excluded_classes_are_not_in_offline_comparisons(self) -> None:
        class DetectorWithFace(Detector):
            def detect(self, frame, confidence_threshold=None):
                return [
                    *super().detect(frame, confidence_threshold),
                    {
                        "label": "face",
                        "confidence": 0.9,
                        "box": {"x1": 15, "y1": 12, "x2": 30, "y2": 30},
                    },
                ]

        registry = ObjectTrackerRegistry()
        registry.register("survng_hybrid", ByteTrackObjectTracker)
        registry.register("ultralytics_fasttrack", ByteTrackObjectTracker)
        runner = TrackingComparisonRunner(
            config=ObjectTrackingConfig(min_confirmations=1, excluded_labels=["face"]),
            detector=DetectorWithFace(),
            tracker_registry=registry,
        )

        result = runner.run(
            CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            [(100.0, np.zeros((100, 120, 3), dtype=np.uint8))],
        )

        for engine in result["engines"].values():
            self.assertEqual(engine["labels"], {"person": 1})

    def test_runs_both_engines_on_the_same_detections_and_reports_metrics(self) -> None:
        registry = ObjectTrackerRegistry()
        registry.register("survng_hybrid", ByteTrackObjectTracker)
        registry.register("ultralytics_fasttrack", ByteTrackObjectTracker)
        config = ObjectTrackingConfig(min_confirmations=1, sample_fps=2.0)
        runner = TrackingComparisonRunner(
            config=config,
            detector=Detector(),
            tracker_registry=registry,
        )
        frames = [
            (100.0 + index * 0.5, np.full((100, 120, 3), index * 2, dtype=np.uint8))
            for index in range(3)
        ]

        result = runner.run(
            CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            frames,
        )

        self.assertEqual(result["frames_processed"], 3)
        self.assertEqual(result["frame_width"], 120)
        self.assertEqual(result["duration_seconds"], 1.0)
        self.assertEqual(result["lost_timeout_seconds"], 3.0)
        for engine in result["engines"].values():
            self.assertEqual(engine["track_count"], 1)
            self.assertEqual(engine["observations"], 3)
            self.assertEqual(engine["fragmentation_proxy"], 0)
            self.assertEqual(len(engine["tracks"][0]["box_history"]), 3)

    def test_rejects_an_empty_or_unreadable_frame_sequence(self) -> None:
        registry = ObjectTrackerRegistry()
        registry.register("survng_hybrid", ByteTrackObjectTracker)
        registry.register("ultralytics_fasttrack", ByteTrackObjectTracker)
        runner = TrackingComparisonRunner(
            config=ObjectTrackingConfig(),
            detector=Detector(),
            tracker_registry=registry,
        )

        with self.assertRaisesRegex(RuntimeError, "no readable frames"):
            runner.run(
                CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
                [],
            )

    def test_appearance_failure_is_reported_without_losing_geometry_results(self) -> None:
        class Encoder:
            enabled = True

            def embed(self, _crop):
                raise RuntimeError("worker unavailable")

        registry = ObjectTrackerRegistry()
        registry.register("survng_hybrid", ByteTrackObjectTracker)
        registry.register("ultralytics_fasttrack", ByteTrackObjectTracker)
        runner = TrackingComparisonRunner(
            config=ObjectTrackingConfig(
                reid_enabled=True,
                reid_model_path="person-reid.xml",
                min_confirmations=1,
            ),
            detector=Detector(),
            tracker_registry=registry,
            appearance_encoder=Encoder(),
        )

        result = runner.run(
            CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            [(100.0, np.zeros((100, 120, 3), dtype=np.uint8))],
        )

        self.assertEqual(result["appearance_failures"], 1)
        self.assertEqual(result["engines"]["survng_hybrid"]["track_count"], 1)

    def test_video_sampler_requires_external_ffmpeg(self) -> None:
        with self.assertRaisesRegex(ValueError, "ffmpeg_path is required"):
            list(sampled_video_frames(
                Path("comparison.mp4"),
                start_epoch=50.0,
                sample_fps=2.0,
                duration_seconds=1.25,
                ffmpeg_path="",
            ))

    def test_ffmpeg_sampler_decodes_only_model_sized_sampled_frames(self) -> None:
        payload = bytes(range(96)) * 2
        process = SimpleNamespace(
            stdout=BytesIO(payload),
            stderr=BytesIO(
                b"[showinfo@source] n: 0 pts: 4500 pts_time:0.05 checksum:AAAA\n"
                b"[showinfo@sampled] n: 0 pts: 0 pts_time:0 checksum:AAAA\n"
                b"[showinfo@source] n: 1 pts: 45000 pts_time:0.5 checksum:BBBB\n"
                b"[showinfo@sampled] n: 1 pts: 1 pts_time:0.5 checksum:BBBB\n"
            ),
            wait=Mock(return_value=0),
            poll=Mock(return_value=0),
            terminate=Mock(),
            kill=Mock(),
        )
        with (
            patch(
                "survng.app.tracking_comparison.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout='{"streams":[{"width":8,"height":4,"time_base":"1/90000"}]}',
                ),
            ) as ffprobe,
            patch("survng.app.tracking_comparison.subprocess.Popen", return_value=process) as popen,
        ):
            frames = list(sampled_video_frames(
                Path("comparison.ffconcat"),
                start_epoch=50.0,
                sample_fps=2.0,
                duration_seconds=1.0,
                ffmpeg_path="ffmpeg",
                maximum_width=4,
                start_offset_seconds=1.25,
                concat_input=True,
                probe_path=Path("first-segment.mp4"),
            ))

        self.assertEqual(ffprobe.call_args.args[0][-1], "first-segment.mp4")
        self.assertEqual([epoch for epoch, _frame in frames], [50.05, 50.5])
        self.assertEqual(frames[1].reference.pts, 45000)
        self.assertEqual(frames[1].reference.time_base_den, 90000)
        self.assertEqual([frame.shape for _epoch, frame in frames], [(4, 8, 3), (4, 8, 3)])
        command = popen.call_args.args[0]
        self.assertIn(
            "scale=8:4,showinfo@source,fps=2.000000,showinfo@sampled",
            command[command.index("-vf") + 1],
        )
        self.assertEqual(command[command.index("-f") + 1], "concat")
        self.assertEqual(command[command.index("-safe") + 1], "0")
        self.assertEqual(command[command.index("-ss") + 1], "1.250")

    def test_exact_frame_reference_redecodes_the_identified_pts(self) -> None:
        reference = VideoFrameReference(
            source_path=Path("segment.mp4"),
            seek_offset_seconds=6.0,
            pts=32871,
            pts_seconds=0.365233,
            time_base_num=1,
            time_base_den=90000,
            captured_at=100.365233,
        )
        raw = bytes(range(96))
        probe = SimpleNamespace(
            returncode=0,
            stdout='{"streams":[{"width":8,"height":4,"time_base":"1/90000"}]}',
        )
        decoded = SimpleNamespace(returncode=0, stdout=raw)

        with patch(
            "survng.app.tracking_comparison.subprocess.run",
            side_effect=[probe, decoded],
        ) as run:
            sample = video_frame_at_reference(
                reference,
                ffmpeg_path="ffmpeg",
                maximum_width=8,
            )

        self.assertIsNotNone(sample)
        self.assertEqual(sample.captured_at, reference.captured_at)
        self.assertIs(sample.reference, reference)
        self.assertEqual(sample.frame.shape, (4, 8, 3))
        command = run.call_args_list[1].args[0]
        self.assertIn(
            "select='eq(pts\\,32871)'",
            command[command.index("-vf") + 1],
        )
        self.assertEqual(command[command.index("-ss") + 1], "6.000")


if __name__ == "__main__":
    unittest.main()
