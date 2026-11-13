from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


from survng.app.recorded_video_frames import sampled_video_frames, video_frame_at_reference
from survng.app.video_frames import VideoFrameReference


class RecordedVideoFramesTest(unittest.TestCase):
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
                b"[showinfo@source] n: 0 pts: 117000 pts_time:1.3 checksum:AAAA\n"
                b"[showinfo@sampled] n: 0 pts: 3 pts_time:1.5 checksum:AAAA\n"
                b"[showinfo@source] n: 1 pts: 157500 pts_time:1.75 checksum:BBBB\n"
                b"[showinfo@sampled] n: 1 pts: 4 pts_time:2 checksum:BBBB\n"
            ),
            wait=Mock(return_value=0),
            poll=Mock(return_value=0),
            terminate=Mock(),
            kill=Mock(),
        )
        with (
            patch(
                "survng.app.recorded_video_frames.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout='{"streams":[{"width":8,"height":4,"time_base":"1/90000"}]}',
                ),
            ) as ffprobe,
            patch("survng.app.recorded_video_frames.subprocess.Popen", return_value=process) as popen,
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
        self.assertEqual(frames[1].reference.pts, 157500)
        self.assertEqual(frames[1].reference.time_base_den, 90000)
        self.assertEqual([frame.shape for _epoch, frame in frames], [(4, 8, 3), (4, 8, 3)])
        command = popen.call_args.args[0]
        before_input = command[:command.index("-i")]
        after_input = command[command.index("-i") + 2:]
        self.assertEqual(before_input[before_input.index("-threads:v") + 1], "2")
        self.assertEqual(before_input[before_input.index("-filter_threads") + 1], "1")
        self.assertEqual(after_input[after_input.index("-threads:v") + 1], "1")
        self.assertIn(
            "showinfo@source,fps=2.000000,showinfo@sampled,scale=8:4,",
            command[command.index("-vf") + 1],
        )
        self.assertNotIn("-hwaccel", command)
        self.assertEqual(command[command.index("-f") + 1], "concat")
        self.assertEqual(command[command.index("-safe") + 1], "0")
        self.assertIn("trim=start=1.250000:end=2.250000", command[command.index("-vf") + 1])
        self.assertNotIn("-ss", command)

    def test_ffmpeg_sampler_scales_on_hardware_before_download(self) -> None:
        process = SimpleNamespace(
            stdout=BytesIO(bytes(16 * 8 * 3)),
            stderr=BytesIO(
                b"[showinfo@source] n: 0 pts: 585000 pts_time:6.5 checksum:AAAA\n"
                b"[showinfo@sampled] n: 0 pts: 13 pts_time:6.5 checksum:AAAA\n"
            ),
            wait=Mock(return_value=0), poll=Mock(return_value=0),
            terminate=Mock(), kill=Mock(),
        )
        probe = SimpleNamespace(
            returncode=0,
            stdout='{"streams":[{"width":16,"height":8,"time_base":"1/90000"}],'
                   '"format":{"start_time":"0.000000"}}',
        )
        with (
            patch("survng.app.recorded_video_frames.subprocess.run", return_value=probe),
            patch("survng.app.recorded_video_frames.subprocess.Popen", return_value=process) as popen,
        ):
            frames = list(sampled_video_frames(
                Path("segment.mp4"), start_epoch=50.0, sample_fps=2.0, duration_seconds=1.0,
                ffmpeg_path="ffmpeg", maximum_width=8, start_offset_seconds=6.5,
                probe_path=Path("segment.mp4"), hardware_acceleration="qsv",
            ))

        self.assertEqual(len(frames), 1)
        self.assertEqual(popen.call_count, 1)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-hwaccel") + 1], "vaapi")
        self.assertLess(command.index("-hwaccel"), command.index("-i"))
        self.assertEqual(command[command.index("-ss") + 1], "6.250000")
        video_filter = command[command.index("-vf") + 1]
        self.assertTrue(video_filter.startswith(
            "trim=start=6.500000:end=7.500000,scale_vaapi=w=16:h=8:format=nv12,hwdownload,format=nv12,"
            "showinfo@source,fps=2.000000,showinfo@sampled,"
        ))

    def test_ffmpeg_sampler_falls_back_to_software_before_the_first_frame(self) -> None:
        failed = SimpleNamespace(
            stdout=BytesIO(b""), stderr=BytesIO(b""), wait=Mock(return_value=1),
            poll=Mock(return_value=1), terminate=Mock(), kill=Mock(),
        )
        decoded = SimpleNamespace(
            stdout=BytesIO(bytes(96)),
            stderr=BytesIO(
                b"[showinfo@source] n: 0 pts: 117000 pts_time:1.3 checksum:AAAA\n"
                b"[showinfo@sampled] n: 0 pts: 3 pts_time:1.5 checksum:AAAA\n"
            ),
            wait=Mock(return_value=0), poll=Mock(return_value=0),
            terminate=Mock(), kill=Mock(),
        )
        probe = SimpleNamespace(
            returncode=0,
            stdout='{"streams":[{"width":8,"height":4,"time_base":"1/90000"}]}',
        )
        with (
            patch("survng.app.recorded_video_frames.subprocess.run", return_value=probe),
            patch(
                "survng.app.recorded_video_frames.subprocess.Popen",
                side_effect=[failed, decoded],
            ) as popen,
            self.assertLogs("survng.app.recorded_video_frames", level="DEBUG"),
        ):
            frames = list(sampled_video_frames(
                Path("segment.mp4"), start_epoch=50.0, sample_fps=2.0, duration_seconds=1.0,
                ffmpeg_path="ffmpeg", maximum_width=8, start_offset_seconds=1.25,
                hardware_acceleration="vaapi",
            ))

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].reference.pts, 117000)
        hardware, software = (call.args[0] for call in popen.call_args_list)
        self.assertIn("scale_vaapi=w=8:h=4:format=nv12", hardware[hardware.index("-vf") + 1])
        self.assertNotIn("-hwaccel", software)

    def test_ffmpeg_sampler_seeks_to_a_keyframe_on_the_source_timeline(self) -> None:
        def sampled_command(probe_stdout):
            process = SimpleNamespace(
                stdout=BytesIO(b""), stderr=BytesIO(b""), wait=Mock(return_value=0),
                poll=Mock(return_value=0), terminate=Mock(), kill=Mock(),
            )
            with (
                patch("survng.app.recorded_video_frames.subprocess.run",
                      return_value=SimpleNamespace(returncode=0, stdout=probe_stdout)),
                patch("survng.app.recorded_video_frames.subprocess.Popen", return_value=process) as popen,
            ):
                list(sampled_video_frames(
                    Path("segment.mp4"), start_epoch=50.0, sample_fps=2.0, duration_seconds=1.0,
                    ffmpeg_path="ffmpeg", maximum_width=4, start_offset_seconds=6.5,
                    probe_path=Path("segment.mp4"),
                ))
            return popen.call_args.args[0]

        streams = '"streams":[{"width":8,"height":4,"time_base":"1/90000"}]'
        command = sampled_command('{%s,"format":{"start_time":"0.000000"}}' % streams)
        self.assertEqual(command[command.index("-ss") + 1], "6.250000")
        self.assertLess(command.index("-copyts"), command.index("-i"))
        self.assertIn("trim=start=6.500000:end=7.500000", command[command.index("-vf") + 1])
        shifted = sampled_command('{%s,"format":{"start_time":"1.400000"}}' % streams)
        self.assertNotIn("-ss", shifted)
        self.assertNotIn("-copyts", shifted)

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
            "survng.app.recorded_video_frames.subprocess.run",
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
        before_input = command[:command.index("-i")]
        after_input = command[command.index("-i") + 2:]
        self.assertEqual(before_input[before_input.index("-threads:v") + 1], "2")
        self.assertEqual(before_input[before_input.index("-filter_threads") + 1], "1")
        self.assertEqual(after_input[after_input.index("-threads:v") + 1], "1")
        self.assertIn(
            "select='eq(pts\\,32871)'",
            command[command.index("-vf") + 1],
        )
        self.assertNotIn("-ss", command)
        self.assertNotIn("fps=", command[command.index("-vf") + 1])


if __name__ == "__main__":
    unittest.main()
