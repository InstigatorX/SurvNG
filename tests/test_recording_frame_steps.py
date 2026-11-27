"""Frame stepping: frame lists, exact frames decoded a run at a time, neighbours."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from survng.app import main
from survng.app.recording_frame_steps import (
    SegmentFrames,
    frame_index_at,
    frame_run,
    parse_packet_listing,
    scaled_size,
)
from survng.app.recording_routes import RecordingRouteDependencies, create_recording_router


class FrameStepHelpersTest(unittest.TestCase):
    def test_packet_listing_keeps_presented_frames_in_display_order(self) -> None:
        listing = json.dumps({
            "packets": [
                # Decode order with B-frames; one packet outside the edit
                # list; one without a time.
                {"pts_time": "0.100000", "flags": "K__"},
                {"pts_time": "0.250000", "flags": "___"},
                {"pts_time": "0.150000", "flags": "___"},
                {"pts_time": "0.200000", "flags": "___"},
                {"pts_time": "0.050000", "flags": "__D"},
                {"pts_time": "N/A", "flags": "___"},
                {"pts_time": "0.300000", "flags": "K__"},
            ],
            "format": {"start_time": "0.100000"},
        })
        frames = parse_packet_listing(listing)
        # Times count from the file's start time, as preview.jpg's do.
        self.assertEqual(frames.times, (0.0, 0.05, 0.1, 0.15, 0.2))
        self.assertEqual(frames.keyframes, (0, 4))
        self.assertEqual(frames.start_time, 0.1)
        self.assertEqual(parse_packet_listing("not json"), SegmentFrames((), ()))

    def test_the_frame_on_screen_is_the_last_at_or_before_a_time(self) -> None:
        times = (0.3, 0.35, 0.4)
        self.assertEqual(frame_index_at(times, 0.0), 0)
        self.assertEqual(frame_index_at(times, 0.349), 0)
        # A frame time printed to the microsecond and added to an epoch
        # still finds its own frame.
        self.assertEqual(frame_index_at(times, 0.35 - 0.0000004), 1)
        self.assertEqual(frame_index_at(times, 9.0), 2)
        with self.assertRaises(ValueError):
            frame_index_at((), 1.0)

    def test_a_run_is_one_gop_or_a_slice_of_a_long_one(self) -> None:
        frames = SegmentFrames(tuple(index / 20 for index in range(100)), (0, 20, 40, 100 - 1))
        self.assertEqual(frame_run(frames, 0), (0, 20))
        self.assertEqual(frame_run(frames, 19), (0, 20))
        self.assertEqual(frame_run(frames, 20), (20, 40))
        # A GOP longer than a run is sliced from its keyframe.
        self.assertEqual(frame_run(frames, 45, max_frames=30), (40, 70))
        self.assertEqual(frame_run(frames, 75, max_frames=30), (70, 99))
        self.assertEqual(frame_run(frames, 99), (99, 100))
        # Without keyframe flags, fixed slices from the first frame.
        self.assertEqual(frame_run(SegmentFrames(frames.times, ()), 65, max_frames=30), (60, 90))
        with self.assertRaises(IndexError):
            frame_run(frames, 100)

    def test_frames_are_never_upscaled_and_keep_an_even_size(self) -> None:
        self.assertEqual(scaled_size((2560, 1920), 1280), (1280, 960))
        self.assertEqual(scaled_size((896, 672), 1280), (896, 672))
        self.assertEqual(scaled_size((2560, 1440), 1001), (1000, 562))
        self.assertIsNone(scaled_size(None, 1280))


# Brightness encodes which frame this is: luma = 30 + 3 * frame number. (The
# frame number rather than geq's T: at a fractional rate T is rounded and two
# frames can share a brightness.)
CLOCK_FPS = "20000/1007"  # about 19.86 fps: frame times are not a 1/20 grid


class FrameStepDecodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg or not shutil.which("ffprobe"):
            self.skipTest("ffmpeg unavailable")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "recordings").mkdir()
        self.manager = SimpleNamespace(
            database_dir=self.root / "database",
            recorder=SimpleNamespace(recordings_dir=self.root / "recordings"),
            config=SimpleNamespace(ffmpeg_path=self.ffmpeg),
        )
        self.runtime = main._recording_media_runtime
        self.runtime.recording_frame_lists.clear()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def clip(self, name: str, *, seconds: float, first_frame: int = 0, extra: tuple[str, ...] = ()) -> Path:
        """A clip whose frame n has luma 30 + 3 * (n + first_frame)."""
        path = self.root / "recordings" / name
        subprocess.run([
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            f"color=c=black:size=320x240:rate={CLOCK_FPS}:duration={seconds},format=gray,"
            f"geq=lum='30+3*(N+{first_frame})'",
            "-c:v", "libx264", "-qp", "0", "-g", "10", "-bf", "2", "-pix_fmt", "yuv420p",
            *extra, "-y", str(path),
        ], check=True, capture_output=True, timeout=120)
        return path

    @staticmethod
    def shown_frame(path: Path) -> int:
        import cv2

        image = cv2.imread(str(path), 0)
        return round((float(image.mean()) - 30) / 3)

    def patched(self):
        return (
            patch.object(main, "manager", self.manager),
            patch.object(self.runtime, "_maintain_recording_preview_cache"),
            patch.object(self.runtime, "_hardware_acceleration_mode", return_value="off"),
        )

    def step_frame(self, row: dict, epoch: float, width: int = 320):
        first, second, third = self.patched()
        with first, second, third:
            return self.runtime._recording_step_frame(row, epoch, width=width)

    def segment_frames(self, row: dict) -> SegmentFrames:
        first, second, third = self.patched()
        with first, second, third:
            return self.runtime._recording_segment_frames(row)

    def test_every_frame_is_the_exact_frame_and_a_gop_is_decoded_once(self) -> None:
        # B-frames, a keyframe every 10 frames.
        path = self.clip("clock.mp4", seconds=2.0)
        row = {"path": str(path), "start_epoch": 1000.0, "end_epoch": 1002.0}
        frames = self.segment_frames(row)
        self.assertEqual(len(frames.times), 40)
        self.assertEqual(frames.keyframes[:2], (0, 10))
        # Real frame times, not multiples of 1/20.
        self.assertAlmostEqual(frames.times[1], 1007 / 20000, places=5)
        decoded = []
        real_decode = self.runtime._decode_frame_run

        def counting(*args, **kwargs):
            decoded.append(args[3:5])
            return real_decode(*args, **kwargs)

        with patch.object(self.runtime, "_decode_frame_run", side_effect=counting):
            for index, offset in enumerate(frames.times):
                epoch = round(1000.0 + offset, 6)
                path_jpg, actual = self.step_frame(row, epoch)
                self.assertAlmostEqual(actual, 1000.0 + offset, places=5)
                self.assertEqual(self.shown_frame(path_jpg), index)
            # A time between two frames shows the earlier one.
            between = 1000.0 + (frames.times[14] + frames.times[15]) / 2
            path_jpg, actual = self.step_frame(row, between)
            self.assertAlmostEqual(actual, 1000.0 + frames.times[14], places=5)
        # One decoder process per GOP, however many frames were shown.
        self.assertEqual(decoded, [(0, 10), (10, 20), (20, 30), (30, 40)])
        # The last frame of the segment is a frame like any other.
        self.assertAlmostEqual(actual, 1000.0 + frames.times[14], places=5)
        last, last_actual = self.step_frame(row, 1001.999)
        self.assertAlmostEqual(last_actual, 1000.0 + frames.times[-1], places=5)
        self.assertEqual(self.shown_frame(last), len(frames.times) - 1)
        # Runs live in the bounded preview cache, as JPEGs with a sidecar.
        cached = sorted(item.suffix for item in (self.root / "database" / "recording-preview-cache").iterdir())
        self.assertEqual(set(cached), {".jpg", ".json"})

    def test_a_video_track_that_starts_after_the_audio(self) -> None:
        plain = self.clip("plain.mp4", seconds=1.5)
        late = self.root / "recordings" / "late.mp4"
        subprocess.run([
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-itsoffset", "0.3",
            "-i", str(plain), "-f", "lavfi", "-i", "sine=frequency=440:duration=1.8",
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-y", str(late),
        ], check=True, capture_output=True, timeout=120)
        row = {"path": str(late), "start_epoch": 1000.0, "end_epoch": 1001.8}
        frames = self.segment_frames(row)
        self.assertAlmostEqual(frames.times[0], 0.3, delta=0.03)
        # Before the first video frame: the first frame there is.
        first, actual = self.step_frame(row, 1000.1)
        self.assertAlmostEqual(actual, 1000.0 + frames.times[0], places=5)
        self.assertEqual(self.shown_frame(first), 0)
        # Every frame keeps its place.
        for index in (1, 9, 10, 17):
            image, actual = self.step_frame(row, 1000.0 + frames.times[index])
            self.assertAlmostEqual(actual, 1000.0 + frames.times[index], places=5)
            self.assertEqual(self.shown_frame(image), index)

    def test_a_file_whose_start_time_is_not_zero(self) -> None:
        # AAC priming puts the container's start time before the first
        # video frame; an offset moves both. Frame times count from the
        # start time, as preview.jpg's do, and every frame is still itself.
        for name, extra in (
            ("offset.mp4", ("-output_ts_offset", "0.5")),
            ("primed.mp4", ()),
        ):
            with self.subTest(name):
                self.runtime.recording_frame_lists.clear()
                path = self.root / "recordings" / name
                subprocess.run([
                    self.ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=c=black:size=320x240:rate={CLOCK_FPS}:duration=1.5,format=gray,geq=lum='30+3*N'",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
                    "-c:v", "libx264", "-qp", "0", "-g", "10", "-bf", "2", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", *extra, "-y", str(path),
                ], check=True, capture_output=True, timeout=120)
                row = {"path": str(path), "start_epoch": 1000.0, "end_epoch": 1001.6}
                frames = self.segment_frames(row)
                if extra:
                    self.assertNotEqual(frames.start_time, 0.0)
                for index in range(len(frames.times)):
                    image, actual = self.step_frame(row, 1000.0 + frames.times[index])
                    self.assertAlmostEqual(actual, 1000.0 + frames.times[index], places=5)
                    self.assertEqual(self.shown_frame(image), index)

    def test_the_segment_muxer_recipe_the_recorder_uses(self) -> None:
        # -f segment with reset timestamps, as the recorder cuts files, from
        # H.264 with B-frames and AAC.
        pattern = self.root / "recordings" / "cut%02d.mp4"
        subprocess.run([
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            f"color=c=black:size=320x240:rate={CLOCK_FPS}:duration=2,format=gray,geq=lum='30+3*N'",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-qp", "0", "-g", "10", "-bf", "2", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-f", "segment", "-segment_time", "1", "-reset_timestamps", "1",
            "-segment_format", "mp4", "-y", str(pattern),
        ], check=True, capture_output=True, timeout=120)
        path = self.root / "recordings" / "cut00.mp4"
        row = {"path": str(path), "start_epoch": 1000.0, "end_epoch": 1001.2}
        frames = self.segment_frames(row)
        for index in range(len(frames.times)):
            image, actual = self.step_frame(row, 1000.0 + frames.times[index])
            self.assertAlmostEqual(actual, 1000.0 + frames.times[index], places=5)
            self.assertEqual(self.shown_frame(image), index)

    def test_a_frame_the_decoder_did_not_produce_is_an_error_not_another_frame(self) -> None:
        path = self.clip("short.mp4", seconds=1.0)
        row = {"path": str(path), "start_epoch": 1000.0, "end_epoch": 1001.0}
        frames = self.segment_frames(row)
        real_decode = self.runtime._decode_frame_run

        def dropping_the_first(*args, **kwargs):
            return real_decode(*args, **kwargs)[1:]

        with patch.object(self.runtime, "_decode_frame_run", side_effect=dropping_the_first):
            with self.assertRaises(HTTPException) as missing:
                self.step_frame(row, 1000.0 + frames.times[0])
        self.assertEqual(missing.exception.status_code, 500)
        # The frames it did decode are still themselves.
        image, actual = self.step_frame(row, 1000.0 + frames.times[1])
        self.assertEqual(self.shown_frame(image), 1)

    def test_the_gpu_decodes_and_scales_under_auto_and_the_cpu_takes_over(self) -> None:
        from survng.app import recording_media_runtime as runtime_module
        from survng.app import recording_frame_runs as frame_runs_module
        from survng.app.recorded_decoder import FAILURE_HARDWARE_INIT, DecodeResult

        path = self.clip("gpu.mp4", seconds=1.0)
        row = {"path": str(path), "start_epoch": 1000.0, "end_epoch": 1001.0}
        frames = self.segment_frames(row)
        commands: list[list[str]] = []
        real_run_decoder = frame_runs_module.run_decoder

        def run_decoder(command, **kwargs):
            commands.append(list(command))
            if "-hwaccel" in command:
                # No such device here: what a GPU that cannot be opened does.
                return DecodeResult(command, kwargs["backend"], 1, None, b"", FAILURE_HARDWARE_INIT, "no VAAPI")
            return real_run_decoder(command, **kwargs)

        def step(epoch: float, *, mode: str = "auto", dimensions=(2560, 1920)):
            with (
                patch.object(main, "manager", self.manager),
                patch.object(self.runtime, "_maintain_recording_preview_cache"),
                patch.object(self.runtime, "_hardware_acceleration_mode", return_value=mode),
                patch.object(
                    self.runtime, "_ffmpeg_vaapi_info",
                    return_value={"available": True, "device": "/dev/dri/renderD129"},
                ),
                patch.object(frame_runs_module, "run_decoder", side_effect=run_decoder),
                patch.object(runtime_module, "mp4_video_dimensions", return_value=dimensions),
            ):
                return self.runtime._recording_step_frame(row, epoch, width=320)

        self.addCleanup(setattr, self.runtime, "recording_strip_hardware_retry_at", 0.0)
        self.runtime.recording_strip_hardware_retry_at = 0.0
        with self.assertLogs("survng.app.recording_frame_runs", level="WARNING"):
            image, actual = step(1000.0 + frames.times[3])
        # Tried on the GPU first, decoding and scaling there, then the CPU.
        self.assertEqual(len(commands), 2)
        gpu, cpu = commands
        self.assertEqual(gpu[gpu.index("-hwaccel") + 1], "vaapi")
        self.assertEqual(gpu[gpu.index("-vaapi_device") + 1], "/dev/dri/renderD129")
        self.assertLess(gpu.index("-hwaccel"), gpu.index("-i"))
        gpu_filter = gpu[gpu.index("-vf") + 1]
        self.assertIn("scale_vaapi=w=320:h=240", gpu_filter)
        self.assertLess(gpu_filter.index("hwdownload"), gpu_filter.index("showinfo"))
        self.assertNotIn("-hwaccel", cpu)
        # The CPU's frame is still the exact one.
        self.assertEqual(self.shown_frame(image), 3)
        self.assertAlmostEqual(actual, 1000.0 + frames.times[3], places=5)
        # A GPU that just failed is not tried again for the next run.
        self.assertGreater(self.runtime.recording_strip_hardware_retry_at, 0.0)
        commands.clear()
        step(1000.0 + frames.times[12])
        self.assertEqual(len(commands), 1)
        self.assertNotIn("-hwaccel", commands[0])
        # Hardware turned off, or a frame of unknown size: never the GPU.
        self.runtime.recording_strip_hardware_retry_at = 0.0
        for mode, dimensions, index in (("off", (2560, 1920), 5), ("auto", None, 15)):
            commands.clear()
            # Nothing cached: each case decodes.
            shutil.rmtree(self.root / "database", ignore_errors=True)
            step(1000.0 + frames.times[index], mode=mode, dimensions=dimensions)
            self.assertEqual(len(commands), 1)
            self.assertNotIn("-hwaccel", commands[0])

    def test_frames_past_100_seconds_into_a_file_are_still_exact(self) -> None:
        # FFmpeg prints frame times to six significant digits: past 100 s
        # that is a millisecond, as coarse as the matching tolerance. A
        # 90 kHz time base at 19.84 fps lands some frames right on it.
        path = self.root / "recordings" / "long.mp4"
        subprocess.run([
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=c=gray:size=64x48:rate=19.84:duration=106", "-c:v", "libx264", "-g", "20",
            "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", "-y", str(path),
        ], check=True, capture_output=True, timeout=120)
        row = {"path": str(path), "start_epoch": 1000.0, "end_epoch": 1106.0}
        frames = self.segment_frames(row)
        for index in (2015, 2077):  # 101.5625 s and 104.6875 s
            _image, actual = self.step_frame(row, 1000.0 + frames.times[index])
            self.assertAlmostEqual(actual, 1000.0 + frames.times[index], places=5)

    def test_a_short_or_failed_gpu_run_falls_back_without_turning_the_gpu_off(self) -> None:
        from survng.app import recording_media_runtime as runtime_module
        from survng.app import recording_frame_runs as frame_runs_module
        from survng.app.recorded_decoder import FAILURE_BITSTREAM, DecodeResult

        path = self.clip("partial.mp4", seconds=1.0)
        row = {"path": str(path), "start_epoch": 1000.0, "end_epoch": 1001.0}
        frames = self.segment_frames(row)
        real_run_decoder = frame_runs_module.run_decoder
        commands: list[list[str]] = []

        def gpu_as(outcome):
            def run_decoder(command, **kwargs):
                commands.append(list(command))
                if "-hwaccel" not in command:
                    return real_run_decoder(command, **kwargs)
                if outcome == "short":
                    # Exit 0 with frames missing, as a GPU decoder can.
                    cpu = [part for part in command]
                    hw = cpu.index("-vaapi_device")
                    del cpu[hw:hw + 6]
                    cpu[cpu.index("-vf") + 1] = "scale=320:240,showinfo"
                    result = real_run_decoder(cpu, **kwargs)
                    result.output = result.output[:3]
                    return result
                return DecodeResult(command, kwargs["backend"], 1, None, b"", FAILURE_BITSTREAM, "corrupt")
            return run_decoder

        self.addCleanup(setattr, self.runtime, "recording_strip_hardware_retry_at", 0.0)
        for outcome, index in (("short", 4), ("corrupt", 14)):
            with self.subTest(outcome):
                self.runtime.recording_strip_hardware_retry_at = 0.0
                commands.clear()
                with (
                    patch.object(main, "manager", self.manager),
                    patch.object(self.runtime, "_maintain_recording_preview_cache"),
                    patch.object(self.runtime, "_hardware_acceleration_mode", return_value="auto"),
                    patch.object(self.runtime, "_ffmpeg_vaapi_info", return_value={"available": True, "device": "/dev/dri/renderD128"}),
                    patch.object(frame_runs_module, "run_decoder", side_effect=gpu_as(outcome)),
                    patch.object(runtime_module, "mp4_video_dimensions", return_value=(2560, 1920)),
                ):
                    image, actual = self.runtime._recording_step_frame(row, 1000.0 + frames.times[index], width=320)
                # The CPU made the whole run; the frame is exact.
                self.assertEqual(len(commands), 2)
                self.assertNotIn("-hwaccel", commands[1])
                self.assertEqual(self.shown_frame(image), index)
                # A problem with this file is not a broken GPU.
                self.assertEqual(self.runtime.recording_strip_hardware_retry_at, 0.0)

    def test_a_run_in_use_stays_fresh_as_a_whole_and_big_runs_bring_maintenance_forward(self) -> None:
        import os

        from survng.app import recording_media_runtime as runtime_module

        path = self.clip("fresh.mp4", seconds=0.5)
        row = {"path": str(path), "start_epoch": 1000.0, "end_epoch": 1000.5}
        frames = self.segment_frames(row)
        self.addCleanup(setattr, self.runtime, "recording_frame_bytes_since_maintenance", 0)
        self.runtime.recording_preview_last_maintenance = 0.0
        with patch.object(runtime_module, "RECORDING_FRAME_MAINTENANCE_BYTES", 1):
            self.step_frame(row, 1000.0 + frames.times[0])
        # Enough new bytes force the next maintenance pass, even soon after boot.
        self.assertEqual(self.runtime.recording_preview_last_maintenance, float("-inf"))
        self.runtime.recording_preview_last_maintenance = 0.0
        cache = self.root / "database" / "recording-preview-cache"
        files = sorted(cache.glob("*.jpg"))
        self.assertGreater(len(files), 2)
        for item in files:
            os.utime(item, (1_000_000, 1_000_000))
        self.step_frame(row, 1000.0 + frames.times[2])
        # Serving one frame keeps every frame of its run, used or not.
        self.assertTrue(all(item.stat().st_mtime > 1_000_000 for item in files))

    def test_a_large_frame_is_scaled_to_the_width_asked_for(self) -> None:
        import cv2

        path = self.clip("wide.mp4", seconds=0.5)
        row = {"path": str(path), "start_epoch": 1000.0, "end_epoch": 1000.5}
        image, _actual = self.step_frame(row, 1000.2, width=320)
        self.assertEqual(cv2.imread(str(image)).shape[:2], (240, 320))
        # Never upscaled past the source.
        image, _actual = self.step_frame(row, 1000.2, width=1280)
        self.assertEqual(cv2.imread(str(image)).shape[:2], (240, 320))

    def route(self, rows: list[dict]):
        manager = SimpleNamespace(
            camera=lambda camera_id: object() if camera_id == "gate" else None,
            recorder=SimpleNamespace(
                recording_rows_between=lambda _camera, start, end, _source, **_kwargs: [
                    row for row in rows if row["start_epoch"] < end and row["end_epoch"] > start
                ],
            ),
        )
        dependencies = RecordingRouteDependencies(
            get_manager=lambda: manager,
            get_config=lambda: SimpleNamespace(recording_segment_seconds=10, cameras=[]),
            get_media_exports=lambda: SimpleNamespace(),
            public_url=lambda value: value,
            recording_rows=lambda *_args, **_kwargs: [],
            recording_day_rows=lambda *_args, **_kwargs: [],
            recording_preview_path=lambda *_args, **_kwargs: Path("preview.jpg"),
            recording_preview_timestamp=lambda _path: (None, "requested_offset"),
            recording_segment_path=lambda *_args, **_kwargs: Path("segment.mp4"),
            recording_day_fmp4_paths=lambda *_args, **_kwargs: (Path("init.mp4"), Path("media.m4s")),
            recording_file_response=lambda *_args, **_kwargs: None,
            event_clip_window=lambda *_args: (5.0, 5.0),
            ensure_event_clip=lambda *_args, **_kwargs: Path("clip.mp4"),
        )
        dependencies = replace(
            dependencies,
            recording_segment_frames=lambda _manager, row: self.segment_frames(row),
            recording_step_frame=lambda _manager, row, epoch, *, width: self.step_frame(row, epoch, width),
        )
        handlers = create_recording_router(dependencies).handlers
        return handlers["recording_frames"], handlers["recording_frame"]

    def test_stepping_continues_into_the_next_segment_and_stops_at_a_gap(self) -> None:
        first = self.clip("a.mp4", seconds=1.0)
        second = self.clip("b.mp4", seconds=1.0, first_frame=20)
        after_gap = self.clip("c.mp4", seconds=1.0, first_frame=40)
        rows = [
            {"path": str(first), "start_epoch": 1000.0, "end_epoch": 1001.0},
            # Indexed 20 ms late, as the recorder's cuts are.
            {"path": str(second), "start_epoch": 1001.02, "end_epoch": 1002.02},
            {"path": str(after_gap), "start_epoch": 1010.0, "end_epoch": 1011.0},
        ]
        frames_json, frame_jpg = self.route(rows)

        listing = json.loads(frames_json("gate", 1000.5, "main").body)
        self.assertEqual(listing["start"], 1000.0)
        self.assertEqual(len(listing["frames"]), 20)
        self.assertIsNone(listing["previous"])
        self.assertEqual(listing["next"], 1001.02)
        # The frame after the last of this segment is the first of the next.
        following = json.loads(frames_json("gate", listing["next"], "main").body)
        self.assertEqual(following["start"], 1001.02)
        self.assertAlmostEqual(following["previous"], 1000.999, places=6)
        response = frame_jpg("gate", following["frames"][0], "main", 320)
        self.assertEqual(response.headers["x-survng-timestamp-source"], "source_pts")
        self.assertEqual(response.headers["x-survng-actual-timestamp"], f"{following['frames'][0]:.6f}")
        self.assertEqual(self.shown_frame(Path(response.path)), 20)
        # And back: the frame before the second segment's first is the
        # first segment's last.
        back = json.loads(frames_json("gate", following["previous"], "main").body)
        response = frame_jpg("gate", back["frames"][-1], "main", 320)
        self.assertEqual(self.shown_frame(Path(response.path)), 19)
        # Eight seconds without recording: no next segment to step into.
        self.assertIsNone(following["next"])
        self.assertIsNone(json.loads(frames_json("gate", 1010.5, "main").body)["previous"])
        # A time in the gap itself has no frames.
        with self.assertRaises(HTTPException) as gap:
            frames_json("gate", 1005.0, "main")
        self.assertEqual(gap.exception.status_code, 404)
        with self.assertRaises(HTTPException) as gap:
            frame_jpg("gate", 1005.0, "main", 320)
        self.assertEqual(gap.exception.status_code, 404)
        with self.assertRaises(HTTPException) as narrow:
            frame_jpg("gate", 1000.5, "main", 100)
        self.assertEqual(narrow.exception.status_code, 400)

    def test_a_run_holds_a_preview_session_not_a_transcode_slot(self) -> None:
        from survng.app.media_sessions import MediaResourceClass

        path = self.clip("session.mp4", seconds=0.5)
        row = {"path": str(path), "start_epoch": 1000.0, "end_epoch": 1000.5}
        requests: list = []
        entered: list[bool] = []

        class Lease:
            def __enter__(self):
                entered.append(True)
                return self

            def __exit__(self, *_exc):
                entered.append(False)
                return False

        self.manager.media_sessions = SimpleNamespace(
            acquire=lambda request, timeout: requests.append(request) or Lease(),
        )
        self.step_frame(row, 1000.1)
        self.step_frame(row, 1000.2)  # same run: from the cache
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].resources, {MediaResourceClass.PREVIEW_PROCESS: 1})
        self.assertEqual(entered, [True, False])


if __name__ == "__main__":
    unittest.main()
