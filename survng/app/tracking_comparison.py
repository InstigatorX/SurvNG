from __future__ import annotations

import copy
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .config import CameraConfig, ObjectTrackingConfig
from .detector import detection_failure
from .object_tracking import (
    ObjectTrackerRegistry,
    _encode_appearance,
    _encoder_supports_label,
    build_builtin_object_tracker_registry,
)
from .zones import apply_detection_zones


class DetectorBackend(Protocol):
    config: Any

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]: ...


class AppearanceEncoder(Protocol):
    enabled: bool

    def embed(self, crop: np.ndarray) -> np.ndarray | None: ...

    def supports_label(self, label: str) -> bool: ...

    def embed_for_label(self, label: str, crop: np.ndarray) -> np.ndarray | None: ...


def sampled_video_frames(
    path: Path,
    *,
    start_epoch: float,
    sample_fps: float,
    duration_seconds: float,
    ffmpeg_path: str = "",
    maximum_width: int = 640,
    start_offset_seconds: float = 0.0,
    concat_input: bool = False,
    probe_path: Path | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    if ffmpeg_path:
        yield from _ffmpeg_sampled_video_frames(
            path,
            start_epoch=start_epoch,
            sample_fps=sample_fps,
            duration_seconds=duration_seconds,
            ffmpeg_path=ffmpeg_path,
            maximum_width=maximum_width,
            start_offset_seconds=start_offset_seconds,
            concat_input=concat_input,
            probe_path=probe_path,
        )
        return
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("comparison video could not be opened")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(source_fps) or source_fps <= 0.0:
            source_fps = 30.0
        interval = 1.0 / max(0.1, float(sample_fps))
        next_sample = 0.0
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            offset = frame_index / source_fps
            frame_index += 1
            if offset > duration_seconds + 1e-6:
                break
            if offset + 1e-6 < next_sample:
                continue
            yield start_epoch + offset, frame
            next_sample += interval
    finally:
        capture.release()


def _ffmpeg_sampled_video_frames(
    path: Path,
    *,
    start_epoch: float,
    sample_fps: float,
    duration_seconds: float,
    ffmpeg_path: str,
    maximum_width: int,
    start_offset_seconds: float,
    concat_input: bool,
    probe_path: Path | None,
) -> Iterator[tuple[float, np.ndarray]]:
    # Opening another cv2.VideoCapture inside the server can contend with all
    # active camera capture threads. ffprobe is isolated and substantially
    # faster under a full camera workload. A constituent file is used when the
    # decoder input itself is an ffconcat manifest.
    source_width, source_height = _ffprobe_video_dimensions(
        probe_path or path,
        ffmpeg_path,
    )
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("comparison video dimensions are unavailable")
    output_width = max(2, min(source_width, max(64, int(maximum_width))))
    output_height = max(2, int(round(source_height * output_width / source_width)))
    output_width -= output_width % 2
    output_height -= output_height % 2
    frame_bytes = output_width * output_height * 3
    input_options = ["-f", "concat", "-safe", "0"] if concat_input else []
    command = [
        ffmpeg_path,
        "-nostdin",
        "-v", "error",
        *input_options,
        "-i", str(path),
        "-ss", f"{max(0.0, float(start_offset_seconds)):.3f}",
        "-t", f"{max(0.1, float(duration_seconds)):.3f}",
        "-vf", f"fps={max(0.1, float(sample_fps)):.6f},scale={output_width}:{output_height}",
        "-an", "-sn", "-dn",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_bytes * 2,
    )
    frame_index = 0
    try:
        if process.stdout is None:
            raise RuntimeError("comparison decoder output is unavailable")
        while True:
            payload = bytearray()
            while len(payload) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise RuntimeError("comparison decoder returned a partial frame")
            frame = np.frombuffer(payload, dtype=np.uint8).reshape((output_height, output_width, 3)).copy()
            yield start_epoch + frame_index / max(0.1, float(sample_fps)), frame
            frame_index += 1
        return_code = process.wait(timeout=5.0)
        if return_code != 0:
            raise RuntimeError("comparison video decoder failed")
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)


def _ffprobe_video_dimensions(path: Path, ffmpeg_path: str) -> tuple[int, int]:
    ffprobe_path = str(Path(ffmpeg_path).with_name("ffprobe"))
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        dimensions = result.stdout.strip().split("x", 1)
        if result.returncode == 0 and len(dimensions) == 2:
            width, height = (int(value) for value in dimensions)
            if width > 0 and height > 0:
                return width, height
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    raise RuntimeError("comparison video dimensions are unavailable")


class TrackingComparisonRunner:
    IMPLEMENTATIONS = ("survng_hybrid", "ultralytics_botsort")

    def __init__(
        self,
        *,
        config: ObjectTrackingConfig,
        detector: DetectorBackend,
        tracker_registry: ObjectTrackerRegistry | None = None,
        appearance_encoder: AppearanceEncoder | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.tracker_registry = tracker_registry or build_builtin_object_tracker_registry()
        self.appearance_encoder = appearance_encoder

    def run(
        self,
        camera: CameraConfig,
        frames: Iterable[tuple[float, np.ndarray]],
    ) -> dict[str, Any]:
        trackers: dict[str, Any] = {}
        initialization_ms: dict[str, float] = {}
        for implementation in self.IMPLEMENTATIONS:
            started = time.perf_counter()
            trackers[implementation] = self.tracker_registry.create(
                implementation,
                self.config.model_copy(update={"implementation": implementation}),
                float(self.detector.config.confidence_threshold),
            )
            initialization_ms[implementation] = (time.perf_counter() - started) * 1000.0
        processing_ms = defaultdict(float)
        maximum_simultaneous: dict[str, Counter[str]] = {
            implementation: Counter() for implementation in trackers
        }
        frames_processed = 0
        detection_ms = 0.0
        appearance_ms = 0.0
        appearance_failures = 0
        frame_decode_ms = 0.0
        first_epoch: float | None = None
        last_epoch: float | None = None
        frame_width = 0
        frame_height = 0

        frame_iterator = iter(frames)
        while True:
            decode_started = time.perf_counter()
            try:
                captured_at, frame = next(frame_iterator)
            except StopIteration:
                break
            frame_decode_ms += (time.perf_counter() - decode_started) * 1000.0
            if first_epoch is None:
                first_epoch = captured_at
            last_epoch = captured_at
            frame_height, frame_width = frame.shape[:2]
            detector_started = time.perf_counter()
            objects = self.detector.detect(
                frame,
                confidence_threshold=self.config.low_confidence_threshold,
            )
            detection_ms += (time.perf_counter() - detector_started) * 1000.0
            failure = detection_failure(objects)
            if failure:
                raise RuntimeError(f"comparison detector failed: {failure}")
            apply_detection_zones(
                camera,
                objects,
                int(frame_width),
                int(frame_height),
                float(self.detector.config.confidence_threshold),
                bool(getattr(self.detector.config, "require_incident_zone", True)),
            )
            appearance_started = time.perf_counter()
            appearance_failures += self._annotate_appearances(frame, objects)
            appearance_ms += (time.perf_counter() - appearance_started) * 1000.0
            for implementation, tracker in trackers.items():
                started = time.perf_counter()
                tracked = tracker.update(
                    copy.deepcopy(objects),
                    captured_at,
                    confirm_new=frames_processed == 0,
                )
                processing_ms[implementation] += (time.perf_counter() - started) * 1000.0
                counts = Counter(
                    str(item.get("label"))
                    for item in tracked
                    if item.get("label")
                )
                for label, count in counts.items():
                    maximum_simultaneous[implementation][label] = max(
                        maximum_simultaneous[implementation][label],
                        count,
                    )
            frames_processed += 1

        if frames_processed == 0 or first_epoch is None or last_epoch is None:
            raise RuntimeError("comparison video contained no readable frames")

        engines: dict[str, Any] = {}
        final_epoch = last_epoch + self.config.lost_timeout_seconds + 0.001
        for implementation, tracker in trackers.items():
            started = time.perf_counter()
            tracker.update([], final_epoch)
            processing_ms[implementation] += (time.perf_counter() - started) * 1000.0
            tracks = tracker.summaries(final_epoch)
            diagnostics_method = getattr(tracker, "diagnostics", None)
            diagnostics = diagnostics_method() if callable(diagnostics_method) else {}
            counts = Counter(str(track.get("label")) for track in tracks if track.get("label"))
            fragment_excess = sum(
                max(0, count - maximum_simultaneous[implementation].get(label, 0))
                for label, count in counts.items()
            )
            engines[implementation] = {
                "implementation": implementation,
                "initialization_ms": round(initialization_ms[implementation], 2),
                "processing_ms": round(processing_ms[implementation], 2),
                "average_ms_per_frame": round(processing_ms[implementation] / frames_processed, 3),
                "track_count": len(tracks),
                "observations": sum(int(track.get("observations") or 0) for track in tracks),
                "reid_recoveries": sum(int(track.get("reid_matches") or 0) for track in tracks),
                "fragmentation_proxy": fragment_excess,
                "labels": dict(sorted(counts.items())),
                "tracks": tracks,
                "reid_diagnostics": diagnostics,
            }

        return {
            "sample_fps": self.config.sample_fps,
            "frames_processed": frames_processed,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "start_epoch": round(first_epoch, 3),
            "end_epoch": round(last_epoch, 3),
            "duration_seconds": round(max(0.0, last_epoch - first_epoch), 3),
            "detection_ms": round(detection_ms, 2),
            "average_detection_ms_per_frame": round(detection_ms / frames_processed, 3),
            "appearance_ms": round(appearance_ms, 2),
            "average_appearance_ms_per_frame": round(appearance_ms / frames_processed, 3),
            "appearance_failures": appearance_failures,
            "frame_decode_ms": round(frame_decode_ms, 2),
            "average_frame_decode_ms": round(frame_decode_ms / frames_processed, 3),
            "engines": engines,
        }

    def _annotate_appearances(
        self,
        frame: np.ndarray,
        objects: list[dict[str, Any]],
    ) -> int:
        encoder = self.appearance_encoder
        if (
            encoder is None
            or not encoder.enabled
            or not self.config.appearance_reid_enabled
        ):
            return 0
        height, width = frame.shape[:2]
        remaining = self.config.reid_max_embeddings_per_frame
        for detected in sorted(
            objects,
            key=lambda item: float(item.get("confidence") or 0.0),
            reverse=True,
        ):
            if remaining <= 0:
                break
            label = str(detected.get("label") or "").lower()
            if not _encoder_supports_label(encoder, self.config, label):
                continue
            box = detected.get("box") or {}
            try:
                x1 = max(0, min(width, int(float(box.get("x1", 0)))))
                y1 = max(0, min(height, int(float(box.get("y1", 0)))))
                x2 = max(0, min(width, int(float(box.get("x2", 0)))))
                y2 = max(0, min(height, int(float(box.get("y2", 0)))))
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 16 or crop.shape[1] < 8:
                continue
            remaining -= 1
            try:
                embedding = _encode_appearance(encoder, label, crop)
            except Exception:
                return 1
            if embedding is not None:
                detected["_tracking_embedding"] = embedding
        return 0
