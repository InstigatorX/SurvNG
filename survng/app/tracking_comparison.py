from __future__ import annotations

import copy
import json
import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
import queue
import re
import subprocess
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

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
from .video_frames import DecodedVideoFrame, VideoFrameReference


TRACKING_COMPARISON_IMPLEMENTATIONS = (
    "survng_hybrid",
    "ultralytics_tracktrack",
    "ultralytics_botsort",
)


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
    ffmpeg_path: str,
    maximum_width: int = 640,
    start_offset_seconds: float = 0.0,
    concat_input: bool = False,
    probe_path: Path | None = None,
) -> Iterator[DecodedVideoFrame]:
    """Sample frames using ``start_epoch`` as the epoch at the seek point.

    FFmpeg resets output timestamps after ``-ss`` on supported builds, so a
    source PTS is relative to ``start_offset_seconds``. Callers sampling a
    segment mid-file must pass the wall-clock epoch of that seek point.
    """
    if not ffmpeg_path:
        raise ValueError("ffmpeg_path is required for video frame sampling")
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
) -> Iterator[DecodedVideoFrame]:
    # ffprobe is isolated and substantially
    # faster under a full camera workload. A constituent file is used when the
    # decoder input itself is an ffconcat manifest.
    source_width, source_height, time_base_num, time_base_den = _ffprobe_video_metadata(
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
    duration = max(0.1, float(duration_seconds))
    command = [
        ffmpeg_path,
        "-nostdin",
        "-v", "info",
        *input_options,
        "-i", str(path),
        "-ss", f"{max(0.0, float(start_offset_seconds)):.3f}",
        "-t", f"{duration:.3f}",
        "-vf", (
            f"scale={output_width}:{output_height},showinfo@source,"
            f"fps={max(0.1, float(sample_fps)):.6f},showinfo@sampled"
        ),
        "-an", "-sn", "-dn",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes * 2,
    )
    timestamps: queue.Queue[tuple[int, float] | None] = queue.Queue()
    frame_pattern = re.compile(
        r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:([-+0-9.eE]+).*"
        r"\bchecksum:([0-9A-Fa-f]+)"
    )

    def read_timestamps() -> None:
        stderr = process.stderr
        if stderr is None:
            timestamps.put(None)
            return
        try:
            source_by_checksum: dict[str, list[tuple[int, float]]] = {}
            last_source_time = float("-inf")
            for raw_line in iter(stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace")
                match = frame_pattern.search(line)
                if not match:
                    continue
                pts = int(match.group(1))
                pts_seconds = float(match.group(2))
                checksum = match.group(3).upper()
                if "showinfo@source" in line:
                    source_by_checksum.setdefault(checksum, []).append(
                        (pts, pts_seconds)
                    )
                    continue
                if "showinfo@sampled" not in line:
                    continue
                candidates = source_by_checksum.get(checksum, [])
                eligible = [
                    item for item in candidates if item[1] >= last_source_time - 1e-9
                ]
                if not eligible:
                    timestamps.put(None)
                    return
                source_pts, source_time = min(
                    eligible,
                    key=lambda item: abs(item[1] - pts_seconds),
                )
                last_source_time = source_time
                candidates.remove((source_pts, source_time))
                timestamps.put((source_pts, source_time))
        finally:
            timestamps.put(None)

    timestamp_thread = threading.Thread(
        target=read_timestamps,
        name="survng-frame-pts",
        daemon=True,
    )
    timestamp_thread.start()
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
            try:
                timestamp = timestamps.get(timeout=5.0)
            except queue.Empty as exc:
                raise RuntimeError("comparison decoder frame timestamp timed out") from exc
            if timestamp is None:
                raise RuntimeError("comparison decoder frame timestamp is unavailable")
            pts, pts_seconds = timestamp
            captured_at = start_epoch + pts_seconds
            yield DecodedVideoFrame(
                captured_at,
                frame,
                VideoFrameReference(
                    source_path=path,
                    seek_offset_seconds=max(0.0, float(start_offset_seconds)),
                    pts=pts,
                    pts_seconds=pts_seconds,
                    time_base_num=time_base_num,
                    time_base_den=time_base_den,
                    captured_at=captured_at,
                ),
            )
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
        timestamp_thread.join(timeout=1.0)
        if process.stderr is not None:
            process.stderr.close()


def _ffprobe_video_metadata(
    path: Path,
    ffmpeg_path: str,
) -> tuple[int, int, int, int]:
    ffprobe_path = str(Path(ffmpeg_path).with_name("ffprobe"))
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,time_base",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") if isinstance(payload, dict) else None
        if result.returncode == 0 and isinstance(streams, list) and streams:
            stream = streams[0]
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            time_base = str(stream.get("time_base") or "1/1").split("/", 1)
            time_base_num = int(time_base[0])
            time_base_den = int(time_base[1]) if len(time_base) > 1 else 1
            if width > 0 and height > 0:
                return width, height, time_base_num, max(1, time_base_den)
    except (json.JSONDecodeError, OSError, TypeError, ValueError, subprocess.TimeoutExpired):
        pass
    raise RuntimeError("comparison video dimensions are unavailable")


def video_frame_at_reference(
    reference: VideoFrameReference,
    *,
    ffmpeg_path: str,
    maximum_width: int,
) -> DecodedVideoFrame | None:
    """Re-decode the exact source PTS identified during recorded sampling."""
    if not reference.exact or maximum_width <= 0:
        return None
    source_width, source_height, _time_base_num, _time_base_den = (
        _ffprobe_video_metadata(reference.source_path, ffmpeg_path)
    )
    output_width = max(2, min(source_width, max(64, int(maximum_width))))
    output_height = max(2, int(round(source_height * output_width / source_width)))
    output_width -= output_width % 2
    output_height -= output_height % 2
    command = [
        ffmpeg_path,
        "-nostdin",
        "-v", "error",
        "-i", str(reference.source_path),
        "-ss", f"{max(0.0, reference.seek_offset_seconds):.3f}",
        "-vf", (
            f"select='eq(pts\\,{reference.pts})',"
            f"scale={output_width}:{output_height},fps=1"
        ),
        "-frames:v", "1",
        "-an", "-sn", "-dn",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    expected_bytes = output_width * output_height * 3
    if result.returncode != 0 or len(result.stdout) != expected_bytes:
        return None
    frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        (output_height, output_width, 3)
    ).copy()
    return DecodedVideoFrame(reference.captured_at, frame, reference)


class TrackingComparisonRunner:
    IMPLEMENTATIONS = TRACKING_COMPARISON_IMPLEMENTATIONS

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
        *,
        sampling_profile: str = "recorded",
    ) -> dict[str, Any]:
        from .tracking_evaluation import (MAX_FRAMES, MAX_OBJECTS, MAX_REPLAY_BYTES, canonical_json, portable_detection, replay_digest)
        samples = []
        replay_bytes = 0
        detection_ms = appearance_ms = frame_decode_ms = 0.0
        appearance_failures = exact_timestamp_frames = 0
        iterator = iter(frames)
        try:
            while True:
                started = time.perf_counter()
                try:
                    sample = next(iterator)
                except StopIteration:
                    break
                frame_decode_ms += (time.perf_counter() - started) * 1000.0
                captured_at, frame = sample
                if len(samples) >= MAX_FRAMES:
                    raise ValueError("comparison exceeds 600-frame replay limit")
                reference = getattr(sample, "reference", None)
                if reference is not None and reference.exact:
                    exact_timestamp_frames += 1
                height, width = frame.shape[:2]
                started = time.perf_counter()
                detect = getattr(self.detector, "detect_offline", self.detector.detect)
                objects = detect(frame, confidence_threshold=self.config.low_confidence_threshold)
                detection_ms += (time.perf_counter() - started) * 1000.0
                failure = detection_failure(objects)
                if failure:
                    raise RuntimeError(f"comparison detector failed: {failure}")
                if any(item.get("status") == "inference_deferred" for item in objects):
                    raise RuntimeError("comparison inference deferred; no empty observation was recorded")
                objects = [item for item in objects if self.config.tracks_label(item.get("label"))]
                if len(objects) > MAX_OBJECTS:
                    raise ValueError("comparison exceeds 100 detections per frame")
                apply_detection_zones(camera, objects, width, height,
                    float(self.detector.config.confidence_threshold),
                    bool(getattr(self.detector.config, "require_incident_zone", True)))
                started = time.perf_counter()
                appearance_failures += self._annotate_appearances(frame, objects)
                appearance_ms += (time.perf_counter() - started) * 1000.0
                portable_frame = {"frame_index": len(samples), "captured_at": float(captured_at),
                    "width": int(width), "height": int(height),
                    "source_pts": bool(reference is not None and reference.exact),
                    "detections": [portable_detection(item) for item in objects]}
                replay_bytes += len(canonical_json(portable_frame).encode())
                if replay_bytes > MAX_REPLAY_BYTES:
                    raise ValueError("comparison replay exceeds 32 MiB")
                samples.append(portable_frame)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        if not samples:
            raise RuntimeError("comparison video contained no readable frames")
        detector_config = getattr(self.detector, "config", None)
        detector_identity = {
            key: str(getattr(detector_config, key))
            for key in ("model_path", "implementation", "device")
            if getattr(detector_config, key, None) is not None
        }
        replay = {"schema_version": 1, "camera_id": camera.id,
            "detector_identity": detector_identity,
            "model_identity_note": "Model paths describe capture configuration; saved detections and embeddings are the exact replay inputs.",
            "tracking_config": self.config.model_dump(mode="json"),
            "high_confidence_threshold": float(self.detector.config.confidence_threshold),
            "timestamp_source": "source_pts" if exact_timestamp_frames == len(samples) else "mixed_or_estimated",
            "source_pts_frames": exact_timestamp_frames,
            "appearance_source": "shared_supplied_embeddings",
            "frames": samples}
        replay["replay_id"] = replay_digest(replay)
        result = self.replay(replay, sampling_profile=sampling_profile,
                             tracker_registry=self.tracker_registry, implementations=self.IMPLEMENTATIONS)
        count = len(samples)
        result.update(replay=replay, capture_frames_processed=count,
            detection_ms=round(detection_ms, 2), average_detection_ms_per_frame=round(detection_ms / count, 3),
            appearance_ms=round(appearance_ms, 2), average_appearance_ms_per_frame=round(appearance_ms / count, 3),
            appearance_failures=appearance_failures, frame_decode_ms=round(frame_decode_ms, 2),
            average_frame_decode_ms=round(frame_decode_ms / count, 3))
        return result

    @staticmethod
    def replay(
        replay: dict[str, Any], *, sampling_profile: str = "recorded",
        labels: dict[str, Any] | None = None,
        tracker_registry: ObjectTrackerRegistry | None = None,
        implementations: tuple[str, ...] = TRACKING_COMPARISON_IMPLEMENTATIONS,
    ) -> dict[str, Any]:
        from .tracking_evaluation import identity_metrics, selected_frames, validate_labels, validate_replay
        validate_replay(replay)
        samples = selected_frames(replay, sampling_profile)
        if labels is not None:
            validate_labels(labels, replay, {frame["frame_index"] for frame in samples})
        config = ObjectTrackingConfig.model_validate(replay["tracking_config"])
        if sampling_profile != "recorded":
            config = config.model_copy(update={"sample_fps": .75 if sampling_profile == "fixed_075fps" else 2.0})
        registry = tracker_registry or build_builtin_object_tracker_registry()
        engines = {}
        for implementation in implementations:
            started = time.perf_counter()
            try:
                tracker = registry.create(implementation, config.model_copy(update={"implementation": implementation}),
                                          replay["high_confidence_threshold"])
            except Exception as error:
                # Optional engines must not prevent baseline capture/replay. Do
                # not expose exception text (paths/URLs may contain credentials).
                engines[implementation] = {"implementation": implementation,
                    "error": f"Tracker unavailable ({type(error).__name__}); check the optional runtime on the server."}
                continue
            initialization_ms = (time.perf_counter() - started) * 1000.0
            processing_ms = 0.0
            simultaneous: Counter = Counter()
            observations = []
            cutoff = None
            ever_live = False
            try:
                for position, frame in enumerate(samples):
                    started = time.perf_counter()
                    tracked = tracker.update(copy.deepcopy(frame["detections"]), frame["captured_at"], confirm_new=position == 0)
                    processing_ms += (time.perf_counter() - started) * 1000.0
                    # Stable per-frame observations are independent of capped
                    # display histories and are retained for labeled evaluation.
                    outputs = [{"track_id": int(item["track_id"]), "label": item["label"], "box": item["box"]}
                               for item in tracked if item.get("track_state") == "confirmed"]
                    observations.append({"frame_index": frame["frame_index"], "captured_at": frame["captured_at"], "objects": outputs})
                    counts = Counter(item["label"] for item in outputs)
                    for label, count in counts.items():
                        simultaneous[label] = max(simultaneous[label], count)
                    live = tracker.has_live_tracks(frame["captured_at"])
                    if ever_live and not live and cutoff is None:
                        cutoff = frame["captured_at"] - samples[0]["captured_at"]
                    ever_live = ever_live or live
                tracks = tracker.summaries(samples[-1]["captured_at"] + config.lost_timeout_seconds + .001)
                diagnostics = getattr(tracker, "diagnostics", lambda: {})()
            except Exception as error:
                engines[implementation] = {"implementation": implementation,
                    "error": f"Tracker failed ({type(error).__name__}); this engine is excluded from scoring."}
                continue
            counts = Counter(item["label"] for item in tracks)
            engine = {"implementation": implementation,
                "initialization_ms": round(initialization_ms, 2), "processing_ms": round(processing_ms, 2),
                "average_ms_per_frame": round(processing_ms / len(samples), 3),
                "track_count": len(tracks), "observations": sum(len(f["objects"]) for f in observations),
                "reid_recoveries": (sum(int(t.get("reid_matches") or 0) for t in tracks)
                                    if implementation.startswith("survng_hybrid") else None),
                "appearance_input_count": sum(item.get("_tracking_embedding") is not None for f in samples for item in f["detections"]),
                "fragmentation_proxy": sum(max(0, count - simultaneous[label]) for label, count in counts.items()),
                "labels": dict(sorted(counts.items())), "tracks": tracks, "reid_diagnostics": diagnostics,
                "frame_observations": observations,
                "simulated_lost_track_cutoff_seconds": cutoff,
                "lifecycle_note": "Backend replay continues after the lost-track predicate; this is not a full production session simulation."}
            if labels is not None:
                try:
                    engine["identity_metrics"] = identity_metrics(observations, labels, replay)
                except (ValueError, TypeError, KeyError) as error:
                    engine = {"implementation": implementation,
                              "error": f"Invalid tracker output for scoring ({type(error).__name__})."}
            engines[implementation] = engine
        try:
            upstream_version = version("ultralytics")
        except PackageNotFoundError:
            upstream_version = None
        dependency_versions = {}
        for package in ("numpy", "torch", "scipy", "lap", "ultralytics"):
            try:
                dependency_versions[package] = version(package)
            except PackageNotFoundError:
                dependency_versions[package] = None
        sources = [Path(__file__), Path(__file__).with_name("ultralytics_tracking.py"),
                   Path(__file__).with_name("tracking_evaluation.py"),
                   *sorted(Path(__file__).with_name("object_track").glob("*.py"))]
        source_hash = hashlib.sha256(b"".join(path.name.encode() + path.read_bytes() for path in sources)).hexdigest()
        first, last = samples[0], samples[-1]
        duration = last["captured_at"] - first["captured_at"]
        return {"sample_fps": config.sample_fps, "sampling_profile": sampling_profile,
            "sampling_note": "Profiles select saved frames at or after target timestamps; gaps skip seconds 5–9 and 15–21. No adaptive policy is simulated.",
            "effective_sample_fps": round((len(samples)-1) / duration, 3) if duration else None,
            "lost_timeout_seconds": config.lost_timeout_seconds, "frames_processed": len(samples),
            "frame_width": first["width"], "frame_height": first["height"],
            "start_epoch": first["captured_at"], "end_epoch": last["captured_at"], "duration_seconds": round(duration, 3),
            "timestamp_source": replay["timestamp_source"],
            "source_pts_frames": sum(bool(frame.get("source_pts", replay["source_pts_frames"] == len(replay["frames"]))) for frame in samples),
            "replay_id": replay["replay_id"], "evaluation_source_sha256": source_hash,
            "ultralytics_version": upstream_version, "appearance_source": replay["appearance_source"],
            "dependency_versions": dependency_versions, "python_version": platform.python_version(),
            "engines": engines}

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
