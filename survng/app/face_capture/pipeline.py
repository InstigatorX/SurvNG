"""Two-pass face capture for one recorded person event.

Pass one finds each person's closest moments on the sub-stream (or, without
one, from the event's stored person boxes, then the refinement plan offsets).
Pass two searches the main stream only around those moments: person boxes,
the upper-body crop of each, and the capture engine's faces in it. Each person
keeps the few best faces by AdaFace norm that pass the size and norm cuts.

Decoding and inference are passed in, so the same core runs in the service
(isolated workers, recording index) and offline (in-process models).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..face_candidates import _padded_crop
from ..face_detection_geometry import upper_body_window
from .moments import Box, choose_moments, link_tracks, merge_windows

# The old face-evidence step reached 0.8 s past the outermost plan offsets.
PLAN_REACH_SECONDS = 0.9
MAX_WINDOW_SECONDS = 30.0
SUB_SAMPLE_FPS = 5.0
# Kept faces of one person are at least this far apart in time.
MIN_FACE_SPACING_SECONDS = 0.25

FrameSource = Callable[[float, float], Iterable[tuple[float, np.ndarray]]]
"""(start, end) seconds relative to the event -> (offset, BGR frame), in order."""


class CaptureCancelled(Exception):
    """The caller asked the capture to stop."""


def _transient(error: BaseException) -> bool:
    """Inference shed for incident work: the whole capture is retried later."""
    from ..inference_runtime.types import InferenceUnavailable

    return isinstance(error, InferenceUnavailable)


@dataclass(frozen=True)
class CaptureSettings:
    min_norm: float = 13.0
    min_size: float = 48.0
    moments_per_track: int = 4
    faces_per_track: int = 3
    margin_seconds: float = 3.0
    moment_radius_seconds: float = 0.5

    @classmethod
    def from_config(cls, config: Any) -> CaptureSettings:
        return cls(
            min_norm=float(config.face_capture_min_norm),
            min_size=float(config.face_capture_min_size),
            moments_per_track=int(config.face_capture_moments_per_track),
            faces_per_track=int(config.face_capture_faces_per_track),
            margin_seconds=float(config.face_capture_margin_seconds),
            moment_radius_seconds=float(config.face_capture_moment_radius_seconds),
        )


@dataclass
class CapturedFace:
    track: int
    offset: float
    box: tuple[float, float, float, float]
    landmarks: list[list[float]]
    confidence: float
    norm: float
    embedding: np.ndarray
    crop: np.ndarray


@dataclass
class CaptureResult:
    faces: list[CapturedFace] = field(default_factory=list)
    moment_source: str = ""
    moments: list[float] = field(default_factory=list)
    window: tuple[float, float] = (0.0, 0.0)
    sub_frames: int = 0
    main_frames: int = 0
    candidates: int = 0
    errors: list[str] = field(default_factory=list)


def event_window(plan_offsets: Iterable[float], margin: float) -> tuple[float, float]:
    """The plan's reach plus ``margin`` either side, at most 30 s long."""
    offsets = list(plan_offsets) or [0.0]
    start = min(offsets) - PLAN_REACH_SECONDS - margin
    end = max(offsets) + PLAN_REACH_SECONDS + margin
    return round(start, 3), round(min(end, start + MAX_WINDOW_SECONDS), 3)


def _scaled(boxes: list[Box], frame: np.ndarray, size: tuple[int, int] | None) -> list[Box]:
    if not size:
        return boxes
    height, width = frame.shape[:2]
    sx, sy = size[0] / width, size[1] / height
    return [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in boxes]


def keep_best(candidates: list[CapturedFace], per_track: int) -> list[CapturedFace]:
    """Each track's highest-norm faces, at least 0.25 s apart."""
    kept: list[CapturedFace] = []
    for track in sorted({face.track for face in candidates}):
        chosen: list[CapturedFace] = []
        for face in sorted((f for f in candidates if f.track == track), key=lambda f: (-f.norm, f.offset)):
            if len(chosen) >= per_track:
                break
            if all(abs(face.offset - other.offset) >= MIN_FACE_SPACING_SECONDS for other in chosen):
                chosen.append(face)
        kept.extend(sorted(chosen, key=lambda f: f.offset))
    return kept


def capture_event(
    *,
    plan_offsets: Iterable[float],
    main_frames: FrameSource,
    detect_people: Callable[[np.ndarray], list[Box]],
    capture_faces: Callable[[np.ndarray], list[dict[str, Any]]],
    settings: CaptureSettings,
    sub_frames: FrameSource | None = None,
    main_size: tuple[int, int] | None = None,
    stored_people: Iterable[tuple[float, list[Box]]] = (),
    cancelled: Callable[[], bool] = lambda: False,
) -> CaptureResult:
    plan = list(plan_offsets)
    result = CaptureResult(window=event_window(plan, settings.margin_seconds))
    start, end = result.window

    def check() -> None:
        if cancelled():
            raise CaptureCancelled()

    # Pass one: where is each person closest?
    if sub_frames is not None:
        samples: list[tuple[float, list[Box]]] = []
        try:
            for offset, frame in sub_frames(start, end):
                check()
                result.sub_frames += 1
                samples.append((offset, _scaled(detect_people(frame), frame, main_size)))
        except (OSError, RuntimeError) as error:
            if _transient(error):
                raise
            result.errors.append(f"sub-stream: {error}")
        result.moments = choose_moments(link_tracks(samples), per_track=settings.moments_per_track)
        result.moment_source = "sub-stream" if result.moments else ""
    if not result.moments:
        stored = [(o, b) for o, b in stored_people if start <= o <= end and b]
        result.moments = choose_moments(link_tracks(stored), per_track=settings.moments_per_track)
        result.moment_source = "stored observations" if result.moments else ""
    if not result.moments:
        result.moments = sorted(round(o, 3) for o in plan)
        result.moment_source = "plan offsets"

    # Pass two: the main stream around those moments.
    found: list[tuple[float, Box, dict[str, Any], np.ndarray]] = []
    people_by_frame: list[tuple[float, list[Box]]] = []
    for low, high in merge_windows(result.moments, settings.moment_radius_seconds):
        try:
            for offset, frame in main_frames(low, high):
                check()
                result.main_frames += 1
                height, width = frame.shape[:2]
                people = detect_people(frame)
                people_by_frame.append((offset, people))
                for person in people:
                    window = upper_body_window(person, width, height)
                    if window is None:
                        continue
                    left, top, right, bottom = window
                    for face in capture_faces(frame[top:bottom, left:right]):
                        x1, y1, x2, y2 = face["box"]
                        box = (x1 + left, y1 + top, x2 + left, y2 + top)
                        result.candidates += 1
                        if face["norm"] < settings.min_norm or min(box[2] - box[0], box[3] - box[1]) < settings.min_size:
                            continue
                        landmarks = [[x + left, y + top] for x, y in face["landmarks"]]
                        found.append((offset, person, {**face, "box": box, "landmarks": landmarks}, frame))
        except (OSError, RuntimeError) as error:
            if _transient(error):
                raise
            result.errors.append(f"main stream {low:+.1f}..{high:+.1f}: {error}")

    track_of: dict[tuple[float, Box], int] = {}
    for index, track in enumerate(link_tracks(people_by_frame)):
        for offset, box in track.samples:
            track_of[(offset, box)] = index
    candidates = []
    for offset, person, face, frame in found:
        cropped = _padded_crop(frame, face["box"])
        if cropped is None:
            continue
        candidates.append(CapturedFace(
            track=track_of.get((offset, person), -1),
            offset=round(offset, 3),
            box=tuple(round(v, 1) for v in face["box"]),
            landmarks=face["landmarks"],
            confidence=float(face["confidence"]),
            norm=float(face["norm"]),
            embedding=np.asarray(face["embedding"], dtype=np.float32),
            crop=np.ascontiguousarray(cropped[0]).copy(),
        ))
    result.faces = keep_best(candidates, settings.faces_per_track)
    return result


def sub_stream_source(
    rows: Callable[[str, float, float], list[dict[str, Any]]],
    event_epoch: float,
    ffmpeg_path: str,
    fps: float = SUB_SAMPLE_FPS,
) -> FrameSource:
    """Sub-stream (``live`` recording) frames sampled at ``fps`` through the shared sampler."""
    from ..recorded_video_frames import sampled_video_frames

    def frames(start: float, end: float) -> Iterator[tuple[float, np.ndarray]]:
        for row in rows("live", event_epoch + start, event_epoch + end):
            segment_start, duration = float(row["start_epoch"]), float(row["duration_seconds"])
            local_start = max(event_epoch + start, segment_start) - segment_start
            local_end = min(event_epoch + end, segment_start + duration) - segment_start
            if local_end - local_start <= 0.1:
                continue
            for decoded in sampled_video_frames(
                Path(row["path"]), start_epoch=segment_start + local_start, sample_fps=fps,
                duration_seconds=local_end - local_start, ffmpeg_path=ffmpeg_path,
                maximum_width=4096, start_offset_seconds=local_start,
            ):
                yield decoded.captured_at - event_epoch, decoded.frame

    return frames


def main_stream_source(
    rows: Callable[[str, float, float], list[dict[str, Any]]],
    event_epoch: float,
    ffmpeg_path: str,
    ffprobe_path: str,
    hardware_acceleration: str | None = None,
) -> FrameSource:
    """Every main-stream frame in a window, decoded a GOP run at a time (frame stepping's decoder)."""
    from ..recorded_decoder import recorded_decode_plans
    from ..recording_frame_runs import decode_frame_run, list_segment_frames
    from ..recording_frame_steps import frame_run
    from ..recording_media import mp4_video_dimensions

    listed: dict[str, Any] = {}

    def frames(start: float, end: float) -> Iterator[tuple[float, np.ndarray]]:
        for row in rows("main", event_epoch + start, event_epoch + end):
            path = Path(row["path"])
            segment_start = float(row["start_epoch"])
            local_start, local_end = event_epoch + start - segment_start, event_epoch + end - segment_start
            if str(path) not in listed:
                listed[str(path)] = list_segment_frames(ffprobe_path, path)
            segment = listed[str(path)]
            wanted = [i for i, t in enumerate(segment.times) if local_start <= t <= local_end]
            if not wanted:
                continue
            size = mp4_video_dimensions(path)
            plans = recorded_decode_plans(hardware_acceleration, scale=size) if size else recorded_decode_plans(None)
            runs = sorted({frame_run(segment, index) for index in wanted})
            for run_start, run_stop in runs:
                run = decode_frame_run(
                    ffmpeg_path, path, segment, run_start, run_stop, size[0] if size else 100_000,
                    size=size, plans=plans, consumer="face_capture",
                    hardware_timeout=5.0, cpu_timeout=20.0,
                )
                for pts, image in run.frames:
                    if local_start <= pts <= local_end:
                        yield segment_start + pts - event_epoch, image

    return frames
