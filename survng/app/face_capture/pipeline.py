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

import heapq
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..face_candidates import _padded_crop
from ..face_detection_geometry import upper_body_window
from .moments import Box, Track, choose_moments, iou, link_tracks, merge_windows

# The old face-evidence step reached 0.8 s past the outermost plan offsets.
PLAN_REACH_SECONDS = 0.9
MAX_WINDOW_SECONDS = 30.0
SUB_SAMPLE_FPS = 5.0
# Kept faces of one person are at least this far apart in time.
MIN_FACE_SPACING_SECONDS = 0.25
# A second-pass person belongs to a first-pass track when their boxes overlap
# this much at the nearest first-pass sample within this time.
TRACK_MATCH_IOU = 0.1
TRACK_MATCH_SECONDS = 1.0
# Second-pass tracks matched to no first-pass track are numbered from here.
UNMATCHED_TRACK_BASE = 1000
# Frames decoded together in one run. A 4K frame is about 25 MB; frame
# stepping's 30-frame runs would hold 750 MB, too much beside the service.
CAPTURE_RUN_MAX_FRAMES = 8
# Candidates held per event (the highest norms), each as a JPEG crop: a
# busy 4K event cannot grow memory beyond a few megabytes.
MAX_CANDIDATES = 300
CANDIDATE_JPEG_QUALITY = 95

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


def person_keys(first: list[Track], second: list[Track]) -> list[int]:
    """For each second-pass track, the first-pass track it continues (by majority of
    its samples), so one person searched in separate windows keeps one id."""
    matched: list[tuple[int | None, int]] = []
    for track in second:
        votes: dict[int, int] = {}
        for offset, box in track.samples:
            best, best_overlap = None, TRACK_MATCH_IOU
            for number, candidate in enumerate(first):
                near = [s for s in candidate.samples if abs(s[0] - offset) <= TRACK_MATCH_SECONDS]
                if not near:
                    continue
                _, nearest = min(near, key=lambda s: abs(s[0] - offset))
                overlap = iou(box, nearest)
                if overlap >= best_overlap:
                    best, best_overlap = number, overlap
            if best is not None:
                votes[best] = votes.get(best, 0) + 1
        best_key = max(votes, key=votes.get) if votes else None
        matched.append((best_key, votes[best_key] if best_key is not None else 0))
    # One person per key at a time: of second-pass tracks that share a key at
    # overlapping times (two people the first pass saw as one), the best
    # matched keeps it and the others get keys of their own.
    base = max(UNMATCHED_TRACK_BASE, len(first))
    keys = [base + index for index in range(len(second))]
    spans: dict[int, list[tuple[float, float]]] = {}
    for index in sorted(range(len(second)), key=lambda i: -matched[i][1]):
        key = matched[index][0]
        samples = second[index].samples
        if key is None or not samples:
            continue
        start, end = samples[0][0], samples[-1][0]
        if all(end < low or start > high for low, high in spans.get(key, ())):
            spans.setdefault(key, []).append((start, end))
            keys[index] = key
    return keys


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
    find_faces: Callable[[np.ndarray], list[dict[str, Any]]] | None = None,
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
    first_tracks: list[Track] = []
    if sub_frames is not None:
        samples: list[tuple[float, list[Box]]] = []
        face_sizes: dict[tuple[float, Box], float] = {}
        try:
            for offset, frame in sub_frames(start, end):
                check()
                result.sub_frames += 1
                people = detect_people(frame)
                scaled = _scaled(people, frame, main_size)
                samples.append((offset, scaled))
                if find_faces is None:
                    continue
                height, width = frame.shape[:2]
                scale = (main_size[0] / width) if main_size else 1.0
                for person, main_box in zip(people, scaled):
                    window = upper_body_window(person, width, height)
                    if window is None:
                        continue
                    left, top, right, bottom = window
                    sides = [min(f["box"][2] - f["box"][0], f["box"][3] - f["box"][1])
                             for f in find_faces(frame[top:bottom, left:right])]
                    if sides:
                        face_sizes[(offset, main_box)] = max(sides) * scale
        except (OSError, RuntimeError) as error:
            if _transient(error):
                raise
            result.errors.append(f"sub-stream: {error}")
        first_tracks = link_tracks(samples)
        result.moments = choose_moments(
            first_tracks, per_track=settings.moments_per_track, face_sizes=face_sizes,
        )
        result.moment_source = "sub-stream" if result.moments else ""
    if not result.moments:
        stored = [(o, b) for o, b in stored_people if start <= o <= end and b]
        first_tracks = link_tracks(stored)
        result.moments = choose_moments(first_tracks, per_track=settings.moments_per_track)
        result.moment_source = "stored observations" if result.moments else ""
    if not result.moments:
        result.moments = sorted(round(o, 3) for o in plan)
        result.moment_source = "plan offsets"

    # Pass two: the main stream around those moments.
    # Only the padded face crop is kept, never the whole frame.
    found: list[tuple[float, int, float, Box, dict[str, Any], bytes]] = []  # heap by norm
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
                        cropped = _padded_crop(frame, box)
                        if cropped is None:
                            continue
                        landmarks = [[x + left, y + top] for x, y in face["landmarks"]]
                        ok, encoded = cv2.imencode(".jpg", cropped[0], [cv2.IMWRITE_JPEG_QUALITY, CANDIDATE_JPEG_QUALITY])
                        if not ok:
                            continue
                        entry = {**face, "box": box, "landmarks": landmarks,
                                 "embedding": np.asarray(face["embedding"], dtype=np.float32)}
                        heapq.heappush(found, (float(face["norm"]), len(found) + result.candidates, offset,
                                               person, entry, encoded.tobytes()))
                        if len(found) > MAX_CANDIDATES:
                            heapq.heappop(found)
        except (OSError, RuntimeError) as error:
            if _transient(error):
                raise
            result.errors.append(f"main stream {low:+.1f}..{high:+.1f}: {error}")

    # Prefixed like the errors above, so a capture that kept no faces while
    # main-stream segments failed is retried rather than recorded as done.
    result.errors.extend(f"main stream {error}" for error in getattr(main_frames, "errors", ()))
    second_tracks = link_tracks(people_by_frame)
    track_of: dict[tuple[float, Box], int] = {}
    for key, track in zip(person_keys(first_tracks, second_tracks), second_tracks):
        for offset, box in track.samples:
            track_of[(offset, box)] = key
    candidates = []
    jpegs: dict[int, bytes] = {}
    for _, _, offset, person, face, jpeg in found:
        candidates.append(CapturedFace(
            track=track_of.get((offset, person), -1),
            offset=round(offset, 3),
            box=tuple(round(v, 1) for v in face["box"]),
            landmarks=face["landmarks"],
            confidence=float(face["confidence"]),
            norm=float(face["norm"]),
            embedding=np.asarray(face["embedding"], dtype=np.float32),
            crop=np.empty((0, 0, 3), np.uint8),
        ))
        jpegs[id(candidates[-1])] = jpeg
    result.faces = keep_best(candidates, settings.faces_per_track)
    for face in result.faces:
        face.crop = cv2.imdecode(np.frombuffer(jpegs[id(face)], np.uint8), cv2.IMREAD_COLOR)
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
    from ..recording_frame_runs import FrameRunError, decode_frame_run, list_segment_frames
    from ..recording_frame_steps import frame_run
    from ..recording_media import mp4_video_dimensions

    listed: dict[str, Any] = {}
    # A hardware decode that failed for other than the file is not tried
    # again for this event, so a broken GPU costs one timeout, not one a run.
    decoder = {"hardware": hardware_acceleration}

    class Frames:
        """The frame source; a segment that cannot be listed or decoded is
        skipped and noted in ``errors``, and the window's other segments
        are still searched."""

        def __init__(self) -> None:
            self.errors: list[str] = []

        def __call__(self, start: float, end: float) -> Iterator[tuple[float, np.ndarray]]:
            for row in rows("main", event_epoch + start, event_epoch + end):
                path = Path(row["path"])
                segment_start = float(row["start_epoch"])
                local_start, local_end = event_epoch + start - segment_start, event_epoch + end - segment_start
                if str(path) not in listed:
                    try:
                        listed[str(path)] = list_segment_frames(ffprobe_path, path)
                    except FrameRunError as error:
                        listed[str(path)] = error
                        self.errors.append(f"{path.name}: {error}")
                segment = listed[str(path)]
                if isinstance(segment, FrameRunError):
                    continue
                wanted = [i for i, t in enumerate(segment.times) if local_start <= t <= local_end]
                if not wanted:
                    continue
                size = mp4_video_dimensions(path)
                runs = sorted({frame_run(segment, index, CAPTURE_RUN_MAX_FRAMES) for index in wanted})
                for run_start, run_stop in runs:
                    plans = (
                        recorded_decode_plans(decoder["hardware"], scale=size) if size
                        else recorded_decode_plans(None)
                    )
                    try:
                        run = decode_frame_run(
                            ffmpeg_path, path, segment, run_start, run_stop, size[0] if size else 100_000,
                            size=size, plans=plans, consumer="face_capture",
                            hardware_timeout=5.0, cpu_timeout=20.0,
                        )
                    except FrameRunError as error:
                        if error.hardware_failed:
                            decoder["hardware"] = None
                        self.errors.append(f"{path.name} frames {run_start}-{run_stop}: {error}")
                        continue
                    if run.hardware_failed:
                        decoder["hardware"] = None
                    for pts, image in run.frames:
                        if local_start <= pts <= local_end:
                            yield segment_start + pts - event_epoch, image

    return Frames()
