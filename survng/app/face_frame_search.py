"""Offline: would searching every frame find bigger faces than the old search?

For recent recorded person events this decodes every frame of the event window
from the main recording, runs the configured person detector and the existing
upper-body face search on each frame, and compares the best face per event
with what the old sampled search (refinement plan frames plus face-evidence
extras around the best anchors) would have seen with the same detectors.

It reads the database and recordings only. It loads its own CPU copies of the
models and leaves the running service, its config and its caches alone.

nice -n 19 python -m survng.app.face_frame_search runtime/database \\
    --cameras boiler,gate --events-per-camera 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# Offsets the old face-evidence step tried around its two best anchors.
EXTRA_DELTAS = (-0.4, 0.4, -0.8, 0.8)
EXTRA_MIN_SEPARATION = 0.15
MAX_ANCHORS = 3
MAX_WINDOW_SECONDS = 30.0


@dataclass
class FrameResult:
    offset: float
    usefulness: float
    face_sides: list[float]
    person_heights: list[float]

    @property
    def best_face(self) -> float:
        return max(self.face_sides, default=0.0)


@dataclass
class EventResult:
    event_id: int
    camera_id: str
    event_epoch: float
    window: tuple[float, float]
    plan_window: tuple[float, float]
    frames: list[FrameResult] = field(default_factory=list)
    inexact_frames: int = 0
    decode_errors: list[str] = field(default_factory=list)
    seconds: float = 0.0
    baseline_offsets: list[float] = field(default_factory=list)


def nearest_frame(frames: list[FrameResult], target: float, tolerance: float) -> FrameResult | None:
    best = min(frames, key=lambda item: abs(item.offset - target), default=None)
    if best is None or abs(best.offset - target) > tolerance:
        return None
    return best


def baseline_frames(
    frames: list[FrameResult],
    plan_offsets: Iterable[float],
    *,
    max_extra: int,
    tolerance: float = 0.1,
) -> list[FrameResult]:
    """Frames the old search looked at, as an upper bound.

    Every planned offset counts as decoded (the live path often stopped early)
    and the face-evidence time budget is ignored, so this favours the old
    search. Extras follow ``_refine_face_evidence``: offsets around the two
    most useful anchors, at least 0.15 s from every other sample.
    """
    samples: list[FrameResult] = []
    for offset in plan_offsets:
        frame = nearest_frame(frames, offset, tolerance)
        if frame is not None and all(frame is not other for other in samples):
            samples.append(frame)
    anchors = sorted(
        (sample for sample in samples if sample.usefulness > 0),
        key=lambda sample: sample.usefulness,
        reverse=True,
    )[:MAX_ANCHORS]
    existing = [sample.offset for sample in samples]
    targets: list[float] = []
    for delta in EXTRA_DELTAS:
        for anchor in anchors[:2]:
            target = round(anchor.offset + delta, 3)
            if all(abs(target - other) >= EXTRA_MIN_SEPARATION for other in existing + targets):
                targets.append(target)
    chosen = list(samples)
    for target in targets[:max_extra]:
        frame = nearest_frame(frames, target, tolerance)
        if frame is None or any(abs(frame.offset - sample.offset) < EXTRA_MIN_SEPARATION for sample in chosen):
            continue
        chosen.append(frame)
    return chosen


def _event_epoch(created_at: str, qualification: dict[str, Any]) -> float:
    timing = qualification.get("event_timing")
    if isinstance(timing, dict) and timing.get("sampling_at"):
        try:
            return datetime.fromisoformat(str(timing["sampling_at"])).timestamp()
        except ValueError:
            pass
    value = qualification.get("evidence_frame_at_epoch")
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(created_at).timestamp()


def select_events(
    connection: sqlite3.Connection,
    cameras: list[str],
    *,
    since: str,
    per_camera: int,
) -> list[tuple[int, str, float, dict[str, Any]]]:
    from .face_funnel import summarize_event

    selected: list[tuple[int, str, float, dict[str, Any]]] = []
    for camera_id in cameras:
        found = 0
        rows = connection.execute(
            "select id, camera_id, objects_json, created_at from events "
            "where camera_id = ? and created_at >= ? and objects_json like '%\"person\"%' "
            "order by id desc",
            (camera_id, since),
        )
        for row in rows:
            objects = json.loads(row[2] or "[]")
            if summarize_event(objects) is None:
                continue
            qualification = next(
                (
                    item.get("motion_qualification")
                    for item in objects
                    if isinstance(item, dict) and item.get("status") == "motion_qualification"
                    and isinstance(item.get("motion_qualification"), dict)
                ),
                {},
            )
            selected.append((int(row[0]), str(row[1]), _event_epoch(str(row[3]), qualification), qualification))
            found += 1
            if found >= per_camera:
                break
    return selected


def _resolve_model_path(value: str) -> str:
    path = Path(value)
    if path.is_file():
        return str(path)
    # The service also looks for /models/... under the working directory.
    if path.is_absolute() and path.parts[1:2] == ("models",):
        local = Path("models", *path.parts[2:])
        if local.is_file():
            return str(local)
    raise FileNotFoundError(f"model not found: {value}")


class FrameSearcher:
    def __init__(self, detector_config: Any) -> None:
        from .detector import OpenVinoDetector
        from .face_detection import OpenVinoFaceDetector

        cpu = {"device": "CPU", "cache_enabled": False}
        self.config = detector_config
        self.person_detector = OpenVinoDetector(detector_config.model_copy(update=cpu))
        self.face_detector = OpenVinoFaceDetector(
            detector_config.model_copy(
                update={
                    **cpu,
                    "face_recognition_enabled": True,
                    "face_recognition_device": "CPU",
                    "face_detection_model_path": _resolve_model_path(
                        detector_config.face_detection_model_path
                    ),
                }
            )
        )
        if not self.face_detector.ready:
            raise RuntimeError(f"face detector unavailable: {self.face_detector.error}")

    def search(self, frame: Any, offset: float) -> FrameResult:
        from .motion_pipeline.object_detection import RecordedMotionObjectDetector, _box
        from .motion_pipeline.recorded_evidence import _image_quality

        objects = self.person_detector.detect(
            frame, confidence_threshold=float(self.config.event_candidate_confidence_threshold)
        )
        if any(isinstance(item, dict) and item.get("status") == "detector_unavailable" for item in objects):
            raise RuntimeError("person detector unavailable")
        people = [
            item for item in objects
            if isinstance(item, dict) and str(item.get("label") or "").lower() == "person"
        ]
        faces = RecordedMotionObjectDetector._detect_faces_in_people(
            frame, people, self.face_detector.detect, max_people=8,
        )
        height, width = frame.shape[:2]
        usefulness = 0.0
        heights: list[float] = []
        for person in people:
            box = _box(person)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            heights.append(y2 - y1)
            # Zones are not evaluated offline; alert confidence stands in for
            # incident eligibility when ranking anchors.
            if float(person.get("confidence") or 0.0) < float(self.config.confidence_threshold):
                continue
            crop = frame[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))]
            if crop.size:
                usefulness = max(usefulness, crop.shape[0] * crop.shape[1] * _image_quality(crop).score)
        sides = []
        for face in faces:
            box = _box(face)
            if box is not None:
                sides.append(min(box[2] - box[0], box[3] - box[1]))
        return FrameResult(round(offset, 3), usefulness, sides, heights)


_SHOWINFO_FRAME = re.compile(
    rb"Parsed_showinfo[^\r\n]*\bn:\s*(\d+)[^\r\n]*\bpts_time:\s*([-+0-9.eE]+)"
)


def every_frame(
    path: Path,
    start_seconds: float,
    end_seconds: float,
    ffmpeg_path: str,
) -> Iterator[tuple[float | None, np.ndarray]]:
    """Every presented frame in ``[start, end)`` of one file, full size, in software.

    Same command shape as the timeline frame-step decoder (PR 370): accurate
    seek with -copyts -start_at_zero, so ``showinfo`` reports file-relative
    PTS. Frames are streamed, not buffered. A frame whose showinfo line was
    lost or torn is yielded with ``None`` rather than a neighbour's time.
    """
    from .recorded_decoder import DecodePlan, StreamingDecode, build_decode_command, warn_decoder_errors
    from .recorded_video_frames import _ffprobe_video_metadata

    width, height, *_ = _ffprobe_video_metadata(path, ffmpeg_path)
    if width <= 0 or height <= 0:
        raise RuntimeError("video dimensions are unavailable")
    times: dict[int, float] = {}
    arrived = threading.Condition()

    def on_line(line: bytes) -> bool:
        match = _SHOWINFO_FRAME.search(line)
        if match is None:
            return b"Parsed_showinfo" in line
        with arrived:
            times[int(match.group(1))] = float(match.group(2))
            arrived.notify_all()
        return True

    def on_end() -> None:
        with arrived:
            times[-1] = 0.0
            arrived.notify_all()

    plan = DecodePlan("cpu")
    command = build_decode_command(
        ffmpeg_path, path, plan=plan, loglevel="info",
        seek=f"{max(0.0, start_seconds - 0.001):.6f}", copyts=True, start_at_zero=True,
        first_video_stream=True,
        video_filter=f"trim=start={start_seconds:.6f}:end={end_seconds:.6f},showinfo",
        passthrough_timestamps=True, transport="raw",
    )
    frame_bytes = width * height * 3
    decoder = StreamingDecode(
        command, backend=plan.backend, buffer_bytes=frame_bytes * 2,
        on_stderr_line=on_line, on_stderr_end=on_end,
    )
    try:
        index = 0
        while True:
            payload = decoder.read_exact(frame_bytes)
            if decoder.stalled or not payload:
                break
            if len(payload) != frame_bytes:
                raise RuntimeError("decoder returned a partial frame")
            with arrived:
                arrived.wait_for(lambda: index in times or -1 in times, timeout=5.0)
                pts = times.pop(index, None)
            frame = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))
            yield pts, frame
            index += 1
        result = decoder.finish()
        warn_decoder_errors("face_frame_search", path, plan.backend, result.stderr)
        if not result.ok:
            raise RuntimeError(f"decoder failed ({result.describe()})")
    finally:
        decoder.close()


def search_event(
    searcher: FrameSearcher,
    recordings: sqlite3.Connection,
    *,
    event_id: int,
    camera_id: str,
    event_epoch: float,
    plan_offsets: list[float],
    ffmpeg_path: str,
    max_extra: int,
    margin: float,
) -> EventResult:
    # The plan window holds every frame the old search could reach.
    plan_start = min(plan_offsets) - max(abs(d) for d in EXTRA_DELTAS) - 0.1
    plan_end = max(plan_offsets) + max(abs(d) for d in EXTRA_DELTAS) + 0.1
    start = plan_start - margin
    end = min(plan_end + margin, start + MAX_WINDOW_SECONDS)
    result = EventResult(
        event_id, camera_id, event_epoch,
        (round(start, 3), round(end, 3)), (round(plan_start, 3), round(plan_end, 3)),
    )
    started = time.monotonic()
    rows = recordings.execute(
        "select path, start_epoch, duration_seconds from recordings "
        "where camera_id = ? and source = 'main' and start_epoch < ? "
        "and start_epoch + duration_seconds > ? order by start_epoch",
        (camera_id, event_epoch + end, event_epoch + start),
    ).fetchall()
    if not rows:
        result.decode_errors.append("no main recording covers the window")
    for path, segment_start, duration in rows:
        local_start = max(event_epoch + start, float(segment_start)) - float(segment_start)
        local_end = min(event_epoch + end, float(segment_start) + float(duration)) - float(segment_start)
        if local_end - local_start <= 0.05:
            continue
        try:
            frames = []
            for pts, frame in every_frame(Path(path), local_start, local_end, ffmpeg_path):
                if pts is None:
                    result.inexact_frames += 1
                    continue
                frames.append(searcher.search(frame, float(segment_start) + pts - event_epoch))
            result.frames.extend(frames)
        except (OSError, RuntimeError) as error:
            result.decode_errors.append(f"{Path(path).name}: {error}")
    result.frames.sort(key=lambda item: item.offset)
    chosen = baseline_frames(result.frames, plan_offsets, max_extra=max_extra)
    result.baseline_offsets = [frame.offset for frame in chosen]
    result.seconds = round(time.monotonic() - started, 1)
    return result


def summarize(results: list[EventResult], face_min_size: int) -> dict[str, Any]:
    cameras: dict[str, dict[str, Any]] = {}
    for result in results:
        camera = cameras.setdefault(result.camera_id, {"events": []})
        by_offset = {frame.offset: frame for frame in result.frames}
        baseline = [by_offset[offset] for offset in result.baseline_offsets]
        full_best = max((frame.best_face for frame in result.frames), default=0.0)
        low, high = result.plan_window
        plan_best = max(
            (frame.best_face for frame in result.frames if low <= frame.offset <= high), default=0.0
        )
        base_best = max((frame.best_face for frame in baseline), default=0.0)
        best_frame = max(result.frames, key=lambda frame: frame.best_face, default=None)
        camera["events"].append({
            "event_id": result.event_id,
            "frames": len(result.frames),
            "baseline_frames": len(baseline),
            "frames_with_face": sum(1 for frame in result.frames if frame.face_sides),
            "frames_with_usable_face": sum(1 for frame in result.frames if frame.best_face >= face_min_size),
            "baseline_best_face": round(base_best, 1),
            "plan_window_best_face": round(plan_best, 1),
            "full_best_face": round(full_best, 1),
            "window": list(result.window),
            "plan_window": list(result.plan_window),
            "full_best_offset": best_frame.offset if best_frame and full_best else None,
            "max_person_height": round(max((h for f in result.frames for h in f.person_heights), default=0.0), 1),
            "inexact": result.inexact_frames,
            "errors": result.decode_errors,
            "seconds": result.seconds,
        })
    for camera in cameras.values():
        events = camera["events"]
        searched = [e for e in events if e["frames"]]
        camera["summary"] = {
            "events": len(events),
            "events_decoded": len(searched),
            "frames": sum(e["frames"] for e in searched),
            "baseline_any_face": sum(1 for e in searched if e["baseline_best_face"] > 0),
            "plan_window_any_face": sum(1 for e in searched if e["plan_window_best_face"] > 0),
            "full_any_face": sum(1 for e in searched if e["full_best_face"] > 0),
            "baseline_usable": sum(1 for e in searched if e["baseline_best_face"] >= face_min_size),
            "plan_window_usable": sum(1 for e in searched if e["plan_window_best_face"] >= face_min_size),
            "full_usable": sum(1 for e in searched if e["full_best_face"] >= face_min_size),
            "gained_usable_plan_window": sum(
                1 for e in searched
                if e["plan_window_best_face"] >= face_min_size > e["baseline_best_face"]
            ),
            "gained_usable": sum(
                1 for e in searched
                if e["full_best_face"] >= face_min_size > e["baseline_best_face"]
            ),
            "median_baseline_best": _median([e["baseline_best_face"] for e in searched if e["baseline_best_face"]]),
            "median_plan_window_best": _median([e["plan_window_best_face"] for e in searched if e["plan_window_best_face"]]),
            "median_full_best": _median([e["full_best_face"] for e in searched if e["full_best_face"]]),
            "seconds": round(sum(e["seconds"] for e in events), 1),
        }
    return {"face_min_size": face_min_size, "cameras": cameras}


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[len(ordered) // 2], 1)


def format_report(report: dict[str, Any]) -> str:
    minimum = report["face_min_size"]
    lines = [
        f"Every frame vs. old sampled search (usable face = shorter side >= {minimum} px)",
        "old = plan frames + face-evidence extras; plan = every frame the old search could reach;",
        "all = every frame including the margin around it",
    ]
    for camera_id, camera in report["cameras"].items():
        s = camera["summary"]
        lines.append("")
        lines.append(f"{camera_id}: {s['events_decoded']}/{s['events']} events decoded, {s['frames']} frames, {s['seconds']} s")
        lines.append(f"  any face:     old {s['baseline_any_face']}  plan {s['plan_window_any_face']}  all {s['full_any_face']}")
        lines.append(
            f"  usable face:  old {s['baseline_usable']}  plan {s['plan_window_usable']}  all {s['full_usable']}"
            f"  (gained: plan {s['gained_usable_plan_window']}, all {s['gained_usable']})"
        )
        lines.append(
            f"  median best face when found: old {s['median_baseline_best']}"
            f"  plan {s['median_plan_window_best']}  all {s['median_full_best']}"
        )
        lines.append("  event   frames old-frames  old-best  plan-best  all-best @offset  usable-frames  person-h")
        for e in camera["events"]:
            offset = "" if e["full_best_offset"] is None else f"@{e['full_best_offset']:+.2f}"
            lines.append(
                f"  {e['event_id']:<7} {e['frames']:>6} {e['baseline_frames']:>10} "
                f"{e['baseline_best_face']:>9} {e['plan_window_best_face']:>10} {e['full_best_face']:>9} {offset:<8} "
                f"{e['frames_with_usable_face']:>13} {e['max_person_height']:>9}"
                + (f"  errors: {'; '.join(e['errors'])}" if e["errors"] else "")
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database_dir", type=Path)
    parser.add_argument("--recording-index", type=Path, default=Path("runtime/recording-index/recordings.sqlite3"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--cameras", default="boiler,sherry-garage,gate,downstairs")
    parser.add_argument("--events-per-camera", type=int, default=10)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--margin", type=float, default=3.0, help="seconds searched beyond the plan window")
    parser.add_argument("--threads", type=int, default=4, help="OpenVINO CPU threads")
    parser.add_argument("--json", type=Path, help="also write the full report here")
    args = parser.parse_args(argv)

    os.environ.setdefault("OV_CPU_THREADS_NUM", str(max(1, args.threads)))
    os.nice(19)
    from .config import DetectorConfig
    from .motion_pipeline.object_detection import (
        _flatten_refinement_stages,
        resolve_recorded_refinement_plan,
    )

    raw = json.loads(args.config.read_text())
    detector_config = DetectorConfig.model_validate(raw["detector"])
    ffmpeg_path = str(raw.get("ffmpeg_path") or "ffmpeg")
    searcher = FrameSearcher(detector_config)

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    events_db = sqlite3.connect(f"file:{args.database_dir / 'survng.sqlite3'}?mode=ro", uri=True)
    recordings_db = sqlite3.connect(f"file:{args.recording_index}?mode=ro", uri=True)
    cameras = [item.strip() for item in args.cameras.split(",") if item.strip()]
    events = select_events(events_db, cameras, since=since, per_camera=args.events_per_camera)
    results: list[EventResult] = []
    for index, (event_id, camera_id, epoch, qualification) in enumerate(events, start=1):
        stages = resolve_recorded_refinement_plan(
            detector_config,
            qualification=qualification,
            event_at=datetime.fromtimestamp(epoch, timezone.utc),
            camera_id=camera_id,
        )[0]
        result = search_event(
            searcher, recordings_db,
            event_id=event_id, camera_id=camera_id, event_epoch=epoch,
            plan_offsets=list(_flatten_refinement_stages(stages)),
            ffmpeg_path=ffmpeg_path,
            max_extra=int(detector_config.face_evidence_max_extra_frames),
            margin=max(0.0, args.margin),
        )
        results.append(result)
        print(
            f"[{index}/{len(events)}] {camera_id} event {event_id}: {len(result.frames)} frames "
            f"in {result.seconds} s" + (f" ({'; '.join(result.decode_errors)})" if result.decode_errors else ""),
            file=sys.stderr, flush=True,
        )
    report = summarize(results, int(detector_config.face_min_size))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
