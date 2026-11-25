"""Offline: does a two-pass face capture keep what the every-frame search finds?

Pass one samples the camera's sub-stream (``live`` recording) at a few frames
per second, runs the configured person detector, links boxes into tracks and
keeps each track's largest moments, scaled to main-stream pixels. Without a
sub-stream it falls back to the event's stored person observations, then to
the refinement plan offsets. Pass two decodes the main stream only within
``--moment-radius`` of those moments and runs the same face search as
``face_frame_search`` (person detector, upper-body crops, face detector, AdaFace
norm), so a comparison with an every-frame report isolates the moment choice.

It reads the database and recordings only.

nice -n 19 python -m survng.app.face_capture_trial runtime/database \\
    --cameras boiler,gate --reference every-frame.json --yunet
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Pass one.
SUB_SAMPLE_FPS = 5.0
TRACK_MIN_IOU = 0.3
TRACK_MAX_GAP_SECONDS = 1.0
MOMENTS_PER_TRACK = 4
MOMENT_MIN_SPACING = 0.5
MAX_MOMENTS = 8
# Pass two.
MOMENT_RADIUS = 0.5

Box = tuple[float, float, float, float]


@dataclass
class Track:
    samples: list[tuple[float, Box]] = field(default_factory=list)


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_tracks(samples: list[tuple[float, list[Box]]]) -> list[Track]:
    """Greedy overlap linking of per-sample person boxes into tracks."""
    tracks: list[Track] = []
    for offset, boxes in sorted(samples, key=lambda item: item[0]):
        open_tracks = [t for t in tracks if offset - t.samples[-1][0] <= TRACK_MAX_GAP_SECONDS]
        pairs = sorted(
            ((iou(t.samples[-1][1], box), ti, bi) for ti, t in enumerate(open_tracks) for bi, box in enumerate(boxes)),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        for overlap, ti, bi in pairs:
            if overlap < TRACK_MIN_IOU or ti in used_tracks or bi in used_boxes:
                continue
            open_tracks[ti].samples.append((offset, boxes[bi]))
            used_tracks.add(ti)
            used_boxes.add(bi)
        for bi, box in enumerate(boxes):
            if bi not in used_boxes:
                tracks.append(Track([(offset, box)]))
    return tracks


def choose_moments(
    tracks: list[Track],
    *,
    per_track: int = MOMENTS_PER_TRACK,
    spacing: float = MOMENT_MIN_SPACING,
    limit: int = MAX_MOMENTS,
) -> list[float]:
    """Each track's tallest samples (person height), spaced apart, best first."""
    candidates: list[tuple[float, float]] = []
    for track in tracks:
        chosen: list[float] = []
        for offset, box in sorted(track.samples, key=lambda item: item[1][3] - item[1][1], reverse=True):
            if len(chosen) >= per_track:
                break
            if all(abs(offset - other) >= spacing for other in chosen):
                chosen.append(offset)
                candidates.append((box[3] - box[1], offset))
    moments: list[float] = []
    for _, offset in sorted(candidates, reverse=True):
        if len(moments) >= limit:
            break
        if all(abs(offset - other) >= spacing / 2 for other in moments):
            moments.append(round(offset, 3))
    return sorted(moments)


def merge_windows(moments: list[float], radius: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for moment in sorted(moments):
        start, end = moment - radius, moment + radius
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    return windows


def _segments(recordings: sqlite3.Connection, camera_id: str, source: str, start: float, end: float):
    return recordings.execute(
        "select path, start_epoch, duration_seconds from recordings where camera_id = ? and source = ? "
        "and start_epoch < ? and start_epoch + duration_seconds > ? order by start_epoch",
        (camera_id, source, end, start),
    ).fetchall()


def _main_size(path: str, ffmpeg_path: str) -> tuple[int, int]:
    from .recorded_video_frames import _ffprobe_video_metadata

    width, height, *_ = _ffprobe_video_metadata(Path(path), ffmpeg_path)
    return width, height


def sub_stream_samples(
    searcher: Any, recordings: sqlite3.Connection, camera_id: str, epoch: float,
    window: tuple[float, float], main_size: tuple[int, int], ffmpeg_path: str,
) -> tuple[list[tuple[float, list[Box]]], int]:
    """Person boxes (main-stream pixels) sampled from the sub-stream."""
    from .recorded_video_frames import sampled_video_frames

    samples: list[tuple[float, list[Box]]] = []
    frames = 0
    for path, segment_start, duration in _segments(recordings, camera_id, "live", epoch + window[0], epoch + window[1]):
        local_start = max(epoch + window[0], segment_start) - segment_start
        local_end = min(epoch + window[1], segment_start + duration) - segment_start
        if local_end - local_start <= 0.1:
            continue
        for decoded in sampled_video_frames(
            Path(path), start_epoch=segment_start + local_start, sample_fps=SUB_SAMPLE_FPS,
            duration_seconds=local_end - local_start, ffmpeg_path=ffmpeg_path,
            maximum_width=4096, start_offset_seconds=local_start,
        ):
            frames += 1
            height, width = decoded.frame.shape[:2]
            sx, sy = main_size[0] / width, main_size[1] / height
            people = searcher.people(decoded.frame)
            samples.append((
                decoded.captured_at - epoch,
                [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in people],
            ))
    return samples, frames


def stored_samples(events: sqlite3.Connection, event_id: int, epoch: float, main_size: tuple[int, int]):
    rows = events.execute(
        "select payload_json, captured_epoch from scene_observations where event_id = ?", (event_id,)
    ).fetchall()
    by_time: dict[float, list[Box]] = {}
    for payload, captured in rows:
        item = json.loads(payload)
        box = item.get("box") or {}
        if item.get("label") != "person" or not box:
            continue
        sx = main_size[0] / float(item.get("detection_frame_width") or main_size[0])
        sy = main_size[1] / float(item.get("detection_frame_height") or main_size[1])
        by_time.setdefault(round(float(captured) - epoch, 3), []).append(
            (box["x1"] * sx, box["y1"] * sy, box["x2"] * sx, box["y2"] * sy)
        )
    return sorted(by_time.items())


def trial_event(
    searcher: Any, events: sqlite3.Connection, recordings: sqlite3.Connection, *,
    event_id: int, camera_id: str, epoch: float, plan_offsets: list[float],
    margin: float, radius: float, ffmpeg_path: str, hardware_acceleration: str | None,
) -> dict[str, Any]:
    from .face_frame_search import EXTRA_DELTAS, MAX_WINDOW_SECONDS, every_frame, passes

    started = time.monotonic()
    plan_start = min(plan_offsets) - max(abs(d) for d in EXTRA_DELTAS) - 0.1
    plan_end = max(plan_offsets) + max(abs(d) for d in EXTRA_DELTAS) + 0.1
    window = (plan_start - margin, min(plan_end + margin, plan_start - margin + MAX_WINDOW_SECONDS))
    result: dict[str, Any] = {"event_id": event_id, "camera_id": camera_id, "errors": []}
    main_rows = _segments(recordings, camera_id, "main", epoch + window[0], epoch + window[1])
    if not main_rows:
        result["errors"].append("no main recording")
        return result
    main_size = _main_size(main_rows[0][0], ffmpeg_path)
    source, sub_frames, moments = "", 0, []
    try:
        samples, sub_frames = sub_stream_samples(
            searcher, recordings, camera_id, epoch, window, main_size, ffmpeg_path,
        )
        moments = choose_moments(link_tracks(samples))
        source = "sub-stream" if moments else ""
    except (OSError, RuntimeError) as error:
        result["errors"].append(f"sub-stream: {error}")
    if not moments:
        stored = [(o, b) for o, b in stored_samples(events, event_id, epoch, main_size) if window[0] <= o <= window[1]]
        moments = choose_moments(link_tracks(stored))
        source = "stored observations" if moments else ""
    if not moments:
        moments, source = sorted(plan_offsets), "plan offsets"
    pass_one = time.monotonic() - started

    faces: list[dict[str, Any]] = []
    main_frames = 0
    for low, high in merge_windows(moments, radius):
        for path, segment_start, duration in _segments(recordings, camera_id, "main", epoch + low, epoch + high):
            local_start = max(epoch + low, segment_start) - segment_start
            local_end = min(epoch + high, segment_start + duration) - segment_start
            if local_end - local_start <= 0.02:
                continue
            try:
                for pts, frame in every_frame(Path(path), local_start, local_end, ffmpeg_path, hardware_acceleration):
                    if pts is None:
                        continue
                    main_frames += 1
                    found = searcher.search(frame, segment_start + pts - epoch)
                    for face in found.faces:
                        face.pop("_thumb", None)
                        face.pop("_embedding", None)
                        faces.append({"offset": found.offset, **face})
            except (OSError, RuntimeError) as error:
                result["errors"].append(f"main {Path(path).name}: {error}")
    best = {
        detector: max(
            (f["side"] for f in faces if f["detector"] == detector and passes(f, "recognisable")), default=0.0
        )
        for detector in {f["detector"] for f in faces} | {"retail"}
    }
    result.update({
        "moment_source": source,
        "moments": moments,
        "sub_frames": sub_frames,
        "main_frames": main_frames,
        "pass_one_seconds": round(pass_one, 1),
        "seconds": round(time.monotonic() - started, 1),
        "best_recognisable": {k: round(v, 1) for k, v in best.items()},
    })
    return result


def compare(results: list[dict[str, Any]], reference: dict[str, Any], detector: str, minimum: float) -> str:
    cameras = reference["detectors"][detector]["cameras"]
    rows = {e["event_id"]: e for cam in cameras.values() for e in cam["events"]}
    lines = [f"Two-pass vs every frame ({detector}, usable = recognisable and >= {minimum:g} px)"]
    total = {"ref": 0, "kept": 0, "gained": 0, "frames": 0, "ref_frames": 0, "seconds": 0.0, "ref_seconds": 0.0}
    by_camera: dict[str, dict[str, Any]] = {}
    for r in results:
        ref = rows.get(r["event_id"])
        if ref is None or "best_recognisable" not in r:
            continue
        cam = by_camera.setdefault(r["camera_id"], {k: 0 for k in total} | {"sources": {}})
        ref_ok = ref["recognisable"]["all_best"] >= minimum
        mine = r["best_recognisable"].get(detector, 0.0)
        for bucket in (cam, total):
            bucket["ref"] += ref_ok
            bucket["kept"] += ref_ok and mine >= minimum
            bucket["gained"] += (not ref_ok) and mine >= minimum
            bucket["frames"] += r["main_frames"]
            bucket["ref_frames"] += ref["frames"]
            bucket["seconds"] += r["seconds"]
            bucket["ref_seconds"] += ref["seconds"]
        cam["sources"][r["moment_source"]] = cam["sources"].get(r["moment_source"], 0) + 1
    for name, c in [*by_camera.items(), ("TOTAL", total)]:
        share = f"{c['kept']}/{c['ref']}" + (f" ({100 * c['kept'] / c['ref']:.0f}%)" if c["ref"] else "")
        lines.append(
            f"{name:14} usable kept {share:12} gained {c['gained']}  main frames {c['frames']}/{c['ref_frames']} "
            f"({100 * c['frames'] / max(1, c['ref_frames']):.0f}%)  time {c['seconds']:.0f}/{c['ref_seconds']:.0f} s"
            + (f"  moments from {c['sources']}" if name != "TOTAL" else "")
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database_dir", type=Path)
    parser.add_argument("--recording-index", type=Path, default=Path("runtime/recording-index/recordings.sqlite3"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--cameras", required=True, help="comma-separated camera ids")
    parser.add_argument("--events-per-camera", type=int, default=10)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--moment-radius", type=float, default=MOMENT_RADIUS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--hw-decode", action="store_true")
    parser.add_argument("--yunet", action="store_true")
    parser.add_argument("--reference", type=Path, help="face_frame_search --json report for the same events")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    os.environ.setdefault("OV_CPU_THREADS_NUM", str(max(1, args.threads)))
    os.nice(19)
    from .config import DetectorConfig
    from .face_frame_search import FrameSearcher, YuNetFaces, YUNET_MODEL, select_events
    from .motion_pipeline.object_detection import _flatten_refinement_stages, resolve_recorded_refinement_plan

    raw = json.loads(args.config.read_text())
    config = DetectorConfig.model_validate(raw["detector"])
    ffmpeg_path = str(raw.get("ffmpeg_path") or "ffmpeg")
    searcher = FrameSearcher(config, YuNetFaces(YUNET_MODEL, 0.6) if args.yunet else None, device=args.device)
    events = sqlite3.connect(f"file:{args.database_dir / 'survng.sqlite3'}?mode=ro", uri=True)
    recordings = sqlite3.connect(f"file:{args.recording_index}?mode=ro", uri=True)
    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    chosen = select_events(events, [c.strip() for c in args.cameras.split(",") if c.strip()],
                           since=since, per_camera=args.events_per_camera)
    results = []
    for index, (event_id, camera_id, epoch, qualification) in enumerate(chosen, start=1):
        stages = resolve_recorded_refinement_plan(
            config, qualification=qualification, event_at=datetime.fromtimestamp(epoch, timezone.utc),
            camera_id=camera_id,
        )[0]
        result = trial_event(
            searcher, events, recordings, event_id=event_id, camera_id=camera_id, epoch=epoch,
            plan_offsets=list(_flatten_refinement_stages(stages)), margin=args.margin,
            radius=args.moment_radius, ffmpeg_path=ffmpeg_path,
            hardware_acceleration=raw.get("hardware_acceleration") if args.hw_decode else None,
        )
        results.append(result)
        print(
            f"[{index}/{len(chosen)}] {camera_id} {event_id}: {result.get('moment_source')} "
            f"{len(result.get('moments', []))} moments, {result.get('sub_frames', 0)} sub + "
            f"{result.get('main_frames', 0)} main frames in {result.get('seconds')} s"
            + (f" ({'; '.join(result['errors'])})" if result["errors"] else ""),
            file=sys.stderr, flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    if args.reference:
        reference = json.loads(args.reference.read_text())
        minimum = float(config.face_min_size)
        print("\n\n".join(compare(results, reference, name, minimum) for name in reference["detectors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
