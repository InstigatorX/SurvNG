"""Run face capture offline on recent person events and show what it keeps.

Uses the same two-pass pipeline as the service, with in-process models instead
of the isolated workers. It reads the event database and recordings only;
captured faces are written only to ``--store-dir`` when given. Keep it niced:

nice -n 19 python -m survng.app.face_capture runtime/database \\
    --cameras gate,foyer --events-per-camera 5 --sheets /tmp/face-capture
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .events import person_events, plan_offsets, stored_people
from .pipeline import (
    CaptureResult,
    CaptureSettings,
    capture_event,
    event_window,
    main_stream_source,
    rows_for_window,
    sub_stream_source,
)

THUMB_HEIGHT = 112


def _rows_reader(index_path: Path, camera_id: str):
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)

    def rows(source: str, start: float, end: float) -> list[dict[str, Any]]:
        return [
            {"path": path, "start_epoch": begin, "duration_seconds": duration}
            for path, begin, duration in connection.execute(
                "select path, start_epoch, duration_seconds from recordings where camera_id = ? and source = ?"
                " and start_epoch < ? and start_epoch > ? and end_epoch > ? order by start_epoch",
                # The lower bound on start_epoch lets the range index work:
                # no segment is anywhere near an hour long.
                (camera_id, source, end, start - 3600.0, start),
            )
        ]

    return rows


def _sheet(camera_id: str, results: list[tuple[int, CaptureResult]], directory: Path) -> Path:
    rows = []
    for event_id, result in results:
        label = np.full((THUMB_HEIGHT, 150, 3), 255, np.uint8)
        cv2.putText(label, str(event_id), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(label, f"{len(result.faces)} kept / {result.candidates}", (4, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(label, result.moment_source[:20], (4, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        tiles = [label]
        for face in result.faces:
            crop = face.crop
            width = max(1, round(crop.shape[1] * THUMB_HEIGHT / crop.shape[0]))
            tile = cv2.resize(crop, (width, THUMB_HEIGHT), interpolation=cv2.INTER_AREA)
            side = min(face.box[2] - face.box[0], face.box[3] - face.box[1])
            cv2.putText(tile, f"t{face.track} n{face.norm:.0f} {side:.0f}px", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    width = max(row.shape[1] for row in rows)
    sheet = np.vstack([
        np.hstack([row, np.full((THUMB_HEIGHT, width - row.shape[1], 3), 255, np.uint8)]) for row in rows
    ])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{camera_id}.jpg"
    cv2.imwrite(str(path), sheet)
    return path


class _People:
    def __init__(self, config: Any, device: str) -> None:
        from ..detector import OpenVinoDetector

        self.config = config
        self.detector = OpenVinoDetector(config.model_copy(update={"device": device, "cache_enabled": False}))

    def __call__(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        from ..motion_pipeline.object_detection import _box

        objects = self.detector.detect(
            frame, confidence_threshold=float(self.config.event_candidate_confidence_threshold),
        )
        if any(isinstance(item, dict) and item.get("status") == "detector_unavailable" for item in objects):
            raise OSError("person detector unavailable")
        return [
            box for item in objects
            if isinstance(item, dict) and str(item.get("label") or "").lower() == "person"
            and (box := _box(item)) is not None
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database_dir", type=Path, help="directory holding survng.sqlite3")
    parser.add_argument("--recording-index", type=Path, default=Path("runtime/recording-index/recordings.sqlite3"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--cameras", required=True, help="comma-separated camera ids, or all")
    parser.add_argument("--events-per-camera", type=int, default=5)
    parser.add_argument("--hours", type=float, default=72.0, help="look this far back for person events")
    parser.add_argument("--device", default="CPU", help="OpenVINO device for the person detector")
    parser.add_argument("--hw-decode", action="store_true", help="decode with the configured hardware_acceleration")
    parser.add_argument("--threads", type=int, default=4, help="OpenVINO CPU threads")
    parser.add_argument("--sheets", type=Path, help="write a contact sheet per camera here")
    parser.add_argument("--json", type=Path, help="write a per-event summary here")
    parser.add_argument("--store-dir", type=Path, help="also record the faces in a face capture store here")
    parser.add_argument("--seed", action="store_true",
                        help="process every person event in --hours, newest first, into --store-dir; events the "
                             "store already finished are skipped, so a stopped run resumes")
    parser.add_argument("--reference", type=Path,
                        help="face_frame_search --json report (with --yunet) to compare against")
    args = parser.parse_args(argv)

    os.environ.setdefault("OV_CPU_THREADS_NUM", str(max(1, args.threads)))
    os.nice(19)
    from ..config import DetectorConfig
    from ..recording_media import mp4_video_dimensions
    from .engine import FaceCaptureEngine
    from .store import FaceCaptureStore

    raw = json.loads(args.config.read_text())
    config = DetectorConfig.model_validate(raw["detector"])
    ffmpeg = str(raw.get("ffmpeg_path") or "ffmpeg")
    ffprobe = str(Path(ffmpeg).with_name("ffprobe")) if "/" in ffmpeg else "ffprobe"
    hardware = raw.get("hardware_acceleration") if args.hw_decode else None
    engine = FaceCaptureEngine(config)
    if not engine.ready:
        print(f"face capture models unavailable: {engine.error}", file=sys.stderr)
        return 2
    people = _People(config, args.device)
    settings = CaptureSettings.from_config(config)
    store = FaceCaptureStore(args.store_dir) if args.store_dir else None
    if args.seed and store is None:
        parser.error("--seed needs --store-dir")
    events = sqlite3.connect(f"file:{args.database_dir / 'survng.sqlite3'}?mode=ro", uri=True)
    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=args.hours)).isoformat()
    until = (now - timedelta(seconds=90)).isoformat()
    wanted = None if args.cameras.strip() == "all" else [c.strip() for c in args.cameras.split(",") if c.strip()]
    if args.seed:
        candidates = person_events(events, since=since, until=until, cameras=wanted, limit=1_000_000)
        seen = store.attempts(event_id for event_id, *_ in candidates)
        queue = [c for c in candidates if _needs_capture(seen.get(c[0]))]
        print(f"seeding {len(queue)} of {len(candidates)} person events (the rest are already captured)", flush=True)
        batches = [(None, queue)]
    else:
        cameras = wanted or sorted({c for _, c, _, _ in person_events(events, since=since, until=until)})
        batches = [
            (camera_id, person_events(events, since=since, until=until, cameras=[camera_id],
                                      limit=args.events_per_camera))
            for camera_id in cameras
        ]
    summary: list[dict[str, Any]] = []
    readers: dict[str, Any] = {}
    done = 0
    seed_started = time.monotonic()
    for camera_label, chosen in batches:
        results: list[tuple[int, CaptureResult]] = []
        for event_id, camera_id, epoch, qualification in chosen:
            rows = readers.setdefault(camera_id, _rows_reader(args.recording_index, camera_id))
            started = time.monotonic()
            if args.seed:
                try:
                    _seed_one(store, engine, people, settings, config, events, rows, ffmpeg, ffprobe, hardware,
                              event_id, camera_id, epoch, qualification)
                except Exception as exc:  # one bad event must not stop seeding
                    store.record_failure(event_id=event_id, camera_id=camera_id, event_epoch=epoch, error=str(exc))
                    print(f"{camera_id} {event_id}: failed: {exc}", flush=True)
                done += 1
                rate = (time.monotonic() - seed_started) / done
                print(f"[{done}/{len(chosen)}] {camera_id} {event_id} in {time.monotonic() - started:.1f} s; "
                      f"about {rate * (len(chosen) - done) / 3600:.1f} h left", flush=True)
                continue
            plan = plan_offsets(config, qualification, epoch, camera_id)
            low, high = event_window(plan, settings.margin_seconds)
            # Reaches past the window by the moment radius, as the service does.
            reach = settings.moment_radius_seconds + 1.0
            window_rows = rows_for_window(rows, epoch + low - reach, epoch + high + reach)
            main_rows = window_rows("main", epoch + low, epoch + high)
            main_size = next(
                (size for row in main_rows if (size := mp4_video_dimensions(Path(row["path"])))), None,
            )
            result = capture_event(
                plan_offsets=plan,
                main_frames=main_stream_source(window_rows, epoch, ffmpeg, ffprobe, hardware),
                sub_frames=sub_stream_source(window_rows, epoch, ffmpeg),
                find_faces=lambda crop: engine.capture(crop, embed=False),
                main_size=main_size,
                stored_people=stored_people(events, event_id, epoch, main_size),
                detect_people=people,
                capture_faces=engine.capture,
                settings=settings,
            )
            seconds = time.monotonic() - started
            if store is not None:
                store.record(event_id=event_id, camera_id=camera_id, event_epoch=epoch, result=result,
                             embedding_model=engine.embedding_model, seconds=seconds)
            results.append((event_id, result))
            summary.append({
                "event_id": event_id, "camera_id": camera_id, "seconds": round(seconds, 1),
                "faces": len(result.faces), "candidates": result.candidates,
                "moment_source": result.moment_source, "moments": result.moments,
                "sub_frames": result.sub_frames, "main_frames": result.main_frames, "errors": result.errors,
                "kept": [{"track": f.track, "offset": f.offset, "norm": round(f.norm, 2),
                          "side": round(min(f.box[2] - f.box[0], f.box[3] - f.box[1]), 1)} for f in result.faces],
            })
            print(
                f"{camera_id} {event_id}: {len(result.faces)} faces kept of {result.candidates} "
                f"({result.moment_source}, {result.sub_frames} sub + {result.main_frames} main frames, "
                f"{seconds:.1f} s)" + (f" errors: {'; '.join(result.errors)}" if result.errors else ""),
                flush=True,
            )
        if args.sheets and results and camera_label:
            print(f"contact sheet: {_sheet(camera_label, results, args.sheets)}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1))
    if args.reference:
        print(compare(summary, json.loads(args.reference.read_text()), float(settings.min_size)))
    return 0


def _needs_capture(previous: tuple[str, int, int] | None) -> bool:
    from .service import MAX_ATTEMPTS
    from .store import CAPTURE_VERSION

    if previous is None:
        return True
    state, attempts, version = previous
    return version < CAPTURE_VERSION or (state == "failed" and attempts < MAX_ATTEMPTS)


def _seed_one(store, engine, people, settings, config, events, rows, ffmpeg, ffprobe, hardware,
              event_id, camera_id, epoch, qualification) -> None:
    from ..recording_media import mp4_video_dimensions

    started = time.monotonic()
    plan = plan_offsets(config, qualification, epoch, camera_id)
    low, high = event_window(plan, settings.margin_seconds)
    reach = settings.moment_radius_seconds + 1.0
    window_rows = rows_for_window(rows, epoch + low - reach, epoch + high + reach)
    main_rows = window_rows("main", epoch + low, epoch + high)
    main_size = next((size for row in main_rows if (size := mp4_video_dimensions(Path(row["path"])))), None)
    result = capture_event(
        plan_offsets=plan,
        main_frames=main_stream_source(window_rows, epoch, ffmpeg, ffprobe, hardware),
        sub_frames=sub_stream_source(window_rows, epoch, ffmpeg),
        find_faces=lambda crop: engine.capture(crop, embed=False),
        main_size=main_size,
        stored_people=stored_people(events, event_id, epoch, main_size),
        detect_people=people,
        capture_faces=engine.capture,
        settings=settings,
    )
    if not result.faces and (result.main_frames == 0 or any(e.startswith("main stream") for e in result.errors)):
        # As the service does: no main-stream frames is a failed attempt, not "no faces".
        store.record_failure(event_id=event_id, camera_id=camera_id, event_epoch=epoch,
                             error="; ".join(result.errors)[:500] or "no main-stream frames")
        return
    store.record(event_id=event_id, camera_id=camera_id, event_epoch=epoch, result=result,
                 embedding_model=engine.embedding_model, seconds=time.monotonic() - started)
    from .groups import FaceGroups

    FaceGroups(store).assign(event_id, float(config.face_group_threshold))
    # Its names reach scene identities when SurvNG next syncs them (at start,
    # or after the next People page edit).
    store.queue_identity_sync([event_id])


def compare(summary: list[dict[str, Any]], reference: dict[str, Any], minimum: float) -> str:
    """Events where every-frame search found a usable YuNet face, and how many capture kept."""
    cameras = reference["detectors"]["yunet"]["cameras"]
    rows = {e["event_id"]: e for cam in cameras.values() for e in cam["events"]}
    lines = [f"Capture vs every frame (YuNet, usable = recognisable and >= {minimum:g} px)"]
    totals: dict[str, list[float]] = {}
    for item in summary:
        ref = rows.get(item["event_id"])
        if ref is None:
            continue
        found = ref["recognisable"]["all_best"] >= minimum
        kept = any(face["side"] >= minimum for face in item["kept"])
        for name in (item["camera_id"], "TOTAL"):
            t = totals.setdefault(name, [0, 0, 0, 0, 0.0, 0.0])
            t[0] += found
            t[1] += found and kept
            t[2] += item["main_frames"]
            t[3] += ref["frames"]
            t[4] += item["seconds"]
            t[5] += ref["seconds"]
    for name, (found, kept, frames, ref_frames, seconds, ref_seconds) in sorted(
        totals.items(), key=lambda kv: kv[0] == "TOTAL",
    ):
        share = f" ({100 * kept / found:.0f}%)" if found else ""
        lines.append(
            f"{name:14} kept {int(kept)}/{int(found)}{share}  main frames {int(frames)}/{int(ref_frames)}"
            f" ({100 * frames / max(1, ref_frames):.0f}%)  time {seconds:.0f}/{ref_seconds:.0f} s"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
