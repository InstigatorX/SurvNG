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
from .pipeline import CaptureResult, CaptureSettings, capture_event, main_stream_source, sub_stream_source

THUMB_HEIGHT = 112


def _rows_reader(index_path: Path, camera_id: str):
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)

    def rows(source: str, start: float, end: float) -> list[dict[str, Any]]:
        return [
            {"path": path, "start_epoch": begin, "duration_seconds": duration}
            for path, begin, duration in connection.execute(
                "select path, start_epoch, duration_seconds from recordings where camera_id = ? and source = ?"
                " and start_epoch < ? and start_epoch + duration_seconds > ? order by start_epoch",
                (camera_id, source, end, start),
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
    parser.add_argument("--cameras", required=True, help="comma-separated camera ids")
    parser.add_argument("--events-per-camera", type=int, default=5)
    parser.add_argument("--hours", type=float, default=72.0, help="look this far back for person events")
    parser.add_argument("--device", default="CPU", help="OpenVINO device for the person detector")
    parser.add_argument("--hw-decode", action="store_true", help="decode with the configured hardware_acceleration")
    parser.add_argument("--threads", type=int, default=4, help="OpenVINO CPU threads")
    parser.add_argument("--sheets", type=Path, help="write a contact sheet per camera here")
    parser.add_argument("--json", type=Path, help="write a per-event summary here")
    parser.add_argument("--store-dir", type=Path, help="also record the faces in a face capture store here")
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
    events = sqlite3.connect(f"file:{args.database_dir / 'survng.sqlite3'}?mode=ro", uri=True)
    now = datetime.now(timezone.utc)
    cameras = [item.strip() for item in args.cameras.split(",") if item.strip()]
    summary: list[dict[str, Any]] = []
    for camera_id in cameras:
        chosen = person_events(
            events, since=(now - timedelta(hours=args.hours)).isoformat(),
            until=(now - timedelta(seconds=90)).isoformat(), cameras=[camera_id], limit=args.events_per_camera,
        )
        rows = _rows_reader(args.recording_index, camera_id)
        results: list[tuple[int, CaptureResult]] = []
        for event_id, _, epoch, qualification in chosen:
            started = time.monotonic()
            main_rows = rows("main", epoch - 1.0, epoch + 1.0)
            main_size = mp4_video_dimensions(Path(main_rows[0]["path"])) if main_rows else None
            result = capture_event(
                plan_offsets=plan_offsets(config, qualification, epoch, camera_id),
                main_frames=main_stream_source(rows, epoch, ffmpeg, ffprobe, hardware),
                sub_frames=sub_stream_source(rows, epoch, ffmpeg),
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
        if args.sheets and results:
            print(f"contact sheet: {_sheet(camera_id, results, args.sheets)}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
