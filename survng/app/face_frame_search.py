"""Offline: would searching every frame find bigger faces than the old search?

For recent recorded person events this decodes every frame of the event window
from the main recording, runs the configured person detector and the existing
upper-body face search on each frame, and compares the best face per event
with what the old sampled search (refinement plan frames plus face-evidence
extras around the best anchors) would have seen with the same detectors.

The 300 px face detector also fires on the backs of heads, caps, torsos and
leaves, and the five-point landmark model fits points to any of them. Each face
is therefore aligned exactly as the live path aligns it (20%-padded crop,
landmarks, similarity transform) and scored with the AdaFace feature norm,
which is low for non-faces and unrecognisable faces. A face that fails
alignment scores 0. Results are reported for any face and for recognisable
faces (norm at least ``--min-norm``). Raw norms go to ``--json`` so the cut can
be retuned without decoding again, and ``--sheets`` writes per-camera contact
sheets of each event's best face under both rules for a visual check.

``--yunet`` also runs the YuNet detector (five built-in landmarks) on the same
upper-body crops of the same frames and reports it separately, aligning its
faces from its own landmarks, so the two detectors differ in nothing else.

The norm comes from ``adaface-ir50-webface4m-norm.xml``, an OpenVINO export
with both the embedding and the norm (``export_openvino_norm.py`` in the model
directory); the embedding-only export drops the norm.

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
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# Offsets the old face-evidence step tried around its two best anchors.
EXTRA_DELTAS = (-0.4, 0.4, -0.8, 0.8)
EXTRA_MIN_SEPARATION = 0.15
MAX_ANCHORS = 3
MAX_WINDOW_SECONDS = 30.0
# AdaFace feature norm separating faces from detector false positives. On
# this host real faces measured 16-18 and heads, caps, torsos and leaves 9-11.
MIN_QUALITY_NORM = 13.0
RULES = ("any", "recognisable")
THUMB_HEIGHT = 112
ADAFACE_DIR = Path("models/face_model/adaface-ir50-webface4m")
YUNET_MODEL = Path("models/face_detector/yunet/face_detection_yunet_2023mar.onnx")
# YuNet cost grows with input size; larger upper-body crops are downscaled.
YUNET_MAX_SIDE = 640


def passes(face: dict[str, Any], rule: str) -> bool:
    return rule == "any" or face.get("norm", 0.0) >= MIN_QUALITY_NORM


@dataclass
class FrameResult:
    offset: float
    usefulness: float
    faces: list[dict[str, Any]]
    person_heights: list[float]

    def best(self, rule: str = "any") -> float:
        return max((face["side"] for face in self.faces if passes(face, rule)), default=0.0)

    @property
    def best_face(self) -> float:
        return self.best("any")


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
    # rule -> (side, offset, thumbnail) of the best face anywhere in the window
    thumbs: dict[str, tuple[float, float, Any]] = field(default_factory=dict)
    # face id -> AdaFace unit embedding (float16), faces that aligned only
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    face_count: int = 0


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


def _quality_models(detector_config: Any) -> tuple[Any, Any]:
    """The live aligner (landmarks + template) and AdaFace with its norm output."""
    from .face_recognition import OpenVinoFaceRecognizer

    model_path = ADAFACE_DIR / "adaface-ir50-webface4m-norm.xml"
    if not model_path.is_file():
        raise FileNotFoundError(f"{model_path} is missing; run export_openvino_norm.py in that directory")
    recognizer = OpenVinoFaceRecognizer(detector_config.model_copy(update={
        "face_recognition_enabled": True,
        "face_recognition_device": "CPU",
        "cache_enabled": False,
        "face_embedding_profile": "adaface",
        "face_embedding_model_path": str(ADAFACE_DIR / "adaface-ir50-webface4m.xml"),
        "face_landmark_model_path": _resolve_model_path(detector_config.face_landmark_model_path),
    }))
    if not recognizer.ready:
        raise RuntimeError(f"face aligner unavailable: {recognizer.error}")
    try:
        from openvino import Core
    except ImportError:
        from openvino.runtime import Core
    threads = max(1, int(os.environ.get("OV_CPU_THREADS_NUM", "4")))
    # f32: the embedding-only export failed parity at the GPU's default f16.
    compiled = Core().compile_model(str(model_path), "CPU", {
        "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1",
        "INFERENCE_NUM_THREADS": threads, "INFERENCE_PRECISION_HINT": "f32",
    })
    return recognizer, compiled


class YuNetFaces:
    """YuNet on one crop; landmarks are kept relative to each face box."""

    def __init__(self, model_path: Path, threshold: float) -> None:
        import cv2

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), float(threshold), 0.3, 50)

    def detect(self, crop: np.ndarray) -> list[dict[str, Any]]:
        import cv2

        height, width = crop.shape[:2]
        scale = min(1.0, YUNET_MAX_SIDE / max(height, width))
        image = crop if scale == 1.0 else cv2.resize(
            crop, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA,
        )
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, rows = self.detector.detect(image)
        faces = []
        for row in () if rows is None else rows:
            values = np.asarray(row[:14], dtype=np.float32) / scale
            x, y, box_width, box_height = (float(value) for value in values[:4])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
            if x2 <= x1 or y2 <= y1:
                continue
            points = values[4:14].reshape(5, 2) - np.asarray([x1, y1], dtype=np.float32)
            faces.append({
                "label": "face",
                "confidence": round(float(row[14]), 4),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "landmarks_from_box": points.tolist(),
            })
        return faces


class FrameSearcher:
    def __init__(self, detector_config: Any, yunet: YuNetFaces | None = None, device: str = "CPU") -> None:
        from .detector import OpenVinoDetector
        from .face_detection import OpenVinoFaceDetector

        cpu = {"device": device, "cache_enabled": False}
        self.config = detector_config
        self.person_detector = OpenVinoDetector(detector_config.model_copy(update=cpu))
        self.face_detector = OpenVinoFaceDetector(
            detector_config.model_copy(
                update={
                    **cpu,
                    "face_recognition_enabled": True,
                    "face_recognition_device": device,
                    "face_detection_model_path": _resolve_model_path(
                        detector_config.face_detection_model_path
                    ),
                }
            )
        )
        if not self.face_detector.ready:
            raise RuntimeError(f"face detector unavailable: {self.face_detector.error}")
        self.recognizer, self.adaface = _quality_models(detector_config)
        self.yunet = yunet

    def quality_norm(self, crop: np.ndarray) -> tuple[float, np.ndarray | None]:
        """AdaFace (norm, unit embedding) of the face aligned as the live path aligns it."""
        try:
            aligned = self.recognizer._align(crop)
        except ValueError:
            return 0.0, None
        return self._norm(aligned)

    def quality_norm_from_points(
        self, frame: np.ndarray, points: np.ndarray,
    ) -> tuple[float, np.ndarray | None]:
        """AdaFace (norm, unit embedding) after aligning on a detector's own landmarks."""
        import cv2

        target = self.recognizer._ARCFACE_TEMPLATE.copy()
        target[:, 0] *= self.recognizer.input_shape[0] / 112.0
        target[:, 1] *= self.recognizer.input_shape[1] / 112.0
        matrix, _ = cv2.estimateAffinePartial2D(points.astype(np.float32), target, method=cv2.LMEDS)
        if matrix is None or not np.all(np.isfinite(matrix)):
            return 0.0, None
        aligned = cv2.warpAffine(
            frame, matrix, self.recognizer.input_shape,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        return self._norm(aligned)

    def _norm(self, aligned: np.ndarray) -> tuple[float, np.ndarray]:
        tensor = ((aligned.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None].copy()
        result = self.adaface([tensor])
        embedding = np.asarray(result[self.adaface.output(0)], dtype=np.float32).reshape(-1)
        norm = float(np.asarray(result[self.adaface.output(1)]).reshape(-1)[0])
        return norm, embedding

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
        faces = [
            {**face, "detector": "retail"}
            for face in RecordedMotionObjectDetector._detect_faces_in_people(
                frame, people, self.face_detector.detect, max_people=8,
            )
        ]
        if self.yunet is not None:
            faces += [
                {**face, "detector": "yunet"}
                for face in RecordedMotionObjectDetector._detect_faces_in_people(
                    frame, people, self.yunet.detect, max_people=8,
                )
            ]
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
        import cv2
        from .face_candidates import _padded_crop

        measured = []
        for face in faces:
            box = _box(face)
            if box is None:
                continue
            cropped = _padded_crop(frame, box)
            if cropped is None:
                continue
            crop = cropped[0]
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            thumb = cv2.resize(
                crop, (max(1, round(crop.shape[1] * THUMB_HEIGHT / crop.shape[0])), THUMB_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            if face["detector"] == "yunet":
                points = np.asarray(face["landmarks_from_box"], dtype=np.float32) + np.asarray(
                    [box[0], box[1]], dtype=np.float32
                )
                norm, embedding = self.quality_norm_from_points(frame, points)
            else:
                norm, embedding = self.quality_norm(crop)
            measured.append({
                "detector": face["detector"],
                "side": round(min(box[2] - box[0], box[3] - box[1]), 1),
                "confidence": round(float(face.get("confidence") or 0.0), 3),
                "norm": round(norm, 2),
                "box": [round(value, 1) for value in box],
                "_embedding": embedding,
                "_thumb": thumb,
            })
        return FrameResult(round(offset, 3), usefulness, measured, heights)


_SHOWINFO_FRAME = re.compile(
    rb"Parsed_showinfo[^\r\n]*\bn:\s*(\d+)[^\r\n]*\bpts_time:\s*([-+0-9.eE]+)"
)


def every_frame(
    path: Path,
    start_seconds: float,
    end_seconds: float,
    ffmpeg_path: str,
    hardware_acceleration: str | None = None,
) -> Iterator[tuple[float | None, np.ndarray]]:
    """Every presented frame in ``[start, end)`` of one file, full size.

    ``hardware_acceleration`` follows recorded-frame policy (explicit ``qsv``
    or ``vaapi`` decodes on the GPU); a hardware attempt that fails before its
    first frame is retried in software.

    Same command shape as the timeline frame-step decoder (PR 370): accurate
    seek with -copyts -start_at_zero, so ``showinfo`` reports file-relative
    PTS. Frames are streamed, not buffered. A frame whose showinfo line was
    lost or torn is yielded with ``None`` rather than a neighbour's time.
    """
    from .recorded_decoder import recorded_decode_plans
    from .recorded_video_frames import _ffprobe_video_metadata

    width, height, *_ = _ffprobe_video_metadata(path, ffmpeg_path)
    if width <= 0 or height <= 0:
        raise RuntimeError("video dimensions are unavailable")
    plans = recorded_decode_plans(hardware_acceleration)
    for attempt, plan in enumerate(plans):
        yielded = False
        try:
            for item in _every_frame_with_plan(path, start_seconds, end_seconds, ffmpeg_path, plan, width, height):
                yielded = True
                yield item
            return
        except (OSError, RuntimeError):
            if yielded or attempt == len(plans) - 1:
                raise


def _every_frame_with_plan(
    path: Path,
    start_seconds: float,
    end_seconds: float,
    ffmpeg_path: str,
    plan: Any,
    width: int,
    height: int,
) -> Iterator[tuple[float | None, np.ndarray]]:
    from .recorded_decoder import StreamingDecode, build_decode_command, warn_decoder_errors

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

    # Hardware frames are brought to system memory before showinfo.
    download = f"{plan.download_filter}," if plan.download_filter else ""
    command = build_decode_command(
        ffmpeg_path, path, plan=plan, loglevel="info",
        seek=f"{max(0.0, start_seconds - 0.001):.6f}", copyts=True, start_at_zero=True,
        first_video_stream=True,
        video_filter=f"trim=start={start_seconds:.6f}:end={end_seconds:.6f},{download}showinfo",
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
    hardware_acceleration: str | None = None,
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
            for pts, frame in every_frame(
                Path(path), local_start, local_end, ffmpeg_path, hardware_acceleration,
            ):
                if pts is None:
                    result.inexact_frames += 1
                    continue
                found = searcher.search(frame, float(segment_start) + pts - event_epoch)
                for face in found.faces:
                    thumb = face.pop("_thumb")
                    embedding = face.pop("_embedding", None)
                    face["id"] = f"{event_id}:{result.face_count}"
                    result.face_count += 1
                    if embedding is not None:
                        result.embeddings[face["id"]] = embedding.astype(np.float16)
                    for rule in RULES:
                        key = f"{face['detector']}/{rule}"
                        best = result.thumbs.get(key)
                        if passes(face, rule) and (best is None or face["side"] > best[0]):
                            result.thumbs[key] = (face["side"], found.offset, thumb)
                frames.append(found)
            result.frames.extend(frames)
        except (OSError, RuntimeError) as error:
            result.decode_errors.append(f"{Path(path).name}: {error}")
    result.frames.sort(key=lambda item: item.offset)
    chosen = baseline_frames(result.frames, plan_offsets, max_extra=max_extra)
    result.baseline_offsets = [frame.offset for frame in chosen]
    result.seconds = round(time.monotonic() - started, 1)
    return result


def for_detector(results: list[EventResult], name: str) -> list[EventResult]:
    """The same events with only one face detector's faces and thumbnails."""
    prefix = f"{name}/"
    return [
        replace(
            result,
            frames=[
                FrameResult(
                    frame.offset, frame.usefulness,
                    [face for face in frame.faces if face.get("detector", "retail") == name],
                    frame.person_heights,
                )
                for frame in result.frames
            ],
            thumbs={key[len(prefix):]: value for key, value in result.thumbs.items() if key.startswith(prefix)},
        )
        for result in results
    ]


def summarize(results: list[EventResult], face_min_size: int) -> dict[str, Any]:
    cameras: dict[str, dict[str, Any]] = {}
    for result in results:
        camera = cameras.setdefault(result.camera_id, {"events": []})
        by_offset = {frame.offset: frame for frame in result.frames}
        baseline = [by_offset[offset] for offset in result.baseline_offsets]
        low, high = result.plan_window
        plan = [frame for frame in result.frames if low <= frame.offset <= high]
        event: dict[str, Any] = {
            "event_id": result.event_id,
            "event_epoch": result.event_epoch,
            "frames": len(result.frames),
            "baseline_frames": len(baseline),
            "baseline_offsets": result.baseline_offsets,
            "window": list(result.window),
            "plan_window": list(result.plan_window),
            "max_person_height": round(max((h for f in result.frames for h in f.person_heights), default=0.0), 1),
            "inexact": result.inexact_frames,
            "errors": result.decode_errors,
            "seconds": result.seconds,
            "faces": [
                {"offset": frame.offset, **face}
                for frame in result.frames for face in frame.faces
            ],
        }
        for rule in RULES:
            best_frame = max(result.frames, key=lambda frame: frame.best(rule), default=None)
            full = best_frame.best(rule) if best_frame else 0.0
            event[rule] = {
                "baseline_best": round(max((f.best(rule) for f in baseline), default=0.0), 1),
                "plan_best": round(max((f.best(rule) for f in plan), default=0.0), 1),
                "all_best": round(full, 1),
                "all_best_offset": best_frame.offset if best_frame and full else None,
                "usable_frames": sum(1 for f in result.frames if f.best(rule) >= face_min_size),
            }
        camera["events"].append(event)
    for camera in cameras.values():
        searched = [e for e in camera["events"] if e["frames"]]
        camera["summary"] = {
            "events": len(camera["events"]),
            "events_decoded": len(searched),
            "frames": sum(e["frames"] for e in searched),
            "seconds": round(sum(e["seconds"] for e in camera["events"]), 1),
        }
        for rule in RULES:
            rows = [e[rule] for e in searched]
            summary: dict[str, Any] = {}
            for name in ("baseline", "plan", "all"):
                key = f"{name}_best"
                summary[f"{name}_any"] = sum(1 for r in rows if r[key] > 0)
                summary[f"{name}_usable"] = sum(1 for r in rows if r[key] >= face_min_size)
                summary[f"{name}_median"] = _median([r[key] for r in rows if r[key]])
            summary["gained_plan"] = sum(1 for r in rows if r["plan_best"] >= face_min_size > r["baseline_best"])
            summary["gained_all"] = sum(1 for r in rows if r["all_best"] >= face_min_size > r["baseline_best"])
            camera["summary"][rule] = summary
    return {
        "face_min_size": face_min_size,
        "min_quality_norm": MIN_QUALITY_NORM,
        "cameras": cameras,
    }


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
        f"recognisable = AdaFace norm >= {report['min_quality_norm']} after the live alignment",
    ]
    for camera_id, camera in report["cameras"].items():
        s = camera["summary"]
        lines.append("")
        lines.append(f"{camera_id}: {s['events_decoded']}/{s['events']} events decoded, {s['frames']} frames, {s['seconds']} s")
        for name in RULES:
            r = s[name]
            lines.append(
                f"  {name:<12} usable: old {r['baseline_usable']}  plan {r['plan_usable']}  all {r['all_usable']}"
                f"  (gained: plan {r['gained_plan']}, all {r['gained_all']});"
                f"  median best: old {r['baseline_median']}  plan {r['plan_median']}  all {r['all_median']}"
            )
        lines.append("  event   frames  old-n | any: old  plan   all @offset | recognisable: old  plan   all @offset  n>=min | person-h")
        for e in camera["events"]:
            cells = []
            for name in RULES:
                r = e[name]
                offset = "" if r["all_best_offset"] is None else f"@{r['all_best_offset']:+.2f}"
                cells.append(f"{r['baseline_best']:>5} {r['plan_best']:>5} {r['all_best']:>5} {offset:<7}")
            lines.append(
                f"  {e['event_id']:<7} {e['frames']:>6} {e['baseline_frames']:>6} | {cells[0]} |"
                f" {cells[1]} {e['recognisable']['usable_frames']:>6} | {e['max_person_height']:>8}"
                + (f"  errors: {'; '.join(e['errors'])}" if e["errors"] else "")
            )
    return "\n".join(lines)


def write_sheets(results: list[EventResult], directory: Path) -> list[Path]:
    """One image per camera: each event's best face, any rule then facing."""
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    cameras: dict[str, list[EventResult]] = {}
    for result in results:
        cameras.setdefault(result.camera_id, []).append(result)
    thumb_width = THUMB_HEIGHT + 90
    row_width = 70 + len(RULES) * (thumb_width + 10)
    for camera_id, events in cameras.items():
        rows = []
        for result in events:
            row = np.full((THUMB_HEIGHT + 22, row_width, 3), 255, dtype=np.uint8)
            cv2.putText(row, str(result.event_id), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            for column, rule in enumerate(RULES):
                left = 70 + column * (thumb_width + 10)
                best = result.thumbs.get(rule)
                label = f"{rule}: none"
                if best is not None:
                    side, offset, thumb = best
                    thumb = thumb[:, :thumb_width]
                    row[20:20 + thumb.shape[0], left:left + thumb.shape[1]] = thumb
                    label = f"{rule} {side:.0f}px {offset:+.1f}s"
                cv2.putText(row, label, (left, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 160), 1)
            rows.append(row)
        if rows:
            path = directory / f"{camera_id}.jpg"
            cv2.imwrite(str(path), np.vstack(rows))
            written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    global MIN_QUALITY_NORM
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database_dir", type=Path)
    parser.add_argument("--recording-index", type=Path, default=Path("runtime/recording-index/recordings.sqlite3"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--cameras", default="boiler,sherry-garage,gate,downstairs")
    parser.add_argument("--events-per-camera", type=int, default=10)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--margin", type=float, default=3.0, help="seconds searched beyond the plan window")
    parser.add_argument("--threads", type=int, default=4, help="OpenVINO CPU threads")
    parser.add_argument("--min-norm", type=float, default=MIN_QUALITY_NORM, help="AdaFace norm for a recognisable face")
    parser.add_argument("--device", default="CPU", help="OpenVINO device for the person and face detectors")
    parser.add_argument("--hw-decode", action="store_true", help="decode with the configured hardware_acceleration")
    parser.add_argument("--yunet", action="store_true", help="also run the YuNet face detector and compare")
    parser.add_argument("--yunet-model", type=Path, default=YUNET_MODEL)
    parser.add_argument("--yunet-threshold", type=float, default=0.6, help="YuNet score threshold")
    parser.add_argument("--json", type=Path, help="also write the full report here")
    parser.add_argument("--embeddings", type=Path, help="write AdaFace embeddings (.npz, ids match face ids in --json)")
    parser.add_argument("--sheets", type=Path, help="write per-camera contact sheets of best faces here")
    args = parser.parse_args(argv)

    os.environ.setdefault("OV_CPU_THREADS_NUM", str(max(1, args.threads)))
    MIN_QUALITY_NORM = args.min_norm
    os.nice(19)
    from .config import DetectorConfig
    from .motion_pipeline.object_detection import (
        _flatten_refinement_stages,
        resolve_recorded_refinement_plan,
    )

    raw = json.loads(args.config.read_text())
    detector_config = DetectorConfig.model_validate(raw["detector"])
    ffmpeg_path = str(raw.get("ffmpeg_path") or "ffmpeg")
    yunet = YuNetFaces(args.yunet_model, args.yunet_threshold) if args.yunet else None
    searcher = FrameSearcher(detector_config, yunet, device=args.device)

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
            hardware_acceleration=raw.get("hardware_acceleration") if args.hw_decode else None,
            margin=max(0.0, args.margin),
        )
        results.append(result)
        print(
            f"[{index}/{len(events)}] {camera_id} event {event_id}: {len(result.frames)} frames "
            f"in {result.seconds} s" + (f" ({'; '.join(result.decode_errors)})" if result.decode_errors else ""),
            file=sys.stderr, flush=True,
        )
    names = ["retail"] + (["yunet"] if yunet is not None else [])
    reports = {
        name: summarize(for_detector(results, name), int(detector_config.face_min_size))
        for name in names
    }
    if args.json:
        args.json.write_text(json.dumps({"detectors": reports}, indent=1))
    if args.embeddings:
        ids = [face_id for result in results for face_id in result.embeddings]
        vectors = [vector for result in results for vector in result.embeddings.values()]
        np.savez_compressed(
            args.embeddings, ids=np.asarray(ids), vectors=np.asarray(vectors, dtype=np.float16).reshape(-1, 512)
        )
    if args.sheets:
        for name in names:
            write_sheets(for_detector(results, name), args.sheets / name)
    print("\n\n".join(f"== face detector: {name} ==\n{format_report(reports[name])}" for name in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
