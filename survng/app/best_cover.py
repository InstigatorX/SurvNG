"""Whole-episode "best cover" pass: a light search independent of tracking.

Once a camera episode has closed, sample its recorded window uniformly at a
low rate on cheap frames (the substream recording when one exists, otherwise
the main recording decoded at low width), run the object detector on at most
``detector.best_cover_max_frames`` of them (no tracker, no ReID), rank the
target subject's views with the recorded-evidence cover score, then re-read
only the winning instant from the main recording at full detail, verify the
subject there, and promote it when it is materially better than the stored
cover.

Motion-guided selection was considered and not used: per-sample foreground
size/position is not retained for a closed episode (the analysis ring holds
seconds, and the only durable cache drops area, expires after ten minutes and
exists only in some trigger modes). See docs/incident-evidence-data-path.md.

Ownership: the per-camera refinement worker runs this at its lowest priority,
under a durable lease in ``event_best_cover_jobs``. Security work preempts it
without charging an attempt. Promotion re-checks lease and evidence revision.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import numpy as np

from .detector import detection_failure
from .event_store.best_cover import best_cover_baseline, best_cover_replaces, best_cover_target
from .event_store.evidence import EventSnapshotChangedError
from .evidence_work import EvidenceWorkPreempted
from .motion_pipeline.recorded_evidence import _normalized_box_metrics, _subject_view_score
from .recorded_video_frames import sampled_video_frames
from .scene_zone_admission import interpret_establishment_observation
from .security import redact_secret_text
from .video_frames import DecodedVideoFrame, VideoFrameReference
from .visual_quality import image_quality

LOGGER = logging.getLogger(__name__)
# Wall-clock budget for one attempt, including main-recording re-reads.
BEST_COVER_ATTEMPT_SECONDS = 20.0
# Cheap search frames are decoded no wider than this.
BEST_COVER_SEARCH_WIDTH = 640
# Ranked candidates verified on the main recording per attempt.
BEST_COVER_MAIN_READS = 2
BEST_COVER_LOG_INTERVAL_SECONDS = 60.0


class BestCoverDetector(Protocol):
    def detect(self, frame: np.ndarray, confidence_threshold: float | None = None) -> list[dict[str, Any]]: ...


class BestCoverRecorder(Protocol):
    ffmpeg_path: str

    def recording_rows_between(self, camera_id: str, start_epoch: float, end_epoch: float, *,
                               source: str, discover_missing: bool = True) -> list[dict[str, Any]]: ...


@dataclass
class SearchCoverage:
    source: str = ""
    window_seconds: float = 0.0
    recorded_seconds: float = 0.0
    decode_failures: int = 0
    frames: int = 0

    @property
    def complete(self) -> bool:
        return (self.decode_failures == 0 and self.window_seconds > 0
                and self.recorded_seconds >= self.window_seconds - 0.5)

    def summary(self) -> dict[str, Any]:
        return {"source": self.source, "window_seconds": round(self.window_seconds, 3),
                "recorded_seconds": round(self.recorded_seconds, 3),
                "decode_failures": self.decode_failures, "frames": self.frames, "complete": self.complete}


def _row_span(row: dict[str, Any]) -> tuple[float, float]:
    start = float(row.get("start_epoch") or 0.0)
    end = row.get("end_epoch")
    if end is None:
        end = start + float(row.get("duration_seconds") or 0.0)
    return start, float(end)


class RecordedBestCoverFrames:
    """Recording access for the pass: cheap uniform search, one full-detail read."""

    def __init__(self, *, camera_id: str, recorder: Callable[[], BestCoverRecorder],
                 main_frame_at: Callable[[float, int, VideoFrameReference | None], np.ndarray | None]) -> None:
        self.camera_id = camera_id
        self._recorder = recorder
        self._main_frame_at = main_frame_at
        self._last_log = float("-inf")

    def rows(self, source: str, start: float, end: float) -> list[dict[str, Any]]:
        rows = self._recorder().recording_rows_between(self.camera_id, start, end, source=source)
        return sorted((dict(row) for row in rows if Path(str(row.get("path") or "")).is_file()),
                      key=lambda row: _row_span(row)[0])

    def search(self, start: float, end: float, *, interval: float, cap: int,
               coverage: SearchCoverage) -> Iterator[DecodedVideoFrame]:
        """Yield at most ``cap`` frames spread uniformly over the recorded window."""
        duration = max(0.0, end - start)
        coverage.window_seconds = duration
        live_rows, main_rows = self.rows("live", start, end), self.rows("main", start, end)

        def covered(rows: list[dict[str, Any]]) -> float:
            return sum(max(0.0, min(end, _row_span(row)[1]) - max(start, _row_span(row)[0])) for row in rows)

        # The substream recording is the cheap search source when it covers
        # the window at least as well as main; otherwise decode main small.
        rows, source = (live_rows, "live") if live_rows and covered(live_rows) >= covered(main_rows) else (main_rows, "main")
        coverage.source = source
        coverage.recorded_seconds = covered(rows)
        if not rows or duration <= 0:
            return
        count = max(1, min(int(cap), int(duration // max(0.1, interval)) + 1))
        fps = max(0.05, count / max(duration, 1e-3))
        recorder = self._recorder()
        for row in rows:
            if coverage.frames >= cap:
                return
            row_start, row_end = _row_span(row)
            segment_start, segment_end = max(start, row_start), min(end, row_end)
            if segment_end <= segment_start:
                continue
            path = Path(str(row["path"]))
            samples = None
            try:
                samples = iter(sampled_video_frames(
                    path, start_epoch=segment_start, sample_fps=fps,
                    duration_seconds=segment_end - segment_start, ffmpeg_path=recorder.ffmpeg_path,
                    maximum_width=BEST_COVER_SEARCH_WIDTH,
                    start_offset_seconds=max(0.0, segment_start - row_start), probe_path=path,
                ))
                for sample in samples:
                    if coverage.frames >= cap:
                        return
                    coverage.frames += 1
                    yield sample
            except (OSError, RuntimeError, ValueError) as error:
                coverage.decode_failures += 1
                self._log("best cover search decode failed for %s/%s: %s", path.name, error)
            finally:
                close = getattr(samples, "close", None)
                if callable(close):
                    close()

    def main_frame(self, captured_at: float, width_hint: int,
                   reference: VideoFrameReference | None) -> np.ndarray | None:
        return self._main_frame_at(captured_at, max(1, int(width_hint)), reference)

    def _log(self, message: str, name: str, error: Exception) -> None:
        now = time.monotonic()
        if now - self._last_log >= BEST_COVER_LOG_INTERVAL_SECONDS:
            self._last_log = now
            LOGGER.warning(message, self.camera_id, name, redact_secret_text(error))


@dataclass(order=True)
class _Candidate:
    rank: tuple[int, float, float, float]
    captured_at: float = field(compare=False)
    label: str = field(compare=False)
    normalized_box: tuple[float, float, float, float] = field(compare=False)
    area: float = field(compare=False)
    clearance: float = field(compare=False)
    reference: VideoFrameReference | None = field(compare=False, default=None)


def _box(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = item.get("box")
    try:
        values = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))  # type: ignore[index]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in values) or values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values  # type: ignore[return-value]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _crop_quality(frame: np.ndarray, box: tuple[float, float, float, float]) -> float:
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
    x2, y2 = min(width, int(math.ceil(box[2]))), min(height, int(math.ceil(box[3])))
    crop = frame[y1:y2, x1:x2]
    return float(image_quality(crop).score) if crop.size else 0.0


class BestCoverPass:
    """Runs one leased best-cover job per call on the refinement worker."""

    def __init__(
        self, *, camera_id: str, store: Any, detector: Callable[[], BestCoverDetector],
        frames: RecordedBestCoverFrames, snapshot_writer: Callable[[np.ndarray, datetime], str],
        config: Callable[[], Any], camera: Callable[[], Any],
        zone_policy: Callable[[], dict[str, Any]] | None = None, lease_owner: str,
    ) -> None:
        self.camera_id = camera_id
        self.store = store
        self.detector = detector
        self.frames = frames
        self.snapshot_writer = snapshot_writer
        self.config = config
        self.camera = camera
        self.zone_policy = zone_policy
        self.lease_owner = lease_owner
        self._last_failure_log = float("-inf")

    def enabled(self) -> bool:
        return bool(getattr(self.config(), "best_cover_pass_enabled", False))

    def run(self, preempted: Callable[[], bool]) -> bool:
        """Claim and settle at most one job. True when a job was handled."""
        if not self.enabled() or preempted():
            return False
        claim = getattr(self.store, "claim_best_cover", None)
        if not callable(claim):
            return False
        job = claim(self.camera_id, lease_owner=self.lease_owner)
        if job is None:
            return False
        event_id = int(job["event_id"])
        try:
            state, reason, result = self._evaluate(job, preempted)
        except EvidenceWorkPreempted:
            self.store.defer_best_cover(event_id, lease_owner=self.lease_owner, reason="security_work_preempted")
            return True
        except Exception as error:
            state, reason, result = "retry", f"best_cover_failed:{type(error).__name__}", {}
            now = time.monotonic()
            if now - self._last_failure_log >= BEST_COVER_LOG_INTERVAL_SECONDS:
                self._last_failure_log = now
                LOGGER.exception("best cover pass failed for %s event %d", self.camera_id, event_id)
        self.store.finish_best_cover(event_id, lease_owner=self.lease_owner, state=state,
                                     reason=reason, result=result)
        return True

    def _check(self, preempted: Callable[[], bool]) -> None:
        if preempted():
            raise EvidenceWorkPreempted()

    def _evaluate(self, job: dict[str, Any], preempted: Callable[[], bool]) -> tuple[str, str, dict[str, Any]]:
        config, camera = self.config(), self.camera()
        if not bool(getattr(camera, "record", True)):
            # A substream-only camera has no full-detail source to save from.
            return "skipped", "main_recording_disabled", {}
        event = job["event"]
        objects = [item for item in _json_list(event.get("objects_json")) if isinstance(item, dict)]
        if any(item.get("snapshot_source") == "object_tracking" for item in objects):
            return "skipped", "tracking_cover_present", {}
        target = best_cover_target(objects)
        if target is None:
            return "skipped", "ambiguous_cover_subject", {}
        episode = job["episode"]
        start = float(episode["start_epoch"])
        end = max(float(episode["end_epoch"]), float(episode["last_activity_epoch"]), start + 1.0)
        if not self.frames.rows("main", start, end):
            return "retry", "main_recording_unavailable", {}
        label = str(target["label"])
        baseline = best_cover_baseline(target)
        policy = _event_zone_policy(objects) or (self.zone_policy() if self.zone_policy else None)
        threshold = float(getattr(config, "event_candidate_confidence_threshold", 0.25))
        cap = max(1, int(getattr(config, "best_cover_max_frames", 12)))
        interval = float(getattr(config, "best_cover_sample_interval_seconds", 1.5))
        deadline = time.monotonic() + BEST_COVER_ATTEMPT_SECONDS
        coverage = SearchCoverage()
        candidates: list[_Candidate] = []
        detector_calls = 0
        result: dict[str, Any] = {"baseline": {k: round(float(v), 6) for k, v in baseline.items()},
                                  "window": [start, end]}
        for sample in self.frames.search(start, end, interval=interval, cap=cap, coverage=coverage):
            self._check(preempted)
            if time.monotonic() >= deadline:
                result.update(search=coverage.summary(), detector_calls=detector_calls)
                return "retry", "processing_deadline", result
            frame = sample.frame
            detections = self._detect(frame, threshold)
            detector_calls += 1
            height, width = frame.shape[:2]
            for item in detections:
                box = _box(item)
                if box is None or str(item.get("label")) != label:
                    continue
                if not _admitted(item, policy, width, height):
                    continue
                clearance, area = _normalized_box_metrics(item, frame)
                quality = _crop_quality(frame, box)
                fully_framed = clearance >= 0.01
                candidates.append(_Candidate(
                    rank=(int(fully_framed), _subject_view_score(area, quality), area,
                          float(item.get("confidence") or 0.0)),
                    captured_at=float(sample.captured_at), label=label,
                    normalized_box=(box[0] / width, box[1] / height, box[2] / width, box[3] / height),
                    area=area, clearance=clearance,
                    reference=sample.reference if coverage.source == "main" else None,
                ))
        result.update(search=coverage.summary(), detector_calls=detector_calls,
                      candidates=len(candidates))
        if coverage.frames == 0:
            return "retry", "search_footage_unavailable", result
        # Without the stored cover's quality, cheap-frame quality is not
        # comparable with main-frame quality; it is checked after the re-read.
        baseline_pixels = float(baseline["pixels"])
        baseline_frame_pixels = (baseline_pixels / float(baseline["area"])) if float(baseline["area"]) > 0 else 0.0
        plausible = sorted((c for c in candidates if best_cover_replaces(
            baseline_fully_framed=bool(baseline["fully_framed"]), baseline_area=float(baseline["area"]),
            baseline_pixels=baseline_pixels, baseline_quality=0.0, candidate_area=c.area,
            candidate_pixels=c.area * max(baseline_frame_pixels, 1.0), candidate_clearance=c.clearance,
            candidate_quality=1.0)), reverse=True)
        if not plausible:
            # A "no better view" verdict is final only for a complete search.
            return ("done" if coverage.complete else "retry"), "not_materially_better", result
        main_failures = 0
        last_reason = "verified_frame_not_better"
        for candidate in plausible[:BEST_COVER_MAIN_READS]:
            self._check(preempted)
            if time.monotonic() >= deadline:
                return "retry", "processing_deadline", result
            frame = self.frames.main_frame(candidate.captured_at, int(target.get("detection_frame_width") or 1),
                                           candidate.reference)
            if frame is None or not getattr(frame, "size", 0):
                main_failures += 1
                last_reason = "main_frame_unavailable"
                continue
            verified = self._verify(frame, candidate, threshold, policy)
            detector_calls += 1
            if verified is None:
                last_reason = "subject_not_verified_on_main"
                continue
            subject, metrics = verified
            height, width = frame.shape[:2]
            if not best_cover_replaces(
                baseline_fully_framed=bool(baseline["fully_framed"]), baseline_area=float(baseline["area"]),
                baseline_pixels=baseline_pixels, baseline_quality=float(baseline["quality"]),
                candidate_area=metrics["snapshot_subject_area_ratio"],
                candidate_pixels=metrics["snapshot_subject_area_ratio"] * width * height,
                candidate_clearance=metrics["snapshot_edge_clearance_ratio"],
                candidate_quality=metrics["snapshot_quality_score"],
            ):
                last_reason = "verified_frame_not_better"
                continue
            self._check(preempted)
            snapshot_path = self.snapshot_writer(frame, datetime.fromtimestamp(candidate.captured_at, timezone.utc))
            diagnostics: dict[str, Any] = {}
            try:
                promoted = self.store.promote_best_cover(
                    int(job["event_id"]), snapshot_path=snapshot_path, captured_at=candidate.captured_at,
                    frame_width=width, frame_height=height, subject=subject, cover_metrics=metrics,
                    expected_revision=int(event["evidence_revision"]), lease_owner=self.lease_owner,
                    diagnostics=diagnostics,
                )
            except EventSnapshotChangedError:
                # Tracking or an operator changed the cover after this job read
                # it. Re-evaluate against the new cover on the next attempt.
                result.update(detector_calls=detector_calls)
                return "retry", "cover_changed_concurrently", result
            result.update(detector_calls=detector_calls, captured_at=candidate.captured_at,
                          metrics=metrics)
            if promoted is None:
                return "done", str(diagnostics.get("reason") or "promotion_declined"), result
            return "done", "promoted", result
        result.update(detector_calls=detector_calls)
        if main_failures and main_failures == min(len(plausible), BEST_COVER_MAIN_READS):
            # Missing full-detail footage is incomplete, never a reason to keep
            # a cheap frame or to fall back to an arbitrary one.
            return "retry", "main_frame_unavailable", result
        return ("done" if coverage.complete else "retry"), last_reason, result

    def _detect(self, frame: np.ndarray, threshold: float) -> list[dict[str, Any]]:
        detections = list(self.detector().detect(frame, confidence_threshold=threshold))
        if any(isinstance(item, dict) and item.get("status") == "inference_deferred" for item in detections):
            # Optional inference was declined under load: yield, do not fail.
            raise EvidenceWorkPreempted()
        failure = detection_failure(detections)
        if failure:
            raise RuntimeError(f"object detector unavailable: {failure[:160]}")
        return [item for item in detections if isinstance(item, dict) and item.get("label")]

    def _verify(self, frame: np.ndarray, candidate: _Candidate, threshold: float,
                policy: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Find the nominated subject in the full-detail frame and measure it there."""
        height, width = frame.shape[:2]
        nx1, ny1, nx2, ny2 = candidate.normalized_box
        expected = (nx1 * width, ny1 * height, nx2 * width, ny2 * height)
        diagonal = math.hypot(width, height)
        best: tuple[float, dict[str, Any], tuple[float, float, float, float]] | None = None
        for item in self._detect(frame, threshold):
            box = _box(item)
            if box is None or str(item.get("label")) != candidate.label or not _admitted(item, policy, width, height):
                continue
            overlap = _iou(expected, box)
            distance = math.hypot((box[0] + box[2] - expected[0] - expected[2]) / 2,
                                  (box[1] + box[3] - expected[1] - expected[3]) / 2) / diagonal
            if overlap < 0.2 and distance > 0.08:
                continue
            score = overlap - distance
            if best is None or score > best[0]:
                best = (score, item, box)
        if best is None:
            return None
        _score, item, box = best
        clearance, area = _normalized_box_metrics(item, frame)
        metrics = {
            "snapshot_primary_subject": True,
            "snapshot_subject_area_ratio": round(area, 6),
            "snapshot_edge_clearance_ratio": round(clearance, 6),
            "snapshot_quality_score": round(_crop_quality(frame, box), 6),
        }
        subject = {"label": candidate.label, "confidence": float(item.get("confidence") or 0.0),
                   "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}}
        return subject, metrics


def _json_list(raw: Any) -> list[Any]:
    try:
        value = json.loads(raw or "[]") if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _event_zone_policy(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The establishment zone snapshot recorded with the event, when present."""
    for item in objects:
        if item.get("status") == "motion_qualification":
            policy = (item.get("motion_qualification") or {}).get("establishment_zone_policy")
            if isinstance(policy, dict):
                return policy
    return None


def _admitted(item: dict[str, Any], policy: dict[str, Any] | None, width: int, height: int) -> bool:
    """Zone admission as recorded for this event; never a looser rule."""
    interpretation = interpret_establishment_observation(item, policy, frame_width=width, frame_height=height)
    return bool(interpretation.get("establishment_eligible")) and bool(interpretation.get("confidence_admitted", True))

