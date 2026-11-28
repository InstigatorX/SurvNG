"""Whole-episode best-cover pass: lifecycle, ranking, and ownership guards."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app import best_cover as best_cover_module
from survng.app.best_cover import BestCoverPass, RecordedBestCoverFrames, SearchCoverage
from survng.app.config import CameraConfig, DetectorConfig
from survng.app.events import EventStore
from survng.app.motion_incidents import MotionIncidentService
from survng.app.scene_zone_admission import establishment_zone_policy
from survng.app.video_frames import DecodedVideoFrame

CAMERA = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera.invalid/main")
POLICY = establishment_zone_policy(CAMERA, require_incident_zone=False, confidence_threshold=0.45)
# An acceptable early cover: fully framed, but small (100x200 of 1920x1080).
EARLY = {"x1": 900, "y1": 400, "x2": 1000, "y2": 600}


def textured(width: int, height: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)


def add_event(store: EventStore, *, age: float = 600.0, objects: list[dict] | None = None) -> dict:
    at = time.time() - age
    objects = objects if objects is not None else [{
        "label": "person", "confidence": .9, "incident_eligible": True,
        "snapshot_primary_subject": True, "temporal_consensus": True,
        "frame_source": "recorded_main", "frame_captured_at_epoch": at,
        "detection_frame_width": 1920, "detection_frame_height": 1080, "box": dict(EARLY),
        "snapshot_subject_area_ratio": 0.009645, "snapshot_edge_clearance_ratio": 0.2,
        "snapshot_quality_score": 0.2,
    }]
    return store.add_event(camera_id="gate", kind="motion", snapshot_path="early.webp",
                           created_at=datetime.fromtimestamp(at, timezone.utc).isoformat(),
                           objects_json=json.dumps(objects))


class FakeFrames:
    """Cheap 640x360 search frames at start+index; boxes are per frame."""

    def __init__(self, boxes: list[dict | None], *, main_rows=True, main_frame=True, complete=True):
        self.boxes = boxes
        self.main_rows = main_rows
        self.main_frame_available = main_frame
        self.complete = complete
        self.main_reads: list[float] = []
        self.frames = [textured(640, 360, index) for index in range(len(boxes))]
        self.start = 0.0

    def rows(self, source, start, end):
        return [{"path": "main.mp4"}] if source == "main" and self.main_rows else []

    def search(self, start, end, *, interval, cap, coverage: SearchCoverage):
        self.start = start
        coverage.source, coverage.window_seconds = "live", end - start
        coverage.recorded_seconds = coverage.window_seconds if self.complete else 0.0
        for index, frame in enumerate(self.frames):
            if coverage.frames >= cap:
                return
            coverage.frames += 1
            yield DecodedVideoFrame(start + index, frame)

    def main_frame(self, captured_at, width_hint, reference):
        self.main_reads.append(captured_at)
        if not self.main_frame_available:
            return None
        return textured(1920, 1080, 1000 + len(self.main_reads))


class FakeDetector:
    """Cheap frames return their nominated box; a main frame returns it at 3x."""

    def __init__(self, frames: FakeFrames, *, on_detect=None):
        self.frames = frames
        self.calls = 0
        self.on_detect = on_detect

    def detect(self, frame, confidence_threshold=None):
        self.calls += 1
        if self.on_detect is not None:
            self.on_detect(self.calls)
        if frame.shape[1] == 1920:
            box = self.frames.boxes[round(self.frames.main_reads[-1] - self.frames.start)]
            scale = 3
        else:
            box = self.frames.boxes[next(i for i, item in enumerate(self.frames.frames) if item is frame)]
            scale = 1
        if box is None:
            return []
        return [{"label": "person", "confidence": .9, "box": {k: v * scale for k, v in box.items()}}]


def make_pass(store, frames, *, detector=None, config=None, camera=CAMERA, writer=None):
    detector = detector or FakeDetector(frames)
    writes: list[str] = []

    def write(frame, at):
        path = f"best-{len(writes)}.webp"
        writes.append(path)
        return path

    cover_pass = BestCoverPass(
        camera_id="gate", store=store, detector=lambda: detector, frames=frames,
        snapshot_writer=writer or write,
        config=lambda: config or DetectorConfig(best_cover_pass_enabled=True),
        camera=lambda: camera, zone_policy=lambda: POLICY, lease_owner="best-cover-test",
    )
    return cover_pass, detector, writes


# Cheap-frame boxes (640x360). EARLY covers ~0.0096 of the frame.
SMALL = {"x1": 300, "y1": 133, "x2": 334, "y2": 200}          # ~0.0099: not better
SLIGHTLY = {"x1": 300, "y1": 120, "x2": 338, "y2": 200}       # ~0.0132: 1.37x, not material
LARGE = {"x1": 250, "y1": 60, "x2": 370, "y2": 300}           # 0.125: clearly better


def never() -> bool:
    return False


def test_later_clearly_better_view_replaces_acceptable_early_cover(tmp_path):
    store = EventStore(tmp_path)
    event = add_event(store)
    episode_before = store.scene_incident(event_id=event["id"])
    frames = FakeFrames([SMALL, None, LARGE, SMALL])
    cover_pass, detector, writes = make_pass(store, frames)

    assert cover_pass.run(never)

    updated = store.get(event["id"])
    assert updated["snapshot_path"] == writes[-1]
    assert updated["evidence_revision"] == event["evidence_revision"] + 1
    subject = json.loads(updated["objects_json"])[0]
    assert subject["snapshot_source"] == "best_cover"
    assert subject["detection_frame_width"] == 1920
    assert subject["snapshot_subject_area_ratio"] > 1.5 * 0.009645
    job = store.best_cover_job(event["id"])
    assert (job["state"], job["reason"]) == ("done", "promoted")
    # Cheap search frames plus exactly one full-detail verification.
    assert detector.calls == len(frames.frames) + 1
    assert len(frames.main_reads) == 1
    # Presentation only: no external notification, no admission/activity change.
    outbox = [row for row in store.pending_evidence_updates() if row["payload"].get("reason") == "best_cover"]
    assert outbox and all(row["payload"]["notify"] is False for row in outbox)
    episode_after = store.scene_incident(event_id=event["id"])
    assert episode_after["state"] == episode_before["state"]
    assert episode_after["last_epoch"] == episode_before["last_epoch"]
    assert not cover_pass.run(never)  # one job per episode


def test_view_that_is_not_materially_better_keeps_cover(tmp_path):
    store = EventStore(tmp_path)
    event = add_event(store)
    frames = FakeFrames([SMALL, SLIGHTLY, SMALL])
    cover_pass, detector, writes = make_pass(store, frames)

    assert cover_pass.run(never)

    assert store.get(event["id"])["snapshot_path"] == "early.webp"
    assert store.get(event["id"])["evidence_revision"] == event["evidence_revision"]
    job = store.best_cover_job(event["id"])
    assert (job["state"], job["reason"]) == ("done", "not_materially_better")
    assert frames.main_reads == [] and writes == []


def test_incomplete_search_without_better_view_retries_instead_of_concluding(tmp_path):
    store = EventStore(tmp_path)
    event = add_event(store)
    cover_pass, _detector, _writes = make_pass(store, FakeFrames([SMALL], complete=False))
    assert cover_pass.run(never)
    job = store.best_cover_job(event["id"])
    assert (job["state"], job["reason"]) == ("pending", "not_materially_better")


def due(store):
    with store._connect() as conn:
        conn.execute("update event_best_cover_jobs set available_at_epoch=0")


def test_security_work_preempts_defers_and_resumes_without_losing_job(tmp_path):
    store = EventStore(tmp_path)
    event = add_event(store)
    frames = FakeFrames([SMALL, LARGE, SMALL])
    service = MotionIncidentService(
        camera_id="gate", decision_processor=Mock(), tracking_enabled=lambda: False,
        has_trackable_objects=lambda _: False, start_tracking=Mock(), prewarm_tracking=Mock(),
        image_reader=Mock(), refinement_store=store,
    )
    detector = FakeDetector(frames, on_detect=lambda calls: calls == 1 and service._security_work_pending.set())
    cover_pass, detector, _writes = make_pass(store, frames, detector=detector)
    service.best_cover_pass = cover_pass.run

    assert service._run_best_cover()
    job = store.best_cover_job(event["id"])
    assert (job["state"], job["reason"], job["attempts"]) == ("pending", "security_work_preempted", 0)
    assert job["lease_owner"] == ""
    assert store.get(event["id"])["snapshot_path"] == "early.webp"
    # Security work owns the worker until it clears the token.
    assert not service._run_best_cover()

    service._security_work_pending.clear()
    detector.on_detect = None
    due(store)
    assert service._run_best_cover()
    assert store.best_cover_job(event["id"])["reason"] == "promoted"
    assert store.get(event["id"])["snapshot_path"] != "early.webp"


def test_concurrent_cover_revision_change_is_not_overwritten(tmp_path):
    store = EventStore(tmp_path)
    event = add_event(store)
    frames = FakeFrames([LARGE])

    def operator_changes_cover(frame, at):
        with store._connect() as conn:
            conn.execute("update events set snapshot_path='operator.webp',"
                         "evidence_revision=evidence_revision+1 where id=?", (event["id"],))
        return "best-raced.webp"

    cover_pass, _detector, _writes = make_pass(store, frames, writer=operator_changes_cover)
    assert cover_pass.run(never)

    assert store.get(event["id"])["snapshot_path"] == "operator.webp"
    job = store.best_cover_job(event["id"])
    assert (job["state"], job["reason"]) == ("pending", "cover_changed_concurrently")


def test_tracking_cover_is_left_alone_and_cannot_be_downgraded_by_tracking(tmp_path):
    store = EventStore(tmp_path)
    tracked = add_event(store, objects=[{
        "label": "person", "confidence": .9, "incident_eligible": True, "snapshot_primary_subject": True,
        "snapshot_source": "object_tracking", "detection_frame_width": 1920,
        "detection_frame_height": 1080, "box": dict(EARLY), "track_id": 1,
    }])
    cover_pass, detector, _writes = make_pass(store, FakeFrames([LARGE]))
    assert cover_pass.run(never)
    assert store.best_cover_job(tracked["id"])["state"] == "skipped"
    assert store.best_cover_job(tracked["id"])["reason"] == "tracking_cover_present"
    assert detector.calls == 0

    # Once the pass owns a cover, a later tracking promotion must be
    # materially better under the same rule, not just better than tracking's
    # own first frame.
    other = EventStore(tmp_path / "second")
    event = add_event(other)
    with other._connect() as conn:
        objects = json.loads(conn.execute("select objects_json from events").fetchone()[0])
        objects[0]["track_id"] = 7
        conn.execute("update events set objects_json=?", (json.dumps(objects),))
    second_pass, _detector, _writes = make_pass(other, FakeFrames([LARGE]))
    assert second_pass.run(never)
    owned = other.get(event["id"])
    area = json.loads(owned["objects_json"])[0]["snapshot_subject_area_ratio"]

    def track_promotion(factor):
        path = tmp_path / "second" / f"tracked-{factor}.webp"
        path.write_bytes(b"x")
        return other.promote_tracking_cover(
            event["id"], snapshot_path=str(path), captured_at=time.time(), frame_width=1920,
            frame_height=1080, tracked_objects=[{"track_id": 7, "box": dict(EARLY), "label": "person"}],
            cover_metrics={"snapshot_subject_area_ratio": area * factor,
                           "snapshot_edge_clearance_ratio": 0.05, "snapshot_quality_score": 0.9},
        )

    assert track_promotion(1.1) is None
    assert other.get(event["id"])["snapshot_path"] == owned["snapshot_path"]
    assert track_promotion(2.0) is not None
    assert json.loads(other.get(event["id"])["objects_json"])[0]["snapshot_source"] == "object_tracking"


def test_substream_only_camera_is_skipped_explicitly(tmp_path):
    store = EventStore(tmp_path)
    event = add_event(store)
    camera = CAMERA.model_copy(update={"record": False, "record_sub": True})
    cover_pass, detector, _writes = make_pass(store, FakeFrames([LARGE]), camera=camera)
    assert cover_pass.run(never)
    job = store.best_cover_job(event["id"])
    assert (job["state"], job["reason"]) == ("skipped", "main_recording_disabled")
    assert detector.calls == 0


@pytest.mark.parametrize(("frames", "reason"), [
    (FakeFrames([LARGE], main_rows=False), "main_recording_unavailable"),
    (FakeFrames([LARGE], main_frame=False), "main_frame_unavailable"),
    (FakeFrames([]), "search_footage_unavailable"),
])
def test_missing_footage_is_incomplete_not_an_arbitrary_cover(tmp_path, frames, reason):
    store = EventStore(tmp_path)
    event = add_event(store)
    cover_pass, _detector, writes = make_pass(store, frames)
    assert cover_pass.run(never)
    job = store.best_cover_job(event["id"])
    assert (job["state"], job["reason"]) == ("pending", reason)
    assert store.get(event["id"])["snapshot_path"] == "early.webp" and writes == []
    for _ in range(2):
        due(store)
        assert cover_pass.run(never)
    assert store.best_cover_job(event["id"])["state"] == "exhausted"


def test_frame_cap_bounds_detector_work(tmp_path):
    store = EventStore(tmp_path)
    add_event(store)
    frames = FakeFrames([SMALL] * 10)
    cover_pass, detector, _writes = make_pass(
        store, frames, config=DetectorConfig(best_cover_pass_enabled=True, best_cover_max_frames=3))
    assert cover_pass.run(never)
    assert detector.calls == 3


def test_processing_deadline_is_incomplete_and_exhausts(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    event = add_event(store)
    monkeypatch.setattr(best_cover_module, "BEST_COVER_ATTEMPT_SECONDS", 0.0)
    cover_pass, _detector, writes = make_pass(store, FakeFrames([LARGE]))
    for _ in range(3):
        due(store)
        assert cover_pass.run(never)
    job = store.best_cover_job(event["id"])
    assert (job["state"], job["reason"], job["attempts"]) == ("exhausted", "processing_deadline", 3)
    assert writes == [] and store.get(event["id"])["snapshot_path"] == "early.webp"


def test_on_demand_runs_while_eager_tracking_owns_the_cover_first(tmp_path):
    store = EventStore(tmp_path)
    on_demand = add_event(store)
    at = datetime.fromisoformat(on_demand["created_at"]).timestamp()
    # on_demand: tracking is deferred until someone opens the incident.
    assert store.enqueue_scene_tracking(on_demand["id"], at - 1, at + 45, deferred=True)["state"] == "deferred"
    cover_pass, _detector, _writes = make_pass(store, FakeFrames([LARGE]))
    assert cover_pass.run(never)
    assert store.best_cover_job(on_demand["id"])["reason"] == "promoted"

    eager_store = EventStore(tmp_path / "eager")
    eager = add_event(eager_store)
    assert eager_store.enqueue_scene_tracking(eager["id"], at - 1, at + 45)["state"] == "queued"
    eager_pass, detector, _writes = make_pass(eager_store, FakeFrames([LARGE]))
    assert not eager_pass.run(never)  # queued/running tracking goes first
    assert detector.calls == 0
    with eager_store._connect() as conn:
        conn.execute("update scene_analysis_jobs set state='complete'")
    assert eager_pass.run(never)
    assert eager_store.best_cover_job(eager["id"])["reason"] == "promoted"


def test_disabled_or_open_episode_does_no_work(tmp_path):
    store = EventStore(tmp_path)
    add_event(store, age=5.0)  # episode still inside its inactivity grace
    frames = FakeFrames([LARGE])
    disabled, detector, _writes = make_pass(store, frames, config=DetectorConfig())
    assert DetectorConfig().best_cover_pass_enabled is False
    assert not disabled.run(never)
    enabled, _detector, _writes = make_pass(store, frames)
    assert not enabled.run(never)
    with store._connect() as conn:
        assert conn.execute("select count(*) from event_best_cover_jobs").fetchone()[0] == 0
    assert detector.calls == 0


def test_recorded_frames_prefer_substream_and_count_decode_failures(tmp_path, monkeypatch):
    for name in ("live.mp4", "main.mp4", "broken.mp4"):
        (tmp_path / name).write_bytes(b"")
    rows = {
        "live": [{"path": str(tmp_path / "live.mp4"), "start_epoch": 100.0, "end_epoch": 110.0},
                 {"path": str(tmp_path / "broken.mp4"), "start_epoch": 110.0, "end_epoch": 120.0}],
        "main": [{"path": str(tmp_path / "main.mp4"), "start_epoch": 100.0, "end_epoch": 120.0}],
    }
    recorder = SimpleNamespace(ffmpeg_path="ffmpeg", recording_rows_between=(
        lambda camera_id, start, end, *, source, discover_missing=True: rows[source]))
    calls = []

    def sampled(path, *, start_epoch, sample_fps, duration_seconds, maximum_width, **_kwargs):
        calls.append((path.name, sample_fps, maximum_width))
        if path.name == "broken.mp4":
            raise RuntimeError("decoder failed")
        count = int(duration_seconds * sample_fps)
        return iter(DecodedVideoFrame(start_epoch + i / sample_fps, np.zeros((2, 2, 3), np.uint8))
                    for i in range(count))

    monkeypatch.setattr(best_cover_module, "sampled_video_frames", sampled)
    frames = RecordedBestCoverFrames(camera_id="gate", recorder=lambda: recorder, main_frame_at=Mock())
    coverage = SearchCoverage()
    yielded = list(frames.search(100.0, 120.0, interval=1.5, cap=6, coverage=coverage))
    assert coverage.source == "live"
    assert len(yielded) <= 6 and all(100.0 <= item.captured_at <= 120.0 for item in yielded)
    assert {name for name, *_ in calls} == {"live.mp4", "broken.mp4"}
    assert all(width == best_cover_module.BEST_COVER_SEARCH_WIDTH for *_, width in calls)
    assert coverage.decode_failures == 1 and not coverage.complete
