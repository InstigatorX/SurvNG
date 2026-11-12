"""Evidence gallery images nominated by object tracking."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from survng.app.config import CameraConfig, ImageStorageConfig, ObjectTrackingConfig
from survng.app.evidence_gallery import MAX_DETECTION_IMAGES, MAX_GALLERY_IMAGES, encode_gallery
from survng.app.events import EventStore
from survng.app.image_storage import DurableImageWriter
from survng.app.motion_pipeline.object_detection import _RecordedDetectionSample
from survng.app.object_track.evidence import (
    TrackedEvidenceCollector,
    save_tracked_evidence,
    select_evidence_views,
)
from survng.app.object_tracking import ObjectTrackingSession
from survng.app.video_frames import VideoFrameReference

WIDTH, HEIGHT = 320, 180


def reference(epoch: float, *, exact: bool = True) -> VideoFrameReference:
    return VideoFrameReference(
        source_path=Path("recordings/gate.mp4"), seek_offset_seconds=epoch - 1000.0,
        pts=int((epoch - 1000.0) * 90000), pts_seconds=epoch - 1000.0,
        time_base_num=1, time_base_den=90000, captured_at=epoch, exact=exact,
    )


def tracked(box, *, track_id=4, label="car", state="confirmed", **extra) -> dict:
    x1, y1, x2, y2 = box
    return {"label": label, "confidence": 0.9, "track_id": track_id, "track_state": state,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, **extra}


def textured(seed: int = 0, blur: int = 0) -> np.ndarray:
    frame = np.random.default_rng(seed).integers(40, 216, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    return cv2.GaussianBlur(frame, (blur, blur), 0) if blur else frame


def collect(moments, *, collector=None) -> TrackedEvidenceCollector:
    collector = collector or TrackedEvidenceCollector()
    for index, (epoch, objects) in enumerate(moments):
        collector.consider(textured(index), epoch, objects, WIDTH, HEIGHT, reference(epoch))
    return collector


def seeded_store(tmp_path) -> tuple[EventStore, dict]:
    store = EventStore(tmp_path)
    event = store.add_event(
        "gate", "motion", created_at=datetime.fromtimestamp(1000, timezone.utc).isoformat(),
        objects_json=json.dumps([{
            "label": "car", "confidence": 0.9, "incident_eligible": True, "confidence_threshold": 0.7,
            "box": {"x1": 10, "y1": 10, "x2": 60, "y2": 50},
            "detection_frame_width": WIDTH, "detection_frame_height": HEIGHT,
        }]),
    )
    return store, event


def snapshot_writer(tmp_path):
    writer = DurableImageWriter(ImageStorageConfig())

    def write(image, at):
        return str(writer.write(tmp_path / "snapshots", f"image-{at.timestamp()}", image) or "")

    return write


def tracking_capacity(store):
    return lambda event_id: store.scene_evidence_capacity(event_id, source="object_tracking")


def save(store, event, views, tmp_path, *, frames=None, known=(), stopped=lambda: False):
    requested = []

    def provide(captured_at, width, frame_reference):
        requested.append((captured_at, width, frame_reference))
        if frames is not None:
            return frames(captured_at)
        # Full detail is twice the tracking resolution.
        return np.full((HEIGHT * 2, WIDTH * 2, 3), 90, np.uint8)

    status = save_tracked_evidence(
        views, event_id=event["id"], camera_id="gate",
        capacity=tracking_capacity(store), writer=store.retain_scene_evidence_images,
        frame_provider=provide, snapshot_writer=snapshot_writer(tmp_path),
        known_frames=list(known), stopped=stopped,
    )
    return status, requested


def test_collector_keeps_the_best_view_per_subject_window_and_needs_exact_frames():
    collector = TrackedEvidenceCollector()
    small, large = (100, 60, 130, 90), (80, 40, 200, 140)
    collector.consider(textured(1), 1000.2, [tracked(small)], WIDTH, HEIGHT, reference(1000.2))
    collector.consider(textured(2), 1000.9, [tracked(large)], WIDTH, HEIGHT, reference(1000.9))
    # No exact frame reference means the full-detail frame cannot be re-read.
    collector.consider(textured(3), 1004.0, [tracked(large)], WIDTH, HEIGHT, None)
    collector.consider(textured(4), 1006.0, [tracked(large)], WIDTH, HEIGHT, reference(1006.0, exact=False))

    views = collector.views()

    assert [view.captured_at for view in views] == [1000.9]
    assert views[0].box == large


def test_collector_ignores_unconfirmed_face_context_and_ignored_zone_subjects():
    box = (80, 40, 200, 140)
    collector = collect([
        (1000.0, [tracked(box, state="tentative")]),
        (1002.0, [tracked(box, label="face")]),
        (1004.0, [tracked(box, activity_role="scene_context")]),
        (1006.0, [tracked(box, zone_admission_reason="ignored_zone")]),
        (1008.0, [tracked(box, track_id=None)]),
    ])

    assert collector.views() == []


def test_selection_spreads_views_and_skips_what_is_already_retained():
    entering, crossing, leaving = (10, 80, 90, 150), (110, 60, 210, 140), (230, 30, 280, 70)
    collector = collect([
        (1000.0, [tracked(entering)]),
        (1002.0, [tracked(crossing)]),
        # Parked: the same place and size for several windows is one picture.
        (1004.0, [tracked(leaving)]),
        (1006.0, [tracked(leaving)]),
        (1008.0, [tracked(leaving)]),
    ])
    retained = [{
        "camera_id": "gate", "captured_epoch": 1002.4,
        "objects": [{"box": dict(zip(("x1", "y1", "x2", "y2"), crossing)),
                     "detection_frame_width": WIDTH, "detection_frame_height": HEIGHT}],
    }]

    chosen = select_evidence_views(collector.views(), "gate", retained, MAX_GALLERY_IMAGES)

    assert [view.box for view in chosen] == [entering, leaving]
    assert [view.captured_at for view in chosen] == sorted(view.captured_at for view in chosen)


def test_selection_prefers_fully_framed_sharp_views_when_slots_are_short():
    clipped, clear, blurred = (0, 20, 150, 170), (160, 30, 260, 120), (40, 30, 150, 130)
    collector = TrackedEvidenceCollector()
    collector.consider(textured(1), 1000.0, [tracked(clipped)], WIDTH, HEIGHT, reference(1000.0))
    collector.consider(textured(2), 1003.0, [tracked(clear)], WIDTH, HEIGHT, reference(1003.0))
    collector.consider(textured(3, blur=15), 1006.0, [tracked(blurred)], WIDTH, HEIGHT, reference(1006.0))

    chosen = select_evidence_views(collector.views(), "gate", [], 1)

    assert [view.box for view in chosen] == [clear]


def test_tracked_views_fill_the_gallery_at_full_detail_without_touching_the_cover(tmp_path):
    store, event = seeded_store(tmp_path)
    before = store.scene_incident(event_id=event["id"])
    notifications = len(store.scene_pending_notifications())
    collector = collect([
        (1001.0, [tracked((10, 80, 90, 150), zones=["Driveway"])]),
        (1003.0, [tracked((110, 60, 210, 140))]),
        (1005.0, [tracked((230, 30, 280, 70))]),
    ])

    status, requested = save(store, event, collector.views(), tmp_path)

    assert status == {"evidence_images_result": "saved", "evidence_images_saved": 3,
                      "evidence_images_nominated": 3}
    assert [call[0] for call in requested] == [1001.0, 1003.0, 1005.0]
    assert all(call[1] == WIDTH and call[2].exact for call in requested)
    after = store.scene_incident(event_id=event["id"])
    images = after["evidence_images"]
    assert [image["captured_epoch"] for image in images] == [1001.0, 1003.0, 1005.0]
    first = images[0]
    assert (first["width"], first["height"]) == (WIDTH * 2, HEIGHT * 2)
    assert first["frame_timestamp_exact"]
    subject = first["objects"][0]
    # Boxes follow the retained image, not the smaller tracking frame.
    assert subject["box"] == {"x1": 20.0, "y1": 160.0, "x2": 180.0, "y2": 300.0}
    assert subject["detection_frame_width"] == WIDTH * 2
    assert subject["track_id"] == 4 and subject["zones"] == ["Driveway"]
    assert subject["evidence_source"] == "object_tracking"
    assert store.scene_evidence_image(first["id"])["snapshot_path"]
    for field in ("snapshot_path", "snapshot_url", "labels", "scene_objects", "alert_decisions",
                  "establishment", "summary", "objects"):
        assert after.get(field) == before.get(field), field
    assert len(store.scene_pending_notifications()) == notifications


def test_tracked_views_respect_the_incident_cap_and_existing_images(tmp_path):
    store, event = seeded_store(tmp_path)
    moments = [(1001.0 + 2 * index, [tracked((10 + 50 * index, 20, 60 + 50 * index, 90))])
               for index in range(6)]

    status, requested = save(store, event, collect(moments).views(), tmp_path)

    assert status["evidence_images_saved"] == MAX_GALLERY_IMAGES
    assert len(requested) == MAX_GALLERY_IMAGES
    assert tracking_capacity(store)(event["id"])["remaining"] == 0
    again, requested = save(store, event, collect(moments).views(), tmp_path)
    assert again["evidence_images_result"] == "gallery_full"
    assert requested == []


def test_tracked_views_do_not_repeat_the_cover_or_an_earlier_tracked_image(tmp_path):
    store, event = seeded_store(tmp_path)
    cover_box, later_box = (110, 60, 210, 140), (230, 30, 280, 70)
    cover = {"camera_id": "gate", "captured_epoch": 1000.0, "objects": [{
        "box": dict(zip(("x1", "y1", "x2", "y2"), cover_box)), "track_id": 4,
        "detection_frame_width": WIDTH, "detection_frame_height": HEIGHT}]}
    moments = [(1004.0, [tracked(cover_box)]), (1008.0, [tracked(later_box)])]

    status, requested = save(store, event, collect(moments).views(), tmp_path, known=[cover])

    assert status["evidence_images_saved"] == 1
    assert [call[0] for call in requested] == [1008.0]
    # A second completed session over the same footage adds nothing new.
    again, requested = save(store, event, collect(moments).views(), tmp_path, known=[cover])
    assert again["evidence_images_result"] == "no_distinct_view"
    assert requested == []


def test_unreadable_frames_and_a_stop_request_are_tolerated(tmp_path):
    store, event = seeded_store(tmp_path)
    moments = [(1001.0, [tracked((10, 80, 90, 150))]), (1004.0, [tracked((230, 30, 280, 70))])]

    def missing_first(captured_at):
        return None if captured_at == 1001.0 else np.full((HEIGHT, WIDTH, 3), 90, np.uint8)

    status, _requested = save(store, event, collect(moments).views(), tmp_path, frames=missing_first)
    assert status["evidence_images_saved"] == 1
    assert status["evidence_images_nominated"] == 2

    store, event = seeded_store(tmp_path / "stopped")
    status, requested = save(store, event, collect(moments).views(), tmp_path, stopped=lambda: True)
    assert status == {"evidence_images_result": "stopped", "evidence_images_saved": 0,
                      "evidence_images_nominated": 2}
    assert requested == []


def test_session_saves_evidence_only_when_a_gallery_store_is_wired(tmp_path):
    store, event = seeded_store(tmp_path)

    def session(**gallery):
        return ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(), detector=object(), frame_provider=lambda: None,
            update_event=lambda *_args: {}, publisher=None, limiter=threading.BoundedSemaphore(1),
            cover_frame_provider=lambda *_args: np.full((HEIGHT * 2, WIDTH * 2, 3), 90, np.uint8),
            snapshot_writer=snapshot_writer(tmp_path), **gallery,
        )

    def run(active):
        active._frame_width, active._frame_height = WIDTH, HEIGHT
        active._note_cover_for_evidence(1000.0, [tracked((110, 60, 210, 140))])
        collect([(1004.0, [tracked((110, 60, 210, 140))]),
                 (1008.0, [tracked((230, 30, 280, 70))])], collector=active._evidence)
        active._save_tracked_evidence(event["id"], threading.Event())
        return active._evidence_status

    assert run(session()) is None
    assert store.scene_incident(event_id=event["id"])["evidence_images"] == []

    status = run(session(evidence_capacity=tracking_capacity(store),
                         evidence_writer=store.retain_scene_evidence_images))
    assert status["evidence_images_saved"] == 1
    images = store.scene_incident(event_id=event["id"])["evidence_images"]
    assert [image["captured_epoch"] for image in images] == [1008.0]


def test_one_detection_pass_still_contributes_at_most_two_images():
    def observation(epoch, x):
        return dict(label="cat", confidence=0.52, captured_at_epoch=epoch, frame_source="recorded_main",
                    box=dict(x1=x, y1=100, x2=x + 80, y2=200), detection_frame_width=640,
                    detection_frame_height=360, temporal_consensus=True, activity_role="active")

    offsets = (0, 2, 4, 6, 9)
    samples = [_RecordedDetectionSample(offset=offset, frame=np.full((360, 640, 3), 60 + offset, np.uint8),
                                        objects=[], recording_path="recordings/animal.mp4",
                                        exact_timestamp=True) for offset in offsets]
    observations = [observation(1000 + offset, 40 + 110 * index)
                    for index, offset in enumerate(offsets[:-1])]

    batch = encode_gallery(samples, samples[-1], observations, camera_id="gate", event_epoch=1000,
                           request={"remaining": MAX_GALLERY_IMAGES}, timing={})
    try:
        assert len(batch.images) == MAX_DETECTION_IMAGES < MAX_GALLERY_IMAGES
    finally:
        batch.close()


def test_completed_recorded_session_nominates_and_saves_evidence(tmp_path):
    import time
    from types import SimpleNamespace

    from survng.app.video_frames import DecodedVideoFrame

    terminal = threading.Event()
    updates: list[dict] = []
    saved: list[tuple[float, tuple[int, int]]] = []
    event_epoch = time.time() - 12.0
    event_at = datetime.fromtimestamp(event_epoch, timezone.utc)

    class Detector:
        config = SimpleNamespace(confidence_threshold=0.7, require_incident_zone=False)

        def detect(self, frame, confidence_threshold=None):
            # The frame's mean encodes how far along the car is.
            step = int(frame[0, 0, 0])
            if step >= 12:
                return []
            x = 5 + step * 6
            return [{"label": "car", "confidence": 0.95, "incident_eligible": True,
                     "box": {"x1": x, "y1": 30, "x2": x + 22, "y2": 60}}]

    def catchup_provider(start_epoch, end_epoch, _sample_fps, _frame_width):
        captured_at = start_epoch
        while captured_at <= min(end_epoch, event_epoch + 9.0):
            step = max(0, int(round((captured_at - event_epoch) * 2)))
            yield DecodedVideoFrame(captured_at, np.full((100, 100, 3), step, np.uint8),
                                    reference(captured_at))
            captured_at += 0.5

    def update_event(_event_id, tracking, _tracked_objects):
        updates.append(tracking)
        if tracking.get("state") == "complete":
            terminal.set()
        return {}

    def writer(_event_id, images, _snapshot_writer):
        for image in images:
            saved.append((image.captured_epoch, (image.metadata["width"], image.metadata["height"])))
        return len(images)

    session = ObjectTrackingSession(
        camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        config=ObjectTrackingConfig(sample_fps=2.0, max_session_seconds=10.0, lost_timeout_seconds=1.0),
        detector=Detector(), frame_provider=lambda: None, catchup_frame_provider=catchup_provider,
        update_event=update_event, publisher=None, limiter=threading.BoundedSemaphore(1),
        cover_frame_provider=lambda *_args: np.full((200, 200, 3), 90, np.uint8),
        snapshot_writer=lambda _frame, _at: "snapshots/gate/evidence.jpg",
        evidence_capacity=lambda _event_id: {"remaining": MAX_GALLERY_IMAGES, "frames": []},
        evidence_writer=writer,
    )
    session.set_accepting(True)

    assert session.start(45, event_at, [{"label": "car", "confidence": 0.95, "incident_eligible": True,
                                         "box": {"x1": 5, "y1": 30, "x2": 27, "y2": 60}}],
                         np.zeros((100, 100, 3), dtype=np.uint8))
    assert terminal.wait(5.0)
    session.stop()

    final = updates[-1]
    assert final["state"] == "complete"
    assert final["evidence_images_result"] == "saved"
    assert 2 <= final["evidence_images_saved"] == len(saved) <= MAX_GALLERY_IMAGES
    # Distinct moments, each re-read at full detail.
    epochs = [epoch for epoch, _size in saved]
    assert epochs == sorted(epochs) and len(set(epochs)) == len(epochs)
    assert all(later - earlier >= 1.0 for earlier, later in zip(epochs, epochs[1:]))
    assert {size for _epoch, size in saved} == {(200, 200)}


def test_detection_cannot_take_the_slots_reserved_for_tracking(tmp_path):
    # Two events in one incident each run a detection pass before on-demand
    # tracking starts. Their images must not fill the whole gallery.
    from survng.app.evidence_gallery import gallery_image

    store, event = seeded_store(tmp_path)
    write = snapshot_writer(tmp_path)

    def detection_image(epoch, x):
        return gallery_image(np.full((HEIGHT, WIDTH, 3), 90, np.uint8), {
            "camera_id": "gate", "captured_epoch": epoch, "width": WIDTH, "height": HEIGHT,
            "frame_timestamp_exact": True, "recording_path": "recordings/gate.mp4", "role": "gallery",
            "objects": [{"label": "car", "confidence": 0.9, "box": {"x1": x, "y1": 20, "x2": x + 40, "y2": 60},
                         "detection_frame_width": WIDTH, "detection_frame_height": HEIGHT}]})

    first_pass = [detection_image(1000.5, 10), detection_image(1001.0, 60)]
    second_pass = [detection_image(1006.0, 110), detection_image(1006.5, 160)]
    assert store.retain_scene_evidence_images(event["id"], first_pass, write) == MAX_DETECTION_IMAGES
    assert store.scene_evidence_capacity(event["id"])["remaining"] == 0
    assert store.retain_scene_evidence_images(event["id"], second_pass, write) == 0
    assert tracking_capacity(store)(event["id"])["remaining"] == MAX_GALLERY_IMAGES - MAX_DETECTION_IMAGES

    moments = [(1010.0 + 2 * index, [tracked((200 + 30 * index, 80, 240 + 30 * index, 150))])
               for index in range(4)]
    status, _requested = save(store, event, collect(moments).views(), tmp_path)

    assert status["evidence_images_saved"] == MAX_GALLERY_IMAGES - MAX_DETECTION_IMAGES
    images = store.scene_incident(event_id=event["id"])["evidence_images"]
    assert len(images) == MAX_GALLERY_IMAGES
    assert sum(image["objects"][0].get("evidence_source") == "object_tracking" for image in images) == 2
    # Tracking images do not free up, or use up, detection's own allowance.
    assert store.scene_evidence_capacity(event["id"])["remaining"] == 0


def test_tracking_may_use_slots_detection_left_empty(tmp_path):
    store, event = seeded_store(tmp_path)
    moments = [(1001.0 + 2 * index, [tracked((10 + 50 * index, 20, 60 + 50 * index, 90))])
               for index in range(6)]

    status, _requested = save(store, event, collect(moments).views(), tmp_path)

    assert status["evidence_images_saved"] == MAX_GALLERY_IMAGES
    # Detection arriving afterwards finds the incident full.
    assert store.scene_evidence_capacity(event["id"])["remaining"] == 0


def test_detection_reservation_is_visible_before_the_images_are_written(tmp_path):
    store, event = seeded_store(tmp_path)
    capacity = store.scene_evidence_capacity(event["id"])
    reservation = store.reserve_scene_evidence_slots(
        event["id"], min(MAX_DETECTION_IMAGES, capacity["remaining"]),
    )
    assert reservation and len(reservation["ids"]) == MAX_DETECTION_IMAGES
    assert tracking_capacity(store)(event["id"])["remaining"] == MAX_GALLERY_IMAGES - MAX_DETECTION_IMAGES
    store.release_scene_evidence_reservation(reservation)
    assert tracking_capacity(store)(event["id"])["remaining"] == MAX_GALLERY_IMAGES
