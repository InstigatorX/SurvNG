import json
from datetime import datetime, timezone

import numpy as np

from survng.app.face_capture.binding import bind_faces
from survng.app.face_capture.groups import FaceGroups
from survng.app.face_capture.pipeline import CapturedFace, CaptureResult
from survng.app.face_capture.store import FaceCaptureStore

T = 1_800_000_000.0


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _person(object_id, *observations):
    return {"id": object_id, "label": "person", "observations": [
        {"event_id": event, "camera_id": camera, "captured_at": _iso(epoch),
         "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
         "detection_frame_width": 1280, "detection_frame_height": 720}
        for event, camera, epoch, box in observations]}


def _face(face_id, event, camera, epoch, box, frame=(3840, 2160)):
    return {"id": face_id, "event_id": event, "camera_id": camera, "captured_epoch": epoch,
            "box_json": json.dumps(box), "frame_width": frame[0] if frame else None,
            "frame_height": frame[1] if frame else None}


def test_a_face_binds_to_the_person_box_around_it_after_scaling():
    # Two people side by side in the detector's 1280x720 frame; faces in 4K.
    left = _person("left", (1, "gate", T, (100, 100, 300, 700)))
    right = _person("right", (1, "gate", T, (700, 100, 900, 700)))
    faces = [_face(1, 1, "gate", T + 0.2, [450, 400, 560, 520]),     # inside left once scaled by 3
             _face(2, 1, "gate", T + 0.2, [2250, 400, 2360, 520])]   # inside right
    assert bind_faces(faces, [left, right], lambda camera: None) == {1: "left", 2: "right"}


def test_no_bind_across_cameras_too_far_in_time_or_outside_every_box():
    # Two people on the gate, so "the only person on the camera" does not apply.
    left = _person("left", (1, "gate", T, (100, 100, 300, 700)))
    right = _person("right", (1, "gate", T, (700, 100, 900, 700)))
    faces = [_face(1, 1, "foyer", T, [450, 400, 560, 520]),          # another camera
             _face(2, 1, "gate", T + 3.0, [450, 400, 560, 520]),      # beyond 2 s, nothing to interpolate
             _face(3, 1, "gate", T, [3000, 400, 3100, 520])]          # outside both boxes
    assert bind_faces(faces, [left, right], lambda camera: None) == {}


def test_a_moving_person_box_is_interpolated_between_sparse_observations():
    walker = _person("walker", (1, "gate", T, (100, 100, 300, 700)), (1, "gate", T + 3.0, (700, 100, 900, 700)))
    still = _person("still", (1, "gate", T, (1000, 100, 1200, 700)))
    # Halfway, the walker is around x 400-600 (1200-1800 in 4K).
    face = _face(1, 1, "gate", T + 1.5, [1450, 400, 1560, 520])
    assert bind_faces([face], [walker, still], lambda camera: None) == {1: "walker"}


def test_same_event_is_preferred_and_the_only_person_on_a_camera_takes_an_unplaced_face():
    near_other_event = _person("other", (2, "gate", T + 0.05, (100, 100, 300, 700)))
    same_event = _person("same", (1, "gate", T + 0.4, (100, 100, 300, 700)))
    face = _face(1, 1, "gate", T, [450, 400, 560, 520], frame=None)
    assert bind_faces([face], [near_other_event, same_event], lambda camera: (3840, 2160)) == {1: "same"}
    # No frame size known (cannot place the box) but only one person tracked: theirs.
    assert bind_faces([face], [same_event], lambda camera: None) == {1: "same"}
    # Two people and no way to place the box: unbound.
    assert bind_faces([face], [near_other_event, same_event], lambda camera: None) == {}


def test_an_incident_lists_which_tracked_person_each_group_is(tmp_path):
    store = FaceCaptureStore(tmp_path)
    groups = FaceGroups(store)
    face = CapturedFace(0, 0.2, (450, 400, 560, 520), [], 0.9, 17.0,
                        np.ones(512, np.float32) / np.sqrt(512), np.zeros((20, 20, 3), np.uint8), frame_size=(3840, 2160))
    store.record(event_id=1, camera_id="gate", event_epoch=T, result=CaptureResult(faces=[face]),
                 embedding_model="m", seconds=1.0)
    groups.assign(1)
    incident = {"event_ids": [1], "scene_objects": [_person("obj-1", (1, "gate", T + 0.1, (100, 100, 300, 700))),
                                                     {"id": "car", "label": "car", "observations": []}]}
    found = groups.for_incident(incident, lambda camera: None)
    [group] = found["groups"]
    assert found["objects"] == {"obj-1": [group["id"]]}
    assert groups.for_incident({"event_ids": [], "scene_objects": []}, lambda camera: None)["objects"] == {}


def test_the_incident_route_binds_through_the_event_store(tmp_path):
    import pytest
    from types import SimpleNamespace

    from fastapi import HTTPException

    from survng.app.face_capture.routes import FaceCaptureRouteDependencies, create_face_capture_router
    from survng.app.security import required_api_scope

    store = FaceCaptureStore(tmp_path)
    groups = FaceGroups(store)
    face = CapturedFace(0, 0.2, (450, 400, 560, 520), [], 0.9, 17.0,
                        np.ones(512, np.float32) / np.sqrt(512), np.zeros((20, 20, 3), np.uint8), frame_size=(3840, 2160))
    store.record(event_id=1, camera_id="gate", event_epoch=T, result=CaptureResult(faces=[face]),
                 embedding_model="m", seconds=1.0)
    groups.assign(1)
    incidents = {"incident-a": {"event_ids": [1], "scene_objects": [_person("obj-1", (1, "gate", T, (100, 100, 300, 700)))]}}
    manager = SimpleNamespace(
        face_capture=SimpleNamespace(store=store, groups=groups, enabled=lambda: True),
        config=SimpleNamespace(cameras=[]),
        events=SimpleNamespace(scene_incident=lambda incident_id: incidents.get(incident_id)),
        recorder=SimpleNamespace(recording_rows_between=lambda *a, **k: []),
    )
    router = create_face_capture_router(FaceCaptureRouteDependencies(get_manager=lambda: manager))
    endpoint = next(r.endpoint for r in router.routes if r.path == "/api/face-captures/by-incident/{incident_id}")
    assert list(endpoint("incident-a")["objects"]) == ["obj-1"]
    with pytest.raises(HTTPException) as error:
        endpoint("incident-missing")
    assert error.value.status_code == 404
    assert required_api_scope("GET", "/api/face-captures/by-incident/incident-a") == "admin"


def test_a_person_box_without_its_detector_frame_size_is_not_placed():
    # Read as main-stream pixels, this box would hold the 4K face.
    unsized = {"id": "unsized", "label": "person", "observations": [
        {"event_id": 1, "camera_id": "gate", "captured_at": _iso(T),
         "box": {"x1": 1400, "y1": 300, "x2": 1700, "y2": 700}}]}
    other = _person("other", (1, "gate", T, (100, 100, 300, 700)))
    face = _face(1, 1, "gate", T, [1450, 400, 1560, 520])
    assert bind_faces([face], [unsized, other], lambda camera: None) == {}


def test_an_incident_past_one_lookup_of_events_still_binds_its_later_faces(tmp_path):
    store = FaceCaptureStore(tmp_path)
    groups = FaceGroups(store)
    face = CapturedFace(0, 0.2, (450, 400, 560, 520), [], 0.9, 17.0,
                        np.ones(512, np.float32) / np.sqrt(512), np.zeros((20, 20, 3), np.uint8), frame_size=(3840, 2160))
    store.record(event_id=650, camera_id="gate", event_epoch=T, result=CaptureResult(faces=[face]),
                 embedding_model="m", seconds=1.0)
    groups.assign(650)
    incident = {"event_ids": list(range(1, 701)),
                "scene_objects": [_person("obj-1", (650, "gate", T + 0.1, (100, 100, 300, 700)))]}
    found = groups.for_incident(incident, lambda camera: None)
    [group] = found["groups"]
    assert found["events"] == {"650": [group["id"]]}
    assert found["objects"] == {"obj-1": [group["id"]]}


def test_a_half_finished_frame_size_upgrade_is_completed(tmp_path):
    import sqlite3

    FaceCaptureStore(tmp_path)
    with sqlite3.connect(tmp_path / "face-captures.sqlite3") as connection:
        connection.execute("alter table face_captures drop column frame_height")
    FaceCaptureStore(tmp_path)
    with sqlite3.connect(tmp_path / "face-captures.sqlite3") as connection:
        columns = {row[1] for row in connection.execute("pragma table_info(face_captures)")}
    assert {"frame_width", "frame_height"} <= columns


def test_a_face_placed_outside_the_only_person_box_stays_unbound():
    only = _person("only", (1, "gate", T, (100, 100, 300, 700)))
    beside = _face(1, 1, "gate", T, [3000, 400, 3100, 520])         # placed, outside the box
    unplaceable = _face(2, 1, "gate", T + 10.0, [450, 400, 560, 520])  # no observation near its moment
    assert bind_faces([beside, unplaceable], [only], lambda camera: None) == {2: "only"}
