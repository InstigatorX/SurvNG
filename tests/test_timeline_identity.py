from pathlib import Path
from types import SimpleNamespace

from survng.app.recording_routes import RecordingRouteDependencies, create_recording_router


class _Recorder:
    def recording_availability_between(self, *_args, **_kwargs):
        return {
            "ranges": [{"start_epoch": 100.0, "end_epoch": 200.0}],
            "segment_count": 1,
        }


class _Events:
    def for_camera_range(self, *_args, **_kwargs):
        return [{
            "id": 7,
            "camera_id": "gate",
            "kind": "object",
            "created_at": "1970-01-01T00:02:30+00:00",
            "snapshot_path": "snapshot.jpg",
            "recording_path": "",
            "objects_json": (
                '[{"label":"person","confidence":0.93,'
                '"incident_eligible":true}]'
            ),
        }]


class _Faces:
    def for_event_ids(self, event_ids):
        assert event_ids == [7]
        return [{
            "observation_id": 91,
            "event_id": 7,
            "person_id": 3,
            "candidate_person_id": None,
            "person_name": "Steve",
            "candidate_person_name": None,
            "match_confidence": 1.0,
            "candidate_confidence": None,
            "consensus": {"candidate_count": 3, "agreement_count": 3},
        }]


class _Manager:
    def __init__(self):
        self.recorder = _Recorder()
        self.events = _Events()
        self.faces = _Faces()
        self.config = SimpleNamespace(cameras=[])

    def camera(self, camera_id):
        return object() if camera_id == "gate" else None


def _dependencies(manager):
    return RecordingRouteDependencies(
        get_manager=lambda: manager,
        get_config=lambda: SimpleNamespace(cameras=[]),
        get_media_exports=lambda: SimpleNamespace(),
        public_url=lambda value: value,
        recording_rows=lambda *_args, **_kwargs: [],
        recording_day_rows=lambda *_args, **_kwargs: [],
        recording_preview_path=lambda *_args, **_kwargs: Path("preview.jpg"),
        recording_preview_timestamp=lambda _path: (None, "requested_offset"),
        recording_day_fmp4_paths=lambda *_args, **_kwargs: (
            Path("init.mp4"), Path("media.m4s")
        ),
        recording_file_response=lambda *_args, **_kwargs: None,
        event_clip_window=lambda *_args: (5.0, 5.0),
        ensure_event_clip=lambda *_args, **_kwargs: Path("clip.mp4"),
    )


def test_recording_day_incident_preserves_projected_identity():
    manager = _Manager()
    bundle = create_recording_router(_dependencies(manager))

    payload = bundle.handlers["recording_day"](
        "gate",
        100.0,
        200.0,
        "main",
    )

    assert len(payload["incidents"]) == 1
    incident = payload["incidents"][0]
    assert incident["identities"][0]["name"] == "Steve"
    assert incident["identities"][0]["status"] == "confirmed"
    assert incident["primary_identity"]["name"] == "Steve"
    assert "faces" not in incident
