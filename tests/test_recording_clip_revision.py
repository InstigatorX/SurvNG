from types import SimpleNamespace
from pathlib import Path
from fastapi import HTTPException

from survng.app.recording_media_runtime import RecordingMediaRuntime


def test_growing_clip_uses_new_frozen_recording_revision(tmp_path):
    rows = [{"path": "first.mp4", "start_epoch": 120, "end_epoch": 150,
             "duration_seconds": 30, "size_bytes": 50, "modified_at": 1}]
    recorder = SimpleNamespace(recording_rows_between=lambda *args, **kwargs: rows)
    manager = SimpleNamespace(recorder=recorder)
    runtime = RecordingMediaRuntime(SimpleNamespace(get_manager=lambda: manager))
    runtime._event_clip_path = lambda *args, **kwargs: tmp_path / "clip.mp4"
    offline = set()
    def storage_path(value, **kwargs):
        if value in offline:
            raise HTTPException(status_code=404, detail="offline")
        return Path(value)
    runtime._recording_storage_path = storage_path
    inputs = []
    def build(event, **kwargs):
        inputs.append(event["_recording_rows"])
        kwargs["output_path"].write_bytes(b"generated")
    runtime._build_event_clip = build
    event = {"id": 120, "camera_id": "gate", "created_at": "1970-01-01T00:02:00+00:00"}
    first = runtime._ensure_event_clip(event, before=0, after=120)
    assert runtime._ensure_event_clip(event, before=0, after=120) == first
    assert len(inputs) == 1
    rows.append({"path": "second.mp4", "start_epoch": 150, "end_epoch": 240,
                 "duration_seconds": 90, "size_bytes": 100, "modified_at": 2})
    second = runtime._ensure_event_clip(event, before=0, after=120)
    assert first != second
    assert len(inputs[0]) == 1 and len(inputs[1]) == 2
    assert runtime._ensure_event_clip(event, before=0, after=120) == second
    assert len(inputs) == 2
    offline.add("second.mp4")
    assert runtime._ensure_event_clip(event, before=0, after=120) == first
    offline.clear()
    assert runtime._ensure_event_clip(event, before=0, after=120) == second
