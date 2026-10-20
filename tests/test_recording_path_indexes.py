from __future__ import annotations

from pathlib import Path

from survng.app.events import EventStore


def test_protected_recording_path_indexes_exist(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    with store._connect() as conn:
        indexes = {
            row["name"]
            for row in conn.execute("select name from sqlite_master where type='index'")
        }
    assert "idx_events_recording_retention" in indexes
    assert "scene_observation_recording" in indexes
    assert "acquired_observation_recording" in indexes
