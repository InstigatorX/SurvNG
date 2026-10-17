import sqlite3

import pytest

from survng.app.event_store import EventStore
from survng.app.face_detection_dataset import FaceDetectionDataset
from survng.app.media_exports import MediaExportStore
from survng.app.recording_process.index import RecordingIndexMixin
from survng.app.telemetry_store import TelemetryStore


class _RecordingIndex(RecordingIndexMixin):
    def __init__(self, index_path):
        self.index_path = index_path


def test_short_lived_database_factories_close_on_context_exit(tmp_path) -> None:
    event_store = EventStore(tmp_path)
    telemetry_store = TelemetryStore(tmp_path)
    export_store = MediaExportStore(tmp_path)
    recording_index = _RecordingIndex(tmp_path / "recording-index.sqlite3")
    face_dataset = FaceDetectionDataset.__new__(FaceDetectionDataset)
    face_dataset.database_path = tmp_path / "face-dataset.sqlite3"

    factories = {
        "events": event_store._connect,
        "jobs": event_store._connect_jobs,
        "telemetry": telemetry_store._connect,
        "recording index": recording_index._index_connection,
        "media exports": export_store._connect,
        "face dataset": face_dataset._connect,
    }
    for factory in factories.values():
        connection = factory()
        with connection:
            assert connection.execute("select 1").fetchone()[0] == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("select 1")
