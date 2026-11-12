"""Regression tests for expected-volume checks before index deletion."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from survng.app.recording_process.recorder import Recorder
from survng.app.volume_identity import (
    VolumeGate,
    VolumeIdentityStore,
    WITNESS_NAME,
    before_final_witness,
)


def _row(path: Path, start: float) -> dict:
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": 8,
        "modified_at": start,
        "start_epoch": start,
        "duration_seconds": 10,
        "end_epoch": start + 10,
        "source": "main",
        "playable": 1,
        "health_error": "",
        "validated": 1,
        "stream_fingerprint": "",
        "fingerprint_checked": 1,
        "playback_duration_seconds": 10,
        "location_id": "default",
    }


class VolumeIdentityTests(unittest.TestCase):
    def test_established_identity_preserves_rows_when_the_mountpoint_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "index"
            media = root / "media"
            media.mkdir()
            clip = media / "clip.mp4"
            clip.write_bytes(b"recording")
            recorder = Recorder("ffmpeg", media, segment_seconds=10, index_dir=index)
            start = time.time() - 30
            recorder._store_recording_rows("gate", "main", [_row(clip, start)])
            store = VolumeIdentityStore(index / "volume-identity.sqlite3")
            with patch("survng.app.volume_identity._mounted", return_value=True):
                store.note_mounted(media)
            clip.unlink()
            (media / WITNESS_NAME).unlink()
            recorder._volume_gate_instance = VolumeGate(store, fallback_root=media)

            result = recorder.recording_at("gate", start + 1)
            with recorder._index_connection() as connection:
                remaining = connection.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
            self.assertIsNone(result)
            self.assertEqual(remaining, 1)
            self.assertTrue(store.get(media).token)

    def test_replacement_disk_is_not_enrolled_by_a_mounted_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"
            media.mkdir()
            store = VolumeIdentityStore(root / "volume-identity.sqlite3")
            with patch("survng.app.volume_identity._mounted", return_value=True):
                store.note_mounted(media)
            original = store.get(media).token
            (media / WITNESS_NAME).write_text("replacement-token")
            with patch("survng.app.volume_identity._mounted", return_value=True):
                store.note_mounted(media)
            gate = VolumeGate(store, fallback_root=media)
            clip = media / "clip.mp4"
            self.assertEqual(store.get(media).token, original)
            self.assertEqual(gate.verdict(clip), "unavailable")
            self.assertTrue(store.reconcile(media, require_mount=False))
            self.assertEqual(store.get(media).token, original)
            self.assertEqual((media / WITNESS_NAME).read_text(), original)
            self.assertEqual(gate.verdict(clip), "confirmed_absent")

    def test_witness_mismatch_between_stat_and_final_read_preserves_the_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"
            media.mkdir()
            clip = media / "clip.mp4"
            clip.write_bytes(b"recording")
            store = VolumeIdentityStore(root / "volume-identity.sqlite3")
            with patch("survng.app.volume_identity._mounted", return_value=True):
                store.note_mounted(media)
            clip.unlink()
            recorder = Recorder("ffmpeg", media, segment_seconds=10, index_dir=root / "index")
            start = time.time() - 30
            recorder._store_recording_rows("gate", "main", [_row(clip, start)])
            recorder._volume_gate_instance = VolumeGate(store, fallback_root=media)

            def drop_witness(_dir_fd: int) -> None:
                os.remove(media / WITNESS_NAME)

            token = before_final_witness.set(drop_witness)
            try:
                result = recorder.recording_at("gate", start + 1)
            finally:
                before_final_witness.reset(token)
            with recorder._index_connection() as connection:
                remaining = connection.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]

        self.assertIsNone(result)
        self.assertEqual(remaining, 1)

    def test_genuine_deletion_on_a_verified_volume_prunes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"
            media.mkdir()
            clip = media / "clip.mp4"
            clip.write_bytes(b"recording")
            store = VolumeIdentityStore(root / "volume-identity.sqlite3")
            with patch("survng.app.volume_identity._mounted", return_value=True):
                store.note_mounted(media)
            clip.unlink()
            recorder = Recorder("ffmpeg", media, segment_seconds=10, index_dir=root / "index")
            start = time.time() - 30
            recorder._store_recording_rows("gate", "main", [_row(clip, start)])
            recorder._volume_gate_instance = VolumeGate(store, fallback_root=media)

            self.assertIsNone(recorder.recording_at("gate", start + 1))
            with recorder._index_connection() as connection:
                remaining = connection.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]

        self.assertEqual(remaining, 0)

    def test_volume_return_serves_the_indexed_file_without_reindex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"
            media.mkdir()
            clip = media / "clip.mp4"
            clip.write_bytes(b"recording")
            store = VolumeIdentityStore(root / "volume-identity.sqlite3")
            with patch("survng.app.volume_identity._mounted", return_value=True):
                store.note_mounted(media)
            witness = (media / WITNESS_NAME).read_text()
            (media / WITNESS_NAME).unlink()
            recorder = Recorder("ffmpeg", media, segment_seconds=10, index_dir=root / "index")
            start = time.time() - 30
            recorder._store_recording_rows("gate", "main", [_row(clip, start)])
            recorder._volume_gate_instance = VolumeGate(store, fallback_root=media)
            self.assertIsNone(recorder.recording_at("gate", start + 1))
            (media / WITNESS_NAME).write_text(witness)
            found = recorder.recording_at("gate", start + 1)

        self.assertEqual(found["path"], str(clip))

    def test_identity_database_stays_readable_when_the_media_root_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"
            media.mkdir()
            store = VolumeIdentityStore(root / "state" / "volume-identity.sqlite3")
            with patch("survng.app.volume_identity._mounted", return_value=True):
                store.note_mounted(media)
            token = store.get(media).token
            for child in media.iterdir():
                child.unlink()
            media.rmdir()
            self.assertEqual(store.get(media).token, token)

    def test_unknown_stat_preserves_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp)
            clip = media / "clip.mp4"
            clip.write_bytes(b"recording")
            recorder = Recorder("ffmpeg", media, segment_seconds=10)
            start = time.time() - 30
            recorder._store_recording_rows("gate", "main", [_row(clip, start)])
            with patch("survng.app.recording_process.index.path_presence", return_value="unknown"):
                self.assertIsNone(recorder.recording_at("gate", start + 1))
                removed = recorder._prune_missing_index_rows()
            with recorder._index_connection() as connection:
                remaining = connection.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
        self.assertEqual(removed, 0)
        self.assertEqual(remaining, 1)


if __name__ == "__main__":
    unittest.main()
