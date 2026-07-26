from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from survng.app import main
from fastapi import HTTPException


class EventApiSerializationTest(unittest.TestCase):
    def test_incident_search_rejects_unsafe_timezone_paths(self) -> None:
        with self.assertRaises(HTTPException) as invalid:
            main.incident_search(time_zone="../../etc/passwd")

        self.assertEqual(invalid.exception.status_code, 422)

    def test_event_row_tolerates_malformed_legacy_object_entries(self) -> None:
        row = main._event_row({
            "id": 1,
            "objects_json": json.dumps([
                "legacy",
                {"label": "person", "confidence": "invalid", "zones": "not-a-list"},
                {"label": "car", "confidence": 0.8, "zones": ["driveway"]},
            ]),
        })

        self.assertEqual(row["labels"], ["car"])
        self.assertEqual(row["zones"], ["driveway"])
        self.assertEqual(len(row["objects"]), 2)

        selected = main._best_incident_event([row])
        self.assertEqual(selected["id"], 1)

    def test_motion_audit_snapshot_status_is_confined_to_storage(self) -> None:
        with tempfile.TemporaryDirectory() as storage, tempfile.TemporaryDirectory() as outside:
            outside_image = Path(outside) / "private.jpg"
            outside_image.write_bytes(b"image")

            row = main._motion_audit_row(
                {
                    "id": 1,
                    "features_json": "[]",
                    "snapshot_path": str(outside_image),
                    "object_detected": 0,
                },
                Path(storage),
            )

        self.assertEqual(row["features"], {})
        self.assertFalse(row["has_snapshot"])
        self.assertFalse(row["object_detected"])


if __name__ == "__main__":
    unittest.main()
