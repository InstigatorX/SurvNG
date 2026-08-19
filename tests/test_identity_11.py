from pathlib import Path
from survng.app.faces import FaceStore


def test_gallery_enrich_missing_person(tmp_path: Path):
    store = FaceStore(tmp_path, start_recognition=False)
    try:
        try:
            store.enrich_person_gallery(999, target_count=8)
        except ValueError as exc:
            assert "person not found" in str(exc)
        else:
            raise AssertionError("expected ValueError")
    finally:
        store.close()
