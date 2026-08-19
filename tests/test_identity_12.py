from pathlib import Path

from survng.app.faces import FaceStore


def test_optimizer_missing_person_or_unready_recognizer(tmp_path: Path):
    store = FaceStore(tmp_path, start_recognition=False)
    try:
        try:
            store.optimize_person_gallery(999)
        except (ValueError, RuntimeError):
            pass
    finally:
        store.close()
