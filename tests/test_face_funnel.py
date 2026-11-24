import json
import sqlite3

import numpy as np

from survng.app import face_funnel
from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector


def _box(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _person(height, consensus=True):
    return {"label": "person", "temporal_consensus": consensus, "box": _box(0, 0, 200, height)}


def _phases(**values):
    return {"status": "motion_qualification",
            "motion_qualification": {"object_detection_phases_ms": values}}


def _database(tmp_path, events, observations=()):
    path = tmp_path / "survng.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        "create table events (id integer primary key, camera_id text, objects_json text, created_at text);"
        "create table face_observations (event_id integer, recognition_outcome text, review_status text,"
        " person_id integer, candidate_person_id integer, observed_at text);"
    )
    connection.executemany("insert into events values (?, ?, ?, ?)", [
        (event_id, camera, json.dumps(objects), created) for event_id, camera, objects, created in events
    ])
    connection.executemany("insert into face_observations values (?, ?, ?, ?, ?, ?)", observations)
    connection.commit()
    return connection


def test_funnel_counts_each_stage_per_camera(tmp_path):
    face = {"label": "face", "box": _box(10, 10, 70, 90)}
    connection = _database(tmp_path, [
        (1, "door", [_person(900), face], "2026-11-20T00:00:00+00:00"),
        (2, "door", [_person(900), _phases(face_search_people=2, face_search_crops=2,
                                           face_search_crops_empty=2, face_search_faces=0,
                                           face_evidence_deadline_reached=1)],
         "2026-11-20T00:00:01+00:00"),
        (3, "door", [_person(200)], "2026-11-20T00:00:02+00:00"),
        (4, "door", [_person(900, consensus=False)], "2026-11-20T00:00:03+00:00"),
        (5, "door", [_person(900)], "2026-11-01T00:00:00+00:00"),
        (6, "yard", [_person(900)], "2026-11-20T00:00:04+00:00"),
    ], [
        (1, "embedded", "confirmed", 7, 7, "2026-11-20T00:00:00+00:00"),
        (1, "too_small", "unknown", None, None, "2026-11-20T00:00:00+00:00"),
        (6, "failed", "unknown", None, None, "2026-11-20T00:00:04+00:00"),
        (5, "embedded", "confirmed", 7, 7, "2026-11-01T00:00:00+00:00"),
    ])
    report = face_funnel.build_report(connection, since="2026-11-10T00:00:00+00:00")
    door, yard = report["cameras"]["door"], report["cameras"]["yard"]

    assert [door[key] for key, _ in face_funnel.STAGES] == [3, 2, 1, 1, 1, 1, 1]
    assert door["outcomes"] == {"embedded": 1, "too_small": 1}
    assert door["face_min_side_px"] == {"count": 1, "median": 60.0, "p90": 60.0, "below_min_size": 0}
    assert door["search_events"] == 1 and door["search_found_nothing"] == 1
    assert door["search"]["crops_empty"] == 2 and door["deadline_reached"] == 1
    # A face stored from an extra frame counts as found without a cover-frame object.
    assert [yard[key] for key, _ in face_funnel.STAGES] == [1, 1, 1, 1, 0, 0, 0]
    text = face_funnel.format_report(report)
    assert "Camera door" in text and "1 events searched and found nothing" in text


def test_cli_reads_the_database_without_writing(tmp_path, capsys):
    _database(tmp_path, []).close()
    before = (tmp_path / "survng.sqlite3").read_bytes()
    assert face_funnel.main([str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cameras"] == {}
    assert (tmp_path / "survng.sqlite3").read_bytes() == before


def test_people_search_records_funnel_counters():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    people = [
        {"label": "person", "confidence": .9, "box": _box(0, 0, 200, 300)},
        {"label": "person", "confidence": .8, "box": _box(200, 0, 400, 300)},
        {"label": "person", "confidence": .7, "box": _box(0, 0, 10, 10)},
        {"label": "person", "confidence": .1, "box": _box(0, 0, 200, 300)},
    ]
    answers = iter([[{"label": "face", "confidence": .9, "box": _box(5, 5, 60, 60)}], []])
    timing = {}
    faces = RecordedMotionObjectDetector._detect_faces_in_people(
        frame, people, lambda _crop: next(answers), max_people=3, timing=timing,
    )
    assert len(faces) == 1
    assert timing == {
        "face_search_people_over_cap": 1, "face_search_people": 3,
        "face_search_windows_too_small": 1, "face_search_crops": 2,
        "face_search_faces": 1, "face_search_crops_empty": 1,
    }
