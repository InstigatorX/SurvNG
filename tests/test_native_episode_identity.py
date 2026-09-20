"""Episode identity stitching from concurrent evidence and motion continuity."""
from survng.app.native_episode_identity import (
    annotate_tracks_with_episode_identities,
    best_objects_by_episode_identity,
    episode_label_counts,
    peak_concurrent_count,
    stitch_motion_identities,
)


def _person_track(track_id, samples, *, confidence=0.9):
    # samples: list[(epoch, x1, y1, x2, y2)]
    return {
        "label": "person",
        "track_id": track_id,
        "confidence": confidence,
        "max_confidence": confidence,
        "detection_frame_width": 896,
        "detection_frame_height": 512,
        "box_history": [list(sample) for sample in samples],
        "first_seen": "unused",
        "last_seen": "unused",
    }


def test_peak_concurrent_count_matches_two_person_scene():
    # Mirrors #73434: many sequential fragments, never more than two at once.
    tracks = [
        _person_track(400, [(100.0, 260, 140, 330, 250), (105.0, 262, 139, 334, 249)]),
        _person_track(401, [(100.0, 80, 130, 120, 180)]),  # co-occur with 400
        _person_track(406, [(112.0, 300, 105, 360, 270), (120.0, 306, 105, 359, 271)]),
        _person_track(415, [(130.0, 440, 98, 497, 213), (140.0, 446, 98, 497, 213)]),
        _person_track(419, [(145.0, 246, 120, 294, 192), (155.0, 250, 118, 290, 190)]),
    ]
    assert peak_concurrent_count(tracks, label="person") == 2
    assert episode_label_counts(tracks)["person"] == 2


def test_stitch_merges_sequential_fragments_but_not_co_occurring_people():
    tracks = [
        # Person A fragments (left/center), sequential
        _person_track(400, [(100.0, 260, 140, 330, 250), (108.0, 270, 140, 340, 255)], confidence=0.97),
        _person_track(415, [(112.0, 280, 135, 350, 250), (125.0, 290, 130, 360, 245)], confidence=0.95),
        # Person B fragments (right), sequential, co-occurs with A early
        _person_track(401, [(100.0, 80, 130, 120, 180), (107.0, 85, 128, 125, 185)], confidence=0.94),
        _person_track(419, [(110.0, 90, 120, 140, 190), (130.0, 95, 118, 145, 188)], confidence=0.96),
    ]
    identities = stitch_motion_identities(tracks, frame_width=896, frame_height=512)
    assert len(identities) == 2
    groups = {
        track_id: identity["episode_identity"]
        for identity in identities
        for track_id in identity["track_ids"]
    }
    # 400 and 401 co-occur at t=100 -> different identities
    assert groups[400] != groups[401]
    assert groups[400] == groups[415]
    assert groups[401] == groups[419]


def test_annotate_and_best_objects_collapse_fragments_for_semantic():
    tracks = [
        _person_track(400, [(100.0, 260, 140, 330, 250), (108.0, 270, 140, 340, 255)], confidence=0.9),
        _person_track(415, [(112.0, 280, 135, 350, 250)], confidence=0.98),
        _person_track(401, [(100.0, 80, 130, 120, 180)], confidence=0.92),
    ]
    annotated, identities, counts = annotate_tracks_with_episode_identities(
        tracks, frame_width=896, frame_height=512,
    )
    assert counts["person"] == 2
    assert len(identities) == 2
    objects = [
        {
            "label": "person",
            "track_id": 400,
            "confidence": 0.9,
            "snapshot_visible": True,
            "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        },
        {
            "label": "person",
            "track_id": 415,
            "confidence": 0.98,
            "snapshot_visible": False,
            "box": {"x1": 5, "y1": 6, "x2": 7, "y2": 8},
        },
        {
            "label": "person",
            "track_id": 401,
            "confidence": 0.92,
            "snapshot_visible": False,
            "box": {"x1": 9, "y1": 10, "x2": 11, "y2": 12},
        },
    ]
    # stamp identities from tracks
    by_tid = {t["track_id"]: t["episode_identity"] for t in annotated}
    for obj in objects:
        obj["episode_identity"] = by_tid[obj["track_id"]]
    best = best_objects_by_episode_identity(objects)
    assert len(best) == 2
    # visible bonus: identity containing 400 prefers visible 400 unless 415 wins on conf+bonus
    # 415 has 0.98 vs 400's 0.9+1.0 visible = 1.9, so 400 wins for that identity
    ids = {obj["track_id"] for obj in best}
    assert 401 in ids
    assert 400 in ids or 415 in ids
