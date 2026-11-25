from survng.app import face_frame_search
from survng.app.face_frame_search import (
    EventResult,
    FrameResult,
    baseline_frames,
    passes,
    summarize,
)


def _face(side, *, recognisable=True):
    return {"side": side, "confidence": 0.95, "norm": 17.0 if recognisable else 10.0}


def _frames(fps=20, start=-4.0, end=12.0, faces=None, useful=None):
    faces = faces or {}
    useful = useful or {}
    count = int(round((end - start) * fps))
    frames = []
    for index in range(count + 1):
        offset = round(start + index / fps, 3)
        frames.append(FrameResult(offset, useful.get(offset, 0.0), faces.get(offset, []), [400.0]))
    return frames


def test_baseline_takes_plan_frames_and_extras_around_the_two_best_anchors():
    frames = _frames(useful={0.0: 5.0, 4.0: 9.0, 8.5: 1.0})
    chosen = baseline_frames(frames, [-0.5, 0.0, 0.5, 4.0, 4.5, 8.0, 8.5], max_extra=4)
    offsets = sorted(frame.offset for frame in chosen)
    # Anchors 4.0 then 0.0. Targets within 0.15 s of a plan sample are
    # skipped (-0.4, 4.4, 0.4); the first four left are 3.6, 3.2, -0.8, 4.8.
    assert offsets == [-0.8, -0.5, 0.0, 0.5, 3.2, 3.6, 4.0, 4.5, 4.8, 8.0, 8.5]


def test_baseline_without_useful_people_is_only_the_plan():
    chosen = baseline_frames(_frames(), [-0.5, 0.0, 0.5], max_extra=4)
    assert sorted(frame.offset for frame in chosen) == [-0.5, 0.0, 0.5]


def test_summary_separates_denser_search_wider_window_and_recognisable():
    faces = {
        2.0: [_face(52.0)],
        -3.0: [_face(70.0)],
        0.0: [_face(30.0)],
        5.0: [_face(300.0, recognisable=False)],
    }
    frames = _frames(faces=faces, useful={0.0: 1.0})
    result = EventResult(1, "gate", 0.0, (-4.0, 12.0), (-1.3, 9.3), frames=frames)
    result.baseline_offsets = [
        frame.offset for frame in baseline_frames(frames, [-0.5, 0.0, 0.5], max_extra=4)
    ]
    report = summarize([result], 48)
    event = report["cameras"]["gate"]["events"][0]
    assert event["any"]["baseline_best"] == 30.0
    assert event["any"]["all_best"] == 300.0
    # A large detection with a low norm (the back of a head) does not count.
    assert event["recognisable"]["plan_best"] == 52.0
    assert event["recognisable"]["all_best"] == 70.0
    assert event["recognisable"]["all_best_offset"] == -3.0
    assert report["cameras"]["gate"]["summary"]["recognisable"]["gained_all"] == 1


def test_recognisable_needs_the_quality_norm_and_failed_alignment_never_passes():
    assert passes({"norm": 13.0}, "recognisable")
    assert not passes({"norm": 12.9}, "recognisable")
    assert not passes({"norm": 0.0}, "recognisable")
    assert passes({"norm": 0.0}, "any")


def test_showinfo_frame_index_and_time_are_read_together():
    line = (
        b"[Parsed_showinfo_1 @ 0x55] n:  17 pts:  77400 pts_time:0.86 "
        b"duration:4500 fmt:yuv420p"
    )
    match = face_frame_search._SHOWINFO_FRAME.search(line)
    assert match is not None
    assert int(match.group(1)) == 17
    assert float(match.group(2)) == 0.86
