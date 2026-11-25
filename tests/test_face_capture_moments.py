from survng.app.face_capture.moments import Track, choose_moments, link_tracks, merge_windows


def _box(x, height):
    return (x, 100.0, x + 50.0, 100.0 + height)


def test_boxes_that_overlap_across_samples_form_one_track():
    samples = [(0.0, [_box(0, 200), _box(500, 100)]), (0.2, [_box(5, 210), _box(505, 100)]), (0.4, [_box(10, 220)])]
    tracks = link_tracks(samples)
    assert sorted(len(t.samples) for t in tracks) == [2, 3]


def test_a_gap_longer_than_a_second_starts_a_new_track():
    tracks = link_tracks([(0.0, [_box(0, 200)]), (1.6, [_box(0, 200)])])
    assert len(tracks) == 2


def test_moments_are_each_tracks_tallest_samples_spaced_apart():
    near = Track([(t / 10, _box(0, 100 + t)) for t in range(30)])  # tallest at the end
    far = Track([(5.0, _box(600, 80))])
    moments = choose_moments([near, far], per_track=2, spacing=0.5)
    assert moments == [2.4, 2.9, 5.0]


def test_overlapping_moment_windows_are_decoded_once():
    assert merge_windows([1.0, 1.6, 4.0], 0.5) == [(0.5, 2.1), (3.5, 4.5)]


def test_a_seen_face_beats_a_taller_person_and_faceless_tracks_fall_back_to_height():
    tall_no_face = (0.0, _box(0, 500))
    shorter_with_face = (2.0, _box(0, 300))
    bigger_face = (3.0, _box(0, 250))
    near = Track([tall_no_face, shorter_with_face, bigger_face])
    other = Track([(6.0, _box(600, 100)), (7.0, _box(600, 120))])  # no face anywhere
    faces = {shorter_with_face: 30.0, bigger_face: 45.0}
    moments = choose_moments([near, other], per_track=2, spacing=0.5, face_sizes=faces)
    # The near track uses its two face sightings, not its tallest sample;
    # the faceless track falls back to its tallest samples.
    assert moments == [2.0, 3.0, 6.0, 7.0]
