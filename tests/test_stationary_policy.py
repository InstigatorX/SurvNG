from survng.app.object_activity import ObjectActivityAttributor
from survng.app.stationary_policy import stationary_object_policy


def test_stationary_presets_coordinate_motion_and_scene_thresholds() -> None:
    light = stationary_object_policy("low")
    standard = stationary_object_policy("balanced")
    strong = stationary_object_policy("high")

    assert light.displacement_ratio < standard.displacement_ratio < strong.displacement_ratio
    assert light.background_learning_seconds > standard.background_learning_seconds > strong.background_learning_seconds
    assert (
        light.scene_stable_displacement_ratio
        < standard.scene_stable_displacement_ratio
        < strong.scene_stable_displacement_ratio
    )


def test_object_activity_reports_the_effective_shared_policy() -> None:
    status = ObjectActivityAttributor("enforce", stationary_tolerance="high").status()

    assert status["stationary_policy"]["name"] == "high"
    assert status["stationary_policy"]["background_learning_seconds"] == 5.0


def _sightings(tolerance, displacements, *, box=None):
    from tests.test_object_activity import observation

    attributor = ObjectActivityAttributor("enforce", stationary_tolerance=tolerance)
    values = {} if box is None else {"box": box}
    return [
        attributor.admit(
            [observation(
                temporal_center_displacement_ratio=displacement,
                temporal_center_path_ratio=displacement,
                **values,
            )],
            {},
            event_key=f"event-{index}",
            observed_at_epoch=1000.0 + index * 60.0,
        )[0].attribution
        for index, displacement in enumerate(displacements)
    ]


def test_stable_limits_scale_with_box_size():
    # A 200 px box in a 1000 px frame needs about 0.011 displacement to be
    # credible movement. Holding to within 0.004 is stillness for that box,
    # though it is above the former fixed 0.0025 limit.
    roles = [item.role.value for item in _sightings("balanced", [0.004] * 3)]

    assert roles == ["indeterminate", "indeterminate", "scene_context"]


def test_a_sighting_between_stable_and_credible_movement_stays_indeterminate():
    roles = [item.role.value for item in _sightings("balanced", [0.008] * 3)]

    assert roles == ["indeterminate"] * 3


def test_a_sighting_classed_active_is_never_remembered_as_stable():
    small = {"x1": 100, "y1": 100, "x2": 140, "y2": 140}
    results = _sightings("high", [0.0035, 0.0035, 0.0005], box=small)

    assert [item.role.value for item in results] == ["active", "active", "indeterminate"]
    assert results[2].evidence.scene_context_memory_sightings == 0


def test_stable_limits_stay_below_credible_movement_for_every_box_size():
    from survng.app.object_motion import temporal_object_motion_evidence
    from tests.test_object_activity import observation

    policies = [stationary_object_policy(name) for name in ("low", "balanced", "high")]
    for side in (10, 40, 100, 200, 400, 900):
        motion = temporal_object_motion_evidence(
            observation(box={"x1": 0, "y1": 0, "x2": side, "y2": side})
        )
        limits = [policy.scene_stable_limits(motion) for policy in policies]
        for displacement, path in limits:
            assert displacement <= 0.8 * motion.movement_threshold
            assert path <= 0.8 * motion.path_threshold
        for policy, (displacement, path) in zip(policies, limits):
            assert displacement >= min(
                policy.scene_stable_displacement_ratio,
                policy.scene_stable_margin * motion.movement_threshold,
            )
        assert limits[0][0] < limits[1][0] < limits[2][0]
        assert limits[0][1] < limits[1][1] < limits[2][1]
