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
