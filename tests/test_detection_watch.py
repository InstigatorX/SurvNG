from __future__ import annotations

from survng.app.config import CameraTransitionRoute
from survng.app.detection_watch import RouteDetectionWatch


def _objects(label: str = "car") -> list[dict[str, object]]:
    return [{"label": label, "confidence": 0.91}]


def test_confirmed_incident_opens_only_configured_directional_window() -> None:
    watches = RouteDetectionWatch([
        CameraTransitionRoute(
            from_camera="gate",
            to_camera="back-left",
            min_seconds=2,
            max_seconds=12,
            name="gate to rear",
        )
    ])

    created = watches.observe_incident(
        camera_id="gate",
        event_id=44,
        event_at=100.0,
        objects=_objects(),
    )

    assert len(created) == 1
    assert watches.match("back-left", 101.9) is None
    assert watches.match("back-left", 102.0) is not None
    assert watches.match("back-left", 112.0) is not None
    assert watches.match("back-left", 112.1) is None
    assert watches.match("gate", 105.0) is None


def test_bidirectional_route_opens_reverse_window() -> None:
    watches = RouteDetectionWatch([
        CameraTransitionRoute(
            from_camera="back-left",
            to_camera="back-middle",
            min_seconds=0,
            max_seconds=20,
            bidirectional=True,
        )
    ])

    watches.observe_incident(
        camera_id="back-middle",
        event_id=8,
        event_at=50.0,
        objects=_objects("person"),
    )

    match = watches.match("back-left", 55.0)
    assert match is not None
    assert match.source_camera_id == "back-middle"
    assert match.labels == ("person",)


def test_duplicate_event_does_not_extend_or_duplicate_watch() -> None:
    watches = RouteDetectionWatch([
        CameraTransitionRoute(
            from_camera="gate",
            to_camera="back-left",
            max_seconds=10,
        )
    ])

    first = watches.observe_incident(
        camera_id="gate", event_id=7, event_at=100.0, objects=_objects()
    )
    duplicate = watches.observe_incident(
        camera_id="gate", event_id=7, event_at=105.0, objects=_objects()
    )

    assert len(first) == 1
    assert duplicate == ()
    assert watches.match("back-left", 111.0) is None


def test_objectless_motion_does_not_open_watch() -> None:
    watches = RouteDetectionWatch([
        CameraTransitionRoute(from_camera="gate", to_camera="back-left")
    ])

    assert watches.observe_incident(
        camera_id="gate", event_id=1, event_at=100.0, objects=[]
    ) == ()
    assert watches.snapshot(100.0) == ()


def test_route_watch_is_consumed_once_and_reports_lifecycle_counters() -> None:
    watches = RouteDetectionWatch([
        CameraTransitionRoute(from_camera="gate", to_camera="back-left")
    ])
    watches.observe_incident(
        camera_id="gate", event_id=9, event_at=100.0, objects=_objects()
    )

    assert watches.match("back-left", 101.0) is not None
    assert watches.consume("back-left", 9) is True
    assert watches.consume("back-left", 9) is False
    assert watches.observe_incident(
        camera_id="gate", event_id=9, event_at=102.0, objects=_objects()
    ) == ()
    status = watches.status(101.0)

    assert status["opened"] == 1
    assert status["matched"] == 1
    assert status["consumed"] == 1
    assert status["active"] == 0


def test_route_lineage_allows_forward_chain_but_blocks_bidirectional_cycle() -> None:
    watches = RouteDetectionWatch([
        CameraTransitionRoute(
            from_camera="gate",
            to_camera="lower-garage",
            bidirectional=True,
        ),
        CameraTransitionRoute(
            from_camera="lower-garage",
            to_camera="upper-garage",
            bidirectional=True,
        ),
        CameraTransitionRoute(
            from_camera="gate",
            to_camera="upper-garage",
            bidirectional=True,
        ),
    ])

    first = watches.observe_incident(
        camera_id="gate", event_id=1, event_at=100.0, objects=_objects()
    )
    lower = next(watch for watch in first if watch.target_camera_id == "lower-garage")
    second = watches.observe_incident(
        camera_id="lower-garage",
        event_id=2,
        event_at=101.0,
        objects=_objects(),
        route_path=lower.route_path,
    )
    assert [watch.target_camera_id for watch in second] == ["upper-garage"]
    upper = second[0]
    assert watches.observe_incident(
        camera_id="upper-garage",
        event_id=3,
        event_at=102.0,
        objects=_objects(),
        route_path=upper.route_path,
    ) == ()
    assert watches.status(102.0)["lineage_blocked"] == 3


def test_origin_consumption_removes_all_alternate_paths_to_one_target() -> None:
    watches = RouteDetectionWatch([
        CameraTransitionRoute(from_camera="gate", to_camera="upper-garage"),
        CameraTransitionRoute(from_camera="lower-garage", to_camera="upper-garage"),
    ])
    direct = watches.observe_incident(
        camera_id="gate", event_id=10, event_at=100.0, objects=_objects()
    )[0]
    lower_to_upper = watches.observe_incident(
        camera_id="lower-garage",
        event_id=11,
        event_at=101.0,
        objects=_objects(),
        route_path=("gate", "lower-garage"),
        origin_camera_id="gate",
        origin_event_id=10,
    )[0]

    assert direct.origin_camera_id == "gate"
    assert direct.origin_event_id == 10
    assert lower_to_upper.origin_camera_id == "gate"
    consumed = watches.consume_origin("upper-garage", "gate", 10)

    assert {watch.source_event_id for watch in consumed} == {10, 11}
    assert watches.match("upper-garage", 102.0) is None
    assert watches.status(102.0)["consumed"] == 2


def test_active_watch_overflow_is_visible_in_status() -> None:
    watches = RouteDetectionWatch([
        CameraTransitionRoute(from_camera="gate", to_camera="back-left")
    ], maximum_watches=16)

    for event_id in range(1, 18):
        watches.observe_incident(
            camera_id="gate",
            event_id=event_id,
            event_at=100.0,
            objects=_objects(),
        )

    status = watches.status(100.0)
    assert status["active"] == 16
    assert status["overflowed"] == 1
