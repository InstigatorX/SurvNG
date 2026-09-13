import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import survng.app.system_routes as system_routes
from survng.app.config import AppConfig
from survng.app.state_events import StateEventBroker
from survng.app.system_routes import SystemRouteDependencies, create_system_router


class StreamRequest:
    def __init__(self, *, header_cursor: str = "", query_cursor: str = "") -> None:
        self.headers = {"last-event-id": header_cursor} if header_cursor else {}
        self.query_params = {"last_event_id": query_cursor} if query_cursor else {}

    async def is_disconnected(self) -> bool:
        return False


def stream_handler(manager, telemetry, get_manager=None, config=None):
    dependencies = SystemRouteDependencies(
        get_manager=get_manager or Mock(return_value=manager),
        get_config=lambda: config or AppConfig(),
        system_telemetry=telemetry,
        ffprobe_path=Mock(),
        ffplay_path=Mock(),
        ffmpeg_qsv_info=Mock(),
        ffmpeg_vaapi_info=Mock(),
        hardware_acceleration_mode=Mock(),
        event_clip_window=Mock(),
        recording_cache_status=Mock(),
        model_evaluation=Mock(),
    )
    return create_system_router(dependencies).handlers["application_event_stream"]


async def read_chunks(handler, request, count):
    response = await handler(request)
    iterator = response.body_iterator
    try:
        return [await anext(iterator) for _ in range(count)]
    finally:
        await iterator.aclose()


def test_stream_resumes_from_query_cursor_without_full_snapshots() -> None:
    broker = StateEventBroker()
    first = broker.publish("camera_state", {"id": "gate", "running": False})
    second = broker.publish("camera_state", {"id": "gate", "running": True})
    manager = SimpleNamespace(state_events=broker, statuses=Mock(return_value=[]))
    telemetry = SimpleNamespace(system_status=Mock(return_value={}))

    chunks = asyncio.run(read_chunks(
        stream_handler(manager, telemetry),
        StreamRequest(query_cursor=first.id),
        3,
    ))

    assert chunks[0] == "retry: 3000\n\n"
    assert f"id: {second.id}" in chunks[1]
    assert "event: camera_state" in chunks[1]
    assert f"id: {second.id}" in chunks[2]
    assert "event: connected" in chunks[2]
    manager.statuses.assert_not_called()
    telemetry.system_status.assert_not_called()


def test_unconsumed_stream_does_not_subscribe_to_the_broker() -> None:
    async def create_without_consuming():
        broker = StateEventBroker()
        manager = SimpleNamespace(state_events=broker, statuses=Mock(return_value=[]))
        telemetry = SimpleNamespace(system_status=Mock(return_value={}))
        get_manager = Mock(return_value=manager)

        response = await stream_handler(manager, telemetry, get_manager)(StreamRequest())

        assert not broker._subscribers
        get_manager.assert_not_called()
        await response.body_iterator.aclose()
        assert not broker._subscribers

    asyncio.run(create_without_consuming())


def test_native_last_event_id_takes_precedence_over_query_cursor() -> None:
    broker = StateEventBroker()
    first = broker.publish("camera_state", {"id": "gate", "running": False})
    second = broker.publish("camera_state", {"id": "gate", "running": True})
    manager = SimpleNamespace(state_events=broker, statuses=Mock(return_value=[]))
    telemetry = SimpleNamespace(system_status=Mock(return_value={}))

    chunks = asyncio.run(read_chunks(
        stream_handler(manager, telemetry),
        StreamRequest(header_cursor=second.id, query_cursor=first.id),
        2,
    ))

    assert chunks[0] == "retry: 3000\n\n"
    assert f"id: {second.id}" in chunks[1]
    assert "event: connected" in chunks[1]
    manager.statuses.assert_not_called()
    telemetry.system_status.assert_not_called()


def test_fresh_stream_sends_one_snapshot_then_a_lightweight_heartbeat(monkeypatch) -> None:
    monkeypatch.setattr(system_routes, "SSE_HEARTBEAT_SECONDS", 0.0)
    broker = StateEventBroker()
    manager = SimpleNamespace(
        state_events=broker,
        statuses=Mock(return_value=[{"id": "gate", "running": True}]),
    )
    telemetry = SimpleNamespace(system_status=Mock(return_value={"cpu_percent": 12}))
    get_manager = Mock(return_value=manager)

    chunks = asyncio.run(read_chunks(
        stream_handler(manager, telemetry, get_manager),
        StreamRequest(),
        5,
    ))

    assert "event: cameras_state" in chunks[1]
    assert "event: system_state" in chunks[2]
    assert "event: connected" in chunks[3]
    assert chunks[4] == ": heartbeat\n\n"
    manager.statuses.assert_called_once_with()
    telemetry.system_status.assert_called_once_with(manager)
    get_manager.assert_called_once_with()


def test_native_incident_snapshot_avoids_camera_and_storage_telemetry():
    broker = StateEventBroker()
    incidents = [{"incident_id": "incident-gate-41", "revision": 3, "state": "complete"}]
    manager = SimpleNamespace(state_events=broker, statuses=Mock(),
                              incident_notification_allowed=lambda item: True,
                              incident_notification_payload=lambda item: {**item, "notifications_enabled": False},
                              incidents=SimpleNamespace(snapshot=Mock(return_value=incidents)))
    telemetry = SimpleNamespace(system_status=Mock())
    request = StreamRequest()
    request.query_params["incidents_only"] = "1"
    chunks = asyncio.run(read_chunks(stream_handler(manager, telemetry), request, 3))
    assert "event: incident_notifications_state" in chunks[1]
    assert '"revision":3' in chunks[1]
    assert '"notifications_enabled":false' in chunks[1]
    assert "event: connected" in chunks[2]
    manager.statuses.assert_not_called()
    telemetry.system_status.assert_not_called()


def test_slow_stream_disconnects_for_replay_instead_of_silently_skipping_events():
    async def run():
        broker = StateEventBroker(subscriber_queue_size=8)
        manager = SimpleNamespace(state_events=broker, incidents=SimpleNamespace(snapshot=lambda: []))
        request = StreamRequest()
        request.query_params["incidents_only"] = "1"
        response = await stream_handler(manager, Mock())(request)
        iterator = response.body_iterator
        for _ in range(3):
            await anext(iterator)
        for revision in range(12):
            broker.publish("incident_lifecycle", {"revision": revision})
        try:
            await anext(iterator)
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("stream silently skipped dropped revisions")
        assert not broker._subscribers
    asyncio.run(run())


def notification_manager():
    from survng.app.config import AppConfig
    from survng.app.manager import AppManager
    manager = object.__new__(AppManager)
    manager.config = AppConfig()
    manager.config.integration_notifications.exclude_motion = True
    manager.state_events = StateEventBroker()
    manager.incidents = SimpleNamespace(snapshot=lambda: [])
    return manager


def test_motion_filter_applies_to_recovery_snapshot():
    manager = notification_manager()
    manager.incidents.snapshot = lambda: [
        {"incident_id": "motion", "has_objects": False},
        {"incident_id": "person", "classes": ["person"]},
    ]
    request = StreamRequest()
    request.query_params["incidents_only"] = "1"
    chunks = asyncio.run(read_chunks(stream_handler(manager, Mock()), request, 3))
    assert '"incident_id":"motion"' not in chunks[1]
    assert '"incident_id":"person"' in chunks[1]


def test_motion_filter_applies_to_replay_and_advances_past_skipped_events():
    manager = notification_manager()
    first = manager.state_events.publish("camera_state", {})
    manager.state_events.publish("incident_lifecycle", {"incident_id": "person", "classes": ["person"]})
    last = manager.state_events.publish("incident_lifecycle", {"incident_id": "motion"})
    request = StreamRequest(header_cursor=first.id)
    request.query_params["incidents_only"] = "1"
    chunks = asyncio.run(read_chunks(stream_handler(manager, Mock()), request, 3))
    assert '"incident_id":"person"' in chunks[1]
    assert '"incident_id":"motion"' not in "".join(chunks)
    assert f"id: {last.id}" in chunks[2]


def test_live_motion_filter_keeps_sequence_and_allows_object_promotion():
    async def run():
        manager = notification_manager()
        request = StreamRequest()
        request.query_params["incidents_only"] = "1"
        response = await stream_handler(manager, Mock())(request)
        iterator = response.body_iterator
        try:
            for _ in range(3):
                await anext(iterator)
            manager.state_events.publish("incident_lifecycle", {"incident_id": "same", "state": "new"})
            manager.state_events.publish("camera_state", {"running": True})
            promoted = manager.state_events.publish("incident_lifecycle", {
                "incident_id": "same", "state": "updated", "classes": ["person"],
            })
            chunk = await asyncio.wait_for(anext(iterator), 1)
            assert f"id: {promoted.id}" in chunk
            assert '"state":"updated"' in chunk
            complete = manager.state_events.publish("incident_lifecycle", {
                "incident_id": "same", "state": "complete", "classes": ["person"],
            })
            assert f"id: {complete.id}" in await asyncio.wait_for(anext(iterator), 1)
            # A hot setting change permits subsequent motion-only notifications.
            manager.config.integration_notifications.exclude_motion = False
            motion = manager.state_events.publish("incident_lifecycle", {"incident_id": "motion"})
            assert f"id: {motion.id}" in await asyncio.wait_for(anext(iterator), 1)
        finally:
            await iterator.aclose()
    asyncio.run(run())


def test_open_stream_stops_after_token_revocation_or_scope_reduction():
    from survng.app.config import ApiTokenConfig
    from survng.app.security import hash_api_token

    async def run(change, replay):
        config = AppConfig()
        config.api_auth.enabled = True
        config.api_auth.tokens = [ApiTokenConfig(id="ha", name="HA", token_hash=hash_api_token("secret"), scopes=["read"])]
        broker = StateEventBroker()
        first = broker.publish("camera_state", {})
        broker.publish("camera_state", {"private": "replay"})
        manager = SimpleNamespace(state_events=broker, statuses=lambda: [])
        request = StreamRequest(header_cursor=first.id if replay else "")
        request.headers["authorization"] = "Bearer secret"
        response = await stream_handler(manager, SimpleNamespace(system_status=lambda _: {}), config=config)(request)
        iterator = response.body_iterator
        for _ in range(1 if replay else 4):
            await anext(iterator)
        change(config)
        broker.publish("camera_state", {"private": "after revocation"})
        try:
            await asyncio.wait_for(anext(iterator), 1)
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("revoked stream delivered data")
        assert not broker._subscribers

    for change in (
        lambda config: config.api_auth.tokens.clear(),
        lambda config: setattr(config.api_auth.tokens[0], "scopes", ["camera:control"]),
        lambda config: setattr(config.api_auth, "enabled", False),
    ):
        for replay in (True, False):
            asyncio.run(run(change, replay))


def test_idle_stream_stops_after_session_revocation():
    from survng.app.config import WebUserConfig
    from survng.app.security import SESSION_COOKIE_NAME, encode_session

    async def run():
        config = AppConfig()
        config.web_auth.enabled = True
        config.web_auth.session_key = "a" * 64
        config.web_auth.users = [WebUserConfig(id="viewer", username="viewer", password_hash="__SURVNG_SECRET_SET__")]
        request = StreamRequest()
        request.headers["cookie"] = f"{SESSION_COOKIE_NAME}={encode_session('viewer', config.web_auth.session_key)}"
        broker = StateEventBroker()
        manager = SimpleNamespace(state_events=broker, statuses=lambda: [])
        response = await stream_handler(manager, SimpleNamespace(system_status=lambda _: {}), config=config)(request)
        iterator = response.body_iterator
        for _ in range(4):
            await anext(iterator)
        config.web_auth.users[0].session_epoch += 1
        try:
            await asyncio.wait_for(anext(iterator), 1)
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("revoked idle session remained open")
        assert not broker._subscribers
    asyncio.run(run())
