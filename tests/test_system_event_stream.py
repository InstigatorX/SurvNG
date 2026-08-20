import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import survng.app.system_routes as system_routes
from survng.app.state_events import StateEventBroker
from survng.app.system_routes import SystemRouteDependencies, create_system_router


class StreamRequest:
    def __init__(self, *, header_cursor: str = "", query_cursor: str = "") -> None:
        self.headers = {"last-event-id": header_cursor} if header_cursor else {}
        self.query_params = {"last_event_id": query_cursor} if query_cursor else {}

    async def is_disconnected(self) -> bool:
        return False


def stream_handler(manager, telemetry, get_manager=None):
    dependencies = SystemRouteDependencies(
        get_manager=get_manager or Mock(return_value=manager),
        get_config=Mock(),
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
