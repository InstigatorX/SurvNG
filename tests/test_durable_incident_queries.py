"""First-class durable incidents on list/detail surfaces."""
from __future__ import annotations

from pathlib import Path

from survng.app.event_store import EventStore
from survng.app.incident_presenter import summary_from_durable
from survng.app.incident_queries import IncidentQueryService
from types import SimpleNamespace


def test_summary_from_durable_uses_incident_time_range_not_gap(tmp_path: Path):
    store = EventStore(tmp_path)
    event = store.add_event(
        camera_id="yard",
        kind="motion",
        topic="native/object-presence",
        message="test",
        created_at="2026-09-20T12:00:00+00:00",
        objects_json='[{"label":"person","confidence":0.9,"zones":["drive"]}]',
    )
    opened = store.open_incident(
        camera_id="yard",
        start_at="2026-09-20T12:00:00+00:00",
        participants=[
            {"label": "person", "zones": ["drive"]},
            {"label": "dog", "zones": ["drive"]},
        ],
        observation_objects=[{"label": "person"}],
        seed_event_id=int(event["id"]),
    )
    store.close_incident(
        opened["id"],
        end_at="2026-09-20T12:10:00+00:00",
        state="complete",
        completion_reason="complete",
    )
    durable = store.get_incident(opened["id"])
    seed = store.get(int(event["id"]))
    summary = summary_from_durable(durable, seed)
    assert summary["durable_incident_id"] == opened["id"]
    assert summary["start_at"] == "2026-09-20T12:00:00+00:00"
    assert summary["end_at"] == "2026-09-20T12:10:00+00:00"
    assert summary["duration_seconds"] == 600.0
    assert summary["labels"] == ["dog", "person"]
    assert summary["representative_event_id"] == int(event["id"])


def test_list_and_resolve_prefer_durable_incidents(tmp_path: Path):
    store = EventStore(tmp_path)
    event = store.add_event(
        camera_id="yard",
        kind="motion",
        topic="native/object-presence",
        message="test",
        created_at="2026-09-20T12:00:00+00:00",
        objects_json='[{"label":"person","confidence":0.9}]',
    )
    opened = store.open_incident(
        camera_id="yard",
        start_at="2026-09-20T12:00:00+00:00",
        participants=[{"label": "person"}],
        observation_objects=[{"label": "person"}],
        seed_event_id=int(event["id"]),
    )
    store.close_incident(
        opened["id"],
        end_at="2026-09-20T12:03:00+00:00",
        state="complete",
    )
    manager = SimpleNamespace(
        events=store,
        storage_dir=tmp_path,
        media_storage=None,
        faces=SimpleNamespace(for_event_ids=lambda ids: []),
        config=SimpleNamespace(cameras=[]),
    )
    summaries = IncidentQueryService.recent_summaries(manager, limit=10, gap_seconds=45)
    assert len(summaries) == 1
    assert summaries[0]["durable_incident_id"] == opened["id"]
    assert summaries[0]["duration_seconds"] == 180.0

    resolved = IncidentQueryService().resolve_event(manager, int(event["id"]))
    assert resolved is not None
    assert resolved["durable_incident_id"] == opened["id"]
    assert resolved["end_at"] == "2026-09-20T12:03:00+00:00"

    detail = IncidentQueryService().detail(
        manager, str(event["id"]), gap_seconds=45
    )
    assert detail["durable_incident_id"] == opened["id"]


def test_legacy_events_without_incidents_still_gap_group(tmp_path: Path):
    store = EventStore(tmp_path)
    first = store.add_event(
        camera_id="gate",
        kind="motion",
        topic="legacy",
        message="a",
        created_at="2026-09-20T12:00:00+00:00",
        objects_json='[{"label":"car"}]',
    )
    second = store.add_event(
        camera_id="gate",
        kind="motion",
        topic="legacy",
        message="b",
        created_at="2026-09-20T12:00:20+00:00",
        objects_json='[{"label":"car"}]',
    )
    manager = SimpleNamespace(
        events=store,
        storage_dir=tmp_path,
        media_storage=None,
        faces=SimpleNamespace(for_event_ids=lambda ids: []),
        config=SimpleNamespace(cameras=[]),
    )
    summaries = IncidentQueryService.recent_summaries(manager, limit=10, gap_seconds=45)
    assert len(summaries) == 1
    assert "durable_incident_id" not in summaries[0]
    assert {int(event["id"]) for event in summaries[0]["events"]} == {
        int(first["id"]),
        int(second["id"]),
    }
