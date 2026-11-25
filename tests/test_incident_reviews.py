"""Incident review verdicts: operator feedback that never rewrites evidence."""
import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from survng.app.events import EventStore
from survng.app.incident_queries import IncidentQueryDependencies, IncidentQueryService, create_incident_query_router
from survng.app.security import required_api_scope

# 2026-05-28 20:26:40 UTC = 16:26:40 in New York, well inside one local day.
BASE = 1780000000
DAY = "2026-05-28"


def person(x=10):
    return {"label": "person", "confidence": .8, "temporal_observations": 5,
            "box": {"x1": x, "y1": 10, "x2": x + 10, "y2": 40}}


def add_incident(store, seconds, camera="driveway"):
    store.add_event(camera, "motion", created_at=datetime.fromtimestamp(BASE + seconds, timezone.utc).isoformat(),
                    objects_json=json.dumps([person()]))
    return store.list_scene_incidents(start_epoch=BASE + seconds - 1, end_epoch=BASE + seconds + 1, summary_only=True)[0]["id"]


@pytest.fixture
def store(tmp_path):
    return EventStore(tmp_path)


def routes(store):
    manager = SimpleNamespace(events=store)
    bundle = create_incident_query_router(IncidentQueryDependencies(lambda: manager, threading.RLock()), IncidentQueryService())
    return {(route.path, method): route.endpoint for route in bundle.router.routes for method in route.methods}


def test_verdict_is_recorded_replaced_and_cleared_without_touching_the_scene(store):
    incident_id = add_incident(store, 0)
    before = store.scene_incident(incident_id)
    assert store.incident_review(incident_id)["verdict"] is None
    recorded = store.set_incident_review(incident_id, "keep", now=BASE + 100)
    assert recorded["verdict"] == "keep"
    assert recorded["reviewed_revision"] == before["revision"]
    assert recorded["changed_since_review"] is False
    assert store.set_incident_review(incident_id, "nothing_here")["verdict"] == "nothing_here"
    after = store.scene_incident(incident_id)
    assert after["revision"] == before["revision"]
    assert after["scene_objects"] == before["scene_objects"]
    assert after["establishment"] == before["establishment"]
    assert store.set_incident_review(incident_id, None)["verdict"] is None
    assert store.incident_review(incident_id)["verdict"] is None


def test_unknown_verdicts_and_incidents_are_rejected(store):
    incident_id = add_incident(store, 0)
    with pytest.raises(ValueError):
        store.set_incident_review(incident_id, "dismissed")
    with pytest.raises(LookupError):
        store.set_incident_review("incident-missing", "reviewed")
    with pytest.raises(LookupError):
        store.incident_review("incident-missing")


def test_later_evidence_marks_a_verdict_as_changed_since_review(store):
    incident_id = add_incident(store, 0)
    store.set_incident_review(incident_id, "reviewed")
    store.add_event("driveway", "motion", created_at=datetime.fromtimestamp(BASE + 5, timezone.utc).isoformat(),
                    objects_json=json.dumps([person(60)]))
    assert store.scene_incident(incident_id)["revision"] > store.incident_review(incident_id)["reviewed_revision"]
    assert store.incident_review(incident_id)["changed_since_review"] is True


def test_queue_follows_list_order_and_counts_what_is_left(store):
    first = add_incident(store, 0)
    second = add_incident(store, 3600, camera="front-door")
    third = add_incident(store, 7200)
    store.set_incident_review(second, "reviewed")
    store.set_incident_review(third, "keep")
    queue = store.incident_review_queue(start_epoch=BASE - 10, end_epoch=BASE + 8000)
    assert [item["incident_id"] for item in queue["items"]] == [third, second, first]
    assert [item["verdict"] for item in queue["items"]] == ["keep", "reviewed", None]
    assert (queue["total"], queue["reviewed"], queue["kept"], queue["remaining"]) == (3, 1, 1, 1)
    assert store.incident_review_queue(start_epoch=BASE - 10, end_epoch=BASE + 8000, camera_id="front-door")["total"] == 1


def test_review_routes(store):
    incident_id = add_incident(store, 0)
    endpoints = routes(store)
    put = endpoints[("/api/incidents/{incident_id}/review", "PUT")]
    get = endpoints[("/api/incidents/{incident_id}/review", "GET")]
    queue = endpoints[("/api/incidents/review-queue", "GET")]
    assert put(incident_id, {"verdict": "reviewed"})["verdict"] == "reviewed"
    assert get(incident_id)["verdict"] == "reviewed"
    result = queue(day=DAY, time_zone="America/New_York", event_type="all")
    assert (result["day"], result["total"], result["remaining"]) == (DAY, 1, 0)
    for body, status in (({}, 422), ({"verdict": "maybe"}, 422)):
        with pytest.raises(HTTPException) as error:
            put(incident_id, body)
        assert error.value.status_code == status
    with pytest.raises(HTTPException) as error:
        put("incident-missing", {"verdict": "reviewed"})
    assert error.value.status_code == 404
    with pytest.raises(HTTPException) as error:
        queue(day="28-05-2026")
    assert error.value.status_code == 422


def test_route_order_and_scopes():
    endpoints = routes(SimpleNamespace())
    paths = [path for path, _ in endpoints]
    # The literal queue path must win over the incident-id catch-all.
    assert paths.index("/api/incidents/review-queue") < paths.index("/api/incidents/{incident_id}")
    assert required_api_scope("GET", "/api/incidents/review-queue") == "read"
    assert required_api_scope("PUT", "/api/incidents/incident-x/review") == "admin"
