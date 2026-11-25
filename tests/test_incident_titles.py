"""Opt-in provider titles: off by default, cached per revision, never blocking."""
import json
import logging
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from survng.app.audit_ai import AuditAiError
from survng.app.config import AuditAiConfig
from survng.app.incident_queries import IncidentQueryDependencies, IncidentQueryService, create_incident_query_router
from survng.app.incident_titles import IncidentTitleGenerator, incident_title_facts


def incident(revision=3, **extra):
    return {
        "incident_id": "incident-a", "revision": revision, "start_at": "2026-11-25T15:24:20+00:00", "duration_seconds": 102,
        "episodes": [{"camera_id": "driveway"}, {"camera_id": "door"}],
        "identities": [{"name": "Alex Example"}],
        "scene_objects": [
            {"id": "object-1", "label": "person", "certainty": "observed", "visible_intervals": [
                {"camera_id": "driveway", "episode_id": "episode-1", "start_at": "2026-11-25T15:24:41+00:00", "end_at": "2026-11-25T15:25:09+00:00", "state": "moving", "zones": ["Driveway"]},
                {"camera_id": "door", "episode_id": "episode-2", "start_at": "2026-11-25T15:25:07+00:00", "end_at": "2026-11-25T15:26:02+00:00", "state": "moving", "zones": ["Entry"]}]},
            {"id": "object-2", "label": "dog", "certainty": "possible", "visible_intervals": []},
        ],
        **extra,
    }


NAMES = {"driveway": "Front Driveway", "door": "Front Door"}
ON = AuditAiConfig(enabled=True, incident_titles_enabled=True, api_key="sk-test")


class Provider:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def __call__(self, config):
        return self

    def complete_json(self, system_prompt, prompt, schema, schema_name):
        self.calls.append(json.loads(prompt))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps({"title": reply})


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def test_off_by_default_and_without_ai_features():
    provider = Provider(["unused"])
    generator = IncidentTitleGenerator(provider_factory=provider)
    assert AuditAiConfig().incident_titles_enabled is False
    assert generator.title(incident(), AuditAiConfig(), NAMES)["status"] == "disabled"
    assert generator.title(incident(), AuditAiConfig(incident_titles_enabled=True, api_key="k"), NAMES)["status"] == "disabled"
    assert provider.calls == []


def test_facts_carry_scene_wording_but_no_identities_or_ids():
    facts = incident_title_facts(incident(), NAMES)
    assert facts["cameras"] == ["Front Driveway", "Front Door"]
    assert facts["objects"][0]["seen"][1] == {"camera": "Front Door", "from_second": 47, "to_second": 102, "state": "moving", "zones": ["Entry"]}
    assert facts["objects"][1]["uncertain"] is True
    encoded = json.dumps(facts)
    for private in ("Alex", "incident-a", "object-1", "episode-1", "driveway\""):
        assert private not in encoded


def test_title_is_cached_per_revision_and_reworded_at_most_once_a_minute():
    provider, clock = Provider(["Person walks from the driveway to the front door.", "Person reaches the door"]), Clock()
    generator = IncidentTitleGenerator(provider_factory=provider, clock=clock)
    first = generator.title(incident(), ON, NAMES)
    assert first == {"incident_id": "incident-a", "status": "ready", "title": "Person walks from the driveway to the front door", "revision": 3, "current": True}
    assert generator.title(incident(), ON, NAMES)["title"] == first["title"]
    # A new revision inside the minute keeps the previous wording, marked not current.
    stale = generator.title(incident(revision=4), ON, NAMES)
    assert (stale["title"], stale["current"]) == (first["title"], False)
    clock.now += 61
    assert generator.title(incident(revision=4), ON, NAMES)["title"] == "Person reaches the door"
    assert len(provider.calls) == 2


def test_failures_fall_back_with_a_logged_reason(caplog):
    provider, clock = Provider([AuditAiError("AI provider request timed out"), "x" * 200, "Fine title"]), Clock()
    generator = IncidentTitleGenerator(provider_factory=provider, clock=clock)
    with caplog.at_level(logging.WARNING, logger="survng.app.incident_titles"):
        failed = generator.title(incident(), ON, NAMES)
    assert (failed["status"], failed["reason"], failed["title"]) == ("failed", "provider_failed", None)
    assert "timed out" in caplog.text and "built-in title" in caplog.text
    # The failure is remembered briefly instead of retried on every poll.
    assert generator.title(incident(), ON, NAMES)["status"] == "failed"
    assert len(provider.calls) == 1
    clock.now += 61
    invalid = generator.title(incident(), ON, NAMES)
    assert (invalid["status"], invalid["reason"]) == ("failed", "invalid_response")
    assert "x" * 50 not in caplog.text, "model output is not echoed into logs"
    clock.now += 61
    assert generator.title(incident(), ON, NAMES)["title"] == "Fine title"


def test_unconfigured_provider_and_empty_scenes_never_call_out():
    provider = Provider([])
    generator = IncidentTitleGenerator(provider_factory=provider)
    assert generator.title(incident(), AuditAiConfig(enabled=True, incident_titles_enabled=True), NAMES)["reason"] == "not_configured"
    assert generator.title(incident(scene_objects=[]), ON, NAMES)["status"] == "skipped"
    assert provider.calls == []


def test_concurrent_requests_for_one_incident_share_one_provider_call():
    release = threading.Event()

    class Slow(Provider):
        def complete_json(self, *args):
            release.wait(5)
            return super().complete_json(*args)

    provider = Slow(["Shared title"])
    generator = IncidentTitleGenerator(provider_factory=provider)
    results = []
    threads = [threading.Thread(target=lambda: results.append(generator.title(incident(), ON, NAMES))) for _ in range(3)]
    for thread in threads:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(10)
    assert [item["title"] for item in results] == ["Shared title"] * 3
    assert len(provider.calls) == 1


def test_route_reads_config_and_reports_missing_incidents():
    events = SimpleNamespace(scene_incident=lambda incident_id: incident() if incident_id == "incident-a" else None)
    camera = SimpleNamespace(id="driveway", name="Front Driveway")
    manager = SimpleNamespace(events=events, config=SimpleNamespace(audit_ai=AuditAiConfig(), cameras=[camera]))
    bundle = create_incident_query_router(IncidentQueryDependencies(lambda: manager, threading.RLock()), IncidentQueryService())
    route = next(item.endpoint for item in bundle.router.routes if item.path == "/api/incidents/{incident_id}/title")
    assert route("incident-a") == {"incident_id": "incident-a", "status": "disabled", "title": None}
    with pytest.raises(HTTPException) as error:
        route("incident-missing")
    assert error.value.status_code == 404
