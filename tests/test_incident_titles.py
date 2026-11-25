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


def incident(revision=3, incident_id="incident-a", **extra):
    return {
        "incident_id": incident_id, "revision": revision, "start_at": "2026-11-25T15:24:20+00:00", "duration_seconds": 102,
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


def test_a_failed_rewording_keeps_the_last_good_title():
    provider, clock = Provider(["First wording", AuditAiError("AI provider request timed out")]), Clock()
    generator = IncidentTitleGenerator(provider_factory=provider, clock=clock)
    assert generator.title(incident(), ON, NAMES)["title"] == "First wording"
    clock.now += 61
    kept = generator.title(incident(revision=4), ON, NAMES)
    assert (kept["status"], kept["title"], kept["current"], kept["refresh"]) == ("ready", "First wording", False, "failed")


def test_an_outage_logs_once_for_all_incidents(caplog):
    provider = Provider([AuditAiError("AI provider unavailable")] * 5)
    generator = IncidentTitleGenerator(provider_factory=provider)
    with caplog.at_level(logging.WARNING, logger="survng.app.incident_titles"):
        for index in range(5):
            generator.title(incident(incident_id=f"incident-{index}"), ON, NAMES)
    assert len(provider.calls) == 5
    assert caplog.text.count("AI incident title failed") == 1


def test_global_budget_reports_busy_without_calling_out():
    clock = Clock()
    provider = Provider([f"Title {index}" for index in range(30)])
    generator = IncidentTitleGenerator(provider_factory=provider, clock=clock)
    results = [generator.title(incident(incident_id=f"incident-{index}"), ON, NAMES) for index in range(25)]
    assert [item["status"] for item in results].count("ready") == 20
    assert {item["status"] for item in results[20:]} == {"busy"}
    clock.now += 61
    assert generator.title(incident(incident_id="incident-24"), ON, NAMES)["status"] == "ready"


def test_operator_typed_labels_are_not_sent():
    corrected = incident()
    corrected["scene_objects"][0]["label"] = "Alex's bike"
    corrected["scene_objects"][0]["observations"] = [{"label": "bicycle"}]
    facts = incident_title_facts(corrected, NAMES)
    assert facts["objects"][0]["label"] == "bicycle"
    assert "Alex" not in json.dumps(facts)


def test_unconfigured_provider_and_empty_scenes_never_call_out():
    provider = Provider([])
    generator = IncidentTitleGenerator(provider_factory=provider)
    assert generator.title(incident(), AuditAiConfig(enabled=True, incident_titles_enabled=True), NAMES)["reason"] == "not_configured"
    assert generator.title(incident(scene_objects=[]), ON, NAMES)["status"] == "skipped"
    assert provider.calls == []


def test_concurrent_requests_never_wait_on_the_provider():
    started, release = threading.Event(), threading.Event()

    class Slow(Provider):
        def complete_json(self, *args):
            started.set()
            release.wait(5)
            return super().complete_json(*args)

    provider = Slow(["Shared title", "Other title"])
    generator = IncidentTitleGenerator(provider_factory=provider)
    results = []
    worker = threading.Thread(target=lambda: results.append(generator.title(incident(), ON, NAMES)))
    worker.start()
    assert started.wait(5)
    # The same incident reports pending; others fill the second slot, then busy.
    assert generator.title(incident(), ON, NAMES)["status"] == "pending"
    other = threading.Thread(target=lambda: results.append(generator.title(incident(incident_id="incident-b"), ON, NAMES)))
    other.start()
    for _ in range(100):
        if len(provider.calls) == 0 and generator._slots._value == 0:
            break
        threading.Event().wait(0.01)
    assert generator.title(incident(incident_id="incident-c"), ON, NAMES)["status"] == "busy"
    release.set()
    worker.join(10); other.join(10)
    assert sorted(item["title"] for item in results) == ["Other title", "Shared title"]
    assert generator.title(incident(), ON, NAMES)["title"] in {"Shared title", "Other title"}


def test_route_skips_the_payload_when_off_and_reports_missing_incidents():
    loads = []

    def scene_incident(incident_id):
        loads.append(incident_id)
        return incident() if incident_id == "incident-a" else None

    camera = SimpleNamespace(id="driveway", name="Front Driveway")
    config = SimpleNamespace(audit_ai=AuditAiConfig(), cameras=[camera])
    manager = SimpleNamespace(events=SimpleNamespace(scene_incident=scene_incident), config=config)
    bundle = create_incident_query_router(IncidentQueryDependencies(lambda: manager, threading.RLock()), IncidentQueryService())
    route = next(item.endpoint for item in bundle.router.routes if item.path == "/api/incidents/{incident_id}/title")
    assert route("incident-a") == {"incident_id": "incident-a", "status": "disabled", "title": None}
    assert loads == [], "the default (off) never builds the incident payload"
    config.audit_ai = AuditAiConfig(enabled=True, incident_titles_enabled=True)
    assert route("incident-a")["reason"] == "not_configured"
    with pytest.raises(HTTPException) as error:
        route("incident-missing")
    assert error.value.status_code == 404
