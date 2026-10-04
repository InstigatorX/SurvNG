from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from survng.app.config import ObjectTrackingConfig
from survng.app.incident_queries import IncidentQueryService
from survng.app.security import required_api_scope


def manager_for_analysis(*, enabled=True, accepting=True, mode="on_demand"):
    status = {"camera_id": "gate", "status": "deferred", "episodes": [{"camera_id": "gate", "admission": "deferred", "status": "deferred"}], "remaining": 1}
    events = Mock()
    events.incident_analysis_status.return_value = status
    events.request_incident_analysis.return_value = {**status, "status": "queued"}
    worker = SimpleNamespace(tracking_lifecycle=SimpleNamespace(
        enabled=lambda: enabled, accepting=lambda: accepting))
    return SimpleNamespace(events=events, workers={"gate": worker}, config=SimpleNamespace(
        detector=SimpleNamespace(tracking=ObjectTrackingConfig(analysis_mode=mode, enabled=enabled))))


def test_mode_defaults_eager_and_rejects_unknown_values():
    assert ObjectTrackingConfig().analysis_mode == "eager"
    with pytest.raises(ValidationError):
        ObjectTrackingConfig(analysis_mode="sometimes")


def test_status_reads_never_admit_work():
    manager = manager_for_analysis()
    assert IncidentQueryService.analysis(manager, "incident")["status"] == "deferred"
    manager.events.request_incident_analysis.assert_not_called()


def test_explicit_request_admits_without_waiting_for_worker():
    manager = manager_for_analysis()
    assert IncidentQueryService.analysis(manager, "incident", request=True)["status"] == "queued"
    manager.events.request_incident_analysis.assert_called_once_with("incident", camera_ids={"gate"})


@pytest.mark.parametrize("enabled,accepting,removed", [(False, True, False), (True, False, False), (True, True, True)])
def test_disabled_or_missing_camera_does_not_queue_forever(enabled, accepting, removed):
    manager = manager_for_analysis(enabled=enabled, accepting=accepting)
    if removed:
        manager.workers.clear()
    result = IncidentQueryService.analysis(manager, "incident", request=True)
    assert result["status"] == "unavailable"
    assert result["enabled"] is False
    manager.events.request_incident_analysis.assert_not_called()


def test_cached_results_still_available_when_detection_off():
    manager = manager_for_analysis(accepting=False)
    manager.events.incident_analysis_status.return_value["status"] = "complete"
    manager.events.incident_analysis_status.return_value["episodes"][0]["status"] = "complete"
    assert IncidentQueryService.analysis(manager, "incident")["status"] == "complete"


def test_explicit_open_can_review_deferred_history_after_opting_out():
    manager = manager_for_analysis(mode="eager")
    assert IncidentQueryService.analysis(manager, "incident", request=True)["status"] == "queued"


def test_eager_automatic_only_incident_has_no_extra_viewer_work():
    manager = manager_for_analysis(mode="eager")
    manager.events.incident_analysis_status.return_value["episodes"][0]["admission"] = "automatic"
    assert IncidentQueryService.analysis(manager, "incident", request=True)["enabled"] is False
    manager.events.request_incident_analysis.assert_not_called()


def test_disabled_first_camera_does_not_prevent_review_of_other_episodes():
    manager = manager_for_analysis()
    status = manager.events.incident_analysis_status.return_value
    status["camera_id"] = "offline"
    status["episodes"].insert(0, {"camera_id": "offline", "admission": "deferred", "status": "deferred"})
    result = IncidentQueryService.analysis(manager, "incident", request=True)
    assert result["enabled"] is True
    assert result["episodes"][0]["status"] == "unavailable"
    manager.events.request_incident_analysis.assert_called_once_with("incident", camera_ids={"gate"})


def test_unknown_incident_returns_404():
    manager = manager_for_analysis()
    manager.events.incident_analysis_status.side_effect = KeyError("missing")
    with pytest.raises(HTTPException) as caught:
        IncidentQueryService.analysis(manager, "missing", request=True)
    assert caught.value.status_code == 404
    manager.events.request_incident_analysis.assert_not_called()


def test_viewer_can_request_only_exact_bounded_analysis_endpoint():
    assert required_api_scope("POST", "/api/incidents/abc/analysis") == "read"
    for path in ("/api/incidents/abc/corrections", "/api/incidents/abc/analysis/reset",
                 "/api/incidents/abc/episodes/analysis", "/api/incidents//analysis"):
        assert required_api_scope("POST", path) == "admin"
    assert required_api_scope("DELETE", "/api/incidents/abc/analysis") == "admin"
