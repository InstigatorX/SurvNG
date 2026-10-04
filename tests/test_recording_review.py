import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from survng.app import recording_review as module
from survng.app.recording_review import RecordingReviewService


EPOCH = 120


def samples(job, stop):
    for timestamp in job["sample_targets"]:
        yield {"timestamp": timestamp, "status": "sampled", "objects": [
            {"label": "person", "confidence": .9, "private": "/secret/file"},
        ]}


def service(tmp_path, *, analyze=samples, manifest=None, identity=None):
    return RecordingReviewService(tmp_path,
        manifest_provider=manifest or (lambda *args: [{"segment_key": "one", "size": 2048, "path": "/private/movie.mp4"}]),
        analysis_identity=identity or (lambda: "model-v1"), analyze=analyze)


def finished(review, epoch=EPOCH):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = review.status("gate", "main", epoch)
        if result["state"] not in {"queued", "analyzing"}:
            return result
        time.sleep(.01)
    raise AssertionError("review did not finish")


def test_status_is_readonly_and_normalizes_closed_minute(tmp_path):
    review = service(tmp_path, analyze=lambda *args: pytest.fail("GET started decoding"))
    before = review.database_path.read_bytes()
    result = review.status("gate", "main", 149)
    assert (result["start_epoch"], result["end_epoch"], result["state"]) == (120, 180, "unreviewed")
    assert review.database_path.read_bytes() == before
    with sqlite3.connect(review.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM reviews").fetchone()[0] == 0
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("epoch", [float("nan"), float("inf"), -1, 1e20])
def test_invalid_or_unclosed_windows(tmp_path, epoch):
    with pytest.raises(ValueError):
        service(tmp_path).request("gate", "main", epoch)


def test_request_deduplicates_and_bounds_queue(tmp_path):
    review = service(tmp_path)
    first = review.request("gate", "main", EPOCH)
    assert review.request("gate", "main", EPOCH + 20)["request_id"] == first["request_id"]
    for index in range(1, module.MAX_PENDING):
        review.request("gate", "main", EPOCH + 60 * index)
    with pytest.raises(RuntimeError, match="queue is full"):
        review.request("gate", "main", EPOCH + 60 * module.MAX_PENDING)


def test_manifest_and_analysis_changes_invalidate_cache(tmp_path):
    current = {"segment_key": "one", "size": 2048}
    identity = ["first"]
    review = service(tmp_path, manifest=lambda *args: [dict(current)], identity=lambda: identity[0])
    first = review.request("gate", "main", EPOCH)
    current["size"] += 1
    assert review.status("gate", "main", EPOCH)["request_id"] != first["request_id"]
    second = review.status("gate", "main", EPOCH)
    identity[0] = "second"
    assert review.status("gate", "main", EPOCH)["request_id"] != second["request_id"]


def test_completed_results_are_reused_without_incident_state(tmp_path):
    calls = []
    def analyze(job, stop):
        calls.append(job["request_id"])
        yield from samples(job, stop)
    review = service(tmp_path, analyze=analyze)
    review.start()
    try:
        review.request("gate", "main", EPOCH)
        result = finished(review)
        assert result["state"] == "sampled"
        assert result["sample_count"] == 12
        assert len(result["observations"]) == 12
        assert result["missing_timestamps"] == []
        assert "private" not in json.dumps(result)
        assert review.request("gate", "main", EPOCH) == result
        assert len(calls) == 1
    finally:
        review.stop()


def test_missing_manifest_never_queues(tmp_path):
    review = service(tmp_path, manifest=lambda *args: [])
    assert review.request("gate", "main", EPOCH)["state"] == "unavailable"
    with sqlite3.connect(review.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM reviews").fetchone()[0] == 0


def test_missing_samples_are_not_claimed_complete(tmp_path):
    def analyze(job, stop):
        for index, timestamp in enumerate(job["sample_targets"]):
            yield {"timestamp": timestamp, "status": "sampled" if index == 0 else "unavailable", "objects": []}
    review = service(tmp_path, analyze=analyze)
    review.start()
    try:
        review.request("gate", "main", EPOCH)
        result = finished(review)
        assert result["state"] == "partial"
        assert result["sample_count"] == 1
        assert len(result["missing_timestamps"]) == 11
    finally:
        review.stop()


def test_restart_does_not_resume_queued_requests(tmp_path):
    review = service(tmp_path)
    review.request("gate", "main", EPOCH)
    restarted = service(tmp_path, analyze=lambda *args: pytest.fail("replayed old work"))
    restarted.start()
    try:
        result = restarted.status("gate", "main", EPOCH)
        assert result["state"] == "failed"
        assert "restart" in result["message"]
    finally:
        restarted.stop()


def test_queued_ttl_is_readonly_and_explicit_request_renews(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    review = service(tmp_path)
    review.request("gate", "main", EPOCH)
    now[0] += 61
    assert review.status("gate", "main", EPOCH)["state"] == "failed"
    assert review.request("gate", "main", EPOCH)["state"] == "queued"
    assert review.status("gate", "main", EPOCH)["state"] == "queued"


def test_running_ttl_stops_after_current_sample(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    def analyze(job, stop):
        yield {"timestamp": job["sample_targets"][0], "status": "sampled", "objects": []}
        now[0] += 61
        yield {"timestamp": job["sample_targets"][1], "status": "sampled", "objects": []}
        pytest.fail("continued after demand expired")
    review = service(tmp_path, analyze=analyze)
    review.start()
    try:
        review.request("gate", "main", EPOCH)
        result = finished(review)
        assert result["state"] == "partial"
        assert result["sample_count"] == 2
    finally:
        review.stop()


def test_shutdown_signals_running_analyzer_and_preserves_partial(tmp_path):
    entered = threading.Event()
    def analyze(job, stop):
        yield {"timestamp": job["sample_targets"][0], "status": "sampled", "objects": []}
        entered.set()
        stop.wait(2)
    review = service(tmp_path, analyze=analyze)
    review.start()
    review.request("gate", "main", EPOCH)
    assert entered.wait(2)
    review.stop()
    assert review.status("gate", "main", EPOCH)["state"] == "partial"


def test_failed_analyzer_does_not_leak_raw_error(tmp_path):
    def analyze(job, stop):
        raise RuntimeError("secret-token-and-stream-url")
    review = service(tmp_path, analyze=analyze)
    review.start()
    try:
        review.request("gate", "main", EPOCH)
        result = finished(review)
        assert result["state"] == "failed"
        assert "secret" not in json.dumps(result)
    finally:
        review.stop()


def test_only_one_worker_can_own_database(tmp_path):
    first = service(tmp_path)
    second = service(tmp_path)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            second.start()
    finally:
        first.stop()
    second.start()
    second.stop()


def test_time_budget_stops_cooperatively(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PROCESSING_BUDGET_SECONDS", -1)
    def analyze(job, stop):
        pytest.fail("started a frame after budget exhaustion")
        yield
    review = service(tmp_path, analyze=analyze)
    review.start()
    try:
        review.request("gate", "main", EPOCH)
        assert finished(review)["state"] == "failed"
    finally:
        review.stop()


def test_expired_queue_does_not_consume_capacity(tmp_path, monkeypatch):
    now = [2000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    review = service(tmp_path)
    for index in range(module.MAX_PENDING):
        review.request("gate", "main", EPOCH + 60 * index)
    now[0] += 61
    assert review.request("gate", "main", EPOCH + 60 * module.MAX_PENDING)["state"] == "queued"


def test_explicit_retry_restarts_partial_but_not_heartbeat(tmp_path):
    calls = []
    def analyze(job, stop):
        calls.append(1)
        yield {"timestamp": job["sample_targets"][0], "status": "sampled", "objects": []}
    review = service(tmp_path, analyze=analyze)
    review.start()
    try:
        review.request("gate", "main", EPOCH)
        assert finished(review)["state"] == "partial"
        assert review.request("gate", "main", EPOCH)["state"] == "partial"
        assert len(calls) == 1
        assert review.request("gate", "main", EPOCH, retry=True)["state"] == "queued"
        assert finished(review)["state"] == "partial"
        assert len(calls) == 2
    finally:
        review.stop()


def test_queued_manifest_change_prevents_analysis(tmp_path):
    manifest = [{"segment_key": "first"}]
    review = service(tmp_path, manifest=lambda *args: list(manifest),
                     analyze=lambda *args: pytest.fail("analyzed stale manifest"))
    payload = review.request("gate", "main", EPOCH)
    manifest[:] = [{"segment_key": "replacement"}]
    # Directly exercise the claimed job so startup recovery is not involved.
    review._process(payload, [{"segment_key": "first"}])
    with sqlite3.connect(review.database_path) as connection:
        stored = json.loads(connection.execute("SELECT payload FROM reviews").fetchone()[0])
    assert stored["state"] == "failed"
    assert "changed" in stored["message"]
    assert review.status("gate", "main", EPOCH)["state"] == "unreviewed"


def test_unowned_stop_cannot_interrupt_other_worker(tmp_path):
    first = service(tmp_path)
    second = service(tmp_path)
    first.start()
    try:
        with pytest.raises(RuntimeError):
            second.start()
        second.stop()
        assert first._thread.is_alive()
    finally:
        first.stop()


def test_changed_identity_heartbeat_never_queues_new_work(tmp_path):
    manifest = [{"segment_key": "first"}]
    review = service(tmp_path, manifest=lambda *args: list(manifest))
    first = review.request("gate", "main", EPOCH)
    manifest[:] = [{"segment_key": "replacement"}]
    result = review.request("gate", "main", EPOCH, expected_request_id=first["request_id"])
    assert result["state"] == "unreviewed"
    with sqlite3.connect(review.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM reviews").fetchone()[0] == 1


def test_busy_resources_are_not_reported_as_missing_footage(tmp_path):
    def analyze(job, stop):
        for timestamp in job["sample_targets"]:
            yield {"timestamp": timestamp, "status": "deferred", "objects": []}
    review = service(tmp_path, analyze=analyze)
    review.start()
    try:
        review.request("gate", "main", EPOCH)
        result = finished(review)
        assert result["state"] == "failed"
        assert result["has_recordings"] is True
        assert "resources were busy" in result["message"]
    finally:
        review.stop()


def test_shutdown_timeout_raises_and_retains_worker_ownership(tmp_path):
    review = service(tmp_path)
    ownership = object()
    review._worker_lock_file = ownership
    review._thread = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True)
    with pytest.raises(RuntimeError, match="shared resources must remain available"):
        review.stop()
    assert review._worker_lock_file is ownership


@pytest.mark.parametrize("completed_samples", [0, 1])
def test_policy_change_discards_sample_from_new_generation(tmp_path, completed_samples):
    identity = ["old-policy"]
    def analyze(job, stop):
        if completed_samples:
            yield {"timestamp": job["sample_targets"][0], "status": "sampled",
                   "objects": [{"label": "old-policy-result", "confidence": .9}]}
        identity[0] = "new-policy"
        yield {"timestamp": job["sample_targets"][completed_samples], "status": "sampled",
               "objects": [{"label": "new-policy-result", "confidence": .9}]}
        pytest.fail("continued analysis after policy change")
    review = service(tmp_path, analyze=analyze, identity=lambda: identity[0])
    payload = review.request("gate", "main", EPOCH)
    review._process(payload, review.manifest_provider("gate", "main", 120, 180))
    with sqlite3.connect(review.database_path) as connection:
        stored = json.loads(connection.execute("SELECT payload FROM reviews").fetchone()[0])
    assert stored["state"] == ("partial" if completed_samples else "failed")
    assert stored["sample_count"] == completed_samples
    assert "settings changed" in stored["message"]
    assert not any(item["label"] == "new-policy-result" for item in stored["observations"])


def test_shutdown_closes_request_admission(tmp_path):
    review = service(tmp_path)
    review.start()
    review.stop()
    with pytest.raises(RuntimeError, match="stopping"):
        review.request("gate", "main", EPOCH)


def test_expiry_during_last_sample_never_reports_completed(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    def analyze(job, stop):
        for index, timestamp in enumerate(job["sample_targets"]):
            if index == 11:
                now[0] += 61
            yield {"timestamp": timestamp, "status": "sampled", "objects": []}
    review = service(tmp_path, analyze=analyze)
    review.start()
    try:
        review.request("gate", "main", EPOCH)
        result = finished(review)
        assert result["sample_count"] == 12
        assert result["state"] == "partial"
        assert "expired" in result["message"]
    finally:
        review.stop()


def test_transient_final_save_failure_does_not_leave_running_job(tmp_path, monkeypatch):
    calls = []
    def analyze(job, stop):
        calls.append(1)
        yield from samples(job, stop)
    review = service(tmp_path, analyze=analyze)
    original_save = review._save
    failures = []
    def save(connection, payload, now):
        if payload["state"] == "sampled" and not failures:
            failures.append(1)
            raise sqlite3.OperationalError("temporary database failure")
        return original_save(connection, payload, now)
    monkeypatch.setattr(review, "_save", save)
    review.start()
    try:
        review.request("gate", "main", EPOCH)
        result = finished(review)
        assert result["state"] == "partial"
        assert result["sample_count"] == 12
        assert "saving results" in result["message"]
        assert len(calls) == 1
    finally:
        review.stop()
