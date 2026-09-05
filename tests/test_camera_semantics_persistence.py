from datetime import datetime, timezone

from survng.app.events import EventStore
from survng.app.motion_incidents import _RefinementJob, _compact_refinement_qualification
from survng.app.motion_pipeline.decision_handler import MotionDecisionOutcome


def test_refinement_recovery_preserves_camera_claim_separately_from_model_result(tmp_path):
    semantics = {"reports": [{
        "topic": "RuleEngine/MyRuleDetector/DogCatDetect",
        "category": "animal",
        "candidate_model_classes": ["dog", "cat"],
    }]}
    # The camera reported an animal; visual inference found a person instead.
    detected = {"label": "person", "confidence": 0.91, "box": [1, 2, 30, 40]}
    job = _RefinementJob(
        topic="RuleEngine/MyRuleDetector/DogCatDetect",
        message="",
        event_at=datetime.now(timezone.utc),
        qualification=_compact_refinement_qualification({"camera_semantics": semantics}),
        existing_event_id=7,
        require_eligible_object=True,
        require_motion_correlation=False,
        callback=None,
        completion_context=None,
        initial_outcome=MotionDecisionOutcome(
            event_id=7, snapshot_path="", object_detected=True,
            detected_objects=(detected,),
        ),
    )
    store = EventStore(tmp_path)
    assert store.enqueue_detection_job(
        job_id=job.job_id("gate"), camera_id="gate",
        dedupe_key=job.dedupe_key(), payload=job.payload(),
    ) == "queued"
    claimed = store.claim_detection_job("gate", lease_owner="first-worker")
    assert claimed is not None
    with store._connect_jobs() as connection:
        connection.execute(
            "update detection_jobs set lease_expires_at = 0 where id = ?",
            (claimed["id"],),
        )

    reopened = EventStore(tmp_path)
    recovered = reopened.claim_detection_job("gate", lease_owner="replacement-worker")
    assert recovered is not None
    restored = _RefinementJob.from_payload(recovered["payload"], None)
    assert restored.qualification["camera_semantics"] == semantics
    assert restored.initial_outcome.detected_objects == (detected,)
    assert recovered["attempts"] == 2
