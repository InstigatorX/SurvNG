"""Read-only native pipeline observations; no image upload inference path."""
from fastapi import APIRouter, HTTPException


def create_native_router(get_manager):
    router = APIRouter()

    @router.get("/api/cameras/{camera_id}/native")
    def native_observations(camera_id: str):
        worker = get_manager().workers.get(camera_id)
        if worker is None:
            raise HTTPException(404, "camera not found")
        status = worker.status()
        return {"camera_id": camera_id, "activity": status["native_activity"],
                "pipeline": status.get("live_pipeline", {}),
                "objects": status.get("live_detections", [])}

    @router.get("/api/events/{event_id}/native-evidence")
    def native_evidence(event_id: int):
        events = get_manager().events
        event = events.get(event_id)
        if event is None:
            raise HTTPException(404, "event not found")
        return {"event_id": event_id, "camera_id": event["camera_id"],
                "attempts": [attempt for attempt in events.evidence_attempts(event_id)
                             if attempt.get("source") == "native_cover"]}

    return router
