"""Stable source identities shared by acquisition and scene projection."""
import hashlib
import json


def observation_identity(camera_id, captured_epoch, observation):
    box = observation.get("box")
    if isinstance(box, dict):
        box = {key: float(value) for key, value in box.items()}
    identity = [camera_id, float(captured_epoch), observation.get("frame_source", "legacy"),
                observation.get("label"), box, observation.get("model_version", "")]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "observation-" + hashlib.sha256(encoded.encode()).hexdigest()[:32]


def analysis_observation_ids(camera_id, captured_epoch, observations):
    return sorted({observation_identity(camera_id, item.get("captured_at_epoch",
                   item.get("frame_captured_at_epoch", captured_epoch)), item)
                   for item in observations})
