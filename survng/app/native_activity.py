"""Native observations own admission and activity; no pixels or model calls.

A native track ID is scoped to a capture session. Only fresh, zone-eligible
observations confirm presence. Predictions may move overlays but cannot create
an event, increment evidence counts, or extend the activity deadline.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import time
from typing import Callable

from .live_detections import DetectionSnapshot
from .native_motion import NativeMotion
from .zones import apply_detection_zones


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def compact_history(samples, limit=1024):
    """Bound memory while preserving the entire episode, including its start."""
    if len(samples) <= limit:
        return samples
    # Preserve recent detail and thin older samples, rather than sliding away
    # the beginning of a long incident.
    split = len(samples) // 2
    return samples[:split:2] + samples[split:]


class NativeActivity:
    def __init__(self, camera, config, events, publish: Callable, snapshot: Callable):
        self.camera, self.config = camera, config
        self.events, self.publish, self.snapshot = events, publish, snapshot
        self.session = ""
        self.identity_epoch = 0
        self.sequence = 0
        self.last_pts = -1.0
        self.last_received = 0.0
        self.last_fresh = 0.0
        self.last_activity = 0.0
        self.last_persist = 0.0
        self.offer_evidence = None
        self.event_id = None
        self.tracks = {}
        self._episode_tracks = {}
        self.counts = Counter()
        self.last_motion_at = ""
        self.health = "waiting_for_metadata"
        self.dimensions = (0, 0)
        self._next_track_id = 1
        self._fresh_times = deque(maxlen=100)

    def consume(self, observation: DetectionSnapshot, *, now: float, epoch: float):
        if not observation.session:
            self.counts["invalid_session"] += 1
            return
        if observation.session != self.session:
            self.finish("stream_reset", now=now)
            self.session = observation.session
            self.identity_epoch = 0
            self.dimensions = (0, 0)
            self.sequence = 0
            self.last_pts = -1.0
        if observation.inference_sequence <= self.sequence or observation.source_pts <= self.last_pts:
            return
        self.sequence = observation.inference_sequence
        self.last_pts = observation.source_pts
        if now - observation.received_monotonic > self.config.native.maximum_observation_age_seconds:
            self.counts["stale_observations"] += 1
            return
        self.last_received = now
        if observation.provenance != "native_fresh_detection":
            self.counts["prediction_frames" if observation.provenance == "native_tracked_prediction" else "unknown_frames"] += 1
            return
        dimensions = (observation.width, observation.height)
        if self.dimensions != (0, 0) and self.dimensions != dimensions:
            self.finish("geometry_changed", now=now)
            self.identity_epoch += 1
            self.counts["geometry_resets"] += 1
        self.last_fresh = now
        self.dimensions = dimensions
        self.health = "healthy"
        self.counts["fresh_frames"] += 1
        self._fresh_times.append(now)
        objects = apply_detection_zones(
            self.camera, observation.scaled_objects(observation.width, observation.height),
            observation.width, observation.height, self.config.confidence_threshold,
            self.config.require_incident_zone, self.config.event_class_confidence_thresholds,
        )
        eligible = [obj for obj in objects
                    if obj.get("detection_provenance") == "native_fresh_detection"
                    and obj.get("incident_eligible")]
        # Live context expires independently of the persisted episode archive.
        for key, track in list(self.tracks.items()):
            if now - track["last_monotonic"] >= max(self.config.native.activity_timeout_seconds, 1.5 / self.fresh_detection_fps):
                del self.tracks[key]
        seen = set()
        for obj in eligible:
            native_id = obj.get("native_track_id")
            if type(native_id) is not int or native_id < 0:
                self.counts["missing_track_id"] += 1
                continue
            key = (native_id, obj["label"])
            if key in seen:
                self.counts["duplicate_track_id"] += 1
                continue
            seen.add(key)
            track = self.tracks.get(key)
            if track is None:
                if len(self.tracks) >= self.config.native.maximum_tracks:
                    self.counts["track_capacity_drops"] += 1
                    continue
                track = {"track_id": self._next_track_id,
                         "native_track_id": native_id,
                         "native_identity": f"{self.camera.id}/{self.session}/{self.identity_epoch}/{native_id}",
                         "label": obj["label"], "first_seen": iso(epoch),
                         "observations": 0, "consecutive": 0, "last_monotonic": now,
                         "box_history": [], "trajectory": [], "_motion": NativeMotion()}
                self.tracks[key] = track
                self._next_track_id += 1
            if now - track["last_monotonic"] > max(self.config.native.maximum_observation_age_seconds, 3 / self.fresh_detection_fps):
                track["consecutive"] = 0
            track.update(last_monotonic=now, last_seen=iso(epoch),
                         box=deepcopy(obj["box"]), confidence=obj["confidence"],
                         zones=obj.get("zones", []), incident_eligible=True)
            track["observations"] += 1
            track["consecutive"] += 1
            required = self.config.event_class_confirmation_frames.get(obj["label"], self.config.event_confirmation_frames)
            track["state"] = "confirmed" if track["consecutive"] >= required else "tentative"
            box = obj["box"]
            track["box_history"].append([epoch, box["x1"], box["y1"], box["x2"], box["y2"]])
            track["trajectory"].append([epoch, (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2])
            track["max_confidence"] = max(track.get("max_confidence", 0), obj["confidence"])
            track["confirmed"] = track.get("confirmed", False) or track["state"] == "confirmed"
            policy = self.config.native.stationary
            previous_motion = track.get("motion_state", "uncertain")
            applies = policy.enabled and obj["label"].lower() in policy.labels
            track["motion_state"] = (track["_motion"].update(
                box, observation.source_pts, policy,
                max(self.config.native.maximum_observation_age_seconds, 3 / self.fresh_detection_fps),
            ) if applies else "presence")
            track["motion_extent"] = round(track["_motion"].extent, 4) if applies else None
            track["activity_eligible"] = track["motion_state"] in {"moving", "presence"}
            if applies and track["motion_state"] != previous_motion:
                self.counts[f"{track['motion_state']}_transitions"] += 1
            if applies and not track["activity_eligible"]:
                self.counts[f"{track['motion_state']}_vehicle_observations"] += 1
            if key in self._episode_tracks:
                self._episode_tracks[key].update({field: track[field] for field in
                                                  ("motion_state", "motion_extent", "activity_eligible")})
            del track["box_history"][:-150]
            del track["trajectory"][:-150]
        for key, track in list(self.tracks.items()):
            if key not in seen:
                track["consecutive"] = 0
        if eligible and not seen:
            self.health = "tracking_unavailable"
        confirmed = [track for key, track in self.tracks.items() if key in seen and track["state"] == "confirmed" and track["activity_eligible"]]
        if confirmed:
            for track in confirmed:
                self._record_activity_track(track, epoch)
            self.last_activity = now
            self.last_motion_at = iso(epoch)
            if self.event_id is None:
                stored = self._objects(self._episode_tracks.values())
                path = self.snapshot(observation, epoch)
                event = self.events.add_event(
                    camera_id=self.camera.id, kind="motion", topic="native/object-presence",
                    message="Confirmed native object presence", created_at=iso(epoch),
                    snapshot_path=path, objects_json=json.dumps(stored),
                    detection_intent_id=f"native:{self.camera.id}:{self.session}:{self.sequence}",
                )
                self.event_id = int(event["id"])
                self.counts["events_created"] += 1
                self.publish("incident", {"camera_id": self.camera.id, "event_id": self.event_id, "timestamp": iso(epoch), "kind": "motion"})
                self.publish("object", {"camera_id": self.camera.id, "event_id": self.event_id,
                             "timestamp": iso(epoch), "objects": stored, "source": "native",
                             "snapshot_path": path})
                self.persist("active", now=now)
            elif now - self.last_persist >= 1.0:
                self.persist("active", now=now)
            if self.event_id is not None and self.offer_evidence is not None:
                self.offer_evidence(observation, epoch, self.event_id, self._objects(confirmed))
        self.tick(now=now)

    @staticmethod
    def _objects(tracks):
        return [{**{k: deepcopy(v) for k, v in track.items()
                   if not k.startswith("_") and k not in {"last_monotonic", "consecutive", "box_history", "trajectory"}},
                 "track_state": track["state"], "track_observations": track["observations"],
                 "detection_provenance": "native_fresh_detection"} for track in tracks]

    def _record_activity_track(self, track, epoch):
        key = (track["native_track_id"], track["label"])
        previous = self._episode_tracks.get(key)
        if previous is None and len(self._episode_tracks) >= self.config.native.maximum_tracks:
            self.counts["episode_track_capacity_drops"] += 1
            return
        stored = {k: deepcopy(v) for k, v in track.items()
                  if not k.startswith("_") and k not in {"last_monotonic", "consecutive", "box_history", "trajectory"}}
        box_history = list(previous["box_history"]) if previous else []
        trajectory = list(previous["trajectory"]) if previous else []
        box_history.append(deepcopy(track["box_history"][-1]))
        trajectory.append(deepcopy(track["trajectory"][-1]))
        stored.update(first_seen=previous["first_seen"] if previous else iso(epoch),
                      observations=previous["observations"] + 1 if previous else 1,
                      max_confidence=max(previous["max_confidence"], track["confidence"]) if previous else track["confidence"],
                      box_history=compact_history(box_history), trajectory=compact_history(trajectory))
        self._episode_tracks[key] = stored

    def persist(self, state: str, *, now: float):
        if self.event_id is None:
            return
        tracks = [{k: deepcopy(v) for k, v in track.items()
                   if k not in {"last_monotonic", "consecutive"}} for track in self._episode_tracks.values()]
        for track in tracks:
            track["duration_seconds"] = max(0, datetime.fromisoformat(track["last_seen"]).timestamp() - datetime.fromisoformat(track["first_seen"]).timestamp())
        payload = {"implementation": "gvatrack", "state": state,
                   "sample_fps": self.fresh_detection_fps,
                   "tracks": tracks, "updated_at": self.last_motion_at,
                   "frame_width": self.dimensions[0], "frame_height": self.dimensions[1],
                   "source": "live", "native_session": self.session,
                   "recording_overlay_compatible": self.camera.native_same_field_of_view or self.camera.live_url() == self.camera.stream_url}
        self.events.update_object_tracking(self.event_id, payload, self._objects(self._episode_tracks.values()))
        self.publish("object_tracking", {"camera_id": self.camera.id, "event_id": self.event_id, **payload})
        if state == "active":
            self.publish("incident", {"camera_id": self.camera.id, "event_id": self.event_id,
                                      "timestamp": self.last_motion_at, "updated": True})
        self.last_persist = now

    @property
    def fresh_detection_fps(self):
        return self.config.live_sample_fps / self.config.native.inference_interval

    def tick(self, *, now: float):
        if self.last_fresh and now - self.last_fresh > max(self.config.native.maximum_observation_age_seconds, 1.5 / self.fresh_detection_fps):
            self.health = "metadata_stale"
        if self.event_id is not None and now - self.last_activity >= self.config.native.activity_timeout_seconds:
            if self.health != "healthy":
                self.finish("metadata_lost", now=now)
            elif self.last_fresh >= self.last_activity + self.config.native.activity_timeout_seconds:
                self.finish("complete", now=now)
            # The timer can cross the deadline between healthy metadata frames.
            # Wait for a fresh observation covering it, or for metadata to become
            # stale above. A normal inter-frame gap is not loss of coverage.

    def finish(self, reason: str, *, now: float):
        self.persist(reason, now=now)
        self.event_id = None
        self._episode_tracks.clear()
        if reason != "complete":
            self.tracks.clear()
            self._next_track_id = 1
        self.last_activity = 0.0

    def status(self):
        elapsed = self._fresh_times[-1] - self._fresh_times[0] if len(self._fresh_times) > 1 else 0
        return {"implementation": "gvatrack", "enabled": self.config.enabled,
                "active": self.event_id is not None, "event_id": self.event_id,
                "health": self.health, "native_session": self.session,
                "effective_fresh_fps": (len(self._fresh_times) - 1) / elapsed if elapsed > 0 else 0,
                "last_fresh_age_seconds": max(0, time.monotonic() - self.last_fresh) if self.last_fresh else None,
                "motion_states": dict(Counter(track.get("motion_state", "uncertain") for track in self.tracks.values())),
                "tracks": self._objects(self.tracks.values()), "counters": dict(self.counts)}
