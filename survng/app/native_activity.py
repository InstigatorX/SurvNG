"""Native observations own admission and activity; no pixels or model calls.

NativeObjectRegistry owns identity, confirmation, and object history once.
Activity policy decides whether confirmed objects start or extend an incident;
NativeIncidentInventory independently records what credible objects were present.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from datetime import datetime
import json
import time
import uuid
from typing import Callable

from .live_detections import DetectionSnapshot
from .native_motion import NativeMotion
from .native_objects import NativeIncidentInventory, NativeObjectRegistry, compact_history, iso
from .zones import apply_detection_zones
from survng.native_spatial import spatial_plan


class NativeActivity:
    def __init__(self, camera, config, events, publish: Callable, snapshot: Callable, *, native_zones=False):
        self.camera, self.config = camera, config
        self.native_zones = native_zones
        self.zone_revision = spatial_plan(camera)["revision"]
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
        self.nominate = None
        self.admission = None
        self.verified_snapshot = None
        self._verification_pending = {}
        self._verification_recent = deque(maxlen=16)
        self.event_id = None

        # Single identity/history owner plus two independent consumers:
        # activity policy and episode inventory.
        self.registry = NativeObjectRegistry(camera.id, config)
        self.inventory = NativeIncidentInventory(config.native.maximum_tracks)
        self._activity_states = {}
        self._seen_keys = set()

        self.counts = Counter()
        self.last_motion_at = ""
        self.health = "waiting_for_metadata"
        self.dimensions = (0, 0)
        self._fresh_times = deque(maxlen=100)

    @property
    def tracks(self):
        """Compatibility/status view of objects participating in activity policy."""
        result = {}
        for key in self._activity_states:
            obj = self._activity_object(key)
            if obj is None:
                continue
            native_id = obj.get("native_track_id")
            public_key = (
                (native_id, obj["label"])
                if type(native_id) is int and native_id >= 0
                else (obj["track_id"], obj["label"])
            )
            result[public_key] = obj
        return result

    def _activity_state(self, key):
        state = self._activity_states.get(key)
        if state is None:
            state = {
                "_motion": NativeMotion(),
                "motion_state": "uncertain",
                "motion_extent": 0.0,
                "activity_eligible": False,
            }
            self._activity_states[key] = state
        return state

    @staticmethod
    def _public_activity_state(state):
        result = {
            "motion_state": state.get("motion_state", "uncertain"),
            "motion_extent": state.get("motion_extent"),
            "activity_eligible": bool(state.get("activity_eligible")),
        }
        if state.get("verification") is not None:
            result["verification"] = deepcopy(state["verification"])
        return result

    def _activity_object(self, key, *, include_history=True):
        obj = self.registry.export(key, include_history=include_history)
        if obj is None:
            return None
        state = self._activity_states.get(key)
        if state is not None:
            obj.update(self._public_activity_state(state))
        else:
            obj.update(motion_state="context", motion_extent=None, activity_eligible=False)
        return obj

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
            self.counts[
                "prediction_frames"
                if observation.provenance == "native_tracked_prediction"
                else "unknown_frames"
            ] += 1
            return

        if self.native_zones:
            revision = self.zone_revision
            valid_ids = {
                str(i)
                for i, zone in enumerate(self.camera.zones)
                if zone.enabled and len(zone.points) >= 3
            }
            if observation.zone_revision != revision or any(
                (
                    type(obj.get("native_track_id")) is int
                    and obj.get("native_track_id") >= 0
                )
                and (
                    obj.get("native_zone_revision") != revision
                    or not isinstance(obj.get("native_zone_ids"), list)
                    or any(
                        not isinstance(zone_id, str) or zone_id not in valid_ids
                        for zone_id in obj.get("native_zone_ids", [])
                    )
                )
                for obj in observation.objects
            ):
                self.counts["invalid_zone_metadata"] += 1
                self.health = "zone_metadata_invalid"
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
            self.camera,
            observation.scaled_objects(observation.width, observation.height),
            observation.width,
            observation.height,
            self.config.confidence_threshold,
            self.config.require_incident_zone,
            self.config.event_class_confidence_thresholds,
            native_membership=self.native_zones,
        )

        self.registry.config = self.config
        self._seen_keys = self.registry.observe(
            objects,
            session=self.session,
            identity_epoch=self.identity_epoch,
            epoch=epoch,
            now=now,
            dimensions=self.dimensions,
            fresh_fps=self.fresh_detection_fps,
        )
        for key in list(self._activity_states):
            if key not in self.registry.tracks:
                del self._activity_states[key]

        confirmed_activity = []
        for key in self._seen_keys:
            track = self.registry.get(key)
            if track is None or not track.get("incident_eligible"):
                continue
            native_id = track.get("native_track_id")
            if type(native_id) is not int or native_id < 0:
                # Fallback association is inventory-only context. Activity
                # admission still requires authoritative native tracker identity.
                self.counts["missing_track_id"] += 1
                continue
            label = str(track.get("label") or "").strip().lower()
            selected = self.config.native.tracking_classes
            if selected is not None and label not in selected:
                continue

            state = self._activity_state(key)
            previous_motion = state.get("motion_state", "uncertain")
            policy = self.config.native.stationary
            applies = policy.enabled and label in policy.labels
            if applies:
                motion_state = state["_motion"].update(
                    track["box"],
                    observation.source_pts,
                    policy,
                    max(
                        self.config.native.maximum_observation_age_seconds,
                        3 / self.fresh_detection_fps,
                    ),
                )
                motion_extent = round(state["_motion"].extent, 4)
            else:
                motion_state, motion_extent = "presence", None
            state.update(
                motion_state=motion_state,
                motion_extent=motion_extent,
                activity_eligible=motion_state in {"moving", "presence"},
            )
            if applies and motion_state != previous_motion:
                self.counts[f"{motion_state}_transitions"] += 1
            if applies and not state["activity_eligible"]:
                self.counts[f"{motion_state}_vehicle_observations"] += 1

            if track.get("state") == "confirmed" and state["activity_eligible"]:
                confirmed_activity.append(key)

        if self.admission is not None and self.config.native.verification_enabled:
            confirmed_activity = self._gate(
                confirmed_activity, observation, epoch, now
            )
        if confirmed_activity:
            self._activate(confirmed_activity, observation, epoch, now)

        if self.event_id is not None:
            self._capture_inventory()
            if now - self.last_persist >= 1.0:
                self.persist("active", now=now)

        self.tick(now=now)

    def _gate(self, confirmed_keys, observation, epoch, now):
        allowed = []
        for key in confirmed_keys:
            track = self.registry.get(key)
            if track is None:
                continue
            state = self._activity_state(key)
            verification = state.get("_verification")
            if verification == "confirmed":
                allowed.append(key)
                continue
            if verification in {"rejected", "unverified"}:
                before = state.get("_verification_box", track["box"])
                box = track["box"]
                scale = max(
                    1,
                    before["x2"] - before["x1"],
                    before["y2"] - before["y1"],
                )
                moved = max(
                    abs((box["x1"] + box["x2"] - before["x1"] - before["x2"]) / 2),
                    abs((box["y1"] + box["y2"] - before["y1"] - before["y2"]) / 2),
                ) > scale * .5
                if not moved and (
                    verification != "unverified"
                    or now < state.get("_verification_retry_at", now)
                ):
                    continue
                state.pop("_verification_token", None)
                state.pop("_verification", None)

            token = state.get("_verification_token")
            if token is None:
                if len(self._verification_pending) >= 32:
                    self.counts["verification_capacity_drops"] += 1
                    continue
                token = state["_verification_token"] = uuid.uuid4().hex
                state["_verification"] = "pending"

            activity = self._public_activity_state(state)
            pending = self._verification_pending.setdefault(token, {"started": now})
            pending.update(
                key=key,
                track=self._activity_object(key),
                activity=activity,
                observation=observation,
                epoch=epoch,
                source_monotonic=track["last_monotonic"],
            )
            self.nominate(
                token,
                observation,
                epoch,
                self._activity_object(key, include_history=False),
            )
        return allowed

    def _poll_verification(self, now):
        if self.admission is None:
            return
        pending_items = sorted(
            self._verification_pending.items(),
            key=lambda item: item[1]["track"]["first_seen"],
        )
        for token, pending in pending_items:
            result = self.admission.poll(token)
            if result is None and now - pending["started"] < 150:
                break
            if result is None:
                self.admission.cancel(token)
                result = {"status": "unverified", "reason": "deadline"}
            del self._verification_pending[token]

            status = result["status"]
            track = pending["track"]
            key = pending["key"]
            self.counts["verification_" + status] += 1
            self._verification_recent.append(
                {
                    "label": track["label"],
                    "status": status,
                    "reason": result.get("reason", ""),
                    "epoch": pending["epoch"],
                    "timestamp": iso(pending["epoch"]),
                    "votes": result.get("votes", []),
                    "checks": result.get("checks", []),
                }
            )

            state = self._activity_states.get(key)
            if state is not None and state.get("_verification_token") == token:
                state["_verification"] = status
                state["_verification_box"] = deepcopy(track["box"])
                state["_verification_retry_at"] = now + 30
                state["verification"] = {
                    name: value for name, value in result.items() if name != "cover"
                }

            if status != "confirmed":
                continue

            pending["activity"]["verification"] = {
                name: value for name, value in result.items() if name != "cover"
            }
            if self.event_id is not None and not self._joins_episode(track):
                self.finish("complete", now=now)
            self._activate(
                [],
                pending["observation"],
                pending["epoch"],
                pending["source_monotonic"],
                cover=result.get("cover"),
                seed=(track, pending["activity"]),
            )

    def _capture_inventory(self):
        self.inventory.capture(
            self.registry,
            self._seen_keys,
            {
                key: self._public_activity_state(state)
                for key, state in self._activity_states.items()
            },
        )

    def _visible_evidence_objects(self):
        values = []
        for key in self.registry.confirmed(self._seen_keys):
            obj = self.registry.export(key, include_history=False)
            if obj is None:
                continue
            state = self._activity_states.get(key)
            if state is not None:
                obj.update(self._public_activity_state(state))
            else:
                obj.update(
                    motion_state="context",
                    motion_extent=None,
                    activity_eligible=False,
                )
            values.append(obj)
        values.sort(
            key=lambda item: (
                bool(item.get("activity_eligible")),
                bool(item.get("incident_eligible")),
                float(item.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        return values

    def _activate(
        self,
        confirmed_keys,
        observation,
        epoch,
        now,
        cover=None,
        seed=None,
    ):
        if self.event_id is None:
            # Episode inventory begins at activity onset. A delayed admission
            # result retains the nominated object's original first observation,
            # but ordinary reactivation never imports stale history from a
            # previous completed incident.
            trigger_epoch = epoch
            if seed is not None:
                try:
                    trigger_epoch = min(
                        trigger_epoch,
                        datetime.fromisoformat(seed[0]["first_seen"]).timestamp(),
                    )
                except (KeyError, TypeError, ValueError):
                    pass
            self.inventory.begin(trigger_epoch, 0)
        if seed is not None:
            self.inventory.record(seed[0], seed[1])
        for key in confirmed_keys:
            track = self.registry.export(key)
            if track is not None:
                self.inventory.record(
                    track,
                    self._public_activity_state(self._activity_state(key)),
                )
        self._capture_inventory()

        self.last_activity = max(self.last_activity, now)
        self.last_motion_at = max(self.last_motion_at, iso(epoch))

        if self.event_id is None:
            visible_ids = {
                self.registry.get(key)["track_id"]
                for key in self.registry.confirmed(self._seen_keys)
                if self.registry.get(key) is not None
            }
            stored = self.inventory.objects(visible_track_ids=visible_ids)
            path = (
                self.verified_snapshot(cover)
                if cover is not None
                else self.snapshot(observation, epoch)
            )
            if cover is not None:
                stored = self.inventory.objects(cover=cover[1])
            created_epoch = (
                self.inventory.first_seen_epoch()
                if cover is not None
                else None
            )
            event = self.events.add_event(
                camera_id=self.camera.id,
                kind="motion",
                topic="native/object-presence",
                message="Confirmed native object presence",
                created_at=iso(created_epoch if created_epoch is not None else epoch),
                snapshot_path=path,
                objects_json=json.dumps(stored),
                detection_intent_id=(
                    f"native:{self.camera.id}:{self.session}:{self.sequence}"
                    + (f":{uuid.uuid4().hex}" if cover is not None else "")
                ),
            )
            self.event_id = int(event["id"])
            self.counts["events_created"] += 1
            self.publish(
                "incident",
                {
                    "camera_id": self.camera.id,
                    "event_id": self.event_id,
                    "timestamp": iso(epoch),
                    "kind": "motion",
                },
            )
            self.publish(
                "object",
                {
                    "camera_id": self.camera.id,
                    "event_id": self.event_id,
                    "timestamp": iso(epoch),
                    "objects": stored,
                    "source": "native",
                    "snapshot_path": path,
                },
            )
            self.persist("active", now=now)

        if self.event_id is not None and self.offer_evidence is not None:
            evidence_objects = self._visible_evidence_objects()
            if evidence_objects:
                self.offer_evidence(
                    observation,
                    epoch,
                    self.event_id,
                    evidence_objects,
                )

    def persist(self, state: str, *, now: float):
        if self.event_id is None:
            return
        tracks = self.inventory.tracking_tracks()
        payload = {
            "implementation": "gvatrack",
            "state": state,
            "sample_fps": self.fresh_detection_fps,
            "tracks": tracks,
            "updated_at": self.last_motion_at,
            "frame_width": self.dimensions[0],
            "frame_height": self.dimensions[1],
            "source": "live",
            "native_session": self.session,
            "recording_overlay_compatible": (
                self.camera.native_same_field_of_view
                or self.camera.live_url() == self.camera.stream_url
            ),
        }
        self.events.update_native_incident_state(
            self.event_id,
            payload,
            self.inventory.objects(),
        )
        self.publish(
            "object_tracking",
            {
                "camera_id": self.camera.id,
                "event_id": self.event_id,
                **payload,
            },
        )
        if state == "active":
            self.publish(
                "incident",
                {
                    "camera_id": self.camera.id,
                    "event_id": self.event_id,
                    "timestamp": self.last_motion_at,
                    "updated": True,
                },
            )
        self.last_persist = now

    @property
    def fresh_detection_fps(self):
        defaults, overrides = self.config.native.budget, self.camera.native_budget
        enabled = defaults.enabled if overrides.enabled is None else overrides.enabled
        idle_fps = (
            defaults.idle_fps
            if overrides.idle_fps is None
            else overrides.idle_fps
        )
        return (
            idle_fps
            if enabled
            else self.config.live_sample_fps / self.config.native.inference_interval
        )

    def _joins_episode(self, track):
        """Group by observed activity, independent of verification completion."""
        tracks = self.inventory.tracking_tracks()
        if not tracks:
            return False
        start = min(
            datetime.fromisoformat(item["first_seen"]).timestamp()
            for item in tracks
        )
        end = max(
            datetime.fromisoformat(item["last_seen"]).timestamp()
            for item in tracks
        )
        timeout = self.config.native.activity_timeout_seconds
        return (
            datetime.fromisoformat(track["first_seen"]).timestamp() <= end + timeout
            and datetime.fromisoformat(track["last_seen"]).timestamp() >= start - timeout
        )

    def tick(self, *, now: float):
        self._poll_verification(now)
        if self.last_fresh and now - self.last_fresh > max(
            self.config.native.maximum_observation_age_seconds,
            (self.config.native.batch_size + .5) / self.fresh_detection_fps,
        ):
            self.health = "metadata_stale"
        if (
            self.event_id is not None
            and now - self.last_activity
            >= self.config.native.activity_timeout_seconds
        ):
            if self.health != "healthy":
                self.finish("metadata_lost", now=now)
            elif (
                self.last_fresh
                >= self.last_activity + self.config.native.activity_timeout_seconds
                and not (
                    now - self.last_activity < 150
                    and any(
                        now - pending["started"] < 150
                        and self._joins_episode(pending["track"])
                        for pending in self._verification_pending.values()
                    )
                )
            ):
                self.finish("complete", now=now)

    def finish(self, reason: str, *, now: float):
        self.persist(reason, now=now)
        self.event_id = None
        self.inventory.clear()
        if reason != "complete":
            if self.admission is not None:
                for token in self._verification_pending:
                    self.admission.cancel(token)
            self._verification_pending.clear()
            self.registry.reset()
            self._activity_states.clear()
            self._seen_keys.clear()
        self.last_activity = 0.0

    def status(self):
        elapsed = (
            self._fresh_times[-1] - self._fresh_times[0]
            if len(self._fresh_times) > 1
            else 0
        )
        counters = Counter(self.counts)
        counters.update(self.registry.counts)
        public_tracks = self.tracks
        return {
            "implementation": "gvatrack",
            "enabled": self.config.enabled,
            "active": self.event_id is not None,
            "event_id": self.event_id,
            "health": self.health,
            "native_session": self.session,
            "effective_fresh_fps": (
                (len(self._fresh_times) - 1) / elapsed if elapsed > 0 else 0
            ),
            "last_fresh_age_seconds": (
                max(0, time.monotonic() - self.last_fresh)
                if self.last_fresh
                else None
            ),
            "motion_states": dict(
                Counter(
                    track.get("motion_state", "uncertain")
                    for track in public_tracks.values()
                )
            ),
            "verification_pending": len(self._verification_pending),
            "verification_recent": list(self._verification_recent),
            "inventory_count": len(self.registry.tracks),
            "tracks": list(public_tracks.values()),
            "counters": dict(counters),
        }
