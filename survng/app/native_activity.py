"""Native observations own admission and activity; no pixels or model calls.

NativeObjectRegistry owns soft spatial/temporal association and confirmation.
Activity policy decides whether confirmed objects start or extend an incident;
NativeIncidentInventory records multi-object participants for the open time range.
Track IDs never admit, extend, or split incidents.
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


_INTERRUPTED_REASONS = frozenset({
    "metadata_lost",
    "stream_reset",
    "geometry_changed",
    "zones_changed",
    "policy_changed",
    "stopped",
})


def terminal_state(reason: str) -> tuple[str, str]:
    """Map lifecycle causes onto a small durable state vocabulary."""
    normalized = str(reason or "").strip().lower()
    if normalized in {"complete", "inactivity"}:
        return "complete", "inactivity"
    if normalized == "failed":
        return "failed", "failed"
    if normalized in _INTERRUPTED_REASONS:
        return "interrupted", normalized
    return "interrupted", normalized or "unknown"


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
        self.route_watch_match = None
        self.consume_route_watch = None
        self._expected_handoffs = {}
        self._verification_pending = {}
        self._verification_recent = deque(maxlen=16)
        self.event_id = None
        self.incident_id = None
        self._observation_seq = 0

        # Soft association/history owner plus two independent consumers:
        # activity policy and multi-object incident inventory.
        self.registry = NativeObjectRegistry(camera.id, config)
        self.inventory = NativeIncidentInventory(config.native.maximum_tracks)
        self._activity_states = {}
        self._seen_keys = set()
        self._observation_samples: deque = deque(maxlen=256)

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
            obj = self._activity_object(key, include_history=False)
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
            self._capture_pending_context(now)
        if confirmed_activity:
            self._activate(confirmed_activity, observation, epoch, now)

        if self.incident_id is not None or self.event_id is not None:
            self._capture_inventory()
            self._record_observation_sample(epoch)
            if now - self.last_persist >= 1.0:
                self.persist("active", now=now, append_observation=True)

        self.tick(now=now)

    @staticmethod
    def _bounded_context_track(track, limit=64):
        """Retain concise pre-admission evidence without pinning live registry history."""
        stored = deepcopy(track)
        for field in ("box_history", "trajectory"):
            values = list(stored.get(field) or [])
            if len(values) > limit:
                values = values[-limit:]
            stored[field] = values
        if stored.get("box_history"):
            stored["first_seen"] = iso(stored["box_history"][0][0])
        return stored

    def _capture_pending_context(self, now):
        """Attach all confirmed scene objects to active admission nominations.

        One object may admit the incident, but every temporally credible
        co-present object is a first-class participant. Context is copied into
        the bounded pending job because the live registry may expire before
        recorded-main verification completes.
        """
        if not self._verification_pending:
            return
        current = {}
        for key in self.registry.confirmed(self._seen_keys):
            track = self.registry.export(key)
            if track is None:
                continue
            state = self._activity_states.get(key)
            if state is not None:
                track.update(self._public_activity_state(state))
            else:
                track.update(
                    motion_state="context",
                    motion_extent=None,
                    activity_eligible=False,
                )
            current[key] = self._bounded_context_track(track)
        if not current:
            return

        timeout = self.config.native.activity_timeout_seconds
        for pending in self._verification_pending.values():
            source = float(
                pending.get("source_monotonic")
                or pending.get("started")
                or now
            )
            if now - source > timeout:
                continue
            context = pending.setdefault("context_tracks", {})
            for key, track in current.items():
                track_id = int(track["track_id"])
                context[track_id] = deepcopy(track)

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
            # Multi-object incidents stay open; a newly verified subject joins
            # the current time range rather than splitting a sibling episode.
            self._activate(
                [],
                pending["observation"],
                pending["epoch"],
                pending["source_monotonic"],
                cover=result.get("cover"),
                seed=(track, pending["activity"]),
                context=list((pending.get("context_tracks") or {}).values()),
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

    def note_expected_handoff(self, watch) -> None:
        """Remember an advisory upstream route watch for status and provenance."""
        if watch is None:
            return
        try:
            source_event_id = int(getattr(watch, "source_event_id", 0) or 0)
            expires_at = float(getattr(watch, "expires_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        if source_event_id <= 0:
            return
        now = time.time()
        expired = [
            key
            for key, item in self._expected_handoffs.items()
            if float(getattr(item, "expires_at", 0.0) or 0.0) < now
        ]
        for key in expired:
            self._expected_handoffs.pop(key, None)
        self._expected_handoffs[source_event_id] = watch

    def expected_handoffs(self, *, now: float | None = None) -> list[dict]:
        when = time.time() if now is None else float(now)
        active = []
        for key, watch in list(self._expected_handoffs.items()):
            expires = float(getattr(watch, "expires_at", 0.0) or 0.0)
            if expires < when:
                self._expected_handoffs.pop(key, None)
                continue
            payload = watch.as_dict() if hasattr(watch, "as_dict") else dict(watch)
            active.append(payload)
        return active

    def _matching_route_watch(self, epoch, objects):
        labels = {
            str(item.get("label") or "").strip().lower()
            for item in objects or ()
            if isinstance(item, dict)
            and item.get("label")
            and item.get("incident_eligible") is not False
        }
        if not labels:
            return None
        watch = None
        if callable(self.route_watch_match):
            try:
                watch = self.route_watch_match(float(epoch))
            except Exception:
                watch = None
        if watch is None:
            when = float(epoch)
            candidates = [
                item
                for item in self._expected_handoffs.values()
                if float(getattr(item, "eligible_at", 0.0) or 0.0)
                <= when
                <= float(getattr(item, "expires_at", 0.0) or 0.0)
            ]
            if candidates:
                watch = max(candidates, key=lambda item: float(item.source_event_at))
        if watch is None:
            return None
        watch_labels = {
            str(label).strip().lower()
            for label in (getattr(watch, "labels", ()) or ())
            if str(label).strip()
        }
        if watch_labels and not (watch_labels & labels):
            return None
        return watch

    def _activate(
        self,
        confirmed_keys,
        observation,
        epoch,
        now,
        cover=None,
        seed=None,
        context=None,
    ):
        opening = self.event_id is None
        if opening:
            # Incident inventory begins at activity onset. A delayed admission
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
            self._observation_samples.clear()
            self._observation_seq = 0
        for track in context or ():
            self.inventory.record(
                track,
                {
                    "motion_state": track.get("motion_state", "context"),
                    "motion_extent": track.get("motion_extent"),
                    "activity_eligible": bool(track.get("activity_eligible")),
                },
            )
        if seed is not None:
            self.inventory.record(seed[0], seed[1])
        for key in confirmed_keys:
            track = self.registry.export(key)
            if track is not None:
                self.inventory.record(
                    track,
                    self._public_activity_state(self._activity_state(key)),
                )
        # A delayed admission owns a source-time context snapshot. The
        # live registry may now describe an unrelated scene; only ordinary
        # live activation may import it into this incident.
        if seed is None:
            self._capture_inventory()

        self.last_activity = max(self.last_activity, now)
        self.last_motion_at = max(self.last_motion_at, iso(epoch))
        self._record_observation_sample(epoch)

        if opening:
            visible_ids = {
                self.registry.get(key)["track_id"]
                for key in self.registry.confirmed(
                    self._seen_keys if seed is None else ()
                )
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
            start_at = iso(created_epoch if created_epoch is not None else epoch)
            route_watch = self._matching_route_watch(epoch, stored)
            route_origin_camera_id = None
            route_origin_event_id = None
            if route_watch is not None:
                handoff = {
                    "status": "native_route_handoff",
                    "route_detection_watch": route_watch.as_dict(),
                }
                stored = [*stored, handoff]
                route_origin_camera_id = str(
                    route_watch.origin_camera_id or route_watch.source_camera_id or ""
                ) or None
                route_origin_event_id = int(
                    route_watch.origin_event_id or route_watch.source_event_id or 0
                ) or None
            participants = [item for item in stored if item.get("label")]
            observation_objects = self._observation_objects(visible_ids)
            event = self.events.add_event(
                camera_id=self.camera.id,
                kind="motion",
                topic="native/object-presence",
                message="Confirmed native object presence",
                created_at=start_at,
                snapshot_path=path,
                objects_json=json.dumps(stored),
                detection_intent_id=(
                    f"native:{self.camera.id}:{self.session}:{self.sequence}"
                    + (f":{uuid.uuid4().hex}" if cover is not None else "")
                ),
                route_origin_camera_id=route_origin_camera_id,
                route_origin_event_id=route_origin_event_id,
            )
            self.event_id = int(event["id"])
            self.counts["events_created"] += 1
            if hasattr(self.events, "open_incident"):
                try:
                    incident = self.events.open_incident(
                        camera_id=self.camera.id,
                        start_at=start_at,
                        participants=participants,
                        observation_objects=observation_objects,
                        snapshot_path=path,
                        seed_event_id=self.event_id,
                    )
                    self.incident_id = int(incident["id"])
                    self._observation_seq = int(
                        incident.get("observation_count") or 1
                    )
                    self.counts["incidents_created"] += 1
                except (TypeError, ValueError, KeyError, AttributeError):
                    self.incident_id = None
                    self._observation_seq = 0
            if route_watch is not None and callable(self.consume_route_watch):
                try:
                    self.consume_route_watch(
                        self.camera.id,
                        int(route_watch.source_event_id),
                    )
                except Exception:
                    pass
                self._expected_handoffs.pop(int(route_watch.source_event_id), None)
                self.counts["route_handoffs"] += 1
            self.publish(
                "incident",
                {
                    "camera_id": self.camera.id,
                    "event_id": self.event_id,
                    "incident_id": self.incident_id,
                    "timestamp": iso(epoch),
                    "kind": "motion",
                    "state": "active",
                },
            )
            self.publish(
                "object",
                {
                    "camera_id": self.camera.id,
                    "event_id": self.event_id,
                    "incident_id": self.incident_id,
                    "timestamp": iso(epoch),
                    "objects": stored,
                    "source": "native",
                    "snapshot_path": path,
                },
            )
            self.persist("active", now=now, append_observation=False)
        elif (
            self.incident_id is not None
            and hasattr(self.events, "append_incident_observation")
        ):
            try:
                visible_ids = {
                    self.registry.get(key)["track_id"]
                    for key in self.registry.confirmed(self._seen_keys)
                    if self.registry.get(key) is not None
                }
                self.events.append_incident_observation(
                    self.incident_id,
                    observed_at=iso(epoch),
                    objects=self._observation_objects(visible_ids),
                    participants=self.inventory.objects(),
                )
                self._observation_seq += 1
            except (TypeError, ValueError, AttributeError):
                pass

        # Delayed results already carry their verified cover. Never pair
        # current-registry boxes with an older nomination observation.
        if seed is None and self.event_id is not None and self.offer_evidence is not None:
            evidence_objects = self._visible_evidence_objects()
            if evidence_objects:
                self.offer_evidence(
                    observation,
                    epoch,
                    self.event_id,
                    evidence_objects,
                )

    def _record_observation_sample(self, epoch: float):
        sample = []
        for key in self.registry.confirmed(self._seen_keys):
            track = self.registry.get(key)
            if track is None or not track.get("label"):
                continue
            box = track.get("box") or {}
            sample.append(
                {
                    "label": track["label"],
                    "track_id": track.get("track_id"),
                    "box": deepcopy(box),
                    "epoch": float(epoch),
                }
            )
        if sample:
            self._observation_samples.append(sample)

    def _observation_objects(self, visible_ids=None):
        visible = set(visible_ids or ())
        objects = []
        for item in self.inventory.objects(
            visible_track_ids=visible if visible_ids is not None else None
        ):
            if not item.get("label"):
                continue
            objects.append(
                {
                    key: item.get(key)
                    for key in (
                        "label",
                        "confidence",
                        "box",
                        "zones",
                        "track_id",
                        "incident_eligible",
                        "activity_eligible",
                        "motion_state",
                        "snapshot_visible",
                        "detection_frame_width",
                        "detection_frame_height",
                    )
                    if key in item
                }
            )
        return objects

    def _episode_counts(self):
        from .native_episode_identity import episode_label_counts

        # Prefer observation bags when present; fall back to inventory histories.
        if self._observation_samples:
            synthetic = []
            for index, sample in enumerate(self._observation_samples):
                for item in sample:
                    box = item.get("box") or {}
                    try:
                        x1, y1, x2, y2 = (
                            float(box["x1"]),
                            float(box["y1"]),
                            float(box["x2"]),
                            float(box["y2"]),
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                    synthetic.append(
                        {
                            "label": item["label"],
                            "track_id": item.get("track_id") or f"{index}:{item['label']}",
                            "box_history": [[float(item["epoch"]), x1, y1, x2, y2]],
                        }
                    )
            return episode_label_counts(synthetic)
        return episode_label_counts(self.inventory.tracking_tracks())

    def persist(
        self,
        state: str,
        *,
        now: float,
        completion_reason: str = "",
        append_observation: bool = False,
    ):
        if self.event_id is None:
            return
        participants = self.inventory.objects()
        if (
            append_observation
            and self.incident_id is not None
            and state == "active"
            and hasattr(self.events, "append_incident_observation")
        ):
            try:
                visible_ids = {
                    item.get("track_id")
                    for item in participants
                    if item.get("snapshot_visible") is not False
                    and item.get("track_id") is not None
                }
                self.events.append_incident_observation(
                    self.incident_id,
                    observed_at=self.last_motion_at or iso(now),
                    objects=self._observation_objects(visible_ids),
                    participants=participants,
                )
                self._observation_seq += 1
            except (TypeError, ValueError, AttributeError):
                pass
        observation_count = self._observation_seq
        if (
            observation_count <= 0
            and self.incident_id is not None
            and hasattr(self.events, "get_incident")
        ):
            try:
                incident = self.events.get_incident(self.incident_id)
                if incident is not None:
                    observation_count = int(incident.get("observation_count") or 0)
            except (TypeError, ValueError, AttributeError):
                pass
        payload = {
            "implementation": "native_observations",
            "state": state,
            "sample_fps": self.fresh_detection_fps,
            "incident_id": self.incident_id,
            "observation_count": observation_count,
            "episode_counts": self._episode_counts(),
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
        if state != "active":
            payload["completion_reason"] = completion_reason or "unknown"
            payload["end_at"] = self.last_motion_at
        if self.incident_id is not None and state != "active" and hasattr(
            self.events, "close_incident"
        ):
            try:
                self.events.close_incident(
                    self.incident_id,
                    end_at=self.last_motion_at or iso(now),
                    state=state,
                    completion_reason=completion_reason or "unknown",
                    participants=participants,
                )
            except (TypeError, ValueError, AttributeError):
                pass
        self.events.update_native_incident_state(
            self.event_id,
            payload,
            participants,
        )
        self.publish(
            "incident",
            {
                "camera_id": self.camera.id,
                "event_id": self.event_id,
                "incident_id": self.incident_id,
                "timestamp": self.last_motion_at,
                "state": state,
                "completion_reason": completion_reason or "",
                "updated": True,
            },
        )
        self.last_persist = now

    def _pending_verification_blocks_completion(self, now: float) -> bool:
        return any(
            now - pending["started"] < 150
            for pending in self._verification_pending.values()
        )

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
                    and self._pending_verification_blocks_completion(now)
                )
            ):
                self.finish("complete", now=now)

    def finish(self, reason: str, *, now: float):
        state, completion_reason = terminal_state(reason)
        self.persist(
            state,
            now=now,
            completion_reason=completion_reason,
        )
        self.event_id = None
        self.incident_id = None
        self._observation_seq = 0
        self._observation_samples.clear()
        self.inventory.clear()
        if state != "complete":
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
            "implementation": "native_observations",
            "enabled": self.config.enabled,
            "active": self.event_id is not None,
            "event_id": self.event_id,
            "incident_id": self.incident_id,
            "observation_count": self._observation_seq,
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
            "expected_route_handoffs": self.expected_handoffs(),
            "inventory_count": len(self.registry.tracks),
            "participants": list(public_tracks.values()),
            "tracks": list(public_tracks.values()),
            "counters": dict(counters),
        }
