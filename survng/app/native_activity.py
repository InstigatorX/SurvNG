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
import uuid
from typing import Callable

from .live_detections import DetectionSnapshot
from .native_motion import NativeMotion
from .zones import apply_detection_zones
from survng.native_spatial import spatial_plan


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
        self.tracks = {}
        self._episode_tracks = {}
        # Incident inventory is deliberately separate from activity admission.
        # All temporally credible fresh detections can describe an incident;
        # zones/stationary policy only decide whether activity starts/continues.
        self.inventory_tracks = {}
        self._episode_inventory = {}
        self._inventory_seen = set()
        self._next_inventory_id = 1
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
        if self.native_zones:
            revision = self.zone_revision
            valid_ids = {str(i) for i, z in enumerate(self.camera.zones) if z.enabled and len(z.points) >= 3}
            if observation.zone_revision != revision or any(obj.get("native_zone_revision") != revision
                   or not isinstance(obj.get("native_zone_ids"), list)
                   or any(not isinstance(i, str) or i not in valid_ids for i in obj["native_zone_ids"])
                   for obj in observation.objects):
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
            self.camera, observation.scaled_objects(observation.width, observation.height),
            observation.width, observation.height, self.config.confidence_threshold,
            self.config.require_incident_zone, self.config.event_class_confidence_thresholds,
            native_membership=self.native_zones,
        )
        self._inventory_seen = self._update_inventory(objects, now=now, epoch=epoch)
        eligible = [obj for obj in objects
                    if obj.get("detection_provenance") == "native_fresh_detection"
                    and obj.get("incident_eligible")
                    and (self.config.native.tracking_classes is None
                         or obj.get("label", "").strip().lower() in self.config.native.tracking_classes)]
        # Live context expires independently of the persisted episode archive.
        for key, track in list(self.tracks.items()):
            if now - track["last_monotonic"] >= max(self.config.native.activity_timeout_seconds, (self.config.native.batch_size + .5) / self.fresh_detection_fps):
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
                inventory = self.inventory_tracks.get(("native", native_id, obj["label"]))
                track_id = inventory["track_id"] if inventory is not None else self._next_track_id
                if inventory is None:
                    self._next_track_id += 1
                track = {"track_id": track_id,
                         "native_track_id": native_id,
                         "native_identity": f"{self.camera.id}/{self.session}/{self.identity_epoch}/{native_id}",
                         "label": obj["label"], "first_seen": iso(epoch),
                         "observations": 0, "consecutive": 0, "last_monotonic": now,
                         "box_history": [], "trajectory": [], "_motion": NativeMotion()}
                self.tracks[key] = track
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
            inventory = self.inventory_tracks.get(("native", native_id, obj["label"]))
            if inventory is not None:
                inventory.update(
                    motion_state=track["motion_state"],
                    motion_extent=track["motion_extent"],
                    activity_eligible=track["activity_eligible"],
                )
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
        if self.admission is not None and self.config.native.verification_enabled:
            confirmed = self._gate(confirmed, observation, epoch, now)
        if confirmed:
            self._activate(confirmed, observation, epoch, now)
        self.tick(now=now)

    @staticmethod
    def _inventory_association_score(previous, current):
        def coords(item):
            box = item.get("box") or {}
            try:
                values = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError):
                return None
            if values[2] <= values[0] or values[3] <= values[1]:
                return None
            return values
        before, after = coords(previous), coords(current)
        if before is None or after is None:
            return None
        ax1, ay1, ax2, ay2 = before
        bx1, by1, bx2, by2 = after
        overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - overlap
        iou = overlap / union if union > 0 else 0.0
        if iou >= .15:
            return 2.0 + iou
        acx, acy = (ax1+ax2)/2, (ay1+ay2)/2
        bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
        scale = max(1.0, ((ax2-ax1)**2 + (ay2-ay1)**2) ** .5,
                    ((bx2-bx1)**2 + (by2-by1)**2) ** .5)
        distance = ((acx-bcx)**2 + (acy-bcy)**2) ** .5 / scale
        return 1.0 - distance/.65 if distance <= .65 else None

    def _inventory_key(self, obj, used):
        label = str(obj.get("label") or "").strip()
        native_id = obj.get("native_track_id")
        if type(native_id) is int and native_id >= 0:
            return ("native", native_id, label)
        scored = []
        for key, track in self.inventory_tracks.items():
            if key in used or key[0] != "inventory" or track.get("label") != label:
                continue
            score = self._inventory_association_score(track, obj)
            if score is not None:
                scored.append((score, key))
        if scored:
            return max(scored)[1]
        key = ("inventory", self._next_inventory_id, label)
        self._next_inventory_id += 1
        return key

    def _update_inventory(self, objects, *, now, epoch):
        ttl = max(self.config.native.activity_timeout_seconds,
                  (self.config.native.batch_size + .5) / self.fresh_detection_fps)
        for key, track in list(self.inventory_tracks.items()):
            if now - track["last_monotonic"] >= ttl:
                del self.inventory_tracks[key]
        seen = set()
        for obj in objects:
            if obj.get("detection_provenance") != "native_fresh_detection":
                continue
            label = str(obj.get("label") or "").strip()
            box = obj.get("box")
            if not label or not isinstance(box, dict):
                continue
            key = self._inventory_key(obj, seen)
            if key in seen:
                self.counts["duplicate_inventory_object"] += 1
                continue
            seen.add(key)
            track = self.inventory_tracks.get(key)
            if track is None:
                if len(self.inventory_tracks) >= self.config.native.maximum_tracks:
                    self.counts["inventory_capacity_drops"] += 1
                    continue
                track = {
                    "track_id": self._next_track_id,
                    "native_track_id": obj.get("native_track_id"),
                    "native_identity": (
                        f"{self.camera.id}/{self.session}/{self.identity_epoch}/{obj.get('native_track_id')}"
                        if type(obj.get("native_track_id")) is int
                        else f"{self.camera.id}/{self.session}/{self.identity_epoch}/inventory-{key[1]}"
                    ),
                    "label": label,
                    "first_seen": iso(epoch),
                    "observations": 0,
                    "consecutive": 0,
                    "last_monotonic": now,
                    "box_history": [],
                    "trajectory": [],
                    "motion_state": "context",
                    "motion_extent": None,
                    "activity_eligible": False,
                }
                self.inventory_tracks[key] = track
                self._next_track_id += 1
            if now - track["last_monotonic"] > max(
                self.config.native.maximum_observation_age_seconds,
                3 / self.fresh_detection_fps,
            ):
                track["consecutive"] = 0
            track.update(
                last_monotonic=now,
                last_seen=iso(epoch),
                box=deepcopy(box),
                confidence=float(obj.get("confidence") or 0.0),
                zones=deepcopy(obj.get("zones", [])),
                incident_eligible=bool(obj.get("incident_eligible")),
                zone_eligible=bool(obj.get("zone_eligible")),
                confidence_eligible=obj.get("confidence_eligible"),
                zone_admission_reason=obj.get("zone_admission_reason"),
                spatial_zones=deepcopy(obj.get("spatial_zones", [])),
                detection_frame_width=self.dimensions[0],
                detection_frame_height=self.dimensions[1],
            )
            track["observations"] += 1
            track["consecutive"] += 1
            required = self.config.event_class_confirmation_frames.get(
                label, self.config.event_confirmation_frames
            )
            track["state"] = "confirmed" if track["consecutive"] >= required else "tentative"
            track["confirmed"] = track.get("confirmed", False) or track["state"] == "confirmed"
            track["max_confidence"] = max(track.get("max_confidence", 0.0), track["confidence"])
            track["box_history"].append([epoch, box["x1"], box["y1"], box["x2"], box["y2"]])
            track["trajectory"].append([epoch, (box["x1"]+box["x2"])/2, (box["y1"]+box["y2"])/2])
            del track["box_history"][:-150]
            del track["trajectory"][:-150]
            if self.event_id is not None and track["state"] == "confirmed":
                self._record_inventory_track(track, epoch)
        for key, track in self.inventory_tracks.items():
            if key not in seen:
                track["consecutive"] = 0
        return seen

    def _record_inventory_track(self, track, epoch):
        key = track["track_id"]
        previous = self._episode_inventory.get(key)
        stored = {k: deepcopy(v) for k, v in track.items()
                  if k not in {"last_monotonic", "consecutive", "box_history", "trajectory"}}
        history = previous["box_history"] if previous else []
        trajectory = previous["trajectory"] if previous else []
        point = deepcopy(track["box_history"][-1])
        if not history or history[-1][0] != point[0]:
            history.append(point)
            trajectory.append(deepcopy(track["trajectory"][-1]))
        stored.update(
            first_seen=previous["first_seen"] if previous else track["first_seen"],
            observations=(previous["observations"] + 1 if previous else 1),
            max_confidence=max(previous["max_confidence"], track["confidence"]) if previous else track["confidence"],
            box_history=compact_history(history),
            trajectory=compact_history(trajectory),
        )
        self._episode_inventory[key] = stored

    def _capture_inventory(self, now):
        freshness = max(self.config.native.maximum_observation_age_seconds,
                        3 / self.fresh_detection_fps)
        for key in self._inventory_seen:
            track = self.inventory_tracks.get(key)
            if (track is not None and track.get("state") == "confirmed"
                    and track["last_monotonic"] >= now - freshness):
                self._record_inventory_track(track, datetime.fromisoformat(track["last_seen"]).timestamp())

    def _incident_objects(self, *, visible_track_ids=None, cover=None):
        objects = self._objects(self._episode_inventory.values())
        if visible_track_ids is not None:
            visible = set(visible_track_ids)
            for item in objects:
                item["snapshot_visible"] = item.get("track_id") in visible
        if cover is not None:
            cover = deepcopy(cover)
            match = next((item for item in objects
                          if item.get("track_id") == cover.get("track_id")), None)
            if match is None:
                match = next((item for item in objects
                              if item.get("label") == cover.get("label")
                              and item.get("native_track_id") == cover.get("native_track_id")), None)
            for item in objects:
                item["snapshot_visible"] = False
            if match is not None:
                match.update(cover)
                match["snapshot_visible"] = True
                match["verification"] = {"status": "confirmed", "source": "main_crop"}
            else:
                cover["snapshot_visible"] = True
                cover["verification"] = {"status": "confirmed", "source": "main_crop"}
                objects.insert(0, cover)
        return objects

    def _gate(self, confirmed, observation, epoch, now):
        allowed = []
        for track in confirmed:
            state = track.get('_verification')
            if state == 'confirmed':
                allowed.append(track)
                continue
            if state in {'rejected', 'unverified'}:
                before, box = track.get('_verification_box', track['box']), track['box']
                scale = max(1, before['x2']-before['x1'], before['y2']-before['y1'])
                moved = max(abs((box['x1']+box['x2']-before['x1']-before['x2'])/2),
                            abs((box['y1']+box['y2']-before['y1']-before['y2'])/2)) > scale*.5
                if not moved and (state != 'unverified' or now < track.get('_verification_retry_at', now)):
                    continue
                track.pop('_verification_token', None)
                track.pop('_verification', None)
            token = track.get('_verification_token')
            if token is None:
                if len(self._verification_pending) >= 32:
                    self.counts['verification_capacity_drops'] += 1
                    continue
                token = track['_verification_token'] = uuid.uuid4().hex
                track['_verification'] = 'pending'
            pending = self._verification_pending.setdefault(token, {'started': now})
            pending.update(track={k: deepcopy(v) for k,v in track.items() if not k.startswith('_')},
                           observation=observation, epoch=epoch)
            self.nominate(token, observation, epoch, self._objects([track])[0])
        return allowed

    def _poll_verification(self, now):
        if self.admission is None:
            return
        # Resolve in observation order so a faster, later result cannot close
        # or replace the episode before an earlier continuation is decided.
        pending_items = sorted(self._verification_pending.items(), key=lambda item: item[1]['track']['first_seen'])
        for token, pending in pending_items:
            result = self.admission.poll(token)
            if result is None and now-pending['started'] < 150:
                break
            if result is None:
                self.admission.cancel(token)
                result = {'status': 'unverified', 'reason': 'deadline'}
            del self._verification_pending[token]
            status = result['status']
            self.counts['verification_'+status] += 1
            track = pending['track']
            self._verification_recent.append({'label': track['label'], 'status': status, 'reason': result.get('reason', ''),
                                              'epoch': pending['epoch'], 'timestamp': iso(pending['epoch']),
                                              'votes': result.get('votes', []), 'checks': result.get('checks', [])})
            current = self.tracks.get((track['native_track_id'], track['label']))
            if current is not None and current.get('_verification_token') == token:
                current['_verification'] = status
                current['_verification_box'] = deepcopy(track['box'])
                current['_verification_retry_at'] = now+30
                current['verification'] = {k:v for k,v in result.items() if k != 'cover'}
            if status != 'confirmed':
                continue
            track['verification'] = {k:v for k,v in result.items() if k != 'cover'}
            if self.event_id is not None and not self._joins_episode(track):
                self.finish('complete', now=now)
            # These observations were fresh at nomination. Verification may finish
            # after the object leaves; retain the original source-time history.
            for point in track['box_history']:
                sample = dict(track, box_history=[point], trajectory=[[point[0], (point[1]+point[3])/2, (point[2]+point[4])/2]], last_seen=iso(point[0]))
                self._record_activity_track(sample, point[0])
            self._activate([], pending['observation'], pending['epoch'], track['last_monotonic'],
                           cover=result.get('cover'))

    def _activate(self, confirmed, observation, epoch, now, cover=None):
        self._capture_inventory(now)
        for track in confirmed:
            self._record_activity_track(track, epoch)
        self.last_activity = max(self.last_activity, now)
        self.last_motion_at = max(self.last_motion_at, iso(epoch))
        if self.event_id is None:
            visible_ids = {
                self.inventory_tracks[key]["track_id"]
                for key in self._inventory_seen
                if key in self.inventory_tracks
                and self.inventory_tracks[key].get("state") == "confirmed"
            }
            stored = self._incident_objects(visible_track_ids=visible_ids)
            path = self.verified_snapshot(cover) if cover is not None else self.snapshot(observation, epoch)
            if cover is not None:
                stored = self._incident_objects(cover=cover[1])
            event = self.events.add_event(
                camera_id=self.camera.id, kind="motion", topic="native/object-presence",
                message="Confirmed native object presence", created_at=min((t["first_seen"] for t in self._episode_tracks.values()), default=iso(epoch)) if cover is not None else iso(epoch),
                snapshot_path=path, objects_json=json.dumps(stored),
                detection_intent_id=f"native:{self.camera.id}:{self.session}:{self.sequence}" + (f":{uuid.uuid4().hex}" if cover is not None else ""),
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
        # Episode history has a single owner. Publication/persistence already
        # takes deep copies; do not recopy up to 1024 entries on each detection.
        box_history = previous["box_history"] if previous else []
        trajectory = previous["trajectory"] if previous else []
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
        payload["inventory_tracks"] = [
            {k: deepcopy(v) for k, v in track.items() if k not in {"box_history", "trajectory"}}
            for track in self._episode_inventory.values()
        ]
        self.events.update_object_tracking(
            self.event_id,
            payload,
            self._incident_objects(),
            replace_objects=True,
        )
        self.publish("object_tracking", {"camera_id": self.camera.id, "event_id": self.event_id, **payload})
        if state == "active":
            self.publish("incident", {"camera_id": self.camera.id, "event_id": self.event_id,
                                      "timestamp": self.last_motion_at, "updated": True})
        self.last_persist = now

    @property
    def fresh_detection_fps(self):
        # Configuration is validated when applied. Reading two effective fields
        # here must not serialize and revalidate the entire motion policy on
        # every health poll, track expiry check, and incident update.
        defaults, overrides = self.config.native.budget, self.camera.native_budget
        enabled = defaults.enabled if overrides.enabled is None else overrides.enabled
        idle_fps = defaults.idle_fps if overrides.idle_fps is None else overrides.idle_fps
        return idle_fps if enabled else self.config.live_sample_fps / self.config.native.inference_interval

    def _joins_episode(self, track):
        """Group by observed activity, independent of verification completion."""
        if not self._episode_tracks:
            return False
        start = min(datetime.fromisoformat(t['first_seen']).timestamp() for t in self._episode_tracks.values())
        end = max(datetime.fromisoformat(t['last_seen']).timestamp() for t in self._episode_tracks.values())
        timeout = self.config.native.activity_timeout_seconds
        return (datetime.fromisoformat(track['first_seen']).timestamp() <= end + timeout
                and datetime.fromisoformat(track['last_seen']).timestamp() >= start - timeout)

    def tick(self, *, now: float):
        self._poll_verification(now)
        if self.last_fresh and now - self.last_fresh > max(self.config.native.maximum_observation_age_seconds, (self.config.native.batch_size + .5) / self.fresh_detection_fps):
            self.health = "metadata_stale"
        if self.event_id is not None and now - self.last_activity >= self.config.native.activity_timeout_seconds:
            if self.health != "healthy":
                self.finish("metadata_lost", now=now)
            elif (self.last_fresh >= self.last_activity + self.config.native.activity_timeout_seconds
                  and not (now - self.last_activity < 150
                           and any(now - pending['started'] < 150 and self._joins_episode(pending['track'])
                                   for pending in self._verification_pending.values()))):
                self.finish("complete", now=now)
            # The timer can cross the deadline between healthy metadata frames.
            # Wait for a fresh observation covering it, or for metadata to become
            # stale above. A normal inter-frame gap is not loss of coverage.

    def finish(self, reason: str, *, now: float):
        self.persist(reason, now=now)
        self.event_id = None
        self._episode_tracks.clear()
        self._episode_inventory.clear()
        if reason != "complete":
            if self.admission is not None:
                for token in self._verification_pending:
                    self.admission.cancel(token)
            self._verification_pending.clear()
            self.tracks.clear()
            self.inventory_tracks.clear()
            self._inventory_seen.clear()
            self._next_inventory_id = 1
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
                "verification_pending": len(self._verification_pending),
                "verification_recent": list(self._verification_recent),
                "inventory_count": len(self.inventory_tracks),
                "tracks": self._objects(self.tracks.values()), "counters": dict(self.counts)}
