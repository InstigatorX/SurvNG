"""Bounded native incident cover selection from existing recordings.

Native tracks nominate frames. Registration checks geometry, and a bounded native
CPU detector confirms the subject in the full-resolution recording before promotion.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from datetime import datetime
import json
import logging
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np

from .stream_alignment import estimate_stream_alignment
from .native_evidence_common import (
    Candidate,
    image_quality as _common_image_quality,
    matches_object_extent,
    resize_objects,
)
from .native_main_frame import NativeMainFrameVerifier
from .native_replay_alignment import estimate_replay_alignment

LOGGER = logging.getLogger(__name__)


def event_tracking(event):
    values = event.get("objects")
    if not isinstance(values, list):
        values = json.loads(event.get("objects_json") or "[]")
    tracking = event.get("object_tracking") or next((x.get("object_tracking", {}) for x in values if x.get("status") == "object_tracking"), {})
    return values, tracking


def image_quality(image):
    """Compatibility wrapper used by cover scoring and existing diagnostics."""
    return _common_image_quality(image)


def candidate_score(image, objects):
    if not any(obj.get("incident_eligible") is not False for obj in objects):
        return None
    quality = image_quality(image)
    if quality is None:
        return None
    height, width = image.shape[:2]
    scores = []
    for obj in objects:
        if obj.get("incident_eligible") is False:
            continue
        box = obj.get("box") or {}
        x1, y1, x2, y2 = [int(box.get(k, 0)) for k in ("x1", "y1", "x2", "y2")]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop_quality = image_quality(image[y1:y2, x1:x2])
        if crop_quality is None:
            continue
        area = (x2-x1)*(y2-y1)/(width*height)
        clearance = min(x1/width, y1/height, (width-x2)/width, (height-y2)/height)
        scores.append(float(obj.get("confidence") or 0) + min(area*8, 2) + crop_quality + min(clearance*10, 0.5))
    return quality + max(scores) if scores else None


def shortlist(candidates, limit=3):
    selected = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if all(abs(candidate.epoch - previous.epoch) >= 1 for previous in selected):
            selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def calibration_epochs(tracking, limit=9):
    """Select bounded track-history timestamps independently of cover images."""
    epochs = []
    for track in tracking.get("tracks") or []:
        for sample in track.get("box_history") or []:
            if len(sample) < 5:
                continue
            values = sample[:5]
            if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in values):
                continue
            epoch, x1, y1, x2, y2 = values
            if x2 <= x1 or y2 <= y1:
                continue
            epochs.append(float(epoch))
    ordered = sorted(set(epochs))
    if len(ordered) <= limit:
        return ordered
    indices = sorted(set(int(round(index)) for index in np.linspace(0, len(ordered) - 1, limit)))
    return [ordered[index] for index in indices]


class NativeEvidenceService:
    def __init__(self, config, events, recorder, image_writer, media_storage, publish=lambda *args: None):
        self.config, self.events, self.recorder = config, events, recorder
        self.image_writer, self.media_storage, self.publish = image_writer, media_storage, publish
        self._condition = threading.Condition()
        self._pending = {}
        self._active = set()
        self._last_offer = {}
        self._threads = []
        self._closed = False
        self.counts = Counter()
        self.recent = deque(maxlen=32)
        self.main_frames = NativeMainFrameVerifier(
            lambda: self.config,
            recorder,
            self.counts,
        )
        # Compatibility handle for diagnostics/tests; scheduling and config
        # ownership live in NativeMainFrameVerifier.
        self.verifier = self.main_frames.detector
        from .native_admission import NativeAdmission
        self.admission = NativeAdmission(self.main_frames)

    def start(self):
        with self._condition:
            if self._threads:
                return
            self.admission.start()
            for number in range(2):
                thread = threading.Thread(target=self._run, name=f"native-evidence-{number}", daemon=True)
                self._threads.append(thread)
                thread.start()

    def stop(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify_all()
        self.main_frames.request_stop()
        self.admission.stop()
        for thread in self._threads:
            thread.join(timeout=35)
        self.main_frames.close()
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("native evidence workers did not stop")

    def status(self):
        with self._condition:
            return {"queued": sum(job["due"] != float("inf") for job in self._pending.values()),
                    "retained": sum(job["due"] == float("inf") for job in self._pending.values()),
                    "active": len(self._active), "counters": dict(self.counts),
                    "recent": list(self.recent), "admission": self.admission.status()}

    def offer(self, event_id, epoch, frame, objects, size):
        # Called at most once a second by the native camera worker. Pixel work
        # retains native live pixels in a bounded shortlist; inference never waits
        # for recording extraction.
        with self._condition:
            if self._closed or epoch - self._last_offer.get(event_id, 0) < 1:
                return
            if event_id not in self._pending and len(self._pending) >= 32:
                self.counts["queue_full"] += 1
                return
            self._last_offer[event_id] = epoch
            if len(self._last_offer) > 1024:
                self._last_offer.pop(next(iter(self._last_offer)))
        height, width = frame.shape[:2]
        objects = resize_objects(objects, size, (width, height))
        score = candidate_score(frame, objects)
        if score is None:
            self.counts["preview_rejected"] += 1
            return
        candidate = Candidate(epoch, frame, objects, score)
        with self._condition:
            if self._closed or (event_id not in self._pending and len(self._pending) >= 32):
                return
            job = self._pending.setdefault(event_id, {"due": time.monotonic()+20, "deadline": time.monotonic()+120, "candidates": []})
            job["candidates"] = shortlist([*job["candidates"], candidate])
            job["version"] = job.get("version", 0) + 1
            job["due"] = min(job["due"], time.monotonic()+20)
            if not job.get("recorded_history"):
                job["deadline"] = time.monotonic()+120
            self._condition.notify_all()

    def enqueue(self, event_id):
        with self._condition:
            if self._closed:
                return
            if event_id not in self._pending and len(self._pending) >= 32:
                self.counts["queue_full"] += 1
                return
            # Preserve completion during preview processing; rescan all tracks.
            job = self._pending.setdefault(event_id, {"due": time.monotonic()+15, "deadline": time.monotonic()+120, "candidates": []})
            job["recorded_history"] = True
            job["version"] = job.get("version", 0) + 1
            job["due"] = min(job["due"], time.monotonic()+15)
            job["deadline"] = time.monotonic()+120
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                if self._closed:
                    return
                now = time.monotonic()
                for key, cached in list(self._pending.items()):
                    if key not in self._active and cached["due"] == float("inf") and cached["deadline"] <= now:
                        del self._pending[key]
                        self.counts["retained_expired"] += 1
                due = [(key, job) for key, job in self._pending.items()
                       if key not in self._active and job["due"] <= now]
                if not due:
                    self._condition.wait(1)
                    continue
                event_id, job = min(due, key=lambda item: item[1]["due"])
                # Keep a single job owner while it runs. Completion/new frames may
                # update it concurrently, but cannot discard the retained shortlist.
                version = job.get("version", 0)
                candidates = list(job["candidates"])
                recorded_history = bool(job.get("recorded_history"))
                job["due"] = float("inf")
                self._active.add(event_id)
            try:
                result = self.process(event_id, candidates, recorded_history=recorded_history)
            except Exception as exc:
                result = {"event_id": event_id, "status": "failed", "reason": "processing_error",
                          "error_type": type(exc).__name__}
                if not self._closed:
                    LOGGER.exception("Native evidence failed for event %s", event_id)
                    self._record_result(result)
            finally:
                with self._condition:
                    self._active.discard(event_id)
            self.counts[result["status"]] += 1
            exhausted = False
            with self._condition:
                if self._closed or self._pending.get(event_id) is not job:
                    continue
                changed = job.get("version", 0) != version
                retry = result["status"] in {"recording_pending", "failed"}
                if retry and time.monotonic() < job["deadline"]:
                    job["due"] = min(job["due"], time.monotonic()+10)
                elif changed:
                    # enqueue/offer already set the next due time and kept the
                    # terminal rescan flag. Never overwrite newer work with a retry.
                    continue
                elif recorded_history or result["status"] in {"event_missing", "not_native"}:
                    self._pending.pop(event_id)
                    exhausted = retry
                else:
                    # A preview may succeed before completion. Retain its three
                    # exact live images, bounded by queue capacity and idle expiry.
                    job["due"] = float("inf")
                    exhausted = retry
            if exhausted:
                self._record_result(dict(result, retry_exhausted=True))

    def read_frame(self, camera_id, epoch, source, maximum_width=0):
        return self.main_frames.read_frame(
            camera_id,
            epoch,
            source,
            maximum_width=maximum_width,
        )

    def recorded_candidates(self, event, tracking, retained=()):
        # Replay metadata nominates a bounded set of timestamps. Native pixels
        # are unavailable for old incidents, so use recorded live evidence.
        width, height = tracking.get("frame_width", 0), tracking.get("frame_height", 0)
        if not width or not height:
            return [], False
        nominated = {}
        tracks = tracking.get("tracks") or []
        for track in tracks:
            for sample in track.get("box_history") or []:
                epoch, x1, y1, x2, y2 = sample[:5]
                area = max(0, x2-x1)*max(0, y2-y1)
                nominated[int(epoch)] = max(nominated.get(int(epoch), (0, epoch)), (area, epoch))
        ordered = sorted(v[1] for v in nominated.values())
        epochs = [ordered[i] for i in sorted(set(int(x) for x in np.linspace(0,len(ordered)-1,min(12,len(ordered)))))] if ordered else []
        candidates = []
        missing = False
        for epoch in epochs:
            if any(abs(epoch - candidate.epoch) < 0.5 for candidate in retained):
                continue
            if self._closed:
                break
            objects = []
            for track in tracks:
                history = track.get("box_history") or []
                if not history:
                    continue
                point = min(history, key=lambda x: abs(x[0]-epoch))
                if abs(point[0]-epoch) > 0.5:
                    continue
                objects.append({"label": track["label"], "confidence": track.get("max_confidence", track.get("confidence", 0)), "track_id": track.get("track_id"), "incident_eligible": True, "box": dict(zip(("x1","y1","x2","y2"), point[1:5]))})
            live = self.read_frame(event["camera_id"], epoch, "live")
            if live is None:
                missing = True
                continue
            objects = resize_objects(objects, (width, height), (live.shape[1], live.shape[0]))
            score = candidate_score(live, objects)
            if score is not None:
                candidates.append(Candidate(epoch, live, objects, score))
        return candidates, missing

    def replay_calibration_observations(self, event, tracking):
        """Detect on bounded main-stream timestamps chosen from full track history."""
        width = tracking.get("frame_width", 0)
        height = tracking.get("frame_height", 0)
        if not width or not height:
            return [], False
        observations = []
        pending = False
        for epoch in calibration_epochs(tracking):
            if self._closed:
                break
            main = self.read_frame(event["camera_id"], epoch, "main")
            if main is None:
                pending = True
                continue
            detections = self.main_frames.detect(main, priority="cover")
            observations.append({
                "epoch": epoch,
                "objects": resize_objects(
                    detections,
                    (main.shape[1], main.shape[0]),
                    (width, height),
                ),
            })
        return observations, pending

    def project_main(self, candidate, main):
        return self.main_frames.project_main(candidate, main)

    def same_fov_main(self, candidate, main, frame_epoch, replay_offset):
        """Map native boxes onto a timestamp-aligned same-FOV main frame.

        Same-FOV is an explicit camera invariant, so once replay timing is
        verified there is no reason to re-register pixels or template-match the
        subject. Preserve normalized geometry and scale only for raster size.
        """
        lh, lw = candidate.image.shape[:2]
        mh, mw = main.shape[:2]
        if not lw or not lh or not mw or not mh:
            return []
        scaled = []
        for obj in resize_objects(candidate.objects, (lw, lh), (mw, mh)):
            box = obj.get("box") or {}
            values = [box.get(k) for k in ("x1", "y1", "x2", "y2")]
            if (
                any(not isinstance(value, (int, float)) for value in values)
                or not 0 <= values[0] < values[2] <= mw
                or not 0 <= values[1] < values[3] <= mh
            ):
                continue
            obj.update(
                frame_source="recorded_main",
                frame_captured_at_epoch=frame_epoch,
                snapshot_visible=True,
                native_alignment={
                    "method": "same_fov_timestamp_aligned",
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "offset_x": 0.0,
                    "offset_y": 0.0,
                    "pixel_scale_x": mw / lw,
                    "pixel_scale_y": mh / lh,
                    "recording_offset_seconds": replay_offset,
                },
            )
            scaled.append(obj)
        return scaled

    def match_main(self, candidate, main):
        alignment = estimate_stream_alignment(candidate.image, main)
        if alignment is None:
            return []
        sx, sy, ox, oy = alignment
        if not (0.25 < sx < 4 and 0.25 < sy < 4 and abs(ox) < 1 and abs(oy) < 1):
            return []
        lh, lw = candidate.image.shape[:2]
        mh, mw = main.shape[:2]
        # Register onto the live-sized coordinate grid. Search locally to
        # account for modest capture/recording timing skew before native verification.
        transformed = cv2.warpAffine(main, np.float32([[lw/(mw*sx), 0, -ox*lw/sx], [0, lh/(mh*sy), -oy*lh/sy]]), (lw, lh))
        matched = []
        for original in candidate.objects:
            box = original["box"]
            x1,y1,x2,y2 = [int(box[k]) for k in ("x1","y1","x2","y2")]
            x1,y1,x2,y2=max(0,x1),max(0,y1),min(lw,x2),min(lh,y2)
            if x2-x1<8 or y2-y1<8:
                continue
            template = cv2.cvtColor(candidate.image[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
            if float(template.std()) < 4:
                continue
            margin = max(16, int(max(x2-x1,y2-y1)*0.4))
            ax,ay,bx,by=max(0,x1-margin),max(0,y1-margin),min(lw,x2+margin),min(lh,y2+margin)
            search=cv2.cvtColor(transformed[ay:by,ax:bx],cv2.COLOR_BGR2GRAY)
            _, confidence, _, location=cv2.minMaxLoc(cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED))
            if confidence < 0.65:
                continue
            px,py=ax+location[0],ay+location[1]
            values=[(px/lw*sx+ox)*mw,(py/lh*sy+oy)*mh,((px+x2-x1)/lw*sx+ox)*mw,((py+y2-y1)/lh*sy+oy)*mh]
            if values[0]<0 or values[1]<0 or values[2]>mw or values[3]>mh:
                continue
            obj=deepcopy(original)
            obj.update(box=dict(zip(("x1","y1","x2","y2"),values)), detection_frame_width=mw,detection_frame_height=mh,frame_source="recorded_main",frame_captured_at_epoch=candidate.epoch,snapshot_visible=True,native_alignment={"scale_x":sx,"scale_y":sy,"offset_x":ox,"offset_y":oy,"template_confidence":confidence})
            matched.append(obj)
        return matched

    def _record_result(self, result):
        summary = {"source": "native_cover", "recorded_at_epoch": time.time(), **result}
        try:
            if result["status"] not in {"event_missing", "not_native"} and self.events.get(result["event_id"]):
                self.events.record_evidence_attempt(result["event_id"], summary)
        except Exception as exc:
            self.counts["diagnostics_failed"] += 1
            LOGGER.warning("Native cover diagnostics unavailable for event %s (%s)",
                           result["event_id"], type(exc).__name__)
        with self._condition:
            self.recent.append(summary)


    def process(self, event_id, candidates=None, *, recorded_history=False):
        assets = []
        try:
            result = self._process(event_id, candidates, assets, recorded_history=recorded_history)
            self._record_result(result)
            return result
        finally:
            # Durable references protect every successfully archived image.
            for path, _ in assets:
                self.events._delete_snapshot_if_unreferenced(path, preserve_archive=True)

    def _process(self, event_id, candidates, assets, *, recorded_history=False):
        event = self.events.get(event_id)
        if not event:
            return {"event_id": event_id, "status": "event_missing"}
        _, tracking = event_tracking(event)
        if tracking.get("implementation") != "gvatrack":
            return {"event_id": event_id, "status": "not_native"}
        pending = False
        retained = list(candidates or [])
        if recorded_history or not retained:
            recorded, pending = self.recorded_candidates(event, tracking, retained)
            candidates = [*retained, *recorded]
        else:
            candidates = retained
        # Cover selection is bounded and independent of calibration history.
        candidates = shortlist(candidates)
        details = {"retained_live_candidates": len(retained),
                   "cover_candidates": len(candidates), "reasons": {}}
        failures = Counter()
        if not candidates:
            return {"event_id": event_id, "status": "recording_pending" if pending else "no_usable_candidate",
                    "reason": "live_recording_unavailable" if pending else "no_cover_candidate", **details}
        best = None
        camera = next((camera for camera in self.config.cameras if camera.id == event["camera_id"]), None)
        same_fov = bool(camera and camera.native_same_field_of_view)
        recording_alignment = tracking.get("recording_alignment") or {}
        replay_offset = recording_alignment.get("offset_seconds")
        same_fov_aligned = bool(
            same_fov
            and recording_alignment.get("source") == "main"
            and recording_alignment.get("verified") is True
            and isinstance(replay_offset, (int, float))
            and -3 < float(replay_offset) < 3
        )
        require_verification = self.config.detector.native.verification_enabled
        calibrate = bool(
            same_fov
            and not same_fov_aligned
            and tracking.get("state") == "complete"
            and tracking.get("native_session")
            and tracking.get("frame_width")
            and tracking.get("frame_height")
        )

        # Main and live streams may have independent recording clocks. Cover
        # selection keeps only three live images, which is intentionally too
        # small for trajectory calibration. Calibrate from the full persisted
        # track history instead, then rerun cover promotion at the corrected
        # main-stream timestamp.
        epochs = calibration_epochs(tracking) if calibrate else []
        details["calibration"] = "verified" if same_fov_aligned else "unverified"
        if calibrate and (len(epochs) < 5 or epochs[-1] - epochs[0] < 3):
            details["calibration"] = "insufficient_history"
            calibrate = False
        if calibrate:
            try:
                timing_observations, calibration_pending = self.replay_calibration_observations(
                    event, tracking
                )
            except Exception as exc:
                details["calibration"] = "unavailable"
                self.counts["calibration_unavailable"] += 1
                LOGGER.warning(
                    "Native replay calibration unavailable for event %s (%s)",
                    event_id,
                    type(exc).__name__,
                )
            else:
                pending = pending or calibration_pending
                alignment = estimate_replay_alignment(tracking, timing_observations)
                if alignment:
                    aligned = self.events.update_native_replay_alignment(
                        event_id, tracking.get("native_session"), alignment
                    )
                    if aligned:
                        self.counts["replay_aligned"] += 1
                        self.publish(
                            "incident_update",
                            {
                                "camera_id": event["camera_id"],
                                "event_id": event_id,
                                "evidence_revision": aligned["evidence_revision"],
                            },
                        )
                        return self._process(event_id, candidates, assets, recorded_history=False)

        frame_verification = same_fov and not same_fov_aligned
        for candidate in candidates:
            if self._closed:
                break
            if frame_verification:
                # A still-image check is not recording-clock calibration. Use
                # only a box detected on the returned main image, never a
                # projected box or another timestamp's annotation.
                try:
                    checked = self.main_frames.verify_candidate(
                        event["camera_id"],
                        candidate,
                        priority="cover",
                    )
                except Exception as exc:
                    failures["main_verification_unavailable"] += 1
                    pending = True
                    LOGGER.warning("Native cover verification unavailable for event %s (%s)",
                                   event_id, type(exc).__name__)
                    break
                cover = checked.get("cover") if checked.get("status") == "confirmed" else None
                if cover is None:
                    votes = [vote for check in checked.get("checks", []) for vote in check.get("votes", [])]
                    if "unavailable" in votes or "unavailable" in checked.get("votes", []):
                        pending = True
                        failures["main_recording_unavailable"] += 1
                    else:
                        failures["main_verification_failed"] += 1
                    continue
                main, detected, main_epoch = cover
                objects = [deepcopy(detected)]
                objects[0].pop("mask_polygon", None)
                objects[0].update(
                    frame_source="recorded_main", frame_captured_at_epoch=main_epoch,
                    detection_frame_width=main.shape[1], detection_frame_height=main.shape[0],
                    snapshot_visible=True, native_cover_verified=True,
                    box_provenance="detected_in_main",
                    verification={"status": "confirmed", "source": "main"},
                    native_alignment={"method": "main_frame_verified", "nomination_epoch": candidate.epoch,
                                      "sample_offset_seconds": main_epoch - candidate.epoch},
                )
            else:
                main_epoch = (
                    candidate.epoch - float(replay_offset)
                    if same_fov_aligned
                    else candidate.epoch
                )
                main = self.read_frame(event["camera_id"], main_epoch, "main")
                if main is None:
                    pending = True
                    failures["main_recording_unavailable"] += 1
                    continue
                objects = (
                    self.same_fov_main(candidate, main, main_epoch, float(replay_offset))
                    if same_fov_aligned
                    else self.match_main(candidate, main)
                )
                detections = []
                if require_verification and objects:
                    detections = self.main_frames.detect(main, priority="cover")
                if require_verification:
                    verified = []
                    for obj in objects:
                        expected = obj["box"]
                        matching = []
                        for detected in detections:
                            threshold = self.config.detector.event_class_confidence_thresholds.get(
                                obj["label"], self.config.detector.confidence_threshold
                            )
                            if (
                                detected.get("label") != obj["label"]
                                or detected.get("confidence", 0) < threshold
                            ):
                                continue
                            actual = detected.get("box") or {}
                            if (
                                all(k in actual for k in ("x1", "y1", "x2", "y2"))
                                and matches_object_extent(expected, actual)
                            ):
                                matching.append(detected)
                        if matching:
                            actual = max(matching, key=lambda x: x.get("confidence", 0))
                            obj.update(
                                box=actual["box"],
                                confidence=actual["confidence"],
                                native_cover_verified=True,
                                box_provenance="detected_in_main",
                                verification={"status": "confirmed", "source": "main"},
                            )
                            verified.append(obj)
                    objects = verified
                else:
                    for obj in objects:
                        obj.update(
                            native_cover_verified=False,
                            box_provenance="projected_from_substream",
                            verification={"status": "confirmed", "source": "substream"},
                        )
            score = candidate_score(main, objects)
            if score is None:
                self.counts["main_rejected"] += 1
                failures["image_quality_rejected"] += 1
                continue
            if main.shape[0] * main.shape[1] <= candidate.image.shape[0] * candidate.image.shape[1]:
                failures["main_not_higher_resolution"] += 1
                continue
            for obj in objects:
                obj["native_cover_score"] = score
                obj["snapshot_quality_score"] = min(1.0, score / 6)
                box = obj["box"]
                obj["snapshot_subject_area_ratio"] = (
                    (box["x2"] - box["x1"])
                    * (box["y2"] - box["y1"])
                    / (main.shape[0] * main.shape[1])
                )
                obj["snapshot_edge_clearance_ratio"] = min(
                    box["x1"] / main.shape[1],
                    box["y1"] / main.shape[0],
                    1 - box["x2"] / main.shape[1],
                    1 - box["y2"] / main.shape[0],
                )
                obj["snapshot_primary_subject"] = True
                obj["temporal_sample_offset_seconds"] = (
                    candidate.epoch - datetime.fromisoformat(event["created_at"]).timestamp()
                )
            if len(assets) >= 3 and score <= min(
                items[1][0]["native_cover_score"] for items in assets
            ):
                continue
            directory = self.media_storage.directory(
                "snapshots", event["camera_id"], event["camera_id"]
            )
            output_path = self.image_writer.write(
                directory, f"native-{event_id}-{uuid.uuid4().hex}", main
            )
            if output_path is None:
                failures["image_write_failed"] += 1
                pending = True
                continue
            assets.append((str(output_path), objects))
            if len(assets) > 3:
                lowest = min(assets, key=lambda item: item[1][0]["native_cover_score"])
                assets.remove(lowest)
                Path(lowest[0]).unlink(missing_ok=True)
            if best is None or score > best[0]:
                best = (score, str(output_path), objects, main.shape[1], main.shape[0])
        details["reasons"] = dict(failures)
        if not best:
            reason = next((name for name in ("main_verification_unavailable", "image_write_failed",
                                           "main_recording_unavailable", "main_verification_failed",
                                           "image_quality_rejected", "main_not_higher_resolution")
                           if failures[name]), "no_usable_candidate")
            return {
                **details,
                "reason": reason,
                "event_id": event_id,
                "status": (
                    "recording_pending"
                    if pending
                    else "no_verified_candidate"
                    if require_verification
                    else "no_usable_candidate"
                ),
            }
        score, output_path, objects, width, height = best
        adoption = {}
        result = self.events.promote_native_evidence(
            event_id, output_path, objects, assets, score, diagnostics=adoption
        )
        if result:
            self.publish(
                "incident_update",
                {
                    "camera_id": event["camera_id"],
                    "event_id": event_id,
                    "evidence_revision": result["evidence_revision"],
                },
            )
        return {
            "event_id": event_id,
            "status": ("promoted" if result else "kept_better_cover"
                       if adoption.get("reason") == "better_cover_retained" else "recording_pending"),
            "reason": (objects[0].get("native_alignment", {}).get("method", "verified_main")
                       if result else adoption.get("reason", "cover_not_adopted")),
            **details,
            "width": width,
            "height": height,
            "candidates": len(assets),
        }
