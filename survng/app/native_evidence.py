"""Bounded native incident cover selection from existing recordings.

Native tracks nominate frames. Registration checks geometry, and a bounded native
CPU detector confirms the subject in the full-resolution recording before promotion.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np

from .stream_alignment import estimate_stream_alignment
from .native_evidence_verifier import NativeEvidenceVerifier
from .native_replay_alignment import estimate_replay_alignment

LOGGER = logging.getLogger(__name__)


def event_tracking(event):
    values = event.get("objects")
    if not isinstance(values, list):
        values = json.loads(event.get("objects_json") or "[]")
    tracking = event.get("object_tracking") or next((x.get("object_tracking", {}) for x in values if x.get("status") == "object_tracking"), {})
    return values, tracking


def image_quality(image):
    """Reject uniform/corrupt-looking frames without rejecting night exposure."""
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (min(320, gray.shape[1]), min(180, gray.shape[0])))
    spread = float(np.percentile(small, 98) - np.percentile(small, 2))
    sharp = float(cv2.Laplacian(small, cv2.CV_32F).var())
    if spread < 10 or sharp < 2:
        return None
    return min(sharp, 500) / 500 + min(spread, 100) / 100


@dataclass
class Candidate:
    epoch: float
    image: object
    objects: list
    score: float


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


def matches_object_extent(expected, actual):
    """A same-class fragment is not confirmation of the nominated object."""
    intersection = max(0, min(expected['x2'], actual['x2'])-max(expected['x1'], actual['x1'])) * max(0, min(expected['y2'], actual['y2'])-max(expected['y1'], actual['y1']))
    area = lambda b: max(1, (b['x2']-b['x1'])*(b['y2']-b['y1']))
    return intersection/area(actual) >= .5 and intersection/(area(actual)+area(expected)-intersection) >= .3


def resize_objects(objects, from_size, to_size):
    fw, fh = from_size
    tw, th = to_size
    result = deepcopy(objects)
    for obj in result:
        box = obj.get("box") or {}
        obj["box"] = {k: float(box.get(k, 0)) * (tw/fw if k.startswith("x") else th/fh) for k in ("x1", "y1", "x2", "y2")}
        obj.update(detection_frame_width=tw, detection_frame_height=th)
    return result


def shortlist(candidates, limit=3):
    selected = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if all(abs(candidate.epoch - previous.epoch) >= 1 for previous in selected):
            selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


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
        self.verifier = NativeEvidenceVerifier(config.detector)
        from .native_admission import NativeAdmission
        self.admission = NativeAdmission(self)

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
        self.verifier.request_stop()
        self.admission.stop()
        for thread in self._threads:
            thread.join(timeout=35)
        self.verifier.close()
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("native evidence workers did not stop")

    def status(self):
        with self._condition:
            return {"queued": len(self._pending), "active": len(self._active), "counters": dict(self.counts), "admission": self.admission.status()}

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
            job["deadline"] = time.monotonic()+120
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                if self._closed:
                    return
                due = [(key, job) for key, job in self._pending.items() if key not in self._active and job["due"] <= time.monotonic()]
                if not due:
                    self._condition.wait(1)
                    continue
                event_id, job = min(due, key=lambda item: item[1]["due"])
                self._pending.pop(event_id)
                self._active.add(event_id)
            try:
                result = self.process(event_id, None if job.get("recorded_history") else job["candidates"])
                self.counts[result["status"]] += 1
                if result["status"] == "recording_pending" and time.monotonic() < job["deadline"]:
                    with self._condition:
                        job["due"] = time.monotonic()+10
                        self._pending.setdefault(event_id, job)
            except Exception:
                if self._closed:
                    return
                self.counts["failed"] += 1
                LOGGER.exception("Native evidence failed for event %s", event_id)
            finally:
                with self._condition:
                    self._active.discard(event_id)

    def read_frame(self, camera_id, epoch, source, maximum_width=0):
        row = self.recorder.recording_at(camera_id, epoch, source=source)
        if not row:
            return None
        command = [self.config.ffmpeg_path, "-nostdin", "-v", "error", "-threads", "1", "-ss", str(max(0, epoch-float(row["start_epoch"]))), "-i", str(row["path"]), "-frames:v", "1", "-an", "-sn"]
        if maximum_width:
            command += ["-vf", f"scale='min({maximum_width},iw)':-2"]
        # This is local IPC, not an archived image. BMP preserves BGR pixels
        # without PNG compression/decompression on every verification sample.
        command += ["-threads", "1", "-f", "image2pipe", "-c:v", "bmp", "-pix_fmt", "bgr24", "pipe:1"]
        result = subprocess.run(command, capture_output=True, timeout=12)
        if result.returncode or not result.stdout:
            self.counts["decode_failed"] += 1
            return None
        return cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)

    def recorded_candidates(self, event, tracking):
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

    def project_main(self, candidate, main):
        """Locate verification crops using scene geometry, not object appearance."""
        alignment = estimate_stream_alignment(candidate.image, main)
        if alignment is None:
            return []
        sx, sy, ox, oy = alignment
        if not (0.25 < sx < 4 and 0.25 < sy < 4 and abs(ox) < 1 and abs(oy) < 1):
            return []
        lh, lw = candidate.image.shape[:2]
        mh, mw = main.shape[:2]
        projected = []
        for original in candidate.objects:
            box = original['box']
            values = [(box['x1']/lw*sx+ox)*mw, (box['y1']/lh*sy+oy)*mh,
                      (box['x2']/lw*sx+ox)*mw, (box['y2']/lh*sy+oy)*mh]
            if not (0 <= values[0] < values[2] <= mw and 0 <= values[1] < values[3] <= mh):
                continue
            obj = deepcopy(original)
            obj.update(box=dict(zip(('x1', 'y1', 'x2', 'y2'), values)),
                       detection_frame_width=mw, detection_frame_height=mh, frame_source='recorded_main',
                       frame_captured_at_epoch=candidate.epoch, snapshot_visible=True,
                       native_alignment={'scale_x': sx, 'scale_y': sy, 'offset_x': ox, 'offset_y': oy})
            projected.append(obj)
        return projected

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

    def process(self, event_id, candidates=None):
        assets = []
        try:
            return self._process(event_id, candidates, assets)
        finally:
            # Durable references protect every successfully archived image.
            for path, _ in assets:
                self.events._delete_snapshot_if_unreferenced(path, preserve_archive=True)

    def _process(self, event_id, candidates, assets):
        event = self.events.get(event_id)
        if not event:
            return {"event_id": event_id, "status": "event_missing"}
        _, tracking = event_tracking(event)
        if tracking.get("implementation") != "gvatrack":
            return {"event_id": event_id, "status": "not_native"}
        pending = False
        if not candidates:
            candidates, pending = self.recorded_candidates(event, tracking)
        best = None
        camera = next((camera for camera in self.config.cameras if camera.id == event["camera_id"]), None)
        calibrate = bool(camera and camera.native_same_field_of_view and tracking.get("state") == "complete"
                         and tracking.get("native_session")
                         and not tracking.get("recording_alignment", {}).get("verified")
                         and tracking.get("frame_width") and tracking.get("frame_height"))
        require_verification = self.config.detector.native.verification_enabled
        timing_observations = []
        calibration_unavailable = False
        for candidate in candidates[:12]:
            if self._closed:
                break
            main = self.read_frame(event["camera_id"], candidate.epoch, "main")
            if main is None:
                pending = True
                continue
            objects = self.match_main(candidate, main)
            self.verifier.config = self.config.detector
            detections = []
            if (require_verification and objects) or (calibrate and not calibration_unavailable):
                try:
                    detections = self.verifier.detect(main)
                except Exception as exc:
                    if require_verification:
                        raise
                    # Optional replay calibration must not prevent image promotion
                    # when incident validation belongs to the substream.
                    calibration_unavailable = True
                    self.counts["calibration_unavailable"] += 1
                    LOGGER.warning("Native replay calibration unavailable for event %s (%s)", event_id, type(exc).__name__)
            if calibrate and not calibration_unavailable:
                timing_observations.append({"epoch": candidate.epoch, "objects": resize_objects(
                    detections, (main.shape[1], main.shape[0]), (tracking["frame_width"], tracking["frame_height"]))})
            if require_verification:
                verified = []
                for obj in objects:
                    expected = obj["box"]
                    matching = []
                    for detected in detections:
                        threshold = self.config.detector.event_class_confidence_thresholds.get(obj["label"], self.config.detector.confidence_threshold)
                        if detected.get("label") != obj["label"] or detected.get("confidence", 0) < threshold:
                            continue
                        actual = detected.get("box") or {}
                        if all(k in actual for k in ("x1", "y1", "x2", "y2")) and matches_object_extent(expected, actual):
                            matching.append(detected)
                    if matching:
                        actual = max(matching,key=lambda x:x.get("confidence",0))
                        obj.update(box=actual["box"], confidence=actual["confidence"], native_cover_verified=True,
                                   box_provenance="detected_in_main", verification={"status": "confirmed", "source": "main"})
                        verified.append(obj)
                objects = verified
            else:
                for obj in objects:
                    obj.update(native_cover_verified=False, box_provenance="projected_from_substream",
                               verification={"status": "confirmed", "source": "substream"})
            score = candidate_score(main, objects)
            if score is None:
                self.counts["main_rejected"] += 1
                continue
            # Require actual source detail; no upscaling preview pixels.
            if main.shape[0]*main.shape[1] <= candidate.image.shape[0]*candidate.image.shape[1]:
                continue
            for obj in objects:
                obj["native_cover_score"] = score
                obj["snapshot_quality_score"] = min(1.0, score/6)
                box = obj["box"]
                obj["snapshot_subject_area_ratio"] = (box["x2"]-box["x1"])*(box["y2"]-box["y1"])/(main.shape[0]*main.shape[1])
                obj["snapshot_edge_clearance_ratio"] = min(box["x1"]/main.shape[1],box["y1"]/main.shape[0],1-box["x2"]/main.shape[1],1-box["y2"]/main.shape[0])
                obj["snapshot_primary_subject"] = True
                obj["temporal_sample_offset_seconds"] = candidate.epoch-datetime.fromisoformat(event["created_at"]).timestamp()
            if len(assets) >= 3 and score <= min(items[1][0]["native_cover_score"] for items in assets):
                continue
            directory = self.media_storage.directory("snapshots", event["camera_id"], event["camera_id"])
            path = self.image_writer.write(directory, f"native-{event_id}-{uuid.uuid4().hex}", main)
            if path is None:
                continue
            assets.append((str(path), objects))
            if len(assets) > 3:
                lowest = min(assets, key=lambda item:item[1][0]["native_cover_score"])
                assets.remove(lowest)
                Path(lowest[0]).unlink(missing_ok=True)
            if best is None or score > best[0]:
                best = (score, str(path), objects, main.shape[1], main.shape[0])
        if calibrate:
            alignment = estimate_replay_alignment(tracking, timing_observations)
            if alignment:
                aligned = self.events.update_native_replay_alignment(event_id, tracking.get("native_session"), alignment)
                if aligned:
                    self.counts["replay_aligned"] += 1
                    self.publish("incident_update", {"camera_id": event["camera_id"], "event_id": event_id,
                                                    "evidence_revision": aligned["evidence_revision"]})
        if not best:
            return {"event_id":event_id,"status":"recording_pending" if pending else "no_verified_candidate" if require_verification else "no_usable_candidate"}
        score, path, objects, width, height = best
        result = self.events.promote_native_evidence(event_id, path, objects, assets, score)
        if result:
            self.publish("incident_update", {"camera_id":event["camera_id"],"event_id":event_id,"evidence_revision":result["evidence_revision"]})
        return {"event_id":event_id,"status":"promoted" if result else "kept_better_cover","width":width,"height":height,"candidates":len(assets)}
