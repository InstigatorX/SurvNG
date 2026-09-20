"""Shared recorded-main frame access and object verification.

Admission and post-incident cover selection use this service independently. The
single CPU verifier remains bounded, with admission work prioritized over cover
refinement when both are waiting.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import subprocess
import threading

import cv2
import numpy as np

from .native_evidence_common import (
    image_quality,
    matches_object_extent,
)
from .native_evidence_verifier import NativeEvidenceVerifier
from .stream_alignment import estimate_stream_alignment


class _VerifierScheduler:
    def __init__(self):
        self._condition = threading.Condition()
        self._busy = False
        self._admission_waiters = 0

    @contextmanager
    def lease(self, priority: str):
        admission = priority == "admission"
        with self._condition:
            if admission:
                self._admission_waiters += 1
            try:
                while self._busy or (not admission and self._admission_waiters):
                    self._condition.wait()
                self._busy = True
            finally:
                if admission:
                    self._admission_waiters -= 1
        try:
            yield
        finally:
            with self._condition:
                self._busy = False
                self._condition.notify_all()


def context_crop(main, box):
    """Keep context and native pixels; never upscale an object for verification."""
    height, width = main.shape[:2]
    x1, y1, x2, y2 = (box[key] for key in ("x1", "y1", "x2", "y2"))
    side = max(192, 3 * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0, int(cx - side / 2))
    top = max(0, int(cy - side / 2))
    right = min(width, int(cx + side / 2))
    bottom = min(height, int(cy + side / 2))
    return main[top:bottom, left:right].copy(), left, top


class NativeMainFrameVerifier:
    def __init__(self, config_provider, recorder, counts):
        self._config_provider = config_provider
        self.recorder = recorder
        self.counts = counts
        self.detector = NativeEvidenceVerifier(self.config.detector)
        self._scheduler = _VerifierScheduler()

    @property
    def config(self):
        return self._config_provider()

    def request_stop(self):
        self.detector.request_stop()

    def close(self):
        self.detector.close()

    def detect(self, image, *, priority="cover"):
        self.detector.config = self.config.detector
        with self._scheduler.lease(priority):
            return self.detector.detect(image)

    def read_frame(self, camera_id, epoch, source, maximum_width=0):
        row = self.recorder.recording_at(camera_id, epoch, source=source)
        if not row:
            return None
        command = [
            self.config.ffmpeg_path,
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-ss",
            str(max(0, epoch - float(row["start_epoch"]))),
            "-i",
            str(row["path"]),
            "-frames:v",
            "1",
            "-an",
            "-sn",
        ]
        if maximum_width:
            command += ["-vf", f"scale='min({maximum_width},iw)':-2"]
        command += [
            "-threads",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "bmp",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        result = subprocess.run(command, capture_output=True, timeout=12)
        if result.returncode or not result.stdout:
            self.counts["decode_failed"] += 1
            return None
        return cv2.imdecode(
            np.frombuffer(result.stdout, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

    @staticmethod
    def project_main(candidate, main):
        """Locate verification crops using scene geometry, not object appearance."""
        alignment = estimate_stream_alignment(candidate.image, main)
        if alignment is None:
            return []
        sx, sy, ox, oy = alignment
        if not (
            0.25 < sx < 4
            and 0.25 < sy < 4
            and abs(ox) < 1
            and abs(oy) < 1
        ):
            return []
        live_height, live_width = candidate.image.shape[:2]
        main_height, main_width = main.shape[:2]
        projected = []
        for original in candidate.objects:
            box = original["box"]
            x1 = (box["x1"] / live_width * sx + ox) * main_width
            y1 = (box["y1"] / live_height * sy + oy) * main_height
            x2 = (box["x2"] / live_width * sx + ox) * main_width
            y2 = (box["y2"] / live_height * sy + oy) * main_height
            if not (x1 < x2 and y1 < y2):
                continue
            # Edge-touching live boxes often project a few pixels past main after
            # modest stream scale/offset. Clip those instead of dropping siblings.
            cx1 = min(max(0.0, x1), float(main_width))
            cy1 = min(max(0.0, y1), float(main_height))
            cx2 = min(max(0.0, x2), float(main_width))
            cy2 = min(max(0.0, y2), float(main_height))
            if not (cx1 < cx2 and cy1 < cy2):
                continue
            projected_area = (x2 - x1) * (y2 - y1)
            clipped_area = (cx2 - cx1) * (cy2 - cy1)
            if projected_area <= 0 or clipped_area / projected_area < 0.5:
                continue
            obj = deepcopy(original)
            obj.update(
                box={"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2},
                detection_frame_width=main_width,
                detection_frame_height=main_height,
                frame_source="recorded_main",
                frame_captured_at_epoch=candidate.epoch,
                snapshot_visible=True,
                native_alignment={
                    "scale_x": sx,
                    "scale_y": sy,
                    "offset_x": ox,
                    "offset_y": oy,
                },
            )
            projected.append(obj)
        return projected

    def _verify_frame(
        self,
        camera_id,
        candidate,
        main,
        frame_epoch,
        *,
        priority,
        cancelled,
        projector=None,
    ):
        aligned = (projector or self.project_main)(candidate, main)
        if not aligned:
            return "unaligned", None
        obj = aligned[0]
        crop, left, top = context_crop(main, obj["box"])
        if image_quality(crop) is None:
            return "unclear", None
        if cancelled is not None and cancelled.is_set():
            return "stopped", None
        detected = self.detect(crop, priority=priority)
        relevant = []
        nearby = False
        for item in detected:
            if item["label"] != obj["label"]:
                continue
            item = deepcopy(item)
            item["box"] = {
                key: value + (left if key.startswith("x") else top)
                for key, value in item["box"].items()
            }
            expected, actual = obj["box"], item["box"]
            nearby |= (
                min(expected["x2"], actual["x2"])
                > max(expected["x1"], actual["x1"])
                and min(expected["y2"], actual["y2"])
                > max(expected["y1"], actual["y1"])
            )
            if matches_object_extent(expected, actual):
                relevant.append(item)

        config = self.config.detector
        threshold = self._object_threshold(camera_id, obj)
        accepted = [
            item for item in relevant
            if item["confidence"] >= threshold
        ]
        if accepted:
            actual = max(accepted, key=lambda item: item["confidence"])
            cover = dict(
                obj,
                box=actual["box"],
                confidence=actual["confidence"],
                native_cover_verified=True,
                frame_captured_at_epoch=frame_epoch,
            )
            if len(aligned) <= 1:
                scene = [deepcopy(cover)]
                scene[0].update(
                    detection_frame_width=main.shape[1],
                    detection_frame_height=main.shape[0],
                    frame_source="recorded_main",
                    snapshot_visible=True,
                    box_provenance=scene[0].get("box_provenance") or "detected_in_main",
                )
            else:
                scene = self.match_scene_detections(
                    camera_id,
                    aligned,
                    main,
                    frame_epoch,
                    primary=cover,
                    detections=None,
                    priority=priority,
                    cancelled=cancelled,
                )
            return "confirmed", (main, cover, frame_epoch, scene)
        return ("ambiguous" if nearby else "negative"), None

    def _object_threshold(self, camera_id, obj):
        config = self.config.detector
        threshold = config.event_class_confidence_thresholds.get(
            obj["label"],
            config.confidence_threshold,
        )
        camera = next(
            (camera for camera in self.config.cameras if camera.id == camera_id),
            None,
        )
        if camera is None:
            return threshold
        zone_thresholds = [
            zone.confidence_threshold
            for zone in camera.zones
            if zone.name in obj.get("zones", [])
            and zone.confidence_threshold is not None
        ]
        return min([threshold, *zone_thresholds])

    def match_scene_detections(
        self,
        camera_id,
        aligned_objects,
        main,
        frame_epoch,
        *,
        primary=None,
        detections=None,
        priority="cover",
        cancelled=None,
    ):
        """Match projected inventory objects onto one verified main raster.

        Extent matches claim first (primary, then siblings) so slot-fill cannot
        steal a detection another scene object already extent-matches. Remaining
        identities may then claim the nearest unused same-label detection within
        a bounded center distance. Unpaired inventory beyond the supplied scene
        objects is never invented.
        """
        from .native_episode_identity import best_objects_by_episode_identity

        if cancelled is not None and cancelled.is_set():
            return [deepcopy(primary)] if primary is not None else []
        if detections is None:
            detections = self.detect(main, priority=priority)
        if cancelled is not None and cancelled.is_set():
            return [deepcopy(primary)] if primary is not None else []

        main_height, main_width = main.shape[:2]
        matched = []
        used_detection_ids = set()
        diagonal = max(1.0, (main_width ** 2 + main_height ** 2) ** 0.5)
        slot_distance_limit = 0.35 * diagonal

        def identity(item):
            if item.get("episode_identity"):
                return ("episode", str(item["episode_identity"]))
            if item.get("track_id") is not None:
                return ("track", item.get("track_id"))
            if item.get("native_identity"):
                return ("native", str(item["native_identity"]))
            if item.get("native_track_id") is not None:
                return (
                    "native_track",
                    str(item.get("label") or ""),
                    item.get("native_track_id"),
                )
            return None

        def center(box):
            return (
                (float(box["x1"]) + float(box["x2"])) * 0.5,
                (float(box["y1"]) + float(box["y2"])) * 0.5,
            )

        def center_distance(left, right):
            lx, ly = center(left)
            rx, ry = center(right)
            return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5

        def claim_extent(obj):
            threshold = self._object_threshold(camera_id, obj)
            candidates = []
            for index, detected in enumerate(detections or ()):
                if index in used_detection_ids:
                    continue
                if detected.get("label") != obj.get("label"):
                    continue
                if float(detected.get("confidence") or 0) < threshold:
                    continue
                actual = detected.get("box") or {}
                if not all(name in actual for name in ("x1", "y1", "x2", "y2")):
                    continue
                if matches_object_extent(obj.get("box") or {}, actual):
                    candidates.append((index, detected))
            if not candidates:
                return None
            index, detected = max(
                candidates,
                key=lambda item: float(item[1].get("confidence") or 0),
            )
            used_detection_ids.add(index)
            return detected

        def claim_slot(obj):
            if not obj.get("box"):
                return None
            expected = obj["box"]
            try:
                expected_area = max(
                    1.0,
                    (float(expected["x2"]) - float(expected["x1"]))
                    * (float(expected["y2"]) - float(expected["y1"])),
                )
            except (KeyError, TypeError, ValueError):
                return None
            threshold = self._object_threshold(camera_id, obj)
            slot_candidates = []
            for index, detected in enumerate(detections or ()):
                if index in used_detection_ids:
                    continue
                if detected.get("label") != obj.get("label"):
                    continue
                if float(detected.get("confidence") or 0) < threshold:
                    continue
                actual = detected.get("box") or {}
                if not all(name in actual for name in ("x1", "y1", "x2", "y2")):
                    continue
                try:
                    actual_area = max(
                        1.0,
                        (float(actual["x2"]) - float(actual["x1"]))
                        * (float(actual["y2"]) - float(actual["y1"])),
                    )
                except (TypeError, ValueError):
                    continue
                # Reject same-class fragments / oversized blobs; slot-fill is for
                # modest projection miss, not a different-sized detection.
                if min(expected_area, actual_area) / max(expected_area, actual_area) < 0.4:
                    continue
                distance = center_distance(expected, actual)
                if distance > slot_distance_limit:
                    continue
                slot_candidates.append((distance, index, detected))
            if not slot_candidates:
                return None
            _distance, index, detected = min(slot_candidates, key=lambda item: item[0])
            used_detection_ids.add(index)
            return detected

        def promote(obj, detected, *, provenance, source):
            item = deepcopy(obj)
            item.update(
                box=deepcopy(detected["box"]),
                confidence=detected["confidence"],
                detection_frame_width=main_width,
                detection_frame_height=main_height,
                frame_source="recorded_main",
                frame_captured_at_epoch=frame_epoch,
                snapshot_visible=True,
                native_cover_verified=True,
                box_provenance=provenance,
                verification={"status": "confirmed", "source": source},
            )
            return item

        def keep_unmatched_primary(item):
            """Keep primary box when no OD claimed; preserve crop-verified provenance."""
            prior = item.get("box_provenance")
            if item.get("native_cover_verified") and prior != "projected_main":
                provenance = prior or "detected_in_main"
            else:
                provenance = "projected_main"
            item.update(
                detection_frame_width=main_width,
                detection_frame_height=main_height,
                frame_source="recorded_main",
                frame_captured_at_epoch=frame_epoch,
                snapshot_visible=True,
                native_cover_verified=True,
                box_provenance=provenance,
            )
            return item

        # Match at most one box per episode identity on this cover frame.
        scene_objects = list(aligned_objects or ())
        if any(
            isinstance(item, dict) and item.get("episode_identity")
            for item in scene_objects
        ):
            scene_objects = best_objects_by_episode_identity(scene_objects)

        primary_key = identity(primary) if primary is not None else None

        def is_primary_duplicate(obj):
            key = identity(obj)
            if primary_key is not None and key is not None and key == primary_key:
                return True
            if primary is not None and key is None:
                expected = primary.get("box") or {}
                actual = obj.get("box") or {}
                if (
                    all(name in expected and name in actual for name in ("x1", "y1", "x2", "y2"))
                    and matches_object_extent(expected, actual)
                    and obj.get("label") == primary.get("label")
                ):
                    return True
            return False

        siblings = [
            obj for obj in scene_objects
            if isinstance(obj, dict)
            and obj.get("label")
            and obj.get("box")
            and not is_primary_duplicate(obj)
        ]

        # Pass 1: exclusive extent claims (primary first, then siblings).
        primary_item = deepcopy(primary) if primary is not None else None
        primary_result = None
        if primary_item is not None:
            detected = claim_extent(primary_item)
            if detected is not None:
                primary_result = promote(
                    primary_item,
                    detected,
                    provenance="detected_in_main",
                    source="main",
                )

        sibling_results = []
        unmatched_siblings = []
        for obj in siblings:
            detected = claim_extent(obj)
            if detected is None:
                unmatched_siblings.append(obj)
                continue
            sibling_results.append(
                promote(obj, detected, provenance="detected_in_main", source="main")
            )

        # Pass 2: identity slot-fill only for leftovers (primary first).
        # Crop-verified primaries keep their confirmed box; still reserve the
        # nearest OD so a sibling cannot slot-claim the crop subject.
        if primary_item is not None and primary_result is None:
            if primary_item.get("native_cover_verified"):
                claim_slot(primary_item)
                primary_result = keep_unmatched_primary(primary_item)
            else:
                detected = claim_slot(primary_item)
                if detected is not None:
                    primary_result = promote(
                        primary_item,
                        detected,
                        provenance="identity_slot_from_main",
                        source="main_identity_slot",
                    )
                else:
                    primary_result = keep_unmatched_primary(primary_item)

        for obj in unmatched_siblings:
            key = identity(obj)
            if key is None:
                continue
            detected = claim_slot(obj)
            if detected is None:
                continue
            sibling_results.append(
                promote(
                    obj,
                    detected,
                    provenance="identity_slot_from_main",
                    source="main_identity_slot",
                )
            )

        # Primary always leads so snapshot_primary_subject stays on the admitted subject.
        matched = []
        if primary_result is not None:
            matched.append(primary_result)
        matched.extend(sibling_results)
        return matched

    def verify_candidate(
        self,
        camera_id,
        candidate,
        *,
        priority="cover",
        cancelled=None,
        frame_reader=None,
        projector=None,
    ):
        frame_votes = []
        cover = None
        for offset in (0.0, .5, -.5, 1.0, -1.0):
            if cancelled is not None and cancelled.is_set():
                return {"status": "unverified", "reason": "stopped"}
            frame_epoch = candidate.epoch + offset
            main = (frame_reader or self.read_frame)(
                camera_id,
                frame_epoch,
                "main",
            )
            if cancelled is not None and cancelled.is_set():
                return {"status": "unverified", "reason": "stopped"}
            if (
                main is None
                or main.shape[0] * main.shape[1]
                <= candidate.image.shape[0] * candidate.image.shape[1]
            ):
                vote, candidate_cover = "unavailable", None
            else:
                vote, candidate_cover = self._verify_frame(
                    camera_id,
                    candidate,
                    main,
                    frame_epoch,
                    priority=priority,
                    cancelled=cancelled,
                    projector=projector,
                )
            if vote == "stopped":
                return {"status": "unverified", "reason": "stopped"}
            frame_votes.append(vote)
            if vote == "confirmed":
                cover = candidate_cover
                break
        vote = next(
            (
                value
                for value in (
                    "confirmed",
                    "ambiguous",
                    "unaligned",
                    "unavailable",
                    "unclear",
                )
                if value in frame_votes
            ),
            "negative",
        )
        result = {
            "status": (
                "confirmed"
                if cover is not None
                else "rejected"
                if vote == "negative"
                else "unverified"
            ),
            "vote": vote,
            "votes": [vote],
            "checks": [{"epoch": candidate.epoch, "votes": frame_votes}],
            "reason": "main_crop_verification",
        }
        if cover is not None:
            main, primary, frame_epoch, scene = cover
            # Admission keeps the historical 3-tuple; cover refinement uses scene.
            result["cover"] = (main, primary, frame_epoch)
            result["cover_objects"] = scene or [primary]
        return result
