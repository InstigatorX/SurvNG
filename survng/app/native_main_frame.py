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
            values = [
                (box["x1"] / live_width * sx + ox) * main_width,
                (box["y1"] / live_height * sy + oy) * main_height,
                (box["x2"] / live_width * sx + ox) * main_width,
                (box["y2"] / live_height * sy + oy) * main_height,
            ]
            if not (
                0 <= values[0] < values[2] <= main_width
                and 0 <= values[1] < values[3] <= main_height
            ):
                continue
            obj = deepcopy(original)
            obj.update(
                box=dict(zip(("x1", "y1", "x2", "y2"), values)),
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
    ):
        aligned = self.project_main(candidate, main)
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
        threshold = config.event_class_confidence_thresholds.get(
            obj["label"],
            config.confidence_threshold,
        )
        camera = next(
            camera for camera in self.config.cameras
            if camera.id == camera_id
        )
        zone_thresholds = [
            zone.confidence_threshold
            for zone in camera.zones
            if zone.name in obj.get("zones", [])
            and zone.confidence_threshold is not None
        ]
        threshold = min([threshold, *zone_thresholds])
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
            return "confirmed", (main, cover, frame_epoch)
        return ("ambiguous" if nearby else "negative"), None

    def verify_candidate(
        self,
        camera_id,
        candidate,
        *,
        priority="cover",
        cancelled=None,
    ):
        frame_votes = []
        cover = None
        for offset in (0.0, .5, -.5, 1.0, -1.0):
            if cancelled is not None and cancelled.is_set():
                return {"status": "unverified", "reason": "stopped"}
            frame_epoch = candidate.epoch + offset
            main = self.read_frame(camera_id, frame_epoch, "main")
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
            result["cover"] = cover
        return result
