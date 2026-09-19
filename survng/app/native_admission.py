"""Bounded asynchronous main-recording verification before incident creation."""
from collections import Counter, deque
import logging
import threading
import time

from .native_evidence_common import Candidate, resize_objects
from .native_main_frame import context_crop

LOGGER = logging.getLogger(__name__)


class NativeAdmission:
    """Own nomination/retry/decision policy, not frame verification mechanics."""

    def __init__(self, main_frames):
        self.main_frames = main_frames
        self.condition = threading.Condition()
        self.jobs = {}
        self.results = {}
        self.counts = Counter()
        self.recent = deque(maxlen=32)
        self.closed = False
        self.thread = None

    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name="native-admission",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        with self.condition:
            self.closed = True
            for job in self.jobs.values():
                job["cancelled"].set()
            self.jobs.clear()
            self.results.clear()
            self.condition.notify_all()
        if self.thread:
            self.thread.join(55)
            if self.thread.is_alive():
                raise RuntimeError("native admission worker did not stop")

    def offer(self, token, camera_id, epoch, frame, obj, size):
        with self.condition:
            if self.closed or token in self.results:
                return
            job = self.jobs.get(token)
            if job is None:
                if len(self.jobs) + len(self.results) >= 32:
                    return
                job = self.jobs[token] = {
                    "camera_id": camera_id,
                    "samples": [],
                    "due": time.monotonic() + 15,
                    "deadline": time.monotonic() + 90,
                    "cancelled": threading.Event(),
                }
                self.counts["nominated"] += 1
            samples = job["samples"]
            if (
                frame is not None
                and (not samples or epoch > samples[-1].epoch)
                and all(abs(epoch - candidate.epoch) >= .4 for candidate in samples)
            ):
                height, width = frame.shape[:2]
                image = frame if not frame.flags.writeable else frame.copy()
                image.setflags(write=False)
                candidate = Candidate(
                    epoch,
                    image,
                    resize_objects([obj], size, (width, height)),
                    0,
                )
                if len(samples) < 3:
                    samples.append(candidate)
                else:
                    samples[-1] = candidate
            self.condition.notify_all()

    def poll(self, token):
        with self.condition:
            return self.results.pop(token, None)

    def cancel(self, token):
        with self.condition:
            job = self.jobs.pop(token, None)
            if job is not None:
                job["cancelled"].set()
            self.results.pop(token, None)

    def status(self):
        with self.condition:
            return {
                "pending": len(self.jobs),
                "counters": dict(self.counts),
                "recent": list(self.recent),
            }

    def verify(self, camera_id, samples, *, cancelled=None):
        votes = []
        checks = []
        best = None
        for candidate in samples[:3]:
            if self.closed or (cancelled is not None and cancelled.is_set()):
                return {"status": "unverified", "reason": "stopped"}
            if self.main_frames is None:
                raise RuntimeError("main-frame verifier is unavailable")
            result = self.main_frames.verify_candidate(
                camera_id,
                candidate,
                priority="admission",
                cancelled=cancelled,
            )
            if result.get("reason") == "stopped":
                return {"status": "unverified", "reason": "stopped"}
            vote = str(result.get("vote") or "unavailable")
            votes.append(vote)
            checks.extend(result.get("checks") or [])
            if result.get("status") == "confirmed":
                best = result.get("cover")
                break
        status = (
            "confirmed"
            if best is not None
            else "rejected"
            if votes.count("negative") >= 3
            else "unverified"
        )
        result = {
            "status": status,
            "votes": votes,
            "checks": checks,
            "reason": "main_crop_verification",
        }
        if best is not None:
            result["cover"] = best
        return result

    def _run(self):
        while True:
            with self.condition:
                if self.closed:
                    return
                ready = [
                    (token, job)
                    for token, job in self.jobs.items()
                    if job["due"] <= time.monotonic()
                ]
                if not ready:
                    self.condition.wait(1)
                    continue
                token, job = min(ready, key=lambda pair: pair[1]["due"])
                samples = list(job["samples"])
                job["due"] = time.monotonic() + 10
            try:
                result = self.verify(
                    job["camera_id"],
                    samples,
                    cancelled=job["cancelled"],
                )
            except Exception as exc:
                result = {
                    "status": "unverified",
                    "reason": type(exc).__name__,
                }
                LOGGER.warning(
                    "Native admission verification unavailable for %s (%s)",
                    job["camera_id"],
                    type(exc).__name__,
                )
            with self.condition:
                if self.jobs.get(token) is not job or self.closed:
                    continue
                if (
                    result["status"] != "confirmed"
                    and time.monotonic() < job["deadline"]
                    and [sample.epoch for sample in samples]
                    != [sample.epoch for sample in job["samples"]]
                ):
                    continue
                if (
                    result["status"] == "unverified"
                    and (
                        not result.get("votes")
                        or any(
                            vote in {"unavailable", "unaligned"}
                            for vote in result["votes"]
                        )
                    )
                    and time.monotonic() < job["deadline"]
                ):
                    continue
                self.jobs.pop(token)
                self.results[token] = result
                self.counts[result["status"]] += 1
                self.recent.append(
                    {
                        "camera_id": job["camera_id"],
                        "status": result["status"],
                        "reason": result["reason"],
                    }
                )
