"""Bounded asynchronous main-recording verification before incident creation."""
from collections import Counter, deque
from copy import deepcopy
import logging
import threading
import time

from .native_evidence import Candidate, image_quality, resize_objects, matches_object_extent as matches

LOGGER = logging.getLogger(__name__)


def context_crop(main, box):
    """Keep context and native pixels; never upscale an object for verification."""
    h, w = main.shape[:2]
    x1, y1, x2, y2 = (box[k] for k in ('x1', 'y1', 'x2', 'y2'))
    side = max(192, 3 * max(x2-x1, y2-y1))
    cx, cy = (x1+x2)/2, (y1+y2)/2
    left, top = max(0, int(cx-side/2)), max(0, int(cy-side/2))
    right, bottom = min(w, int(cx+side/2)), min(h, int(cy+side/2))
    return main[top:bottom, left:right].copy(), left, top


class NativeAdmission:
    def __init__(self, evidence):
        self.evidence = evidence
        self.condition = threading.Condition()
        self.jobs = {}
        self.results = {}
        self.counts = Counter()
        self.recent = deque(maxlen=32)
        self.closed = False
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name='native-admission', daemon=True)
        self.thread.start()

    def stop(self):
        with self.condition:
            self.closed = True
            for job in self.jobs.values():
                job['cancelled'].set()
            self.jobs.clear()
            self.results.clear()
            self.condition.notify_all()
        if self.thread:
            self.thread.join(55)
            if self.thread.is_alive():
                raise RuntimeError('native admission worker did not stop')

    def offer(self, token, camera_id, epoch, frame, obj, size):
        with self.condition:
            if self.closed or token in self.results:
                return
            job = self.jobs.get(token)
            if job is None:
                if len(self.jobs) + len(self.results) >= 32:
                    return  # Caller expires this as unverified, never as rejection.
                job = self.jobs[token] = {'camera_id': camera_id, 'samples': [], 'due': time.monotonic()+15,
                                         'deadline': time.monotonic()+90, 'cancelled': threading.Event()}
                self.counts['nominated'] += 1
            samples = job['samples']
            if (frame is not None and (not samples or epoch > samples[-1].epoch)
                    and all(abs(epoch-c.epoch) >= .4 for c in samples)):
                h, w = frame.shape[:2]
                # Capture publishes immutable allocations. Multiple objects in
                # one frame can retain those pixels instead of duplicating them.
                image = frame if not frame.flags.writeable else frame.copy()
                image.setflags(write=False)
                candidate = Candidate(epoch, image, resize_objects([obj], size, (w,h)), 0)
                if len(samples) < 3:
                    samples.append(candidate)
                else:
                    # Keep early evidence plus a recent view. Freezing all
                    # three at entry misses later, unobstructed confirmation.
                    samples[-1] = candidate
            self.condition.notify_all()

    def poll(self, token):
        with self.condition:
            return self.results.pop(token, None)

    def cancel(self, token):
        with self.condition:
            job = self.jobs.pop(token, None)
            if job is not None:
                job['cancelled'].set()
            self.results.pop(token, None)

    def status(self):
        with self.condition:
            return {'pending': len(self.jobs), 'counters': dict(self.counts), 'recent': list(self.recent)}

    def verify(self, camera_id, samples, *, cancelled=None):
        votes, best = [], None
        for candidate in samples:
            if self.closed or (cancelled is not None and cancelled.is_set()):
                return {'status': 'unverified', 'reason': 'stopped'}
            main = self.evidence.read_frame(camera_id, candidate.epoch, 'main')
            if self.closed or (cancelled is not None and cancelled.is_set()):
                return {'status': 'unverified', 'reason': 'stopped'}
            if main is None:
                votes.append('unavailable')
                continue
            if main.shape[0]*main.shape[1] <= candidate.image.shape[0]*candidate.image.shape[1]:
                votes.append('unavailable')
                continue
            aligned = self.evidence.match_main(candidate, main)
            frame_epoch = candidate.epoch
            if not aligned:
                # Independently recorded streams can differ by a fraction of a
                # second. Keep the appearance and extent safeguards, but look
                # for the matching pose within a bounded recording window.
                for offset in (.5, -.5, 1., -1.):
                    if self.closed or (cancelled is not None and cancelled.is_set()):
                        return {'status': 'unverified', 'reason': 'stopped'}
                    alternate = self.evidence.read_frame(camera_id, candidate.epoch + offset, 'main')
                    if self.closed or (cancelled is not None and cancelled.is_set()):
                        return {'status': 'unverified', 'reason': 'stopped'}
                    if alternate is None or alternate.shape[0]*alternate.shape[1] <= candidate.image.shape[0]*candidate.image.shape[1]:
                        continue
                    aligned = self.evidence.match_main(candidate, alternate)
                    if aligned:
                        main, frame_epoch = alternate, candidate.epoch + offset
                        break
            if not aligned:
                votes.append('unaligned')
                continue
            obj = aligned[0]
            crop, left, top = context_crop(main, obj['box'])
            if image_quality(crop) is None:
                votes.append('unclear')
                continue
            if self.closed or (cancelled is not None and cancelled.is_set()):
                return {'status': 'unverified', 'reason': 'stopped'}
            detected = self.evidence.verifier.detect(crop)
            relevant = []
            nearby = False
            for item in detected:
                if item['label'] != obj['label']:
                    continue
                item = deepcopy(item)
                item['box'] = {k: v+(left if k.startswith('x') else top) for k,v in item['box'].items()}
                a, b = obj['box'], item['box']
                nearby |= min(a['x2'], b['x2']) > max(a['x1'], b['x1']) and min(a['y2'], b['y2']) > max(a['y1'], b['y1'])
                if matches(obj['box'], item['box']):
                    relevant.append(item)
            config = self.evidence.config.detector
            threshold = config.event_class_confidence_thresholds.get(obj['label'], config.confidence_threshold)
            # Preserve a lower explicit zone threshold already used for nomination.
            camera = next(c for c in self.evidence.config.cameras if c.id == camera_id)
            thresholds = [z.confidence_threshold for z in camera.zones if z.name in obj.get('zones', []) and z.confidence_threshold is not None]
            threshold = min([threshold, *thresholds])
            accepted = [d for d in relevant if d['confidence'] >= threshold]
            if accepted:
                actual = max(accepted, key=lambda d: d['confidence'])
                cover = dict(obj, box=actual['box'], confidence=actual['confidence'], native_cover_verified=True,
                             frame_captured_at_epoch=frame_epoch)
                best = (main, cover, frame_epoch)
                votes.append('confirmed')
                # One clear positive decides admission. Remaining samples are
                # only needed to establish rejection, not to reconfirm success.
                break
            else:
                votes.append('ambiguous' if nearby else 'negative')
        status = 'confirmed' if votes.count('confirmed') >= 1 else 'rejected' if votes.count('negative') >= 3 else 'unverified'
        result = {'status': status, 'votes': votes, 'reason': 'main_crop_verification'}
        if status == 'confirmed':
            result['cover'] = best
        return result

    def _run(self):
        while True:
            with self.condition:
                if self.closed:
                    return
                ready = [(token, job) for token,job in self.jobs.items() if job['due'] <= time.monotonic()]
                if not ready:
                    self.condition.wait(1)
                    continue
                token, job = min(ready, key=lambda pair: pair[1]['due'])
                samples = list(job['samples'])
                job['due'] = time.monotonic()+10
            try:
                result = self.verify(job['camera_id'], samples, cancelled=job['cancelled'])
            except Exception as exc:
                result = {'status': 'unverified', 'reason': type(exc).__name__}
                LOGGER.warning('Native admission verification unavailable for %s (%s)', job['camera_id'], type(exc).__name__)
            with self.condition:
                if self.jobs.get(token) is not job or self.closed:
                    continue  # Camera/session/policy canceled while verification ran.
                if (result['status'] != 'confirmed' and time.monotonic() < job['deadline']
                        and [s.epoch for s in samples] != [s.epoch for s in job['samples']]):
                    continue  # Evaluate a newer view before finalizing failure.
                if result['status'] == 'unverified' and (not result.get('votes') or any(v in {'unavailable', 'unaligned'} for v in result['votes'])) and time.monotonic() < job['deadline']:
                    continue
                self.jobs.pop(token)
                self.results[token] = result
                self.counts[result['status']] += 1
                self.recent.append({'camera_id': job['camera_id'], 'status': result['status'], 'reason': result['reason']})
