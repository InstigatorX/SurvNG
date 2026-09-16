"""Bounded asynchronous main-recording verification before incident creation."""
from collections import Counter, deque
from copy import deepcopy
import logging
import threading
import time

from .native_evidence import Candidate, image_quality, resize_objects

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


def matches(expected, actual):
    intersection = max(0, min(expected['x2'], actual['x2'])-max(expected['x1'], actual['x1'])) * max(0, min(expected['y2'], actual['y2'])-max(expected['y1'], actual['y1']))
    area = lambda b: max(1, (b['x2']-b['x1'])*(b['y2']-b['y1']))
    return intersection/area(actual) >= .5 and intersection/(area(actual)+area(expected)-intersection) >= .3


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
                                         'deadline': time.monotonic()+90}
                self.counts['nominated'] += 1
            if frame is not None and len(job['samples']) < 3 and all(abs(epoch-c.epoch) >= .4 for c in job['samples']):
                h, w = frame.shape[:2]
                job['samples'].append(Candidate(epoch, frame.copy(), resize_objects([obj], size, (w,h)), 0))
            self.condition.notify_all()

    def poll(self, token):
        with self.condition:
            return self.results.pop(token, None)

    def cancel(self, token):
        with self.condition:
            self.jobs.pop(token, None)
            self.results.pop(token, None)

    def status(self):
        with self.condition:
            return {'pending': len(self.jobs), 'counters': dict(self.counts), 'recent': list(self.recent)}

    def verify(self, camera_id, samples):
        votes, best = [], None
        for candidate in samples:
            if self.closed:
                return {'status': 'unverified', 'reason': 'stopped'}
            main = self.evidence.read_frame(camera_id, candidate.epoch, 'main')
            if main is None:
                votes.append('unavailable')
                continue
            if main.shape[0]*main.shape[1] <= candidate.image.shape[0]*candidate.image.shape[1]:
                votes.append('unavailable')
                continue
            aligned = self.evidence.match_main(candidate, main)
            if not aligned:
                votes.append('unaligned')
                continue
            obj = aligned[0]
            crop, left, top = context_crop(main, obj['box'])
            if image_quality(crop) is None:
                votes.append('unclear')
                continue
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
                cover = dict(obj, box=actual['box'], confidence=actual['confidence'], native_cover_verified=True)
                best = (main, cover, candidate.epoch)
                votes.append('confirmed')
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
                result = self.verify(job['camera_id'], samples)
            except Exception as exc:
                result = {'status': 'unverified', 'reason': type(exc).__name__}
                LOGGER.warning('Native admission verification unavailable for %s (%s)', job['camera_id'], type(exc).__name__)
            with self.condition:
                if self.jobs.get(token) is not job or self.closed:
                    continue  # Camera/session/policy canceled while verification ran.
                if result['status'] == 'unverified' and (not result.get('votes') or any(v in {'unavailable', 'unaligned'} for v in result['votes'])) and time.monotonic() < job['deadline']:
                    continue
                self.jobs.pop(token)
                self.results[token] = result
                self.counts[result['status']] += 1
                self.recent.append({'camera_id': job['camera_id'], 'status': result['status'], 'reason': result['reason']})
