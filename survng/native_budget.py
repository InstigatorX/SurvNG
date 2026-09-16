"""Per-stream inference admission. Motion and predictions are never evidence."""
from __future__ import annotations

from collections import Counter
import math
import threading

from survng.motion_regions import MotionRegions

BUDGETS = {}  # Owned by native graph lifetime; gvapython adapters borrow entries.
MOTION_PROPERTIES = ('block_size', 'motion_threshold', 'min_persistence', 'max_miss',
                     'iou_threshold', 'smooth_alpha', 'confirm_frames', 'pixel_diff_threshold', 'min_rel_area')


def _inside(x, y, points):
    inside = False
    a = points[-1]
    for b in points:
        if (a['y'] > y) != (b['y'] > y) and x < (b['x']-a['x'])*(y-a['y'])/(b['y']-a['y'])+a['x']:
            inside = not inside
        a = b
    return inside


def overlaps(rect, points, padding):
    """Rectangle vs polygon, including edges and an approach margin."""
    x1, y1, x2, y2 = rect
    x1, y1, x2, y2 = x1-padding, y1-padding, x2+padding, y2+padding
    if any(x1 <= p['x'] <= x2 and y1 <= p['y'] <= y2 for p in points):
        return True
    if any(_inside(x, y, points) for x, y in ((x1,y1),(x1,y2),(x2,y1),(x2,y2))):
        return True
    # Clip each polygon edge against the rectangle (Liang-Barsky).
    a = points[-1]
    for b in points:
        dx, dy = b['x']-a['x'], b['y']-a['y']
        low, high = 0.0, 1.0
        for p, q in ((-dx,a['x']-x1),(dx,x2-a['x']),(-dy,a['y']-y1),(dy,y2-a['y'])):
            if p == 0:
                if q < 0:
                    high = -1
                    break
            elif p < 0:
                low = max(low, q/p)
            else:
                high = min(high, q/p)
        if low <= high:
            return True
        a = b
    return False


class NativeBudget:
    def __init__(self, plan):
        self.plan = plan
        self.config = plan['budget']
        self.zones = [z for z in plan.get('zones', []) if z.get('enabled', True)
                      and z.get('behavior', 'incident') == 'incident' and len(z.get('points', [])) >= 3]
        exclusions = [z for z in plan.get('zones', []) if z.get('enabled', True)
                      and z.get('exclude_from_ema', False) and len(z.get('points', [])) >= 3]
        self.motion_regions = MotionRegions(self.zones, exclusions, self.config['approach_padding']) if exclusions else None
        self.lock = threading.RLock()
        self.counts = Counter()
        self.latest_pts = None
        self.active_until = None
        self.next_due = None
        self.mode = 'active'
        self.selected_full_frame = True

    def relevant(self, rect, label=None, confidence=1):
        if not self.zones:
            threshold = self.plan.get('class_confidence', {}).get(label, self.plan.get('object_confidence', 0))
            return label is None or confidence >= threshold
        for zone in self.zones:
            classes = [c.strip().lower() for c in zone.get('object_classes', [])]
            if label is not None:
                if classes and label.lower() not in classes:
                    continue
                threshold = zone.get('confidence_threshold')
                if threshold is None:
                    threshold = self.plan.get('class_confidence', {}).get(label, self.plan.get('object_confidence', 0))
                if confidence < threshold:
                    continue
            if overlaps(rect, zone['points'], self.config['approach_padding']):
                return True
        return False

    def wake(self, pts, reason):
        hold = max(self.config['cooldown_seconds'], self.config.get('confirmation_hold_seconds', 0))
        self.active_until = max(self.active_until or pts, pts + hold)
        self.counts[reason] += 1

    def motion_relevant(self, rectangles):
        relevant = False
        for rect in rectangles:
            if not self.relevant(rect):
                continue
            if self.motion_regions is not None and not self.motion_regions.allows(rect):
                self.counts['excluded_motion_regions'] += 1
            else:
                relevant = True
        return relevant

    def select(self, pts, motion=()):
        with self.lock:
            if not math.isfinite(pts) or pts < 0:
                raise ValueError('invalid budget PTS')
            if self.latest_pts is None or pts < self.latest_pts:
                self.next_due = None
                self.active_until = None
                self.wake(pts, 'startup')
            self.latest_pts = pts
            if self.motion_relevant(motion):
                self.wake(pts, 'motion_wakes')
            mode = 'active' if pts <= self.active_until else 'idle'
            if mode != self.mode:
                self.next_due = None
                self.counts[mode + '_transitions'] += 1
            self.mode = mode
            self.counts['sampled_frames'] += 1
            rate = self.config[mode + '_fps']
            if self.next_due is not None and pts + 1e-6 < self.next_due:
                self.counts['skipped_frames'] += 1
                return False
            self.next_due = max((self.next_due if self.next_due is not None else pts) + 1/rate, pts)
            self.selected_full_frame = mode == 'idle'
            self.counts['admitted_frames'] += 1
            return True

    def objects(self, objects, width, height, pts):
        with self.lock:
            # Feedback expires in source time. Predictions never enter this API.
            for obj in objects:
                box = obj['box']
                x, y = (box['x1']+box['x2'])/2/width, box['y2']/height
                if self.relevant((x,y,x,y), obj['label'], obj['confidence']):
                    self.wake(pts, 'object_wakes')
                    break

    def status(self):
        with self.lock:
            return {'mode': self.mode, 'target_fps': self.config[self.mode+'_fps'],
                    'motion_enabled': self.config['motion_enabled'],
                    'idle_fps': self.config['idle_fps'], 'active_fps': self.config['active_fps'],
                    'cooldown_remaining_seconds': max(0, (self.active_until or 0)-(self.latest_pts or 0)),
                    **self.counts}
