"""Bounded image evidence, independent of admission and representative covers.

Only existing decoded samples are considered. Encoding happens before the
recorded workflow releases its memory lease; no raw frames leave this module.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
import weakref
from dataclasses import dataclass

from .image_storage import DurableImageWriter, EncodedImage

MAX_GALLERY_IMAGES = 2
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
# At most 16 MiB of completed optional images can await persistence process-wide.
_PENDING_BATCHES = threading.BoundedSemaphore(2)


@dataclass(frozen=True)
class GalleryImage:
    key: str
    captured_epoch: float
    image: EncodedImage
    metadata: dict


class GalleryBatch:
    """Own a nonblocking capacity permit until persistence or result disposal."""

    def __init__(self):
        self.images: list[GalleryImage] = []
        self._release = weakref.finalize(self, _PENDING_BATCHES.release)

    def close(self):
        self.images.clear()
        self._release()


def _box(item):
    try:
        w, h = float(item['detection_frame_width']), float(item['detection_frame_height'])
        b = item['box']
        result = tuple(float(b[k]) / d for k, d in zip(('x1','y1','x2','y2'), (w,h,w,h)))
        if all(math.isfinite(v) for v in result) and result[2] > result[0] and result[3] > result[1]:
            return result
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    return None


def _useful(item):
    # This is evidence selection, not a repeat of the incident confidence gate.
    return (bool(item.get('label')) and _box(item) is not None
            and item.get('activity_role') != 'scene_context'
            and item.get('zone_admission_reason') != 'ignored_zone'
            and 'ignored_zone' not in (item.get('alert_reasons') or []))


def _distinct(candidate, previous):
    """Prefer different subjects/views, without image hashes or another model."""
    for old in previous:
        if candidate['camera_id'] != old.get('camera_id'):
            continue
        if abs(candidate['captured_epoch'] - old['captured_epoch']) < .1:
            return False
        if abs(candidate['captured_epoch'] - old['captured_epoch']) >= 2:
            continue
        a, b = candidate['objects'], old.get('objects', [])
        for current in a:
            box = _box(current)
            if not box:
                continue
            for prior in b:
                other = _box(prior)
                if not other:
                    continue
                # Spatial redundancy applies even when the classifier changes
                # cat to dog. It does not assert that the two labels are equal.
                distance = math.hypot((box[0]+box[2]-other[0]-other[2])/2,
                                      (box[1]+box[3]-other[1]-other[3])/2)
                area = (box[2]-box[0])*(box[3]-box[1])
                old_area = (other[2]-other[0])*(other[3]-other[1])
                if distance < .025 and .7 < area / old_area < 1.4:
                    return False
    return True


def encode_gallery(samples, selected, observations, *, camera_id, event_epoch, request, timing):
    limit = min(MAX_GALLERY_IMAGES, max(0, int((request or {}).get('remaining', 0))))
    if not limit:
        return None
    if not _PENDING_BATCHES.acquire(blocking=False):
        timing['gallery_capacity_skips'] = 1
        return None
    batch = GalleryBatch()
    started = time.monotonic()
    try:
        candidates = []
        previous = list((request or {}).get('frames', []))
        for sample in samples:
            if sample.frame is None or sample.frame.shape[0] * sample.frame.shape[1] > MAX_IMAGE_PIXELS:
                continue
            epoch = event_epoch + sample.offset
            objects = [dict(o) for o in observations
                       if abs(float(o.get('captured_at_epoch', -1))-epoch) < 1e-6 and _useful(o)]
            if not objects:
                continue
            metadata = {'camera_id':camera_id, 'captured_epoch':epoch, 'objects':objects,
                        'width':int(sample.frame.shape[1]), 'height':int(sample.frame.shape[0]),
                        'frame_source':'recorded_main', 'frame_timestamp_exact':sample.exact_timestamp,
                        'recording_path':sample.recording_path, 'role':'gallery'}
            if sample is selected:
                previous.append(metadata)
                continue
            def rank(o):
                x1,y1,x2,y2 = _box(o)
                clearance = min(x1,y1,1-x2,1-y2)
                return (o.get('activity_role') == 'active', bool(o.get('temporal_consensus')),
                        clearance > .002, (x2-x1)*(y2-y1), clearance, float(o.get('confidence', 0)))
            candidates.append((max(rank(o) for o in objects), sample, metadata))
        # If there is a moving subject, do not spend remaining gallery slots
        # on unrelated stationary context elsewhere in the scene.
        if any(rank[0] for rank, _, _ in candidates):
            candidates = [candidate for candidate in candidates if candidate[0][0]]
        for _, sample, metadata in sorted(candidates, key=lambda c:c[0], reverse=True):
            if len(batch.images) >= limit or time.monotonic()-started > .15:
                break
            if not _distinct(metadata, previous):
                continue
            # Encode just the winners at a fixed, economical JPEG quality.
            data = DurableImageWriter._encode('jpeg', sample.frame, 85)
            if not data or len(data) > MAX_IMAGE_BYTES:
                timing['gallery_encode_skips'] = timing.get('gallery_encode_skips', 0) + 1
                continue
            key = hashlib.sha256(f'{camera_id}|{sample.recording_path}|{metadata["captured_epoch"]:.6f}'.encode()).hexdigest()
            batch.images.append(GalleryImage(key, metadata['captured_epoch'], EncodedImage(data), metadata))
            previous.append(metadata)
        timing['gallery_encoded_images'] = len(batch.images)
        timing['gallery_encoded_bytes'] = sum(len(item.image.data) for item in batch.images)
        if batch.images:
            return batch
        batch.close()
        return None
    except BaseException:
        batch.close()
        raise
    finally:
        timing['gallery_encode_ms'] = (time.monotonic()-started)*1000
