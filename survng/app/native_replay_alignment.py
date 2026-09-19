"""Estimate a recording/track clock offset from independently detected boxes.

No pixel inference runs here. Ambiguous/static scenes deliberately yield no
correction; an offset belongs to one completed episode and recording source.
"""
from bisect import bisect_left
import math
from statistics import mean, median


def _box_at(history, times, epoch):
    index = bisect_left(times, epoch)
    if index == len(times) or epoch < times[0]:
        return None
    if times[index] == epoch:
        return history[index][1:5]
    left, right = history[index-1], history[index]
    if right[0] - left[0] > 2:
        return None
    weight = (epoch-left[0])/(right[0]-left[0])
    return [a+(b-a)*weight for a, b in zip(left[1:5], right[1:5])]


def _iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/max(1, union)


def estimate_replay_alignment(tracking, observations):
    """Observations contain recording epochs and boxes in native track geometry.

    Positive offset means look up *later* saved track timestamps for the current
    recording frame. Require a distinct peak supported across multiple times.
    """
    tracks = []
    for track in tracking.get("tracks", []):
        history = sorted((sample[:5] for sample in track.get("box_history", [])
                          if len(sample) >= 5 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in sample[:5])), key=lambda x: x[0])
        if len(history) < 2 or history[-1][0]-history[0][0] < 1:
            continue
        centers = [((s[1]+s[3])/2, (s[2]+s[4])/2) for s in history]
        scale = median(math.hypot(s[3]-s[1], s[4]-s[2]) for s in history)
        spread = math.hypot(max(x for x,y in centers)-min(x for x,y in centers),
                            max(y for x,y in centers)-min(y for x,y in centers))
        # Presence boundaries and ID fragmentation are not clock evidence.
        if spread < max(10, scale*.35):
            continue
        tracks.append((track.get("label"), history, [sample[0] for sample in history]))
    frames = []
    for observation in observations:
        boxes = []
        for obj in observation.get("objects", []):
            values = [(obj.get("box") or {}).get(k) for k in ("x1", "y1", "x2", "y2")]
            if (obj.get("confidence", 0) >= .5 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values)
                    and values[2] > values[0] and values[3] > values[1]):
                boxes.append((obj.get("label"), values))
        epoch = observation.get("epoch")
        if boxes and isinstance(epoch, (int, float)) and math.isfinite(epoch):
            frames.append((epoch, boxes))
    if len(frames) < 5 or max(x[0] for x in frames)-min(x[0] for x in frames) < 3:
        return None
    offsets = [index/10 for index in range(-30, 31)]
    scores = []
    per_frame = []
    for offset in offsets:
        values = []
        for epoch, boxes in frames:
            predicted = [(label, _box_at(history, times, epoch+offset)) for label, history, times in tracks]
            pairs = sorted(((_iou(box, expected), i, j) for i, (name, box) in enumerate(boxes)
                            for j, (label, expected) in enumerate(predicted)
                            if label == name and expected is not None), reverse=True)
            used_boxes, used_tracks, total = set(), set(), 0.0
            for overlap, i, j in pairs:
                if i not in used_boxes and j not in used_tracks:
                    used_boxes.add(i)
                    used_tracks.add(j)
                    total += overlap
            values.append(total/len(boxes))
        per_frame.append(values)
        scores.append(mean(values))
    best = max(range(len(scores)), key=lambda i: scores[i])
    offset, score = offsets[best], scores[best]
    # A boundary peak may mean the true lag lies outside the bounded search.
    if best in (0, len(scores)-1) or score < .55:
        return None
    alternatives = [s for o, s in zip(offsets, scores) if abs(o-offset) >= .5]
    if score-max(alternatives) < .08:
        return None
    baseline = scores[30]
    if abs(offset) >= .3 and score-baseline < .12:
        return None
    votes = [offsets[max(range(len(offsets)), key=lambda i: per_frame[i][j])] for j in range(len(frames))]
    if sum(abs(v-offset) <= .3 for v in votes) < math.ceil(len(votes)*.6):
        return None
    if abs(median(votes)-offset) > .2:
        return None
    return {"source": "main", "verified": True, "offset_seconds": offset,
            "mean_iou": round(float(score), 4), "baseline_iou": round(float(baseline), 4),
            "observations": len(frames), "method": "recorded_detection_trajectory_v1"}
