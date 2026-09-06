"""Portable offline tracking inputs and explicitly defined identity metrics.

Run with ``python -m survng.app.tracking_evaluation replay.json --output result.json``.
No recording, detector, encoder or live incident mutations are needed for replay.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import ObjectTrackingConfig
from .object_track.geometry import _box, _iou

SCHEMA_VERSION = 1
MAX_FRAMES = 600
MAX_OBJECTS = 100
MAX_IDENTITIES = 1000
MAX_REPLAY_BYTES = 32 * 1024 * 1024
PROFILES = ("recorded", "fixed_2fps", "fixed_075fps", "sparse_gaps")
DETECTION_FIELDS = ("label", "confidence", "box", "incident_eligible", "zones", "_tracking_embedding")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def replay_digest(replay: dict[str, Any]) -> str:
    content = {key: value for key, value in replay.items() if key != "replay_id"}
    return hashlib.sha256(canonical_json(content).encode()).hexdigest()


def portable_detection(item: dict[str, Any]) -> dict[str, Any]:
    result = {key: copy.deepcopy(item[key]) for key in DETECTION_FIELDS if key in item}
    embedding = result.get("_tracking_embedding")
    if embedding is not None:
        result["_tracking_embedding"] = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    return result


def validate_replay(replay: dict[str, Any]) -> None:
    if not isinstance(replay, dict) or replay.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported replay schema")
    if len(canonical_json(replay).encode()) > MAX_REPLAY_BYTES:
        raise ValueError("replay exceeds 32 MiB limit")
    if replay.get("replay_id") != replay_digest(replay):
        raise ValueError("replay checksum mismatch")
    frames = replay.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError("replay needs 1–600 frames")
    previous = -math.inf
    dimensions = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("invalid replay frame")
        epoch = frame.get("captured_at")
        if type(epoch) not in (float, int) or not math.isfinite(epoch) or not 0 <= epoch <= 253402300799 or epoch <= previous:
            raise ValueError("replay timestamps must be finite and strictly increasing")
        previous = epoch
        if type(frame.get("frame_index")) is not int or frame["frame_index"] != index:
            raise ValueError("replay frame indexes must be consecutive")
        if not all(type(frame.get(key)) is int and 0 < frame[key] <= 16384 for key in ("width", "height")):
            raise ValueError("invalid replay frame dimensions")
        current_dimensions = (frame["width"], frame["height"])
        if dimensions is not None and current_dimensions != dimensions:
            raise ValueError("replay frame dimensions must remain constant")
        dimensions = current_dimensions
        objects = frame.get("detections")
        if not isinstance(objects, list) or len(objects) > MAX_OBJECTS:
            raise ValueError("replay allows at most 100 detections per frame")
        for item in objects:
            if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not item["label"].strip() or _box(item.get("box")) is None:
                raise ValueError("invalid replay detection")
            if item.get("status") == "inference_deferred":
                raise ValueError("deferred inference cannot be a replay detection")
            confidence = item.get("confidence")
            if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("invalid replay confidence")
            embedding = item.get("_tracking_embedding")
            if embedding is not None and (not isinstance(embedding, list) or not 1 <= len(embedding) <= 4096 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in embedding)):
                raise ValueError("invalid replay embedding")
    ObjectTrackingConfig.model_validate(replay.get("tracking_config"))
    if not all(isinstance(replay.get(key), str) and replay[key] for key in ("timestamp_source", "appearance_source")):
        raise ValueError("replay needs timestamp and appearance provenance")
    if type(replay.get("source_pts_frames")) is not int or not 0 <= replay["source_pts_frames"] <= len(frames):
        raise ValueError("invalid source timestamp frame count")
    threshold = replay.get("high_confidence_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid detector confidence threshold")


def selected_frames(replay: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    if profile not in PROFILES:
        raise ValueError("unknown sampling profile")
    frames = replay["frames"]
    if profile == "recorded":
        return frames
    fps = .75 if profile == "fixed_075fps" else 2.0
    start = frames[0]["captured_at"]
    next_sample = start
    selected = []
    for frame in frames:
        epoch = frame["captured_at"]
        if epoch + 1e-6 < next_sample:
            continue
        # Jump directly to the next cadence slot. Advancing one slot at a time
        # makes a valid replay with a long recording gap effectively hang.
        next_sample = start + (math.floor((epoch - start + 1e-6) * fps) + 1) / fps
        elapsed = epoch - start
        if profile == "sparse_gaps" and (5 <= elapsed < 9 or 15 <= elapsed < 21):
            continue
        selected.append(frame)
    return selected


def validate_labels(
    labels: dict[str, Any], replay: dict[str, Any], selected_indexes: Iterable[int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Validate annotations independently of any engine's predictions."""
    if not isinstance(labels, dict) or labels.get("schema_version") != 1 or labels.get("replay_id") != replay["replay_id"]:
        raise ValueError("labels must identify this exact replay")
    if len(canonical_json(labels).encode()) > MAX_REPLAY_BYTES:
        raise ValueError("labels exceed 32 MiB limit")
    frames = labels.get("frames")
    if not isinstance(frames, list) or len(frames) > MAX_FRAMES:
        raise ValueError("labels need a bounded list of frames")
    by_frame = {}
    identities = set()
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("invalid labeled frame")
        index = frame.get("frame_index")
        if type(index) is not int or not 0 <= index < len(replay["frames"]) or index in by_frame:
            raise ValueError("invalid or duplicate labeled frame index")
        objects = frame.get("objects")
        if not isinstance(objects, list) or len(objects) > MAX_OBJECTS:
            raise ValueError("labeled frames require an explicit objects list with at most 100 objects")
        seen = set()
        for obj in objects:
            if not isinstance(obj, dict):
                raise ValueError("invalid ground-truth object")
            identity = obj.get("identity")
            if not isinstance(identity, str) or not identity.strip() or identity in seen or not isinstance(obj.get("label"), str) or not obj["label"].strip() or _box(obj.get("box")) is None:
                raise ValueError("each ground-truth object needs a unique identity, label and valid box")
            seen.add(identity)
            identities.add((obj["label"], identity))
        by_frame[index] = objects
    if len(identities) > MAX_IDENTITIES:
        raise ValueError("labels exceed 1000 identity limit")
    if selected_indexes is not None and any(index not in by_frame for index in selected_indexes):
        raise ValueError("label every evaluated frame; an empty objects list means no visible objects")
    return by_frame


def identity_metrics(observations: list[dict[str, Any]], labels: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    """Class-aware IoU>=.5 potential matches and global assignment for IDF1.

    IDF1 counts every admissible GT/track coexistence before global identity
    assignment, as in TrackEval's identity metric. Separate per-frame matching
    defines switches (including gaps), fragments (tracked/untracked/tracked
    while GT remains present), and false merges (IDs matched to multiple GTs).
    These explicit event counts are not the full TrackEval/CLEAR benchmark.
    """
    from .object_track.assignment import maximum_weight_assignment
    by_frame = validate_labels(labels, replay)
    pair_counts: Counter = Counter()
    potential_counts: Counter = Counter()
    gt_count = pred_count = switches = fragments = 0
    last_identity = {}
    ever_matched = set()
    missed = set()
    all_tracks = set()
    previous_index = -1
    for frame in observations:
        index = frame["frame_index"]
        if type(index) is not int or index <= previous_index:
            raise ValueError("evaluated frames must have unique increasing indexes")
        previous_index = index
        if index not in by_frame:
            raise ValueError("label every evaluated frame; an empty objects list means no visible objects")
        gt = by_frame[index]
        pred = frame["objects"]
        if not isinstance(pred, list) or len(pred) > MAX_OBJECTS:
            raise ValueError("invalid predicted objects list")
        seen_tracks = set()
        for item in pred:
            if not isinstance(item, dict) or type(item.get("track_id")) is not int or item["track_id"] in seen_tracks or not isinstance(item.get("label"), str) or not item["label"].strip() or _box(item.get("box")) is None:
                raise ValueError("predictions need unique track IDs, labels and valid boxes")
            seen_tracks.add(item["track_id"])
            all_tracks.add(item["track_id"])
        if len(all_tracks) > MAX_IDENTITIES:
            raise ValueError("predictions exceed 1000 identity limit")
        gt_count += len(gt)
        pred_count += len(pred)
        weights = []
        bonus = min(len(gt), len(pred)) + 1
        for truth in gt:
            row = []
            for item in pred:
                overlap = _iou(_box(truth["box"]), _box(item["box"])) if truth["label"] == item["label"] else 0
                if overlap >= .5:
                    potential_counts[((truth["label"], truth["identity"]), item["track_id"])] += 1
                row.append(bonus + overlap if overlap >= .5 else 0)
            weights.append(row)
        matches = maximum_weight_assignment(weights) if gt and pred else []
        matched_gt = set()
        for i, j in matches:
            identity = (gt[i]["label"], gt[i]["identity"])
            track = pred[j]["track_id"]
            matched_gt.add(identity)
            pair_counts[(identity, track)] += 1
            if identity in last_identity and last_identity[identity] != track:
                switches += 1
            if identity in missed:
                fragments += 1
                missed.remove(identity)
            last_identity[identity] = track
            ever_matched.add(identity)
        present = {(item["label"], item["identity"]) for item in gt}
        missed.update((present & ever_matched) - matched_gt)
        # A GT absence (true occlusion/out-of-frame) is not itself tracker fragmentation.
        missed.intersection_update(present)
    identities = sorted({identity for identity, _ in potential_counts})
    tracks = sorted({track for _, track in potential_counts})
    weights = [[potential_counts[(identity, track)] for track in tracks] for identity in identities]
    idtp = sum(weights[i][j] for i, j in maximum_weight_assignment(weights)) if identities and tracks else 0
    idfp, idfn = pred_count - idtp, gt_count - idtp
    merges = sum(len({identity for identity, candidate in pair_counts if candidate == track}) > 1 for track in tracks)
    denominator = 2 * idtp + idfp + idfn
    return {"definition": "survng_identity_v1_iou_0.5", "idf1": round(2 * idtp / denominator, 6) if denominator else None,
            "idtp": idtp, "idfp": idfp, "idfn": idfn, "id_switches": switches, "fragmentations": fragments,
            "false_merges": merges, "ground_truth_observations": gt_count, "predicted_observations": pred_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="recorded")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-template", action="store_true", help="write blank annotation frames instead of executing trackers")
    args = parser.parse_args()
    if args.replay.stat().st_size > MAX_REPLAY_BYTES:
        parser.error("replay exceeds 32 MiB")
    if args.labels and args.labels.stat().st_size > MAX_REPLAY_BYTES:
        parser.error("labels exceed 32 MiB")
    replay = json.loads(args.replay.read_text())
    validate_replay(replay)
    if args.label_template:
        result = {"schema_version": 1, "replay_id": replay["replay_id"], "frames": [{"frame_index": f["frame_index"], "objects": [{"identity": None, "label": d["label"], "box": d["box"]} for d in f["detections"]]} for f in replay["frames"]]}
    else:
        from .tracking_comparison import TrackingComparisonRunner
        result = TrackingComparisonRunner.replay(replay, sampling_profile=args.profile, labels=json.loads(args.labels.read_text()) if args.labels else None)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
