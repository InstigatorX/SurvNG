"""Offline, held-out face evaluation; never reads or writes the live gallery.

python -m survng.app.face_evaluation manifest.json --detector-config model.json --output report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import DetectorConfig
from .face_recognition import OpenVinoFaceRecognizer
from .face_store.quality import _face_quality


SPLITS = {"enrollment", "calibration", "held_out"}


def validate_manifest(manifest: dict, root: Path) -> list[dict]:
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Expected manifest version 1")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("A labeled sample list is required")
    days, visits, images, ids = {}, {}, {}, set()
    validated = []
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("split") not in SPLITS:
            raise ValueError("Every sample needs an enrollment, calibration or held_out split")
        if any(
            not isinstance(sample.get(key), str) or not sample[key].strip()
            for key in ("id", "path", "day", "visit", "track")
        ):
            raise ValueError("Samples require nonempty id, path, day, visit and track strings")
        if sample["id"] in ids:
            raise ValueError("Sample IDs must be unique")
        ids.add(sample["id"])
        if "person" not in sample:
            raise ValueError("Every sample needs an explicit person label or null")
        person = sample["person"]
        if person is not None and (not isinstance(person, str) or not person.strip()):
            raise ValueError("Person must be a nonempty label or null for an unknown visitor")
        if sample["split"] == "enrollment" and person is None:
            raise ValueError("Enrollment requires confirmed person labels")
        path = (root / sample["path"]).resolve()
        content = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Sample {sample['id']} is not a readable image")
        # Hash decoded pixels too: re-encoding identical imagery is still leakage.
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        for values, key in ((days, sample["day"]), (visits, sample["visit"]), (images, digest)):
            previous = values.setdefault(key, sample["split"])
            if previous != sample["split"]:
                raise ValueError("Days, visits and duplicate images must not cross splits")
        validated.append({**sample, "resolved_path": path, "image_digest": digest})
    gallery = {sample["person"] for sample in validated if sample["split"] == "enrollment"}
    for split in ("calibration", "held_out"):
        labels = [sample["person"] for sample in validated if sample["split"] == split]
        if None not in labels or not any(label in gallery for label in labels):
            raise ValueError(f"{split} must contain enrolled people and unknown visitors")
        if any(label is not None and label not in gallery for label in labels):
            raise ValueError("Use null for people who are absent from enrollment")
    identities = {}
    for sample in validated:
        key = (sample["visit"], sample["track"])
        if key in identities and identities[key] != sample["person"]:
            raise ValueError("A labeled track must have one identity")
        identities[key] = sample["person"]
    return validated


def score_tracks(samples: list[dict], split: str, threshold: float) -> list[dict]:
    """Replay top-three gallery matching and quality-weighted candidate voting.

    Metrics describe suggestions at the supplied score threshold, not production
    auto-identification (which also requires margins/reference-count gates).
    """
    gallery = defaultdict(list)
    probes = defaultdict(list)
    for sample in samples:
        if sample["split"] == "enrollment" and sample.get("embedding") is not None:
            gallery[sample["person"]].append(sample["embedding"])
        elif sample["split"] == split:
            probes[(sample["visit"], sample["track"])].append(sample)
    result = []
    for key, frames in sorted(probes.items()):
        votes = defaultdict(list)
        for frame in frames:
            if frame.get("embedding") is None:
                continue
            scores = []
            for person, references in gallery.items():
                similarities = sorted(
                    (float(frame["embedding"] @ reference) for reference in references),
                    reverse=True,
                )[:3]
                scores.append((float(np.mean(similarities)), person))
            scores.sort(reverse=True)
            if not scores or max(0.0, min(1.0, scores[0][0])) < threshold:
                continue
            score, person = scores[0]
            votes[person].append((
                round(max(0.0, min(1.0, score)), 4),
                max(.05, min(1., frame["quality"])),
            ))
        winner = max(votes, key=lambda person: (
            len(votes[person]), np.mean([v[0] for v in votes[person]]), person,
        )) if votes else None
        score = (
            sum(score * weight for score, weight in votes[winner])
            / sum(weight for _, weight in votes[winner])
            if winner else None
        )
        result.append({
            "visit": key[0], "track": key[1], "expected": frames[0]["person"],
            "predicted": winner, "score": score,
            "usable_frames": sum(frame.get("embedding") is not None for frame in frames),
        })
    return result


def metrics(rows: list[dict]) -> dict:
    total = len(rows)
    named = sum(row["predicted"] is not None for row in rows)
    correct = sum(row["predicted"] is not None and row["predicted"] == row["expected"] for row in rows)
    unknown = sum(row["expected"] is None for row in rows)
    false_unknown = sum(row["expected"] is None and row["predicted"] is not None for row in rows)
    known = total - unknown
    return {
        "tracks": total, "named": named, "correct_names": correct,
        "wrong_names": named - correct, "unresolved": total - named,
        "unknown_tracks": unknown, "unknown_false_names": false_unknown,
        "known_tracks": known, "precision": correct / named if named else None,
        "known_coverage": correct / known if known else None,
        "unknown_false_name_rate": false_unknown / unknown if unknown else None,
    }


def evaluate(manifest: dict, root: Path, recognizer) -> dict:
    samples = validate_manifest(manifest, root)
    for sample in samples:
        image = cv2.imread(str(sample["resolved_path"]))
        if image is None:
            raise ValueError(f"Sample {sample['id']} disappeared during evaluation")
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if digest != sample["image_digest"]:
            raise ValueError("Corpus changed during evaluation; run again on a stable copy")
        sample["quality"] = _face_quality(image, 1.0).score
        sample["embedding"] = None
        if min(image.shape[:2]) < recognizer.config.face_min_size:
            sample["failure"] = "too_small"
            continue
        try:
            vector = np.asarray(recognizer.embed(image), dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(vector)
            if not vector.size or not np.isfinite(norm) or norm <= 1e-9:
                raise ValueError("Invalid embedding")
            sample["embedding"] = vector / norm
        except ValueError as error:
            sample["failure"] = str(error)
    if not any(sample["split"] == "enrollment" and sample["embedding"] is not None for sample in samples):
        raise ValueError("No enrollment image produced a usable embedding")
    candidates = []
    for threshold in np.linspace(0, 1, 101):
        measured = metrics(score_tracks(samples, "calibration", float(threshold)))
        if measured["wrong_names"] == 0 and measured["correct_names"] > 0:
            candidates.append((measured["correct_names"], float(threshold)))
    chosen = max(candidates, key=lambda item: (item[0], -item[1]))[1] if candidates else None
    baseline = float(recognizer.config.face_match_threshold)
    selected = chosen if chosen is not None else baseline
    held_out = score_tracks(samples, "held_out", selected)
    return {
        "version": 1, "matcher": "top3_gallery_quality_weighted_votes_v1",
        "model_fingerprint": recognizer.model_fingerprint,
        "profile": recognizer.config.face_embedding_profile,
        "manifest_digest": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "images_digest": hashlib.sha256("".join(sample["image_digest"] for sample in samples).encode()).hexdigest(),
        "calibrated_suggestion_threshold": chosen,
        "calibration_status": "candidate_found" if chosen is not None else "no_threshold_with_correct_names_and_zero_wrong_names",
        "evaluation_threshold": selected,
        "baseline_threshold": baseline,
        "baseline_held_out": metrics(score_tracks(samples, "held_out", baseline)),
        "calibration": metrics(score_tracks(samples, "calibration", selected)),
        "held_out": metrics(held_out), "held_out_predictions": held_out,
        "failed_samples": [
            {"id": sample["id"], "reason": sample["failure"]}
            for sample in samples if "failure" in sample
        ],
        "limitation": (
            "Suggestion evaluation with all supplied enrollment crops; production "
            "gallery selection and automatic-identification guards are not replayed. "
            "No live settings are changed. Small corpora do not certify rare-error rates."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--detector-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = DetectorConfig.model_validate(json.loads(args.detector_config.read_text()))
    config.face_recognition_enabled = True
    recognizer = OpenVinoFaceRecognizer(config)
    if not recognizer.ready:
        parser.error(recognizer.error)
    report = evaluate(json.loads(args.manifest.read_text()), args.manifest.parent, recognizer)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
