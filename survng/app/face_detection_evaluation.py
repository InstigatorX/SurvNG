"""Score a face detector IR against detection labels. This is not face recognition."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .face_detection import parse_ssd_face_detections
from .face_detection_geometry import box_tuple


IOU_THRESHOLD = 0.5
DEFAULT_THRESHOLD = 0.60
MISS_RECALL_MINIMUM = 0.80
MISS_RECALL_GAIN = 0.10
PASS_CALLS = 12
PASS_P95_SECONDS = 2.0
FULL_CALLS = 44
FULL_BUDGET_SECONDS = 4.0

Detection = dict[str, Any]
Detector = Callable[[np.ndarray, float], list[Detection]]


def intersection_over_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    overlap = width * height
    if overlap <= 0:
        return 0.0
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def match_detections(
    ground_truth: list[tuple[float, float, float, float]],
    predictions: list[Detection],
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[int, int]:
    """Greedy one-to-one match. Returns matched ground-truth count and unmatched predictions."""
    ordered = sorted(predictions, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    unmatched = set(range(len(ground_truth)))
    unmatched_predictions = 0
    for prediction in ordered:
        box = box_tuple(prediction.get("box") if isinstance(prediction, dict) else None)
        if box is None:
            unmatched_predictions += 1
            continue
        best_index = None
        best_iou = iou_threshold
        for index in unmatched:
            score = intersection_over_union(ground_truth[index], box)
            if score >= best_iou:
                best_index = index
                best_iou = score
        if best_index is None:
            unmatched_predictions += 1
            continue
        unmatched.remove(best_index)
    return len(ground_truth) - len(unmatched), unmatched_predictions


def _boxes(annotations: list[dict[str, Any]], kind: str) -> list[tuple[float, float, float, float]]:
    boxes = []
    for annotation in annotations:
        if annotation.get("kind") != kind:
            continue
        box = box_tuple(annotation.get("box") if isinstance(annotation, dict) else None)
        if box is not None:
            boxes.append(box)
    return boxes


def _filter_threshold(predictions: list[Detection], threshold: float) -> list[Detection]:
    kept = []
    for prediction in predictions:
        try:
            confidence = float(prediction.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence >= threshold:
            kept.append(prediction)
    return kept


def score_split(
    samples: list[dict[str, Any]],
    *,
    prediction_key: str,
    threshold: float,
    split: str,
) -> dict[str, Any]:
    """Score one split. Dropped and unreviewed samples are ignored."""
    chosen = [
        sample
        for sample in samples
        if sample.get("split") == split and sample.get("disposition") == "reviewed"
    ]
    ground_truth_faces = 0
    matched_faces = 0
    negative_crops = 0
    negative_false = 0
    unmatched_predictions = 0
    for sample in chosen:
        annotations = list(sample.get("annotations") or [])
        kinds = {item.get("kind") for item in annotations}
        predictions = _filter_threshold(list(sample.get(prediction_key) or []), threshold)
        if "face" in kinds:
            truth = _boxes(annotations, "face")
            matched, unmatched = match_detections(truth, predictions)
            ground_truth_faces += len(truth)
            matched_faces += matched
            unmatched_predictions += unmatched
        elif "no_face" in kinds:
            negative_crops += 1
            if predictions:
                negative_false += 1
                unmatched_predictions += len(predictions)
    return {
        "split": split,
        "threshold": threshold,
        "crops": len(chosen),
        "ground_truth_faces": ground_truth_faces,
        "matched_faces": matched_faces,
        "recall": (matched_faces / ground_truth_faces) if ground_truth_faces else None,
        "negative_crops": negative_crops,
        "negative_false_crops": negative_false,
        "negative_crop_fpr": (negative_false / negative_crops) if negative_crops else None,
        "unmatched_predictions": unmatched_predictions,
    }


def miss_set_recall(
    samples: list[dict[str, Any]],
    *,
    candidate_key: str,
    baseline_key: str,
    candidate_threshold: float,
    baseline_threshold: float = DEFAULT_THRESHOLD,
    split: str = "held_out",
) -> dict[str, Any]:
    """Recall on held-out faces the baseline detector missed at its threshold."""
    missed = 0
    recovered = 0
    for sample in samples:
        if sample.get("split") != split or sample.get("disposition") != "reviewed":
            continue
        truth = _boxes(list(sample.get("annotations") or []), "face")
        if not truth:
            continue
        baseline = _filter_threshold(list(sample.get(baseline_key) or []), baseline_threshold)
        candidate = _filter_threshold(list(sample.get(candidate_key) or []), candidate_threshold)
        baseline_matches = _matched_indices(truth, baseline)
        candidate_matches = _matched_indices(truth, candidate)
        for index in range(len(truth)):
            if index in baseline_matches:
                continue
            missed += 1
            if index in candidate_matches:
                recovered += 1
    return {
        "missed_faces": missed,
        "recovered_faces": recovered,
        "recall": (recovered / missed) if missed else None,
    }


def _matched_indices(
    ground_truth: list[tuple[float, float, float, float]],
    predictions: list[Detection],
) -> set[int]:
    ordered = sorted(predictions, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    unmatched = set(range(len(ground_truth)))
    matched: set[int] = set()
    for prediction in ordered:
        box = box_tuple(prediction.get("box") if isinstance(prediction, dict) else None)
        if box is None:
            continue
        best_index = None
        best_iou = IOU_THRESHOLD
        for index in unmatched:
            score = intersection_over_union(ground_truth[index], box)
            if score >= best_iou:
                best_index = index
                best_iou = score
        if best_index is None:
            continue
        unmatched.remove(best_index)
        matched.add(best_index)
    return matched


def latency_gate(call_seconds: list[float]) -> dict[str, Any]:
    """p95 of one detect call, scaled to a 12-call pass and a 44-call evidence budget."""
    if not call_seconds:
        return {
            "calls": 0,
            "p95_seconds": None,
            "estimated_pass_seconds": None,
            "estimated_full_seconds": None,
            "passed": False,
            "reasons": ["No face-detector timings were recorded"],
        }
    ordered = sorted(float(value) for value in call_seconds)
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    p95 = ordered[index]
    estimated_pass = p95 * PASS_CALLS
    estimated_full = p95 * FULL_CALLS
    reasons = []
    if estimated_pass > PASS_P95_SECONDS:
        reasons.append(
            f"Estimated 12-call face pass is {estimated_pass:.3f}s; the gate is {PASS_P95_SECONDS:.1f}s"
        )
    if estimated_full > FULL_BUDGET_SECONDS:
        reasons.append(
            f"Estimated 44-call evidence budget is {estimated_full:.3f}s; the gate is {FULL_BUDGET_SECONDS:.1f}s"
        )
    return {
        "calls": len(ordered),
        "p95_seconds": p95,
        "estimated_pass_seconds": estimated_pass,
        "estimated_full_seconds": estimated_full,
        "passed": not reasons,
        "reasons": reasons,
    }


def promotion_gate(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    miss: dict[str, Any],
    latency: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    miss_recall = miss.get("recall")
    baseline_miss = 0.0
    if not miss.get("missed_faces"):
        reasons.append("Held-out miss set is empty, so a recall gain cannot be shown")
    elif miss_recall is None or miss_recall < MISS_RECALL_MINIMUM:
        reasons.append(
            f"Miss-set recall {miss_recall} is below {MISS_RECALL_MINIMUM:.2f}"
        )
    elif miss_recall < baseline_miss + MISS_RECALL_GAIN:
        reasons.append(
            f"Miss-set recall {miss_recall} is not {MISS_RECALL_GAIN:.2f} above the baseline"
        )
    candidate_recall = candidate.get("recall")
    baseline_recall = baseline.get("recall")
    if candidate_recall is None or baseline_recall is None:
        reasons.append("Held-out face recall needs ground-truth faces for both models")
    elif candidate_recall < baseline_recall:
        reasons.append("Held-out recall is below the baseline detector")
    candidate_fpr = candidate.get("negative_crop_fpr")
    baseline_fpr = baseline.get("negative_crop_fpr")
    if candidate_fpr is None or baseline_fpr is None:
        reasons.append("Held-out no-face crops are required to compare false positives")
    elif candidate_fpr > baseline_fpr:
        reasons.append("False-positive rate on no-face crops is above the baseline")
    if not latency.get("passed"):
        reasons.extend(str(reason) for reason in latency.get("reasons") or ["Latency gate failed"])
    return {
        "passed": not reasons,
        "reasons": reasons,
        "miss_recall_minimum": MISS_RECALL_MINIMUM,
        "miss_recall_gain": MISS_RECALL_GAIN,
        "iou": IOU_THRESHOLD,
    }


def suggest_threshold(
    samples: list[dict[str, Any]],
    *,
    candidate_key: str = "candidate",
    baseline_key: str = "baseline",
) -> float | None:
    """Pick a calibration threshold. Held-out samples are not read."""
    baseline = score_split(samples, prediction_key=baseline_key, threshold=DEFAULT_THRESHOLD, split="calibration")
    baseline_fpr = baseline.get("negative_crop_fpr")
    if baseline_fpr is None:
        return None
    best: tuple[float, float] | None = None
    for step in range(1, 100):
        threshold = step / 100
        measured = score_split(samples, prediction_key=candidate_key, threshold=threshold, split="calibration")
        miss = miss_set_recall(
            samples,
            candidate_key=candidate_key,
            baseline_key=baseline_key,
            candidate_threshold=threshold,
            split="calibration",
        )
        fpr = measured.get("negative_crop_fpr")
        recall = miss.get("recall")
        if fpr is None or recall is None or fpr > baseline_fpr:
            continue
        if best is None or recall > best[0]:
            best = (recall, threshold)
    return None if best is None else best[1]


def evaluate_predictions(
    samples: list[dict[str, Any]],
    *,
    candidate_threshold: float = DEFAULT_THRESHOLD,
    threshold_source: str = "default",
    call_seconds: list[float] | None = None,
) -> dict[str, Any]:
    candidate = score_split(
        samples,
        prediction_key="candidate",
        threshold=candidate_threshold,
        split="held_out",
    )
    baseline = score_split(
        samples,
        prediction_key="baseline",
        threshold=DEFAULT_THRESHOLD,
        split="held_out",
    )
    miss = miss_set_recall(
        samples,
        candidate_key="candidate",
        baseline_key="baseline",
        candidate_threshold=candidate_threshold,
    )
    latency = latency_gate(list(call_seconds or []))
    gate = promotion_gate(candidate, baseline, miss, latency)
    return {
        "threshold": candidate_threshold,
        "threshold_source": threshold_source,
        "candidate": candidate,
        "baseline": baseline,
        "miss_set": miss,
        "latency": latency,
        "gate": gate,
    }


def run_detector_on_samples(
    samples: list[dict[str, Any]],
    detector: Detector,
    *,
    image_reader: Callable[[dict[str, Any]], np.ndarray],
    threshold: float = 0.01,
) -> tuple[dict[int, list[Detection]], list[float]]:
    import time

    predictions: dict[int, list[Detection]] = {}
    timings: list[float] = []
    for sample in samples:
        if sample.get("disposition") != "reviewed" or not sample.get("split"):
            continue
        image = image_reader(sample)
        started = time.perf_counter()
        detections = detector(image, threshold)
        timings.append(time.perf_counter() - started)
        predictions[int(sample["id"])] = list(detections or [])
    return predictions, timings


def attach_predictions(
    samples: list[dict[str, Any]],
    *,
    candidate: dict[int, list[Detection]],
    baseline: dict[int, list[Detection]],
) -> list[dict[str, Any]]:
    attached = []
    for sample in samples:
        item = dict(sample)
        item["candidate"] = list(candidate.get(int(sample["id"]), []))
        item["baseline"] = list(baseline.get(int(sample["id"]), []))
        attached.append(item)
    return attached


def evaluate_samples(
    samples: list[dict[str, Any]],
    candidate_detector: Detector,
    baseline_detector: Detector,
    *,
    image_reader: Callable[[dict[str, Any]], np.ndarray],
    threshold: float | None = None,
) -> dict[str, Any]:
    candidate_predictions, timings = run_detector_on_samples(
        samples, candidate_detector, image_reader=image_reader
    )
    baseline_predictions, _baseline_timings = run_detector_on_samples(
        samples, baseline_detector, image_reader=image_reader
    )
    attached = attach_predictions(
        samples, candidate=candidate_predictions, baseline=baseline_predictions
    )
    chosen = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    source = "default" if threshold is None else "requested"
    if threshold is None:
        suggested = suggest_threshold(attached)
        held = evaluate_predictions(attached, candidate_threshold=DEFAULT_THRESHOLD, call_seconds=timings)
        if not held["gate"]["passed"] and suggested is not None and abs(suggested - DEFAULT_THRESHOLD) > 1e-9:
            chosen = suggested
            source = "calibration"
        else:
            return held
    return evaluate_predictions(
        attached,
        candidate_threshold=chosen,
        threshold_source=source,
        call_seconds=timings,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ir_contract(model_path: Path) -> dict[str, Any]:
    """Reject an IR the current face worker cannot parse. This does not prove baked BGR."""
    path = Path(model_path)
    if not path.is_file() or path.suffix.lower() != ".xml":
        raise ValueError("Face detector IR must be an existing OpenVINO .xml file")
    try:
        from openvino import Core
    except ImportError:
        try:
            from openvino.runtime import Core
        except ImportError as exc:
            raise ValueError("OpenVINO is not installed") from exc
    model = Core().read_model(model=path)
    try:
        shape = [int(value) for value in model.input(0).shape]
    except (TypeError, ValueError) as exc:
        raise ValueError("Face detector input must be a static NCHW shape") from exc
    if len(shape) != 4 or shape[1] != 3 or shape[2] <= 0 or shape[3] <= 0:
        raise ValueError(f"Expected NCHW face detector input, got {shape}")
    output = model.output(0)
    elements = 1
    try:
        for value in output.shape:
            elements *= int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Face detector output must be static SSD rows of 7 floats") from exc
    if elements <= 0 or elements % 7 != 0:
        raise ValueError("Face detector output must be static SSD rows of 7 floats")
    return {
        "input_shape": shape,
        "output_elements": elements,
        "sha256": file_sha256(path),
        "parser_check": "ssd-7",
    }


def detections_from_ssd(raw: Any, width: int, height: int, threshold: float) -> list[Detection]:
    return parse_ssd_face_detections(raw, width, height, threshold)
