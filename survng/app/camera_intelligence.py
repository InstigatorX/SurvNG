from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable

from .audit_ai import AuditAiChange


def select_balanced_samples(
    candidates: Iterable[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Round-robin review categories so common successes cannot hide rare misses."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        buckets[str(candidate.get("category") or "other")].append(candidate)
    priority = [
        "possible_miss",
        "visual_backup",
        "motion_filtered",
        "motion_only_incident",
        "recognized_incident",
        "other",
    ]
    ordered_categories = [name for name in priority if buckets.get(name)]
    ordered_categories.extend(sorted(set(buckets) - set(ordered_categories)))
    selected: list[dict[str, Any]] = []
    while len(selected) < max(1, int(limit)):
        added = False
        for category in ordered_categories:
            if buckets[category] and len(selected) < limit:
                selected.append(buckets[category].pop(0))
                added = True
        if not added:
            break
    return selected


def aggregate_camera_intelligence(
    analyses: Iterable[dict[str, Any]],
    *,
    records_considered: int,
    selected_images: int,
    failed: int,
    hours: float,
) -> dict[str, Any]:
    rows = list(analyses)
    verdicts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    subjects: Counter[str] = Counter()
    detector: Counter[str] = Counter()
    tracking: Counter[str] = Counter()
    grouped_changes: dict[tuple[str, str], dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    for row in rows:
        verdicts[str(row.get("verdict") or "uncertain")] += 1
        categories[str(row.get("category") or "other")] += 1
        detector[str(row.get("detector_assessment") or "unavailable")] += 1
        tracking[str(row.get("tracking_assessment") or "unavailable")] += 1
        subjects.update(
            str(subject).strip().lower()
            for subject in row.get("visible_subjects") or []
            if str(subject).strip()
        )
        samples.append({
            key: row.get(key)
            for key in (
                "kind", "record_id", "event_id", "audit_id", "created_at",
                "category", "verdict", "confidence", "summary", "image_url",
                "detector_assessment", "tracking_assessment",
            )
        })
        for raw_change in row.get("changes") or []:
            change = (
                raw_change
                if isinstance(raw_change, AuditAiChange)
                else AuditAiChange.model_validate(raw_change)
            )
            if change.scope != "camera":
                continue
            value_key = json.dumps(change.value, sort_keys=True, separators=(",", ":"))
            key = (change.setting, value_key)
            grouped = grouped_changes.setdefault(key, {
                "scope": "camera",
                "setting": change.setting,
                "value": change.value,
                "support_count": 0,
                "confidence_total": 0.0,
                "reasons": [],
                "evidence": [],
            })
            grouped["support_count"] += 1
            grouped["confidence_total"] += float(row.get("confidence") or 0)
            if change.reason not in grouped["reasons"] and len(grouped["reasons"]) < 4:
                grouped["reasons"].append(change.reason)
            if len(grouped["evidence"]) < 8:
                grouped["evidence"].append({
                    "kind": row.get("kind"),
                    "id": row.get("record_id"),
                    "image_url": row.get("image_url"),
                })

    minimum_support = 2 if len(rows) >= 4 else 1
    by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for grouped in grouped_changes.values():
        support = int(grouped.pop("support_count"))
        confidence_total = float(grouped.pop("confidence_total"))
        if support < minimum_support:
            continue
        by_setting[str(grouped["setting"])].append({
            **grouped,
            "support_count": support,
            "average_confidence": round(confidence_total / support, 4),
        })
    recommendations: list[dict[str, Any]] = []
    for choices in by_setting.values():
        choices.sort(
            key=lambda item: (item["support_count"], item["average_confidence"]),
            reverse=True,
        )
        if (
            len(choices) > 1
            and choices[0]["support_count"] == choices[1]["support_count"]
        ):
            continue
        recommendations.append(choices[0])
    recommendations.sort(
        key=lambda item: (item["support_count"], item["average_confidence"]),
        reverse=True,
    )

    analyzed = len(rows)
    return {
        "review_type": "camera_intelligence",
        "summary": (
            f"Reviewed {analyzed} balanced image sample{'' if analyzed == 1 else 's'} "
            f"from {records_considered} recent camera records covering up to {hours:g} hours. "
            f"Found {verdicts.get('likely_miss', 0)} likely miss{'' if verdicts.get('likely_miss', 0) == 1 else 'es'}, "
            f"{verdicts.get('likely_false_alarm', 0)} likely false alarm{'' if verdicts.get('likely_false_alarm', 0) == 1 else 's'}, "
            f"and {verdicts.get('uncertain', 0)} uncertain result{'' if verdicts.get('uncertain', 0) == 1 else 's'}."
        ),
        "hours": hours,
        "records_considered": records_considered,
        "selected_images": selected_images,
        "analyzed": analyzed,
        "failed": failed,
        "verdict_counts": dict(verdicts),
        "category_counts": dict(categories),
        "visible_subject_counts": dict(subjects.most_common(12)),
        "detector_assessments": dict(detector),
        "tracking_assessments": dict(tracking),
        "recommendations": recommendations[:8],
        "samples": samples,
    }
