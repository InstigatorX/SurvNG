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
    category_verdicts: dict[str, Counter[str]] = defaultdict(Counter)
    grouped_changes: dict[tuple[str, str], dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    for row in rows:
        verdict = str(row.get("verdict") or "uncertain")
        category = str(row.get("category") or "other")
        verdicts[verdict] += 1
        categories[category] += 1
        category_verdicts[category][verdict] += 1
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
        "category_verdict_counts": {
            category: dict(counts)
            for category, counts in sorted(category_verdicts.items())
        },
        "visible_subject_counts": dict(subjects.most_common(12)),
        "detector_assessments": dict(detector),
        "tracking_assessments": dict(tracking),
        "recommendations": recommendations[:8],
        "samples": samples,
    }


def compare_camera_intelligence_results(
    baseline: dict[str, Any],
    followup: dict[str, Any],
) -> dict[str, Any]:
    """Compare two balanced reviews without pretending they are exhaustive counts."""
    before_total = max(1, int(baseline.get("analyzed") or 0))
    after_total = max(1, int(followup.get("analyzed") or 0))
    before_counts = baseline.get("verdict_counts") or {}
    after_counts = followup.get("verdict_counts") or {}
    definitions = (
        ("likely_miss", "Likely missed subjects"),
        ("likely_false_alarm", "Likely nuisance alerts"),
        ("likely_misclassification", "Likely wrong labels"),
        ("consistent", "Results that look correct"),
    )
    metrics: list[dict[str, Any]] = []
    for key, label in definitions:
        before_count = int(before_counts.get(key) or 0)
        after_count = int(after_counts.get(key) or 0)
        before_rate = before_count / before_total
        after_rate = after_count / after_total
        metrics.append({
            "key": key,
            "label": label,
            "before_count": before_count,
            "after_count": after_count,
            "before_rate": round(before_rate, 4),
            "after_rate": round(after_rate, 4),
            "change_points": round((after_rate - before_rate) * 100, 1),
        })

    issue_keys = {"likely_miss", "likely_false_alarm", "likely_misclassification"}
    before_strata = baseline.get("category_verdict_counts") or {}
    after_strata = followup.get("category_verdict_counts") or {}
    strata: list[dict[str, Any]] = []
    weighted_before = 0.0
    weighted_after = 0.0
    matched_support = 0
    for category in sorted(set(before_strata) & set(after_strata)):
        before_category = before_strata.get(category) or {}
        after_category = after_strata.get(category) or {}
        before_category_total = sum(int(value or 0) for value in before_category.values())
        after_category_total = sum(int(value or 0) for value in after_category.values())
        if before_category_total <= 0 or after_category_total <= 0:
            continue
        support = min(before_category_total, after_category_total)
        before_rate = sum(
            int(before_category.get(key) or 0) for key in issue_keys
        ) / before_category_total
        after_rate = sum(
            int(after_category.get(key) or 0) for key in issue_keys
        ) / after_category_total
        weighted_before += before_rate * support
        weighted_after += after_rate * support
        matched_support += support
        strata.append({
            "category": category,
            "matched_support": support,
            "before_rate": round(before_rate, 4),
            "after_rate": round(after_rate, 4),
            "change_points": round((after_rate - before_rate) * 100, 1),
        })
    sufficiently_comparable = (
        matched_support >= 4
        and int(baseline.get("analyzed") or 0) >= 4
        and int(followup.get("analyzed") or 0) >= 4
    )
    before_issues = weighted_before / matched_support if matched_support else 0.0
    after_issues = weighted_after / matched_support if matched_support else 0.0
    change_points = round((after_issues - before_issues) * 100, 1)
    if not sufficiently_comparable:
        outcome = "inconclusive"
        summary = (
            "There is not enough category-matched evidence to measure this setting change yet."
        )
    elif change_points <= -5:
        outcome = "improved"
        summary = (
            f"The reviewed issue rate fell from {before_issues * 100:.0f}% to "
            f"{after_issues * 100:.0f}% after the setting change."
        )
    elif change_points >= 5:
        outcome = "worsened"
        summary = (
            f"The reviewed issue rate rose from {before_issues * 100:.0f}% to "
            f"{after_issues * 100:.0f}% after the setting change."
        )
    else:
        outcome = "inconclusive"
        summary = (
            f"The reviewed issue rate was broadly unchanged "
            f"({before_issues * 100:.0f}% before and {after_issues * 100:.0f}% after)."
        )
    return {
        "outcome": outcome,
        "summary": summary,
        "before_analyzed": int(baseline.get("analyzed") or 0),
        "after_analyzed": int(followup.get("analyzed") or 0),
        "before_issue_rate": round(before_issues, 4),
        "after_issue_rate": round(after_issues, 4),
        "issue_rate_change_points": change_points,
        "comparison_basis": "category_matched_balanced_samples",
        "matched_sample_support": matched_support,
        "category_comparisons": strata,
        "metrics": metrics,
        "caution": (
            "This compares like-for-like review categories using matched sample support, "
            "not every camera frame. Treat small changes as directional evidence rather than proof."
        ),
    }
