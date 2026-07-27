from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from .audit_ai import AuditAiAdvice


def aggregate_motion_ai_review(
    analyses: Iterable[tuple[dict[str, Any], AuditAiAdvice]],
    *,
    audits_considered: int,
    images_available: int,
    failed: int,
    current_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic camera report from independently validated AI responses."""
    rows = list(analyses)
    verdicts: Counter[str] = Counter()
    subjects: Counter[str] = Counter()
    audit_reasons: Counter[str] = Counter()
    grouped_changes: dict[tuple[str, str], dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    for audit, advice in rows:
        verdicts[advice.verdict] += 1
        audit_reasons[str(audit.get("reason") or "unknown")] += 1
        subjects.update(str(subject).strip().lower() for subject in advice.visible_subjects if str(subject).strip())
        samples.append({
            "audit_id": int(audit["id"]),
            "created_at": audit.get("created_at"),
            "reason": audit.get("reason"),
            "verdict": advice.verdict,
            "confidence": round(float(advice.confidence), 4),
            "summary": advice.summary,
        })
        for change in advice.changes:
            if change.scope != "camera":
                continue
            normalized_value = json.dumps(change.value, sort_keys=True, separators=(",", ":"))
            key = (change.setting, normalized_value)
            grouped = grouped_changes.setdefault(key, {
                "scope": "camera",
                "setting": change.setting,
                "value": change.value,
                "support_count": 0,
                "confidence_total": 0.0,
                "reasons": [],
                "evidence_audit_ids": [],
            })
            grouped["support_count"] += 1
            grouped["confidence_total"] += float(advice.confidence)
            if change.reason not in grouped["reasons"] and len(grouped["reasons"]) < 4:
                grouped["reasons"].append(change.reason)
            if len(grouped["evidence_audit_ids"]) < 8:
                grouped["evidence_audit_ids"].append(int(audit["id"]))

    minimum_support = 2 if len(rows) >= 5 else 1
    recommendations: list[dict[str, Any]] = []
    for grouped in grouped_changes.values():
        support_count = int(grouped.pop("support_count"))
        confidence_total = float(grouped.pop("confidence_total"))
        if support_count < minimum_support:
            continue
        recommendations.append({
            **grouped,
            "current_value": (current_settings or {}).get(str(grouped["setting"])),
            "support_count": support_count,
            "average_confidence": round(confidence_total / support_count, 4),
        })
    recommendations.sort(
        key=lambda item: (item["support_count"], item["average_confidence"]),
        reverse=True,
    )

    analyzed = len(rows)
    return {
        "summary": (
            f"Analyzed {analyzed} retained image{'' if analyzed == 1 else 's'} from the latest "
            f"{audits_considered} motion audit{'' if audits_considered == 1 else 's'}. "
            f"Found {verdicts.get('real_motion', 0)} likely real-motion, "
            f"{verdicts.get('noise', 0)} likely nuisance, and "
            f"{verdicts.get('uncertain', 0)} uncertain sample{'' if verdicts.get('uncertain', 0) == 1 else 's'}."
        ),
        "audits_considered": audits_considered,
        "images_available": images_available,
        "analyzed": analyzed,
        "failed": failed,
        "verdict_counts": dict(verdicts),
        "visible_subject_counts": dict(subjects.most_common(12)),
        "audit_reason_counts": dict(audit_reasons.most_common()),
        "recommendations": recommendations[:8],
        "samples": samples,
    }
