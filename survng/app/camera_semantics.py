"""Normalize advisory object-class claims made by camera event metadata."""

from __future__ import annotations

import re
from typing import Iterable
from xml.etree import ElementTree

from .motion_topics import normalize_motion_topic, semantic_motion_kind

_CATEGORY_LABELS = {
    "person": {"person", "people", "human", "pedestrian"},
    "vehicle": {"vehicle", "car", "truck", "deliverytruck", "bus", "motorcycle", "motorbike", "bicycle", "bike", "van", "suv", "train", "boat"},
    "animal": {"animal", "dog", "cat", "bird", "horse", "sheep", "cow", "rabbit", "squirrel", "deer", "elephant", "bear", "zebra", "giraffe"},
    "face": {"face"},
}
_VALUE_FIELDS = {"class", "classification", "objectclass", "objecttype"}
# IsInside is the state of an ONVIF FieldDetector (Dahua IVS intrusion) rule.
_STATE_FIELDS = {"state", "ismotion", "motion", "isinside"}
_INACTIVE = {"false", "0", "off", "inactive"}
_ACTIVE = {"true", "1", "on", "active"}
_BROAD_CLASSES = {"people", "human"}
_SIMPLE_ITEM_TAG = re.compile(r"<(?:[A-Za-z_][\w.-]*:)?SimpleItem\b([^<>]*?)/?>")
_XML_ATTRIBUTE = re.compile(r"\b(Name|Value)\s*=\s*\"([^\"]*)\"", re.IGNORECASE)

CAMERA_NOTICE_ADMISSION_VERSION = 1
# Categories whose affirmative camera claim may stand in for SurvNG's own
# multi-frame temporal consensus. A face implies a person is present.
SEMANTIC_FAST_PATH_CATEGORIES = frozenset({"person", "vehicle", "animal", "face"})


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _topic_category(topic: str) -> str | None:
    category = semantic_motion_kind(topic)
    return category if category in _CATEGORY_LABELS else None


def _explicit_items(message: str) -> list[tuple[str, str]]:
    """Read explicit Name/Value pairs from valid XML or a bounded Zeep repr."""
    text = str(message or "").strip()
    if not text:
        return []
    if text.startswith("<"):
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError:
            # Fragments with an undeclared prefix (``<tt:SimpleItem .../>``)
            # are what the ONVIF parser accepts; read only whole SimpleItem tags.
            pairs: list[tuple[str, str]] = []
            for attributes in _SIMPLE_ITEM_TAG.findall(text[:20000])[:32]:
                values = {
                    name.lower(): value
                    for name, value in _XML_ATTRIBUTE.findall(attributes)
                }
                if "name" in values and "value" in values:
                    pairs.append((values["name"], values["value"]))
            return pairs
        return [
            (str(element.attrib.get("Name") or ""), str(element.attrib.get("Value") or ""))
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "SimpleItem"
            and element.attrib.get("Name") is not None
            and element.attrib.get("Value") is not None
        ]
    items: list[tuple[str, str]] = []
    for container in re.findall(r"\{[^{}]{0,500}\}", text)[:32]:
        attributes = {
            name.lower(): value.strip()
            for name, value in re.findall(
                r"['\"]?\b(name|value)['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]",
                container,
                flags=re.IGNORECASE,
            )
        }
        if "name" in attributes and "value" in attributes:
            items.append((attributes["name"], attributes["value"]))
    return items


def camera_semantic_reports(topic: str, message: str, configured_labels: Iterable[object] = ()) -> list[dict[str, object]]:
    """Return camera claims as provenance, never as detector observations."""
    topic_category = _topic_category(topic)
    items = _explicit_items(message)
    states = [value.strip().lower() for name, value in items if _key(name) in _STATE_FIELDS]
    if states and all(value in _INACTIVE for value in states):
        return []

    labels = [str(label).strip() for label in configured_labels if str(label).strip()]
    label_keys = {_key(label): label for label in labels}
    typed: list[tuple[str, str]] = []
    for name, value in items[:32]:
        if _key(name) not in _VALUE_FIELDS:
            continue
        clean = value.strip()
        value_key = _key(clean)
        category = next((candidate for candidate, aliases in _CATEGORY_LABELS.items() if value_key in aliases), None)
        category = category or topic_category
        if category and value_key and len(clean) <= 64 and (category, value_key) not in {(item[0], _key(item[1])) for item in typed}:
            typed.append((category, clean))

    topic_reported = topic_category if topic_category in {"person", "face"} else None
    claims: list[tuple[str, str | None]] = typed or (
        [(topic_category, topic_reported)] if topic_category else []
    )
    reports: list[dict[str, object]] = []
    normalized_topic = normalize_motion_topic(topic)
    for category, reported_class in claims[:16]:
        reported_key = _key(reported_class) if reported_class else ""
        broad = reported_key in {category, *_BROAD_CLASSES}
        if reported_class and not broad:
            exact = label_keys.get(reported_key)
            candidates = [exact] if exact is not None else []
        else:
            allowed = _CATEGORY_LABELS[category]
            if "dogcatdetect" in normalized_topic:
                allowed = {"dog", "cat"}
            candidates = [label for label in labels if _key(label) in allowed]
        report: dict[str, object] = {"topic": str(topic), "category": category, "candidate_model_classes": candidates}
        if reported_class:
            report["reported_class"] = reported_class
        reports.append(report)
    return reports


def active_camera_semantic_kind(topic: str, message: str) -> str | None:
    """Classify an active semantic notice using both topic and payload state."""
    reports = camera_semantic_reports(topic, message)
    return str(reports[0]["category"]) if reports else None


def _family_labels(category: str, labels: list[str]) -> list[str]:
    allowed = _CATEGORY_LABELS[category]
    if not labels:
        return sorted(allowed)
    return [label for label in labels if _key(label) in allowed]


def _compatible_labels(report: dict[str, object], labels: list[str]) -> list[str]:
    """Detector labels that would corroborate one camera claim.

    A broad or topic-only claim keeps the configured candidates the report
    already resolved (DogCatDetect stays dog/cat). A specific known alias such
    as "car" admits its configured family, because cameras and detectors
    routinely disagree within a family. An unknown specific class stays exact.
    """
    category = str(report.get("category") or "")
    if category not in _CATEGORY_LABELS:
        return []
    if category == "face":
        return [*_family_labels("person", labels), *_family_labels("face", labels)]
    candidates = [str(item) for item in report.get("candidate_model_classes") or [] if str(item)]
    reported = _key(report.get("reported_class"))
    if reported and reported not in {category, *_BROAD_CLASSES}:
        if reported in _CATEGORY_LABELS[category]:
            return _family_labels(category, labels)
        return candidates
    return candidates if labels else _family_labels(category, labels)


def camera_notice_admission(
    topic: str, message: str, configured_labels: Iterable[object] = ()
) -> dict[str, object]:
    """Decide whether one camera notice is an affirmative semantic claim.

    Only an affirmative person/vehicle/animal/face claim qualifies. Cleared or
    ambiguous state, an unparseable payload, an unknown topic and generic
    motion all fall back to ``generic``: ordinary validation, never a bypass.
    An empty payload on a semantic topic is a start, matching the ONVIF
    parser, which has already filtered explicit stops.
    """
    labels = [str(label).strip() for label in configured_labels if str(label).strip()]
    generic = {"version": CAMERA_NOTICE_ADMISSION_VERSION, "kind": "generic"}
    text = str(message or "").strip()
    items = _explicit_items(text)
    if text and not items:
        return {**generic, "reason": "unparsed_payload"}
    states = [value.strip().lower() for name, value in items if _key(name) in _STATE_FIELDS]
    if states and not any(value in _ACTIVE for value in states):
        return {**generic, "reason": "state_not_affirmative"}
    reports = [
        report for report in camera_semantic_reports(topic, text, labels)
        if report.get("category") in SEMANTIC_FAST_PATH_CATEGORIES
    ]
    if not reports:
        return {**generic, "reason": "no_semantic_claim"}
    compatible: list[str] = []
    for report in reports:
        for label in _compatible_labels(report, labels):
            if label not in compatible:
                compatible.append(label)
    return {
        "version": CAMERA_NOTICE_ADMISSION_VERSION,
        "kind": "semantic",
        "reason": "affirmative_semantic_notice",
        "categories": list(dict.fromkeys(str(report["category"]) for report in reports)),
        "compatible_labels": compatible,
    }


def semantic_label_compatible(label: object, compatible_labels: Iterable[object]) -> bool:
    key = _key(label)
    return bool(key) and key in {_key(item) for item in compatible_labels}
