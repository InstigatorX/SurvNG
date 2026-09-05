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
_INACTIVE = {"false", "0", "off", "inactive"}


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
            return []
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
    states = [value.strip().lower() for name, value in items if _key(name) in {"state", "ismotion", "motion"}]
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
        broad = reported_key in {category, "people", "human"}
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
