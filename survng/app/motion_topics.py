"""Shared interpretation of camera motion topics across parser and policy."""

from __future__ import annotations


def normalize_motion_topic(topic: str) -> str:
    return "/".join(
        segment.rsplit(":", 1)[-1].strip().lower()
        for segment in str(topic or "").strip().split("/")
        if segment.strip()
    )


def semantic_motion_kind(topic: str) -> str | None:
    """Classify known semantic notices without inferring classes from XML data.

    Keep the established vendor-neutral aliases and explicitly include
    Reolink's DogCatDetect, whose topic does not contain the word animal.
    Unknown topics remain eligible for the parser's generic motion fallback.
    """
    normalized = normalize_motion_topic(topic)
    if normalized.startswith("manual"):
        return "manual"
    terminal = "".join(character for character in normalized.rsplit("/", 1)[-1] if character.isalnum())
    for kind, aliases in (
        ("person", {"person", "persondetect", "persondetected", "persondetection", "persondetector", "people", "peopledetect", "humandetect"}),
        ("vehicle", {"vehicle", "vehicledetect", "vehicledetected", "vehicledetection", "vehicledetector"}),
        ("animal", {"animal", "animaldetect", "animaldetected", "animaldetection", "animaldetector", "dogcatdetect"}),
        ("face", {"face", "facedetect", "facedetected", "facedetection", "facedetector"}),
    ):
        if terminal in aliases:
            return kind
    return None
