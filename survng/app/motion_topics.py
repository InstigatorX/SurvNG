"""Shared interpretation of camera motion topics across parser and policy."""

from __future__ import annotations

from enum import StrEnum


# Notice topic used when confirmed speech asks for an incident check. It is
# SurvNG's own topic, never a camera/ONVIF topic, so it carries no semantics.
SPEECH_TOPIC = "survng/speech"
# Notice topic for a recognized doorbell press. Like speech it is SurvNG's own
# topic; the camera's ONVIF topic is kept on the trigger as provenance.
DOORBELL_TOPIC = "survng/doorbell"
# Topics SurvNG assigns to triggers raised by its own EMA analysis. They are
# persisted with motion triggers and events, so the strings are a stable
# format; classify them with ``motion_trigger_kind`` rather than comparing.
ADAPTIVE_TOPIC_PREFIX = "adaptive/"
ADAPTIVE_MOTION_TOPIC = "adaptive/motion"
ADAPTIVE_VISUAL_BACKUP_TOPIC = "adaptive/visual_backup"
ADAPTIVE_ACTIVE_FOLLOWUP_TOPIC = "adaptive/active_followup"
# Topic of the Admin test trigger; any topic starting with "manual" is manual.
MANUAL_TEST_TOPIC = "manual/test"


class MotionTriggerKind(StrEnum):
    """What raised a motion trigger, derived from its topic."""

    CAMERA = "camera"
    MANUAL = "manual"
    SPEECH = "speech"
    DOORBELL = "doorbell"
    EMA_MOTION = "ema_motion"
    EMA_VISUAL_BACKUP = "ema_visual_backup"
    EMA_ACTIVE_FOLLOWUP = "ema_active_followup"

    @property
    def is_ema(self) -> bool:
        return self in _EMA_KINDS


_EMA_KINDS = frozenset({
    MotionTriggerKind.EMA_MOTION,
    MotionTriggerKind.EMA_VISUAL_BACKUP,
    MotionTriggerKind.EMA_ACTIVE_FOLLOWUP,
})
_ADAPTIVE_KINDS = {
    ADAPTIVE_VISUAL_BACKUP_TOPIC: MotionTriggerKind.EMA_VISUAL_BACKUP,
    ADAPTIVE_ACTIVE_FOLLOWUP_TOPIC: MotionTriggerKind.EMA_ACTIVE_FOLLOWUP,
}


def motion_trigger_kind(topic: str) -> MotionTriggerKind:
    """Classify a trigger topic. Camera/ONVIF topics are the default.

    SurvNG's own ``adaptive/...`` topics are matched exactly as written (they
    are never namespaced). Manual and speech topics use the same normalized
    form as ``semantic_motion_kind``.
    """
    raw = str(topic or "")
    if raw.startswith(ADAPTIVE_TOPIC_PREFIX):
        return _ADAPTIVE_KINDS.get(raw, MotionTriggerKind.EMA_MOTION)
    normalized = normalize_motion_topic(raw)
    if normalized.startswith("manual"):
        return MotionTriggerKind.MANUAL
    if normalized == SPEECH_TOPIC:
        return MotionTriggerKind.SPEECH
    if normalized == DOORBELL_TOPIC:
        return MotionTriggerKind.DOORBELL
    return MotionTriggerKind.CAMERA


def is_speech_topic(topic: str) -> bool:
    return motion_trigger_kind(topic) is MotionTriggerKind.SPEECH


def is_doorbell_topic(topic: str) -> bool:
    return motion_trigger_kind(topic) is MotionTriggerKind.DOORBELL


def is_manual_topic(topic: str) -> bool:
    return motion_trigger_kind(topic) is MotionTriggerKind.MANUAL


def normalize_motion_topic(topic: str) -> str:
    return "/".join(
        segment.rsplit(":", 1)[-1].strip().lower()
        for segment in str(topic or "").strip().split("/")
        if segment.strip()
    )


def semantic_motion_kind(topic: str) -> str | None:
    """Classify known semantic notices without inferring classes from XML data.

    Keep the established vendor-neutral aliases and explicitly include
    Reolink's DogCatDetect, whose topic does not contain the word animal, and
    Dahua Smart Motion Detection's SmartMotionHuman/SmartMotionVehicle.
    Unknown topics remain eligible for the parser's generic motion fallback.
    """
    normalized = normalize_motion_topic(topic)
    if normalized.startswith("manual"):
        return "manual"
    terminal = "".join(character for character in normalized.rsplit("/", 1)[-1] if character.isalnum())
    for kind, aliases in (
        ("person", {"person", "persondetect", "persondetected", "persondetection", "persondetector", "people", "peopledetect", "humandetect", "smartmotionhuman"}),
        ("vehicle", {"vehicle", "vehicledetect", "vehicledetected", "vehicledetection", "vehicledetector", "smartmotionvehicle"}),
        ("animal", {"animal", "animaldetect", "animaldetected", "animaldetection", "animaldetector", "dogcatdetect", "petdetect"}),
        ("face", {"face", "facedetect", "facedetected", "facedetection", "facedetector"}),
    ):
        if terminal in aliases:
            return kind
    return None
