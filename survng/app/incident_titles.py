"""Optional provider-worded incident titles.

Off by default (``audit_ai.incident_titles_enabled``). The incident page always
has a deterministic title built from objects, cameras and zones; this only
offers a better-worded one when an administrator opts in. A failure never
blocks the page: the caller gets ``status="failed"`` and keeps its own title,
and the failure is logged (rate limited per incident).

Only scene facts leave SurvNG: object labels and certainty, camera names,
zone names, movement states and relative times. No images, recognized names,
stream URLs or identifiers are sent.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from .assistant import AssistantProvider
from .audit_ai import AuditAiError, ai_provider_configured
from .config import AuditAiConfig
from .growing_playlists import RateLimitedLog

LOGGER = logging.getLogger(__name__)
MAX_TITLE_CHARS = 90
MAX_FACT_OBJECTS = 12
# A live incident gains revisions quickly; reword it at most this often.
REGENERATE_SECONDS = 60.0
MAX_CACHED_INCIDENTS = 512
MAX_CONCURRENT_REQUESTS = 2

SYSTEM_PROMPT = (
    "You title home security camera incidents. Given structured facts, write one short, "
    "plain-language title (at most 12 words) describing what happened, in the order it "
    "happened, like a news headline without a trailing period. Use only the facts given. "
    "Do not invent people, identities, intent, or objects. Describe uncertain objects as "
    "possible. Use camera names as given."
)
TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string", "maxLength": MAX_TITLE_CHARS}},
    "required": ["title"],
    "additionalProperties": False,
}


class IncidentTitle(BaseModel):
    title: str = Field(min_length=3, max_length=MAX_TITLE_CHARS)

    @field_validator("title", mode="before")
    @classmethod
    def single_line(cls, value: object) -> str:
        text = " ".join(str(value or "").split()).strip().rstrip(".")
        return text


def _epoch(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def incident_title_facts(incident: Mapping[str, Any], camera_names: Mapping[str, str]) -> dict[str, Any]:
    """The bounded, identifier-free facts sent to the provider."""
    start = _epoch(incident.get("start_at")) or 0.0
    name = lambda camera_id: camera_names.get(camera_id) or "camera"  # noqa: E731
    objects = []
    for subject in (incident.get("scene_objects") or [])[:MAX_FACT_OBJECTS]:
        label = str(subject.get("label") or "object").replace("_", " ")
        if label == "face":
            continue
        intervals = []
        for interval in subject.get("visible_intervals") or []:
            begin, end = _epoch(interval.get("start_at")), _epoch(interval.get("end_at"))
            intervals.append({
                "camera": name(interval.get("camera_id")),
                "from_second": round(begin - start) if begin is not None else None,
                "to_second": round(end - start) if end is not None else None,
                "state": interval.get("state"),
                "zones": [str(zone) for zone in interval.get("zones") or []][:4],
            })
        objects.append({"label": label, "uncertain": subject.get("certainty") in {"possible", "uncertain"}, "seen": intervals[:8]})
    episodes = [{"camera": name(episode.get("camera_id"))} for episode in incident.get("episodes") or []]
    return {
        "duration_seconds": round(float(incident.get("duration_seconds") or 0)),
        "cameras": list(dict.fromkeys(item["camera"] for item in episodes)),
        "objects": objects,
        "person_recognized": bool(incident.get("identities")),
    }


class IncidentTitleGenerator:
    """Caches one provider title per incident revision; never raises to callers."""

    def __init__(
        self,
        *,
        provider_factory: Callable[[AuditAiConfig], Any] = AssistantProvider,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._provider_factory = provider_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._inflight: dict[str, threading.Event] = {}
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)
        self._log = RateLimitedLog(LOGGER, interval_seconds=300.0)

    def title(self, incident: Mapping[str, Any], config: AuditAiConfig, camera_names: Mapping[str, str]) -> dict[str, Any]:
        incident_id = str(incident.get("incident_id") or incident.get("id") or "")
        revision = incident.get("revision")
        if not (config.enabled and config.incident_titles_enabled):
            return {"incident_id": incident_id, "status": "disabled", "title": None}
        if not ai_provider_configured(config):
            return self._failed(incident_id, revision, "not_configured", "AI title unavailable: the AI provider is not configured")
        if not incident.get("scene_objects"):
            # Nothing observed to describe; the built-in title is the honest one.
            return {"incident_id": incident_id, "status": "skipped", "title": None}
        while True:
            with self._lock:
                cached = self._cache.get(incident_id)
                recent = bool(cached) and self._clock() - cached["at"] < REGENERATE_SECONDS
                # A title stays while its revision is current; a failure is
                # retried after the interval even when nothing changed.
                fresh = recent or bool(cached and cached["status"] == "ready" and cached["revision"] == revision)
                if fresh and cached["status"] == "ready":
                    self._cache.move_to_end(incident_id)
                    return {"incident_id": incident_id, "status": "ready", "title": cached["title"],
                            "revision": cached["revision"], "current": cached["revision"] == revision}
                if fresh and cached["status"] == "failed":
                    return {"incident_id": incident_id, "status": "failed", "title": None, "reason": cached["reason"],
                            "message": "AI title unavailable"}
                waiting = self._inflight.get(incident_id)
                if waiting is None:
                    done = self._inflight[incident_id] = threading.Event()
                    break
            waiting.wait(timeout=float(config.timeout_seconds) + 5)
        try:
            return self._generate(incident_id, revision, incident, config, camera_names)
        finally:
            with self._lock:
                self._inflight.pop(incident_id, None)
            done.set()

    def _generate(self, incident_id, revision, incident, config, camera_names) -> dict[str, Any]:
        if not self._slots.acquire(timeout=float(config.timeout_seconds)):
            return {"incident_id": incident_id, "status": "failed", "title": None, "reason": "busy", "message": "AI title unavailable"}
        try:
            prompt = json.dumps(incident_title_facts(incident, camera_names), separators=(",", ":"), ensure_ascii=False)
            raw = self._provider_factory(config).complete_json(SYSTEM_PROMPT, prompt, TITLE_SCHEMA, "survng_incident_title")
            title = IncidentTitle.model_validate_json(raw).title
        except (AuditAiError, ValidationError, ValueError) as exc:
            reason = "provider_failed" if isinstance(exc, AuditAiError) else "invalid_response"
            # Provider errors are already bounded and redacted; validation
            # errors may echo model output, so only their type is logged.
            detail = str(exc) if isinstance(exc, AuditAiError) else type(exc).__name__
            self._log.log(logging.WARNING, "incident_title", incident_id,
                          "AI incident title failed for %s (%s: %s); showing the built-in title",
                          incident_id, reason, detail)
            return self._failed(incident_id, revision, reason, "AI title unavailable")
        except Exception:
            LOGGER.exception("AI incident title failed unexpectedly for %s; showing the built-in title", incident_id)
            return self._failed(incident_id, revision, "unexpected_error", "AI title unavailable")
        finally:
            self._slots.release()
        self._store(incident_id, {"status": "ready", "title": title, "revision": revision, "at": self._clock()})
        return {"incident_id": incident_id, "status": "ready", "title": title, "revision": revision, "current": True}

    def _failed(self, incident_id, revision, reason, message) -> dict[str, Any]:
        if reason != "not_configured":
            self._store(incident_id, {"status": "failed", "reason": reason, "revision": revision, "at": self._clock()})
        else:
            self._log.log(logging.WARNING, "incident_title", "not_configured",
                          "AI incident titles are enabled but the AI provider is not configured; showing built-in titles")
        return {"incident_id": incident_id, "status": "failed", "title": None, "reason": reason, "message": message}

    def _store(self, incident_id, entry) -> None:
        with self._lock:
            self._cache[incident_id] = entry
            self._cache.move_to_end(incident_id)
            while len(self._cache) > MAX_CACHED_INCIDENTS:
                self._cache.popitem(last=False)
