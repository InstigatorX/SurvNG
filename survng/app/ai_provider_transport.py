"""Shared, bounded HTTP transport behavior for external AI providers."""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request


LOGGER = logging.getLogger(__name__)
MAX_PROVIDER_ERROR_BYTES = 64 * 1024
TRANSIENT_HTTP_CODES = {408, 429, 500, 502, 503, 504}
_PROVIDER_REQUEST_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class AiProviderTransportError(RuntimeError):
    """A bounded provider failure safe to expose through SurvNG's API."""

    message: str
    status_code: int | None = None
    category: str = "request_failed"

    def __str__(self) -> str:
        return self.message


def _duration_seconds(value: object) -> float | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    try:
        if text.endswith("ms"):
            return max(0.0, float(text[:-2]) / 1000.0)
        if text.endswith("s"):
            return max(0.0, float(text[:-1]))
        return max(0.0, float(text))
    except (TypeError, ValueError):
        return None


def _http_error_metadata(error: HTTPError) -> tuple[str, float | None]:
    """Extract only classification hints; never reflect provider text to callers."""
    message = ""
    retry_after = _duration_seconds(error.headers.get("Retry-After") if error.headers else None)
    try:
        raw = error.read(MAX_PROVIDER_ERROR_BYTES + 1)
        if len(raw) <= MAX_PROVIDER_ERROR_BYTES:
            payload = json.loads(raw.decode("utf-8"))
            provider_error = payload.get("error", {}) if isinstance(payload, Mapping) else {}
            if isinstance(provider_error, Mapping):
                message = str(provider_error.get("message") or "").lower()
                details = provider_error.get("details")
                if isinstance(details, list):
                    for detail in details:
                        if not isinstance(detail, Mapping):
                            continue
                        candidate = _duration_seconds(detail.get("retryDelay"))
                        if candidate is not None:
                            retry_after = candidate
                            break
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return message, retry_after


def _http_failure(error: HTTPError) -> tuple[AiProviderTransportError, float | None, bool]:
    provider_message, retry_after = _http_error_metadata(error)
    status_code = int(error.code)
    spend_cap = any(marker in provider_message for marker in (
        "monthly spending cap",
        "project spend cap",
        "spending limit",
        "billing hard limit",
        "insufficient_quota",
    ))
    if status_code == 429 and spend_cap:
        return (
            AiProviderTransportError(
                "AI provider spending cap reached; increase the project spending cap or wait for it to reset",
                status_code=429,
                category="spending_cap",
            ),
            None,
            False,
        )
    if status_code == 429:
        return (
            AiProviderTransportError(
                "AI provider is temporarily rate limited; try again shortly",
                status_code=429,
                category="rate_limited",
            ),
            retry_after,
            True,
        )
    return (
        AiProviderTransportError(
            f"AI provider returned HTTP {status_code}",
            status_code=status_code,
            category="http_error",
        ),
        retry_after,
        status_code in TRANSIENT_HTTP_CODES,
    )


def request_provider_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    timeout_seconds: float,
    max_response_bytes: int,
    opener: Callable[..., object],
    max_attempts: int = 3,
    sleeper: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> dict[str, object]:
    """Send one serialized provider request with shared throttling and safe retries."""
    request_data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    attempts = max(1, min(int(max_attempts), 3))
    for attempt in range(attempts):
        request = Request(
            url,
            data=request_data,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            # AI features have separate route-level limiters. This shared gate
            # prevents those independent workflows from creating provider spikes.
            with _PROVIDER_REQUEST_LOCK:
                response_context = opener(request, timeout=float(timeout_seconds))
                with response_context as response:
                    raw = response.read(max_response_bytes + 1)
            if len(raw) > max_response_bytes:
                raise AiProviderTransportError("AI provider response was too large")
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise AiProviderTransportError("AI provider returned an invalid response object")
            return decoded
        except HTTPError as error:
            failure, retry_after, transient = _http_failure(error)
            if not transient or attempt + 1 >= attempts:
                raise failure from error
            base_delay = retry_after if retry_after is not None else float(2**attempt)
            delay = min(8.0, max(0.25, base_delay) + jitter(0.0, 0.25))
            LOGGER.info(
                "AI provider transient HTTP %s; retrying attempt %s/%s in %.2fs",
                error.code,
                attempt + 2,
                attempts,
                delay,
            )
            sleeper(delay)
        except AiProviderTransportError:
            raise
        except (URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AiProviderTransportError(
                f"AI provider request failed ({type(error).__name__})"
            ) from error
    raise AiProviderTransportError("AI provider request failed")
