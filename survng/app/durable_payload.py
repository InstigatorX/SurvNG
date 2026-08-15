from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    """Convert only well-defined scalar metadata at durable boundaries."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    scalar = getattr(value, "item", None)
    if callable(scalar):
        converted = scalar()
        if converted is value:
            raise TypeError(f"unsupported durable value: {type(value).__name__}")
        return converted
    raise TypeError(f"unsupported durable value: {type(value).__name__}")


def durable_json_dumps(value: Any, *, sort_keys: bool = False) -> str:
    """Encode durable work without silently stringifying invalid payloads."""
    encoded = json.dumps(
        value,
        default=_json_default,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    )
    # json's default hook can return a non-finite native float (for example
    # from a numpy scalar), so validate the normalized tree once more.
    normalized = json.loads(encoded)

    def validate(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("durable payload contains a non-finite number")
        if isinstance(item, list):
            for child in item:
                validate(child)
        elif isinstance(item, dict):
            for child in item.values():
                validate(child)

    validate(normalized)
    return encoded


def durable_json_copy(value: Any) -> Any:
    """Return the exact JSON-compatible value that will survive recovery."""
    return json.loads(durable_json_dumps(value))
