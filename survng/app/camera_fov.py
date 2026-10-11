"""Horizontal field-of-view lookup for fixed surveillance cameras.

Values are manufacturer horizontal angles in degrees. The catalog is a starter
list, not a complete vendor database. An unknown model stays a manual entry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_CATALOG_PATH = Path(__file__).with_name("data") / "camera_fov_catalog.json"
_TOKEN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class FovEntry:
    make: str
    model: str
    hfov_deg: float


def normalize_model_token(value: str) -> str:
    return _TOKEN.sub("", str(value or "").casefold())


@lru_cache(maxsize=1)
def load_fov_catalog(path: str = "") -> tuple[FovEntry, ...]:
    catalog_path = Path(path) if path else _CATALOG_PATH
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = payload.get("cameras") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("camera field-of-view catalog is invalid")
    entries: list[FovEntry] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("camera field-of-view catalog is invalid")
        make = str(row.get("make") or "").strip()
        model = str(row.get("model") or "").strip()
        try:
            hfov = float(row.get("hfov_deg"))
        except (TypeError, ValueError) as error:
            raise ValueError("camera field-of-view catalog is invalid") from error
        if not make or not model or not (1 <= hfov <= 180):
            raise ValueError("camera field-of-view catalog is invalid")
        entries.append(FovEntry(make=make, model=model, hfov_deg=hfov))
    return tuple(entries)


def lookup_fov(make: str, model: str, entries: tuple[FovEntry, ...] | None = None) -> dict:
    catalog = entries if entries is not None else load_fov_catalog()
    model_key = normalize_model_token(model)
    if not model_key:
        return {"matched": False, "reason": "model is required"}
    make_key = normalize_model_token(make)
    exact = [
        entry for entry in catalog
        if normalize_model_token(entry.make) == make_key
        and normalize_model_token(entry.model) == model_key
    ]
    if len(exact) == 1:
        return _match(exact[0], "exact")
    by_model = [
        entry for entry in catalog
        if normalize_model_token(entry.model) == model_key
        and (not make_key or normalize_model_token(entry.make) == make_key)
    ]
    if len(by_model) == 1:
        return _match(by_model[0], "model")
    if len(by_model) > 1:
        return {
            "matched": False,
            "reason": "model matches more than one camera",
            "candidates": [_public(entry) for entry in by_model],
        }
    return {"matched": False, "reason": "no catalog entry"}


def _match(entry: FovEntry, how: str) -> dict:
    return {"matched": True, "match": how, **_public(entry)}


def _public(entry: FovEntry) -> dict:
    return {"make": entry.make, "model": entry.model, "hfov_deg": entry.hfov_deg}
