"""Isolated ReID domain-adaptation training corpus (optional)."""

from __future__ import annotations

from .collector import ReidTrainingBuffer, ReidTrainingCollector
from .store import ReidTrainingStore

__all__ = [
    "ReidTrainingBuffer",
    "ReidTrainingCollector",
    "ReidTrainingStore",
]
