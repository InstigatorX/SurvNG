"""Compatibility name for the Hybrid behavior promoted into production."""

from __future__ import annotations

from .hybrid import HybridObjectTracker


class HybridCandidateObjectTracker(HybridObjectTracker):
    """Backward-compatible candidate name for saved comparisons and imports."""
