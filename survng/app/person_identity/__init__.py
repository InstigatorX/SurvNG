"""Fused face + body people identity.

Named people identity is no longer face-only. SurvNG matches trusted person
galleries using face embeddings when available and whole-body ReID embeddings
when the face is weak, small, or absent. Track continuity ReID remains a
separate subsystem; this package owns durable named-person policy.
"""

from .fusion import (
    FusionDecision,
    FusionMatch,
    FusionPolicy,
    fuse_identity_match,
)

__all__ = [
    "FusionDecision",
    "FusionMatch",
    "FusionPolicy",
    "fuse_identity_match",
]
