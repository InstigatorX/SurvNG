"""Shared, bounded image-quality measurements for stored visual evidence."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VisualQuality:
    score: float
    sharpness: float
    exposure: float
    contrast: float
    edge_detail: float


def image_quality(frame: np.ndarray, *, max_dimension: int = 512) -> VisualQuality:
    """Return bounded quality signals without saturating on textured scenes."""
    if frame is None or not getattr(frame, "size", 0):
        return VisualQuality(0.0, 0.0, 0.0, 0.0, 0.0)
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        height, width = gray.shape[:2]
        largest = max(height, width)
        if largest > max_dimension:
            scale = max_dimension / largest
            gray = cv2.resize(
                gray,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        pixels = gray.astype(np.float32, copy=False)
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_strength = float(np.percentile(cv2.magnitude(gradient_x, gradient_y), 90.0))
        laplacian_detail = laplacian_variance / (laplacian_variance + 900.0)
        edge_detail = edge_strength / (edge_strength + 120.0)
        sharpness = max(0.0, min(1.0, 0.60 * laplacian_detail + 0.40 * edge_detail))
        clipped = float(np.mean((pixels <= 4.0) | (pixels >= 251.0)))
        exposure = max(0.0, min(1.0, 1.0 - clipped))
        contrast = max(0.0, min(1.0, float(pixels.std()) / 64.0))
        score = 0.75 * sharpness + 0.15 * exposure + 0.10 * contrast
        return VisualQuality(score, sharpness, exposure, contrast, edge_detail)
    except (cv2.error, TypeError, ValueError):
        return VisualQuality(0.0, 0.0, 0.0, 0.0, 0.0)
