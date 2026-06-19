"""
Module 2 — Brick 2.0  |  brick20_quality_gate.py
Image Quality Gate — Server-side safety net for runtime patient uploads.

The primary quality control happens client-side (Module 6):
  - Guided oval overlay enforces capture distance (10-15cm close, 30-40cm wide)
  - Live Laplacian enforces sharpness before capture button activates
  - Live brightness indicator enforces exposure
  - camera2 API locks white balance, disables ISP sharpening

This brick re-checks sharpness and exposure on the ACTUAL uploaded frame.
Motion blur can occur at the tap moment even if the preview was sharp.
This is a safety net, not the primary gate.

Applied independently to BOTH close-up and wide-angle streams.

Thresholds (tunable via env vars):
  QUALITY_BLUR_THRESHOLD    default 20.0   Laplacian variance on centre crop
  QUALITY_DARK_THRESHOLD    default 30.0   mean brightness too dark
  QUALITY_BRIGHT_THRESHOLD  default 230.0  mean brightness overexposed
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from loguru import logger


BLUR_THRESHOLD   = float(os.getenv("QUALITY_BLUR_THRESHOLD",   "20.0"))
DARK_THRESHOLD   = float(os.getenv("QUALITY_DARK_THRESHOLD",   "30.0"))
BRIGHT_THRESHOLD = float(os.getenv("QUALITY_BRIGHT_THRESHOLD", "230.0"))


@dataclass
class QualityResult:
    passed:        bool
    quality_score: float
    reason:        str
    stream:        str   # "closeup" | "wideangle" | "unknown"


def _centre_crop_gray(image_bgr: np.ndarray) -> np.ndarray:
    """
    Returns the central 60% crop as grayscale.
    Avoids penalising smooth perilesional skin backgrounds that drag
    down the Laplacian score on clean close-up images.
    """
    gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.2), int(w * 0.8)
    return gray[y0:y1, x0:x1]


def compute_laplacian_variance(image_bgr: np.ndarray) -> float:
    """Sharpness via Laplacian variance on central 60% crop."""
    crop = _centre_crop_gray(image_bgr)
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def check_exposure(image_bgr: np.ndarray) -> tuple[float, str]:
    """Mean brightness check. Returns (mean, failure_reason_or_empty)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean < DARK_THRESHOLD:
        return mean, f"image too dark (mean={mean:.1f}, threshold={DARK_THRESHOLD})"
    if mean > BRIGHT_THRESHOLD:
        return mean, f"image overexposed (mean={mean:.1f}, threshold={BRIGHT_THRESHOLD})"
    return mean, ""


def run_quality_gate(
    image_bgr: np.ndarray,
    stream:    str = "unknown",
) -> QualityResult:
    """
    Sharpness + exposure check on a BGR image.
    Applied to both close-up and wide-angle streams independently.
    Never raises.
    """
    if image_bgr is None or image_bgr.size == 0:
        return QualityResult(
            passed=False, quality_score=0.0,
            reason="image could not be loaded", stream=stream
        )

    quality_score = compute_laplacian_variance(image_bgr)
    if quality_score < BLUR_THRESHOLD:
        reason = (
            f"image too blurry "
            f"(score={quality_score:.1f}, threshold={BLUR_THRESHOLD})"
        )
        logger.warning(f"[{stream}] Quality gate FAILED: {reason}")
        return QualityResult(
            passed=False, quality_score=quality_score,
            reason=reason, stream=stream
        )

    _, exposure_reason = check_exposure(image_bgr)
    if exposure_reason:
        logger.warning(f"[{stream}] Quality gate FAILED: {exposure_reason}")
        return QualityResult(
            passed=False, quality_score=quality_score,
            reason=exposure_reason, stream=stream
        )

    logger.info(f"[{stream}] Quality gate PASSED: score={quality_score:.1f}")
    return QualityResult(
        passed=True, quality_score=quality_score,
        reason="", stream=stream
    )


def run_quality_gate_from_path(
    image_path: str,
    stream:     str = "unknown",
) -> QualityResult:
    """Convenience wrapper — loads from disk then runs quality gate."""
    img = cv2.imread(image_path)
    if img is None:
        return QualityResult(
            passed=False, quality_score=0.0,
            reason=f"could not read file: {image_path}", stream=stream
        )
    return run_quality_gate(img, stream=stream)