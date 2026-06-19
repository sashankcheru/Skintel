"""
Module 2 — Brick 2.2  |  brick22_segmentation.py
Lesion Segmentation + SFV Extraction

Applied ONLY to the close-up stream (10-15cm capture).
Wide-angle images skip this brick — SFV on a body-wide shot is
geometrically meaningless and clinically unreliable.

Why close-up only:
  The guided capture forces the lesion to be centred in frame at 10-15cm.
  SAM centre-point prompt is almost always on the lesion.
  Wide-angle shows body distribution, not lesion geometry — different
  clinical question, different model (EfficientNet/MobileNetV3).

Segmentation pipeline:
  Stage A — SAM ViT-B with centre-point prompt (zero-shot)
            Best available pretrained segmentation for skin images.
            No fine-tuning needed. Outperforms GrabCut significantly.
            Weights: /app/models/checkpoints/sam_vit_b_01ec64.pth
            Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

  Stage B — GrabCut fallback
            Used when SAM weights absent or SAM fails.
            Always available, no weights needed.

SFV (Shape Feature Vector) — 4 geometry features for Module 5 fusion:
  sfv_border_irregularity  (ABCDE → Border)
  sfv_asymmetry_index      (ABCDE → Asymmetry)
  sfv_fractal_dimension    (border roughness)
  sfv_color_gradient       (ABCDE → Color)
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from loguru import logger


SAM_PATH = os.getenv("SAM_PATH", "/app/models/checkpoints/sam_vit_b_01ec64.pth")


@dataclass
class SegmentationResult:
    mask:                    np.ndarray
    method:                  str          # "sam" | "grabcut" | "grabcut_failed"
    sfv_border_irregularity: float
    sfv_asymmetry_index:     float
    sfv_fractal_dimension:   float
    sfv_color_gradient:      float


# ── SAM STATE ─────────────────────────────────────────────────────────────────
class _SAMState:
    """Singleton SAM predictor — loaded once, reused for all calls."""

    def __init__(self):
        self._predictor = None
        self._loaded    = False
        self._failed    = False

    def load(self) -> bool:
        if self._loaded:
            return True
        if self._failed:
            return False
        if not os.path.exists(SAM_PATH):
            logger.info(
                f"SAM weights not found at {SAM_PATH} — using GrabCut fallback. "
                f"Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )
            self._failed = True
            return False
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam    = sam_model_registry["vit_b"](checkpoint=SAM_PATH)
            sam.to(device)
            self._predictor = SamPredictor(sam)
            self._loaded    = True
            logger.info(f"SAM ViT-B loaded on {device}")
            return True
        except Exception as exc:
            logger.warning(f"SAM load failed: {exc} — using GrabCut fallback")
            self._failed = True
            return False

    @property
    def predictor(self):
        return self._predictor


_sam = _SAMState()


# ── STAGE A — SAM ─────────────────────────────────────────────────────────────
def _segment_sam(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    SAM ViT-B with single centre-point prompt.
    Returns binary mask (0/255) or None on failure.

    Centre-point works reliably for SkinTel close-up images because
    the guided capture centres the lesion in frame at 10-15cm distance.
    """
    if not _sam.load():
        return None

    try:
        import numpy as np

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        _sam.predictor.set_image(image_rgb)

        h, w         = image_bgr.shape[:2]
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])  # 1 = foreground

        masks, scores, _ = _sam.predictor.predict(
            point_coords     = point_coords,
            point_labels     = point_labels,
            multimask_output = True,
        )

        best_mask = masks[np.argmax(scores)]
        binary    = (best_mask * 255).astype(np.uint8)

        logger.debug(f"SAM segmentation succeeded: score={scores.max():.3f}")
        return binary

    except Exception as exc:
        logger.warning(f"SAM predict failed: {exc} — falling back to GrabCut")
        return None


# ── STAGE B — GRABCUT FALLBACK ────────────────────────────────────────────────
def _segment_grabcut(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Otsu + GrabCut fallback when SAM unavailable or fails.
    Returns binary mask or None if GrabCut itself fails.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        rect = (int(w * 0.3), int(h * 0.3), int(w * 0.4), int(h * 0.4))
    else:
        largest       = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        pad = 0.10
        x   = max(0, int(x - bw * pad))
        y   = max(0, int(y - bh * pad))
        bw  = min(w - x, int(bw * (1 + 2 * pad)))
        bh  = min(h - y, int(bh * (1 + 2 * pad)))
        rect = (x, y, bw, bh)

    bgd     = np.zeros((1, 65), np.float64)
    fgd     = np.zeros((1, 65), np.float64)
    mask_gc = np.zeros((h, w), np.uint8)

    try:
        cv2.grabCut(image_bgr, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)
        logger.debug("GrabCut fallback mask computed")
        return mask
    except cv2.error as exc:
        logger.warning(f"GrabCut failed: {exc}")
        return None


# ── SFV COMPUTATION ───────────────────────────────────────────────────────────
def _compute_sfv(image_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """
    4 SFV geometry metrics from segmentation mask.
    Implements ABCDE dermatology criteria geometrically.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return dict(
            sfv_border_irregularity = 0.0,
            sfv_asymmetry_index     = 0.0,
            sfv_fractal_dimension   = 1.0,
            sfv_color_gradient      = 0.0,
        )

    largest = max(contours, key=cv2.contourArea)
    area    = float(cv2.contourArea(largest))
    perim   = float(cv2.arcLength(largest, closed=True))

    # Border Irregularity — ABCDE B
    if perim > 0:
        circularity  = (4 * np.pi * area) / (perim ** 2)
        border_irreg = float(np.clip(1.0 - circularity, 0.0, 1.0))
    else:
        border_irreg = 0.0

    # Asymmetry Index — ABCDE A
    h, w  = mask.shape
    top   = mask[:h//2, :]
    bot   = np.flipud(mask[h//2:, :])
    left  = mask[:, :w//2]
    right = np.fliplr(mask[:, w//2:])

    def _diff(a: np.ndarray, b: np.ndarray) -> float:
        min_h = min(a.shape[0], b.shape[0])
        min_w = min(a.shape[1], b.shape[1])
        diff  = np.abs(
            a[:min_h, :min_w].astype(float) -
            b[:min_h, :min_w].astype(float)
        )
        denom = max(
            a[:min_h, :min_w].astype(float).sum() +
            b[:min_h, :min_w].astype(float).sum(), 1
        )
        return float(diff.sum() / denom)

    asymmetry = float(
        np.clip((_diff(top, bot) + _diff(left, right)) / 2.0, 0.0, 1.0)
    )

    # Fractal Dimension — box counting on border
    border   = cv2.Canny(mask, 50, 150)
    min_side = min(border.shape)
    sizes    = [2**i for i in range(2, int(np.log2(min_side)))]
    counts   = []
    for s in sizes:
        ds = cv2.resize(
            border,
            (border.shape[1] // s, border.shape[0] // s),
            interpolation=cv2.INTER_NEAREST,
        )
        counts.append(int(np.count_nonzero(ds)))

    if len(sizes) >= 2 and counts[-1] > 0:
        try:
            coeffs  = np.polyfit(np.log(sizes), np.log(np.maximum(counts, 1)), 1)
            fractal = float(np.clip(abs(coeffs[0]), 1.0, 2.0))
        except Exception:
            fractal = 1.2
    else:
        fractal = 1.2

    # Colour Gradient — ABCDE C
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated     = cv2.dilate(mask, kernel, iterations=2)
    eroded      = cv2.erode(mask, kernel, iterations=2)
    border_zone = cv2.bitwise_and(dilated, cv2.bitwise_not(eroded))
    lab         = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    grad        = np.sqrt(
        cv2.Sobel(lab[:, :, 0], cv2.CV_64F, 1, 0) ** 2 +
        cv2.Sobel(lab[:, :, 0], cv2.CV_64F, 0, 1) ** 2
    )
    border_pixels = grad[border_zone > 0]
    if len(border_pixels) > 0:
        max_grad   = float(grad.max()) if grad.max() > 0 else 1.0
        color_grad = float(np.clip(border_pixels.mean() / max_grad, 0.0, 1.0))
    else:
        color_grad = 0.0

    return dict(
        sfv_border_irregularity = round(border_irreg, 4),
        sfv_asymmetry_index     = round(asymmetry,    4),
        sfv_fractal_dimension   = round(fractal,      4),
        sfv_color_gradient      = round(color_grad,   4),
    )


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────
def segment_and_extract_sfv(image_bgr: np.ndarray) -> SegmentationResult:
    """
    Segmentation + SFV for close-up stream only.
    SAM → GrabCut fallback → zero defaults on complete failure.
    Never raises on segmentation failure.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty")

    mask   = _segment_sam(image_bgr)
    method = "sam"

    if mask is None:
        mask   = _segment_grabcut(image_bgr)
        method = "grabcut"

    if mask is None:
        logger.warning("All segmentation failed — SFV zero defaults")
        return SegmentationResult(
            mask                    = np.zeros(image_bgr.shape[:2], dtype=np.uint8),
            method                  = "grabcut_failed",
            sfv_border_irregularity = 0.0,
            sfv_asymmetry_index     = 0.0,
            sfv_fractal_dimension   = 1.0,
            sfv_color_gradient      = 0.0,
        )

    sfv = _compute_sfv(image_bgr, mask)

    logger.info(
        f"SFV [{method}]: "
        f"border={sfv['sfv_border_irregularity']:.3f} | "
        f"asym={sfv['sfv_asymmetry_index']:.3f} | "
        f"fractal={sfv['sfv_fractal_dimension']:.3f} | "
        f"color_grad={sfv['sfv_color_gradient']:.3f}"
    )

    return SegmentationResult(
        mask                    = mask,
        method                  = method,
        sfv_border_irregularity = sfv["sfv_border_irregularity"],
        sfv_asymmetry_index     = sfv["sfv_asymmetry_index"],
        sfv_fractal_dimension   = sfv["sfv_fractal_dimension"],
        sfv_color_gradient      = sfv["sfv_color_gradient"],
    )