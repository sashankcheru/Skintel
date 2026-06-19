"""
Module 2 — Brick 2.3  |  brick23_color_normalizer.py
Colour Normalisation + Resize — Runtime patient uploads only.

Applied to BOTH close-up and wide-angle streams at runtime.
Training datasets skip this entirely — model learns from natural variation.

The camera layer (Module 6) is the primary fix for colour consistency:
  - react-native-vision-camera locks white balance via camera2 API
  - ISP sharpening and saturation boost disabled
This brick is the server-side safety net for residual ISP differences.

Pipeline:
  Stage 1 — Gray World white balance
            Neutralises residual device colour cast.
            No reference image dependency. Always deterministic.

  Stage 2 — Gamma correction targeting brightness=160
            Computed from skin pixels only (YCrCb mask).
            Covers Fitzpatrick I-VI and Monk MST 4-9.
            Normalises exposure without washing out lesion colour.

  Stage 3 — Resize to 224×224
            ImageNet standard. Matches DermaCon-IN Swin-T checkpoint.
            INTER_LANCZOS4 — best for medical images, avoids aliasing.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from loguru import logger


TARGET_SIZE       = (224, 224)
TARGET_BRIGHTNESS = float(os.getenv("TARGET_BRIGHTNESS", "160.0"))


@dataclass
class ColorNormResult:
    image:         np.ndarray
    original_size: tuple
    method:        str    # "gray_world_gamma" | "resize_only_fallback"
    mean_before:   float
    mean_after:    float


def _gray_world(image_bgr: np.ndarray) -> np.ndarray:
    """
    Gray World white balance.
    Scales each BGR channel so all three means equal their overall mean.
    """
    b, g, r             = cv2.split(image_bgr.astype(np.float32))
    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    overall             = (b_mean + g_mean + r_mean) / 3.0

    b_scale = overall / b_mean if b_mean > 0 else 1.0
    g_scale = overall / g_mean if g_mean > 0 else 1.0
    r_scale = overall / r_mean if r_mean > 0 else 1.0

    balanced = cv2.merge([
        np.clip(b * b_scale, 0, 255).astype(np.uint8),
        np.clip(g * g_scale, 0, 255).astype(np.uint8),
        np.clip(r * r_scale, 0, 255).astype(np.uint8),
    ])
    logger.debug(
        f"Gray World: means=({b_mean:.1f},{g_mean:.1f},{r_mean:.1f}) "
        f"scales=({b_scale:.3f},{g_scale:.3f},{r_scale:.3f})"
    )
    return balanced


def _skin_mask_ycrcb(image_bgr: np.ndarray) -> np.ndarray:
    """
    YCrCb skin pixel mask — excludes lesion centre and background.
    Standard range: Cr 133-173, Cb 77-127.
    Covers Fitzpatrick I-VI and Monk MST 4-9.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(
        ycrcb,
        np.array([0,   133,  77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )


def _gamma_correction(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Gamma correction targeting TARGET_BRIGHTNESS on skin pixels only.
    Returns (corrected_image, mean_before, mean_after).
    """
    gray        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    skin_mask   = _skin_mask_ycrcb(image_bgr)
    skin_pixels = gray[skin_mask > 0]

    if len(skin_pixels) < 100:
        logger.debug("Gamma: insufficient skin pixels — using global mean")
        skin_pixels = gray.flatten()

    mean_before = float(skin_pixels.mean())
    if mean_before < 1.0:
        return image_bgr, mean_before, mean_before

    gamma = np.log(TARGET_BRIGHTNESS / 255.0) / np.log(mean_before / 255.0)
    gamma = float(np.clip(gamma, 0.4, 2.5))

    lut = np.array([
        min(255, int((i / 255.0) ** gamma * 255))
        for i in range(256)
    ], dtype=np.uint8)

    corrected       = cv2.LUT(image_bgr, lut)
    corrected_gray  = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    skin_after      = corrected_gray[skin_mask > 0]
    mean_after      = float(skin_after.mean()) if len(skin_after) > 0 else mean_before

    logger.debug(
        f"Gamma: {mean_before:.1f} → {mean_after:.1f} (gamma={gamma:.3f})"
    )
    return corrected, mean_before, mean_after


def normalise_and_resize(image_bgr: np.ndarray) -> ColorNormResult:
    """
    Gray World + gamma + 224×224 resize.
    Applied to both streams at runtime. Never raises.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty")

    original_size = (image_bgr.shape[0], image_bgr.shape[1])

    try:
        balanced                           = _gray_world(image_bgr)
        corrected, mean_before, mean_after = _gamma_correction(balanced)
        resized = cv2.resize(
            corrected, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4
        )
        logger.info(
            f"Colour norm done: {original_size} → {TARGET_SIZE}, "
            f"brightness {mean_before:.1f} → {mean_after:.1f}"
        )
        return ColorNormResult(
            image         = resized,
            original_size = original_size,
            method        = "gray_world_gamma",
            mean_before   = mean_before,
            mean_after    = mean_after,
        )
    except Exception as exc:
        logger.error(f"Colour norm failed: {exc} — resizing original")
        return ColorNormResult(
            image         = cv2.resize(
                image_bgr, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4
            ),
            original_size = original_size,
            method        = "resize_only_fallback",
            mean_before   = 0.0,
            mean_after    = 0.0,
        )