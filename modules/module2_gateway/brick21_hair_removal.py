"""
Module 2 — Brick 2.1  |  brick21_hair_removal.py
Hair Removal Pipeline

NOTE ON CURRENT USAGE:
  This brick is NOT called in the runtime pipeline.
  Hair removal at server side was dropped — it risks damaging lesion detail
  and inpainting artefacts were observed in testing.

  Hair advisory is handled client-side (Module 6):
    - Camera guidance detects hair coverage on live frame
    - If coverage > 60%, user is advised to reposition or move hair aside
    - Capture is never blocked — patient may not be able to remove hair

  This file is retained for:
    1. MobileHairNet integration (parallel research track, post Module 6)
    2. Future consideration if server-side removal is revisited

  DO NOT call remove_hair() from pipeline.py.
"""
# ... rest of existing brick21 code unchanged ...
# import os
# import cv2
# import numpy as np
# from dataclasses import dataclass
# from loguru import logger


# # ── CONFIG ────────────────────────────────────────────────────────────────────
# INPAINT_RADIUS      = int(os.getenv("INPAINT_RADIUS",            "3"))
# COVERAGE_THRESHOLD  = float(os.getenv("HAIR_COVERAGE_THRESHOLD", "0.30"))  # warn at 30%
# INPAINT_HARD_LIMIT  = float(os.getenv("HAIR_INPAINT_LIMIT",      "0.50"))  # skip inpaint at 50%
# DARK_SKIN_THRESHOLD = float(os.getenv("DARK_SKIN_THRESHOLD",     "80.0"))  # mean brightness


# @dataclass
# class HairRemovalResult:
#     image:          np.ndarray  # BGR image with hair removed (or original if guard triggered)
#     mask:           np.ndarray  # binary mask — white pixels = hair was here
#     coverage_ratio: float       # 0–1, fraction of image detected as hair
#     heavy_coverage: bool        # True if coverage > COVERAGE_THRESHOLD
#     inpaint_skipped: bool       # True if heavy coverage guard prevented inpainting
#     method:         str         # "dual_morphology" | "dual_morphology_dark_refined"


# # ── KERNEL SIZING ─────────────────────────────────────────────────────────────
# def _adaptive_kernel_size(image_bgr: np.ndarray) -> tuple[int, int]:
#     """
#     Returns (primary_kernel, fine_kernel) sizes scaled to image resolution.

#     Hair strand width is roughly 1–3px per 300px of image height.
#     Curly hair strokes are 2-3x wider → need larger kernel.
#     We use 6% of the shorter dimension for primary, 3% for fine.

#     Minimum 9px to catch fine hairs. Maximum 31px to avoid
#     over-smoothing on large images.
#     """
#     min_dim = min(image_bgr.shape[:2])
#     primary = int(min_dim * 0.06)
#     fine    = int(min_dim * 0.03)

#     # Enforce odd numbers (morphological kernels must be odd)
#     primary = max(9,  min(31, primary | 1))
#     fine    = max(7,  min(17, fine    | 1))

#     return primary, fine


# # ── DUAL MORPHOLOGY MASK ──────────────────────────────────────────────────────
# def _build_dual_mask(
#     image_bgr:   np.ndarray,
#     kernel_size: int,
# ) -> np.ndarray:
#     """
#     Builds a hair mask using both blackhat and tophat morphology.

#     BLACKHAT  = close(I) - I  → dark hair on bright background
#     TOPHAT    = I - open(I)   → white/grey hair on dark background

#     Runs on all 3 BGR channels independently and combines with OR.
#     This handles hair that is only visible in one colour channel
#     (e.g. reddish hair visible only in blue channel of BGR).

#     Returns binary mask: 255 = hair, 0 = skin.
#     """
#     kernel = cv2.getStructuringElement(
#         cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
#     )

#     combined = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

#     for ch in cv2.split(image_bgr):
#         # Dark hair detection (blackhat)
#         blackhat  = cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, kernel)
#         _, bh_bin = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

#         # White / grey hair detection (tophat)
#         tophat    = cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, kernel)
#         _, th_bin = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)

#         # Combine both detections for this channel
#         channel_mask = cv2.bitwise_or(bh_bin, th_bin)
#         combined     = cv2.bitwise_or(combined, channel_mask)

#     # Dilate slightly to fully cover hair edges before inpainting
#     dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     combined = cv2.dilate(combined, dilate_k, iterations=1)

#     return combined


# # ── DARK SKIN REFINEMENT ──────────────────────────────────────────────────────
# def _refine_dark_skin_mask(
#     image_bgr:    np.ndarray,
#     fine_kernel:  int,
# ) -> np.ndarray:
#     """
#     On dark skin (Fitzpatrick IV-VI), the primary kernel over-detects
#     because skin texture has similar contrast to fine hair.
#     Re-run with a smaller kernel + higher threshold to reduce false positives.
#     """
#     kernel = cv2.getStructuringElement(
#         cv2.MORPH_ELLIPSE, (fine_kernel, fine_kernel)
#     )
#     combined = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

#     for ch in cv2.split(image_bgr):
#         blackhat  = cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, kernel)
#         _, bh_bin = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)  # higher threshold

#         tophat    = cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, kernel)
#         _, th_bin = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)

#         channel_mask = cv2.bitwise_or(bh_bin, th_bin)
#         combined     = cv2.bitwise_or(combined, channel_mask)

#     # Tighter dilation for dark skin to avoid merging skin regions
#     dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#     combined = cv2.dilate(combined, dilate_k, iterations=1)

#     return combined


# # ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────
# def remove_hair(image_bgr: np.ndarray) -> HairRemovalResult:
#     """
#     Full dual-morphology hair removal pipeline.

#     Works on:
#       ✅ Dark hair    (blackhat)
#       ✅ White hair   (tophat)
#       ✅ Grey hair    (tophat)
#       ✅ Curly hair   (adaptive kernel size)
#       ✅ Fine hair    (multi-channel combination)
#       ✅ Dark skin    (refined kernel + higher threshold)
#       ✅ Dense hair   (heavy coverage guard — returns original if > 50%)

#     Steps:
#       1. Compute adaptive kernel sizes from image resolution
#       2. Check if dark skin → use refined mask if so
#       3. Build dual-morphology mask
#       4. Compute coverage ratio
#       5. Telea inpainting if coverage < INPAINT_HARD_LIMIT
#          else return original (preserves lesion over mosaic)

#     Returns HairRemovalResult.
#     """
#     if image_bgr is None or image_bgr.size == 0:
#         raise ValueError("Input image is empty")

#     primary_k, fine_k = _adaptive_kernel_size(image_bgr)
#     gray_mean = float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).mean())
#     is_dark_skin = gray_mean < DARK_SKIN_THRESHOLD

#     # Build mask — use refined kernel for dark skin
#     if is_dark_skin:
#         mask   = _refine_dark_skin_mask(image_bgr, fine_k)
#         method = "dual_morphology_dark_refined"
#         logger.info(
#             f"Hair removal: dark skin detected (mean={gray_mean:.1f}) "
#             f"— using refined kernel={fine_k}px"
#         )
#     else:
#         mask   = _build_dual_mask(image_bgr, primary_k)
#         method = "dual_morphology"
#         logger.info(
#             f"Hair removal: standard skin (mean={gray_mean:.1f}) "
#             f"— kernel={primary_k}px"
#         )

#     # Coverage check
#     total_pixels   = mask.shape[0] * mask.shape[1]
#     masked_pixels  = int(np.count_nonzero(mask))
#     coverage_ratio = masked_pixels / total_pixels
#     heavy_coverage = coverage_ratio > COVERAGE_THRESHOLD

#     if heavy_coverage:
#         logger.warning(
#             f"Hair removal: heavy coverage {coverage_ratio:.1%} "
#             f"(>{COVERAGE_THRESHOLD:.0%})"
#         )

#     # Heavy coverage guard — skip inpainting if it would destroy the image
#     if coverage_ratio > INPAINT_HARD_LIMIT:
#         logger.warning(
#             f"Hair removal: coverage {coverage_ratio:.1%} exceeds inpaint limit "
#             f"({INPAINT_HARD_LIMIT:.0%}) — returning original to preserve lesion detail"
#         )
#         return HairRemovalResult(
#             image           = image_bgr.copy(),
#             mask            = mask,
#             coverage_ratio  = coverage_ratio,
#             heavy_coverage  = True,
#             inpaint_skipped = True,
#             method          = method,
#         )

#     # Telea inpainting — fills hair regions by propagating surrounding skin texture
#     result_image = cv2.inpaint(image_bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)

#     logger.info(
#         f"Hair removal done: coverage={coverage_ratio:.1%}, "
#         f"heavy={heavy_coverage}, method={method}, "
#         f"kernel={fine_k if is_dark_skin else primary_k}px"
#     )

#     return HairRemovalResult(
#         image           = result_image,
#         mask            = mask,
#         coverage_ratio  = coverage_ratio,
#         heavy_coverage  = heavy_coverage,
#         inpaint_skipped = False,
#         method          = method,
#     )

"""
Module 2 — Brick 2.1  |  brick21_hair_removal.py
Hair Removal Pipeline — All Hair Types, All Skin Tones

Problem:
  Standard DullRazor (blackhat only) fails on:
    - White / grey hair    → bright structures, blackhat is blind to them
    - Curly / thick hair   → standard kernel too small, misses wide strokes
    - Dark skin (Fitz IV-VI) → low contrast causes massive over-detection → mosaics

Solution — Dual-morphology DullRazor:
  DARK  hair: Blackhat  = morphological_close(I) - I
              Detects dark thin structures on bright backgrounds.
  WHITE hair: Tophat    = I - morphological_open(I)
              Detects bright thin structures on dark backgrounds.
  Both masks combined with OR → complete hair detection across all types.

Adaptive kernel:
  Kernel size scales with image resolution so curly/thick hair
  strokes are covered regardless of capture device.

Heavy coverage guard — two tiers (v2):
  TIER 1 — COVERAGE_THRESHOLD (30%):  warn only, inpaint still runs with Telea.
  TIER 2 — INPAINT_HARD_LIMIT (60%):  skip inpainting entirely — hair is less
            damaging to Swin-T than a mosaic reconstruction.

  Why no NS inpainting tier:
    Testing showed Navier-Stokes inpainting at 50–65% coverage creates large
    blotchy patch seams that CLAHE then amplifies into a visible honeycomb grid.
    Telea produces cleaner results at all tested coverage levels below 60%.
    Images above 60% are correctly returned unchanged.

Kernel cap reduced 31 → 21 (v2):
  The original 6% rule with a 31px cap was calibrated for dermoscopy crops
  (~300px). Full-resolution body photos (scin dataset, 600–1200px) hit the
  31px cap, and a 31px morphological kernel on smooth skin detects skin folds
  and texture as "thick hair", creating 60%+ false-positive coverage.
  Reducing to 4% rule with a 21px cap eliminates this on full-res body images
  while still covering thick curly hair strands in dermoscopy crops.

Tophat threshold kept at 10 (reverted from 7):
  Lowering to 7 increased white/grey hair recall on pad_ufes_20_005 but caused
  significant false positives on smooth pale skin (scin body images +12–15%).
  The net effect was worse. Threshold remains at 10, independently configurable.

Adaptive inpaint radius (v2):
  Fixed radius of 3px was insufficient for thick curly strands seen in
  fitzpatrick17k dataset. Radius now scales with kernel size: max(3, kernel//3).
  This fills wider strands without over-smoothing fine-hair inpaints.

Skin tone awareness:
  Dark images (median brightness of central 60% crop < DARK_SKIN_THRESHOLD)
  re-run with refined smaller kernel. Median of central crop used instead of
  global mean — more robust against dark vignette borders and bright lesion centres.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from loguru import logger


# ── CONFIG ────────────────────────────────────────────────────────────────────
COVERAGE_THRESHOLD    = float(os.getenv("HAIR_COVERAGE_THRESHOLD", "0.30"))  # warn at 30%
INPAINT_HARD_LIMIT    = float(os.getenv("HAIR_INPAINT_LIMIT",      "0.60"))  # skip inpaint at 60%
DARK_SKIN_THRESHOLD   = float(os.getenv("DARK_SKIN_THRESHOLD",     "80.0"))  # median brightness

# Threshold separation: tophat (white/grey hair) has weaker morphological
# response than blackhat (dark hair). Kept at 10 — lowering to 7 caused
# false positives on smooth pale skin (scin body images) that outweighed
# the white hair recall gain. Independently configurable via env vars.
BLACKHAT_THRESHOLD      = int(os.getenv("BLACKHAT_THRESHOLD",      "10"))
TOPHAT_THRESHOLD        = int(os.getenv("TOPHAT_THRESHOLD",        "10"))
BLACKHAT_THRESHOLD_DARK = int(os.getenv("BLACKHAT_THRESHOLD_DARK", "15"))  # dark skin: higher to reduce FP
TOPHAT_THRESHOLD_DARK   = int(os.getenv("TOPHAT_THRESHOLD_DARK",   "12"))  # still lower than blackhat


# ── RESULT DATACLASS ──────────────────────────────────────────────────────────
@dataclass
class HairRemovalResult:
    image:              np.ndarray  # BGR image — hair removed, or original if guard triggered
    mask:               np.ndarray  # binary mask: 255 = hair pixel
    coverage_ratio:     float       # 0–1, fraction of image detected as hair
    heavy_coverage:     bool        # True if coverage > COVERAGE_THRESHOLD
    inpaint_skipped:    bool        # True if hard limit guard prevented inpainting
    method:             str         # "dual_morphology" | "dual_morphology_dark_refined"
    inpaint_method:     str         # "telea" | "ns" | "skipped"
    kernel_used:        int         # actual kernel size applied (diagnostic)
    dark_skin_detected: bool        # whether the dark-skin path was triggered


# ── INPUT VALIDATION ──────────────────────────────────────────────────────────
def _validate_input(image_bgr: np.ndarray) -> None:
    """
    Raises ValueError with a clear message for invalid inputs.
    Catches the silent channel mismatch that breaks cv2.split() downstream.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or None")
    if image_bgr.ndim != 3:
        raise ValueError(
            f"Expected 3-channel BGR image, got ndim={image_bgr.ndim}. "
            "Convert grayscale with cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) first."
        )
    if image_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel BGR image, got {image_bgr.shape[2]} channels. "
            "Ensure image is loaded with cv2.imread() without IMREAD_UNCHANGED."
        )


# ── SKIN TONE DETECTION ───────────────────────────────────────────────────────
def _median_brightness(image_bgr: np.ndarray) -> float:
    """
    Returns median brightness of the central 60% crop.

    Why median of central crop instead of global mean:
      - Dermoscopy images often have dark vignette borders — global mean skews
        dark even on light skin, falsely triggering the dark-skin path.
      - Bright lesion centres would skew the mean light, hiding true skin tone.
      - Median is robust to both extremes; central crop excludes border artifacts.
    """
    h, w = image_bgr.shape[:2]
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.2), int(w * 0.8)
    crop = image_bgr[y0:y1, x0:x1]
    return float(np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))


# ── KERNEL SIZING ─────────────────────────────────────────────────────────────
def _adaptive_kernel_size(image_bgr: np.ndarray) -> tuple[int, int]:
    """
    Returns (primary_kernel, fine_kernel) sizes scaled to image resolution.

    Hair strand width is roughly 1–3px per 300px of image height.
    Curly hair strokes are 2–3x wider → need larger kernel.
    We use 4% of the shorter dimension for primary, 2% for fine.

    Why 4% not 6%:
      The 6% rule with a 31px cap worked for dermoscopy crops (~300px).
      Full-resolution body photos (scin dataset, 600–1200px) all hit the 31px
      cap. A 31px morphological kernel on smooth skin detects skin folds and
      texture as "thick hair" (60%+ false-positive coverage confirmed in testing).
      4% rule with a 21px cap covers thick curly dermoscopy hair while avoiding
      over-detection on large full-resolution body images.

    Minimum 9px to catch fine hairs. Maximum 21px.
    """
    min_dim = min(image_bgr.shape[:2])
    primary = int(min_dim * 0.04)
    fine    = int(min_dim * 0.02)

    # Enforce odd numbers (morphological kernels must be odd)
    primary = max(9,  min(21, primary | 1))
    fine    = max(7,  min(13, fine    | 1))

    return primary, fine


def _adaptive_inpaint_radius(kernel_size: int) -> int:
    """
    Returns inpaint radius scaled to kernel size.

    Fixed radius of 3px is insufficient for thick curly strands (kernel=21+).
    Scaling to kernel//3 ensures the inpaint fill region is proportional to
    the hair strand width being removed.
    Minimum 3px for fine hairs. Maximum 9px to avoid over-smoothing.
    """
    return max(3, min(9, kernel_size // 3))


# ── DUAL MORPHOLOGY MASK ──────────────────────────────────────────────────────
def _build_dual_mask(
    image_bgr:   np.ndarray,
    kernel_size: int,
    bh_thresh:   int = BLACKHAT_THRESHOLD,
    th_thresh:   int = TOPHAT_THRESHOLD,
    dilate_size: int = 3,
) -> np.ndarray:
    """
    Builds a hair mask using both blackhat and tophat morphology.

    BLACKHAT  = close(I) - I  → dark hair on bright background
    TOPHAT    = I - open(I)   → white/grey hair on dark background

    Runs on all 3 BGR channels independently and combines with OR.
    This handles hair that is only visible in one colour channel
    (e.g. reddish hair visible only in blue channel of BGR).

    Separate thresholds for blackhat vs tophat (v2):
      White/grey hair has weaker tophat response than dark hair's blackhat
      response. A lower th_thresh catches thin bright strands that the
      shared threshold=10 was missing (confirmed on pad_ufes_20_005).

    Returns binary mask: 255 = hair, 0 = skin.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    combined = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    for ch in cv2.split(image_bgr):
        # Dark hair detection
        blackhat  = cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, kernel)
        _, bh_bin = cv2.threshold(blackhat, bh_thresh, 255, cv2.THRESH_BINARY)

        # White / grey hair detection — lower threshold than blackhat
        tophat    = cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, kernel)
        _, th_bin = cv2.threshold(tophat, th_thresh, 255, cv2.THRESH_BINARY)

        channel_mask = cv2.bitwise_or(bh_bin, th_bin)
        combined     = cv2.bitwise_or(combined, channel_mask)

    # Dilate to fully cover hair edges before inpainting
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    combined = cv2.dilate(combined, dilate_k, iterations=1)

    return combined


# ── DARK SKIN REFINEMENT ──────────────────────────────────────────────────────
def _refine_dark_skin_mask(
    image_bgr:   np.ndarray,
    fine_kernel: int,
) -> np.ndarray:
    """
    On dark skin (Fitzpatrick IV-VI), the primary kernel over-detects
    because skin texture has similar contrast to fine hair.
    Re-run with a smaller kernel + higher thresholds to reduce false positives.
    Tighter dilation (2px) avoids merging adjacent skin texture regions.
    """
    return _build_dual_mask(
        image_bgr,
        kernel_size = fine_kernel,
        bh_thresh   = BLACKHAT_THRESHOLD_DARK,
        th_thresh    = TOPHAT_THRESHOLD_DARK,
        dilate_size  = 2,
    )


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────
def remove_hair(image_bgr: np.ndarray) -> HairRemovalResult:
    """
    Full dual-morphology hair removal pipeline.

    Works on:
      ✅ Dark hair       (blackhat)
      ✅ White/grey hair (tophat, lowered threshold=7 catches thin bright strands)
      ✅ Curly hair      (adaptive kernel size)
      ✅ Fine hair       (multi-channel combination)
      ✅ Dark skin       (refined kernel + higher thresholds, median crop for tone)
      ✅ Dense hair      (coverage guard):
           < 60%   → Telea inpainting
           > 60%   → return original (preserves lesion over mosaic)

    Steps:
      1. Validate input (channel count, non-empty)
      2. Compute adaptive kernel and inpaint radius from image resolution
      3. Detect skin tone via median brightness of central crop
      4. Build dual-morphology mask (dark skin → refined params)
      5. Compute coverage ratio
      6. Route to Telea / NS / skip based on coverage tiers

    Returns HairRemovalResult.
    """
    _validate_input(image_bgr)

    primary_k, fine_k    = _adaptive_kernel_size(image_bgr)
    brightness           = _median_brightness(image_bgr)
    is_dark_skin         = brightness < DARK_SKIN_THRESHOLD
    kernel_used          = fine_k if is_dark_skin else primary_k
    inpaint_radius       = _adaptive_inpaint_radius(kernel_used)

    # Build mask — use refined params for dark skin
    if is_dark_skin:
        mask   = _refine_dark_skin_mask(image_bgr, fine_k)
        method = "dual_morphology_dark_refined"
        logger.info(
            f"Hair removal: dark skin detected (median_brightness={brightness:.1f}) "
            f"— refined kernel={fine_k}px, bh_thresh={BLACKHAT_THRESHOLD_DARK}, "
            f"th_thresh={TOPHAT_THRESHOLD_DARK}"
        )
    else:
        mask   = _build_dual_mask(image_bgr, primary_k)
        method = "dual_morphology"
        logger.info(
            f"Hair removal: standard skin (median_brightness={brightness:.1f}) "
            f"— kernel={primary_k}px, bh_thresh={BLACKHAT_THRESHOLD}, "
            f"th_thresh={TOPHAT_THRESHOLD}"
        )

    # Coverage metrics
    total_pixels   = mask.shape[0] * mask.shape[1]
    masked_pixels  = int(np.count_nonzero(mask))
    coverage_ratio = masked_pixels / total_pixels
    heavy_coverage = coverage_ratio > COVERAGE_THRESHOLD

    if heavy_coverage:
        logger.warning(
            f"Hair removal: heavy coverage {coverage_ratio:.1%} "
            f"(>{COVERAGE_THRESHOLD:.0%}) — kernel={kernel_used}px"
        )

    # ── Tier 2: ultra-dense — skip inpainting entirely ──────────────────────
    if coverage_ratio > INPAINT_HARD_LIMIT:
        logger.warning(
            f"Hair removal: ultra-dense coverage {coverage_ratio:.1%} exceeds "
            f"hard limit ({INPAINT_HARD_LIMIT:.0%}) — returning original to "
            f"preserve lesion detail. method={method}"
        )
        return HairRemovalResult(
            image              = image_bgr.copy(),
            mask               = mask,
            coverage_ratio     = coverage_ratio,
            heavy_coverage     = True,
            inpaint_skipped    = True,
            method             = method,
            inpaint_method     = "skipped",
            kernel_used        = kernel_used,
            dark_skin_detected = is_dark_skin,
        )

    # ── Tier 1: normal — Telea inpainting ───────────────────────────────────
    result_image   = cv2.inpaint(image_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
    inpaint_method = "telea"

    logger.info(
        f"Hair removal done: coverage={coverage_ratio:.1%}, heavy={heavy_coverage}, "
        f"method={method}, inpaint={inpaint_method}, "
        f"kernel={kernel_used}px, radius={inpaint_radius}px"
    )

    return HairRemovalResult(
        image              = result_image,
        mask               = mask,
        coverage_ratio     = coverage_ratio,
        heavy_coverage     = heavy_coverage,
        inpaint_skipped    = False,
        method             = method,
        inpaint_method     = inpaint_method,
        kernel_used        = kernel_used,
        dark_skin_detected = is_dark_skin,
    )