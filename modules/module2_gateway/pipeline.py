"""
Module 2 — Orchestrator  |  pipeline.py
Runtime patient upload preprocessing — dual stream.

Handles TWO image streams from guided dual capture:
  close-up   (10-15cm) — Swin-T input
  wide-angle (30-40cm) — EfficientNet/MobileNetV3 input

Per-stream pipeline:
  BOTH streams:
    1. Quality Gate         sharpness + exposure recheck on uploaded frame
    2. Colour Normalisation Gray World + gamma (ISP safety net)
    3. Resize               224×224 Lanczos

  CLOSE-UP only:
    4. SAM Segmentation     centre-point prompt → lesion mask
    5. SFV Extraction       4 geometry features → Module 5 fusion

Wide-angle skips SFV — body distribution, not lesion geometry.

No batch preprocessing. Training datasets fed raw to Swin-T.
Phase 1: single-stream Swin-T on 27k images.
Phase 2: dual-stream after real paired patient data collected via deployment.
"""

import io
import os
import cv2
import numpy as np
from dataclasses import dataclass
from loguru import logger
from minio import Minio
from dotenv import load_dotenv

from modules.module2_gateway.brick20_quality_gate     import run_quality_gate
from modules.module2_gateway.brick22_segmentation     import segment_and_extract_sfv
from modules.module2_gateway.brick23_color_normalizer import normalise_and_resize

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT",      "minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ROOT_USER",      "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_ROOT_PASSWORD",  "minioadmin123")
MINIO_SECURE   = os.getenv("MINIO_SECURE",         "false").lower() == "true"
BUCKET_RUNTIME = "skintel-runtime"
TARGET_SIZE    = (224, 224)


@dataclass
class StreamResult:
    """Result of preprocessing one image stream."""
    stream:                  str
    quality_passed:          bool
    quality_score:           float
    quality_fail_reason:     str
    color_method:            str
    mean_brightness_before:  float
    mean_brightness_after:   float
    processed_path:          str
    image_array:             np.ndarray   # 224×224 BGR — final model input
    # Close-up only — zero/skipped for wide-angle
    sfv_border_irregularity: float = 0.0
    sfv_asymmetry_index:     float = 0.0
    sfv_fractal_dimension:   float = 1.0
    sfv_color_gradient:      float = 0.0
    seg_method:              str   = "skipped"


@dataclass
class DualStreamResult:
    """Combined result for both streams — one patient submission."""
    submission_id: str
    closeup:       StreamResult
    wideangle:     StreamResult


def _get_minio() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key = MINIO_ACCESS,
        secret_key = MINIO_SECRET,
        secure     = MINIO_SECURE,
    )


def _ensure_bucket(minio: Minio, bucket: str) -> None:
    try:
        if not minio.bucket_exists(bucket):
            minio.make_bucket(bucket)
            logger.info(f"Created bucket: {bucket}")
    except Exception as exc:
        logger.warning(f"Could not ensure bucket {bucket}: {exc}")


def _upload(
    minio:         Minio,
    submission_id: str,
    stream:        str,
    image:         np.ndarray,
) -> str:
    """Upload 224×224 image to MinIO runtime bucket. Returns path or ''."""
    obj_key   = f"runtime/{submission_id}/{stream}.png"
    img_bytes = cv2.imencode(".png", image)[1].tobytes()
    try:
        _ensure_bucket(minio, BUCKET_RUNTIME)
        minio.put_object(
            BUCKET_RUNTIME, obj_key,
            io.BytesIO(img_bytes), len(img_bytes),
            content_type="image/png",
        )
        path = f"{BUCKET_RUNTIME}/{obj_key}"
        logger.info(f"[{submission_id}/{stream}] Uploaded: {path}")
        return path
    except Exception as exc:
        logger.error(f"[{submission_id}/{stream}] Upload failed: {exc}")
        return ""


def _process_stream(
    submission_id: str,
    image_bgr:     np.ndarray,
    stream:        str,
    minio:         Minio,
) -> StreamResult:
    """
    Processes one stream through Module 2 pipeline.
    Close-up: quality gate → colour norm → SAM SFV → upload
    Wide-angle: quality gate → colour norm → upload
    """
    # Quality Gate
    quality = run_quality_gate(image_bgr, stream=stream)
    if not quality.passed:
        return StreamResult(
            stream              = stream,
            quality_passed      = False,
            quality_score       = quality.quality_score,
            quality_fail_reason = quality.reason,
            color_method        = "skipped",
            mean_brightness_before = 0.0,
            mean_brightness_after  = 0.0,
            processed_path      = "",
            image_array         = cv2.resize(
                image_bgr, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4
            ),
        )

    # Colour Normalisation + Resize
    color = normalise_and_resize(image_bgr)
    final = color.image   # 224×224 BGR

    # SFV — close-up only
    sfv_border = 0.0
    sfv_asym   = 0.0
    sfv_frac   = 1.0
    sfv_color  = 0.0
    seg_method = "skipped"

    if stream == "closeup":
        try:
            seg        = segment_and_extract_sfv(image_bgr)
            sfv_border = seg.sfv_border_irregularity
            sfv_asym   = seg.sfv_asymmetry_index
            sfv_frac   = seg.sfv_fractal_dimension
            sfv_color  = seg.sfv_color_gradient
            seg_method = seg.method
        except Exception as exc:
            logger.error(f"[{submission_id}/closeup] SFV failed: {exc}")

    # Upload
    processed_path = _upload(minio, submission_id, stream, final)

    return StreamResult(
        stream                  = stream,
        quality_passed          = True,
        quality_score           = quality.quality_score,
        quality_fail_reason     = "",
        color_method            = color.method,
        mean_brightness_before  = color.mean_before,
        mean_brightness_after   = color.mean_after,
        processed_path          = processed_path,
        image_array             = final,
        sfv_border_irregularity = sfv_border,
        sfv_asymmetry_index     = sfv_asym,
        sfv_fractal_dimension   = sfv_frac,
        sfv_color_gradient      = sfv_color,
        seg_method              = seg_method,
    )


def process_dual_stream(
    submission_id:   str,
    image_closeup:   np.ndarray,
    image_wideangle: np.ndarray,
) -> DualStreamResult:
    """
    Main Module 2 entry point at runtime.
    Called by routes.py when a patient submission arrives.
    Both images must be BGR numpy arrays (cv2.imdecode output).
    Returns DualStreamResult — callers check quality_passed per stream.
    """
    minio = _get_minio()
    logger.info(f"[{submission_id}] Module 2 dual stream start")

    closeup_result   = _process_stream(submission_id, image_closeup,   "closeup",   minio)
    wideangle_result = _process_stream(submission_id, image_wideangle, "wideangle", minio)

    logger.info(
        f"[{submission_id}] Complete: "
        f"closeup={'PASS' if closeup_result.quality_passed else 'FAIL'} "
        f"wideangle={'PASS' if wideangle_result.quality_passed else 'FAIL'}"
    )

    return DualStreamResult(
        submission_id = submission_id,
        closeup       = closeup_result,
        wideangle     = wideangle_result,
    )