"""
Module 2 — Brick 2.5  |  routes.py
FastAPI router — Runtime image preprocessing endpoints

Endpoints:
  GET  /api/v1/gateway/health    Service health check
  POST /api/v1/gateway/process   Dual stream upload — close-up + wide-angle
"""

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from loguru import logger

from modules.module2_gateway.pipeline import process_dual_stream

router = APIRouter()


@router.get("/health")
async def gateway_health():
    return {"status": "ok", "module": "module2_gateway"}


@router.post("/process")
async def process_submission(
    submission_id:   str,
    image_closeup:   UploadFile = File(...),
    image_wideangle: UploadFile = File(...),
):
    """
    Processes both streams for one patient submission.

    close-up   → quality gate + colour norm + SAM SFV + resize 224×224
    wide-angle → quality gate + colour norm + resize 224×224

    Returns quality status, SFV values (close-up), processed MinIO paths.
    If quality_passed=False for a stream, the app should prompt the patient
    to retake that image.
    """
    # Decode close-up
    closeup_bytes = await image_closeup.read()
    closeup_arr   = np.frombuffer(closeup_bytes, np.uint8)
    closeup_bgr   = cv2.imdecode(closeup_arr, cv2.IMREAD_COLOR)
    if closeup_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode close-up image")

    # Decode wide-angle
    wideangle_bytes = await image_wideangle.read()
    wideangle_arr   = np.frombuffer(wideangle_bytes, np.uint8)
    wideangle_bgr   = cv2.imdecode(wideangle_arr, cv2.IMREAD_COLOR)
    if wideangle_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode wide-angle image")

    try:
        result = process_dual_stream(submission_id, closeup_bgr, wideangle_bgr)
    except Exception as exc:
        logger.error(f"process_dual_stream failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    cu = result.closeup
    wa = result.wideangle

    return {
        "submission_id": result.submission_id,
        "closeup": {
            "quality_passed":          cu.quality_passed,
            "quality_score":           round(cu.quality_score, 2),
            "quality_fail_reason":     cu.quality_fail_reason,
            "color_method":            cu.color_method,
            "mean_brightness_before":  round(cu.mean_brightness_before, 1),
            "mean_brightness_after":   round(cu.mean_brightness_after, 1),
            "processed_path":          cu.processed_path,
            "seg_method":              cu.seg_method,
            "sfv_border_irregularity": cu.sfv_border_irregularity,
            "sfv_asymmetry_index":     cu.sfv_asymmetry_index,
            "sfv_fractal_dimension":   cu.sfv_fractal_dimension,
            "sfv_color_gradient":      cu.sfv_color_gradient,
        },
        "wideangle": {
            "quality_passed":          wa.quality_passed,
            "quality_score":           round(wa.quality_score, 2),
            "quality_fail_reason":     wa.quality_fail_reason,
            "color_method":            wa.color_method,
            "mean_brightness_before":  round(wa.mean_brightness_before, 1),
            "mean_brightness_after":   round(wa.mean_brightness_after, 1),
            "processed_path":          wa.processed_path,
        },
    }