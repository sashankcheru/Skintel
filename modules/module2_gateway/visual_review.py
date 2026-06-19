"""
Module 2 — visual_review.py
Visual sanity check for the dual-stream runtime pipeline.
Saves side-by-side panels: Original | Processed (224×224)
Labels: PASS (green), PASS+SFV (teal), FAIL (red)

Run:
    docker cp test_module2.py skintel_backend:/app/test_module2.py
    winpty docker-compose exec backend python3 //app/test_module2.py
Output: data/processed/visual_review/
"""

import cv2
import numpy as np
import os
import shutil
import glob

from modules.module2_gateway.brick20_quality_gate     import run_quality_gate
from modules.module2_gateway.brick22_segmentation     import segment_and_extract_sfv
from modules.module2_gateway.brick23_color_normalizer import normalise_and_resize

TARGET_SIZE = (224, 224)
OUTPUT_DIR  = "data/processed/visual_review"
PANEL_W     = 224
PANEL_H     = 224
BAR_H       = 40

# Test folders — override as needed
TEST_CASES = {
    "closeup":   "data/raw/fitzpatrick17k/images",
    "wideangle": "data/raw/dermaconin/images",
}


def make_panel(image_bgr: np.ndarray, label: str, color: tuple) -> np.ndarray:
    panel = cv2.resize(image_bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LANCZOS4)
    bar   = np.zeros((BAR_H, PANEL_W, 3), dtype=np.uint8)
    bar[:] = color
    cv2.putText(bar, label, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return np.vstack([panel, bar])


def run_visual_review():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for stream, img_dir in TEST_CASES.items():
        if not os.path.exists(img_dir):
            print(f"[{stream}] directory not found: {img_dir}")
            continue

        paths = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                glob.glob(os.path.join(img_dir, "*.png"))
        paths = paths[:10]

        print(f"\n── {stream} ({len(paths)} images) ──")

        for img_path in paths:
            fname = os.path.basename(img_path)
            orig  = cv2.imread(img_path)
            if orig is None:
                continue

            quality = run_quality_gate(orig, stream=stream)

            if not quality.passed:
                orig_panel   = make_panel(
                    orig,
                    f"ORIGINAL score={quality.quality_score:.1f}",
                    (40, 40, 40),
                )
                failed_panel = make_panel(
                    np.full_like(cv2.resize(orig, TARGET_SIZE), 60),
                    f"FAIL {quality.reason[:30]}",
                    (0, 0, 180),
                )
                panel = np.hstack([orig_panel, failed_panel])
                out   = os.path.join(OUTPUT_DIR, f"FAIL_{stream}_{fname}.png")
                cv2.imwrite(out, panel)
                print(f"  FAIL | {fname[:35]} | {quality.reason}")
                continue

            color = normalise_and_resize(orig)

            sfv_label  = ""
            proc_color = (0, 140, 0)

            if stream == "closeup":
                try:
                    seg       = segment_and_extract_sfv(orig)
                    sfv_label = (
                        f"b={seg.sfv_border_irregularity:.2f} "
                        f"a={seg.sfv_asymmetry_index:.2f}"
                    )
                    proc_color = (0, 140, 0) if seg.method != "grabcut_failed" \
                                             else (0, 140, 140)
                except Exception as e:
                    sfv_label  = f"SFV ERR"
                    proc_color = (0, 100, 100)

            orig_panel = make_panel(
                orig,
                f"ORIGINAL q={quality.quality_score:.1f}",
                (40, 40, 40),
            )
            proc_panel = make_panel(
                color.image,
                f"PASS {sfv_label}" if sfv_label else "PASS",
                proc_color,
            )
            panel = np.hstack([orig_panel, proc_panel])
            out   = os.path.join(OUTPUT_DIR, f"PASS_{stream}_{fname}.png")
            cv2.imwrite(out, panel)
            print(f"  PASS | {fname[:35]} | q={quality.quality_score:.1f} | {sfv_label}")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_visual_review()