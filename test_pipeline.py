import os
import cv2
import numpy as np
import sys
import shutil
sys.path.insert(0, "/app")

from modules.module2_gateway.brick20_quality_gate import (
    run_quality_gate,
    compute_laplacian_variance,
    compute_fft_sharpness,
    BLUR_THRESHOLD,
    FFT_THRESHOLD,
)
from modules.module2_gateway.brick22_segmentation import segment_and_extract_sfv

TARGET_SIZE = (224, 224)
OUTPUT_DIR  = "/app/data/processed/visual_review"

# Clear old review images before each run
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

PANEL_W = 224
PANEL_H = 224
BAR_H   = 40

# ── Edge case sharpness check ─────────────────────────────────────────────────
print("── Sharpness check on known edge cases ──")
edge_cases = [
    "/app/data/raw/pad_ufes_20/images/PAT_100_393_595.png",   # false reject at 27.1
    "/app/data/raw/pad_ufes_20/images/PAT_1000_31_620.png",   # correct reject at 13.7
    "/app/data/raw/pad_ufes_20/images/PAT_1006_53_385.png",   # correct pass at 57.5
]
for p in edge_cases:
    img = cv2.imread(p)
    if img is None:
        print(f"  NOT FOUND: {p}")
        continue
    lap      = compute_laplacian_variance(img)
    fft      = compute_fft_sharpness(img)
    lap_pass = lap >= BLUR_THRESHOLD
    fft_pass = fft >= FFT_THRESHOLD
    status   = "PASS" if (lap_pass or fft_pass) else "FAIL"
    print(f"  {os.path.basename(p):30s} lap={lap:.1f} fft={fft:.3f}  {status}")
print()


# ── Panel helper ──────────────────────────────────────────────────────────────
def make_panel(image_bgr: np.ndarray, label: str, color: tuple) -> np.ndarray:
    panel = cv2.resize(image_bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LANCZOS4)
    bar   = np.zeros((BAR_H, PANEL_W, 3), dtype=np.uint8)
    bar[:] = color
    cv2.putText(bar, label, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.vstack([panel, bar])


# ── Dataset loop ──────────────────────────────────────────────────────────────
datasets = {
    "FK17": "/app/data/raw/fitzpatrick17k/images",
    "PU20": "/app/data/raw/pad_ufes_20/images",
    "SC":   "/app/data/raw/scin/images",
    "DC":   "/app/data/raw/dermaconin/images",
}

for dataset, img_dir in datasets.items():
    if not os.path.exists(img_dir):
        print(f"[{dataset}] directory not found — skipping")
        continue

    files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:10]

    print(f"\n── {dataset} ({len(files)} images) ──────────────────────")

    for fname in files:
        path  = os.path.join(img_dir, fname)
        image = cv2.imread(path)

        if image is None:
            print(f"  {fname[:40]:40s} | ERROR: could not read")
            continue

        # ── Quality gate ──────────────────────────────────────────────────────
        quality = run_quality_gate(image)

        if not quality.passed:
            orig_panel   = make_panel(
                image,
                f"ORIGINAL  lap={quality.quality_score:.1f} fft={quality.fft_score:.3f}",
                (40, 40, 40),
            )
            failed_panel = make_panel(
                np.full_like(cv2.resize(image, TARGET_SIZE), 60),
                f"FAILED  {quality.reason[:35]}",
                (0, 0, 180),
            )
            side_by_side = np.hstack([orig_panel, failed_panel])
            out_path = os.path.join(OUTPUT_DIR, f"FAIL_{dataset}_{fname}.png")
            cv2.imwrite(out_path, side_by_side)
            print(f"  {fname[:40]:40s} | FAIL | lap={quality.quality_score:.1f} fft={quality.fft_score:.3f} | {quality.reason}")
            continue

        # ── SFV ───────────────────────────────────────────────────────────────
        try:
            seg       = segment_and_extract_sfv(image)
            sfv_label = (
                f"b={seg.sfv_border_irregularity:.2f} "
                f"a={seg.sfv_asymmetry_index:.2f} "
                f"fg={seg.sfv_color_gradient:.2f}"
            )
            sfv_ok = seg.method == "grabcut"
        except Exception as e:
            sfv_label = f"SFV ERR: {str(e)[:30]}"
            sfv_ok    = False

        # ── Processed panel ───────────────────────────────────────────────────
        processed  = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        orig_panel = make_panel(
            image,
            f"ORIGINAL  lap={quality.quality_score:.1f} fft={quality.fft_score:.3f}",
            (40, 40, 40),
        )
        proc_color = (0, 140, 0) if sfv_ok else (0, 140, 140)
        proc_panel = make_panel(processed, f"PASS  {sfv_label}", proc_color)

        side_by_side = np.hstack([orig_panel, proc_panel])
        out_path = os.path.join(OUTPUT_DIR, f"PASS_{dataset}_{fname}.png")
        cv2.imwrite(out_path, side_by_side)
        print(f"  {fname[:40]:40s} | PASS | lap={quality.quality_score:.1f} fft={quality.fft_score:.3f} | {sfv_label}")

print(f"\nOutput: {OUTPUT_DIR}")
print("Green bar  = PASS + SFV computed")
print("Teal bar   = PASS + SFV zero (GrabCut failed)")
print("Blue bar   = FAILED quality gate")