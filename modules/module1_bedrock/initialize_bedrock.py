"""
Module 1 — Brick 1.1  |  initialize_bedrock.py
Initialises skintel_bedrock.parquet with a flat 36-field schema.
Idempotent: skips creation if file already exists.
Run this once before the ETL ingestor.
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "/app/data/processed"
PARQUET   = os.path.join(DATA_PATH, "skintel_bedrock.parquet")

# ─────────────────────────────────────────────────────────────────────────────
# FLAT SCHEMA — 36 columns
#
# Blood markers rationale:
#   cbc_wbc          bacterial vs viral gate
#   cbc_hemoglobin   anaemia / iron-deficiency skin signs
#   cbc_platelets    purpura, lupus, vasculitis
#   cbc_eosinophils  allergic diseases + scabies
#   inf_crp          universal inflammation gate
#   inf_esr          autoimmune: psoriasis, lupus, scleroderma
#   cmp_creatinine   uremic pruritus (kidney → skin)
#   cmp_glucose      acanthosis nigricans (diabetes → skin)
#
#   REMOVED: cbc_neutrophils (derivable from WBC), cbc_lymphocytes (narrow use),
#            cmp_alt/ast (no liver disease in v1 training set)
#
# SFV fields (Shape Feature Vector):
#   NULL at ingest. Module 2 fills after segmentation.
#   Renamed from tnf_* — SFV is the correct term per roadmap.
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = pa.schema([

    # ── IDENTITY ──────────────────────────────────────────────────────────────
    ("patient_id",          pa.string()),       # FK17-A1B2 | PU20-C3D4 | SC-E5F6 | DC-G7H8
    ("source_dataset",      pa.string()),       # fitzpatrick17k | pad_ufes_20 | scin | dermaconin
    ("abha_id",             pa.string()),       # NULL for training data; populated for patient submissions
    ("ingest_timestamp",    pa.timestamp("ms")),

    # ── DATASET SPLIT ─────────────────────────────────────────────────────────
    ("split",               pa.string()),       # train | val | test
    ("is_augmented",        pa.bool_()),        # True if record is an augmented copy

    # ── IMAGE ─────────────────────────────────────────────────────────────────
    ("image_path",          pa.string()),       # MinIO key or local path
    ("original_resolution", pa.string()),       # "1920x1440" or "unknown"
    ("quality_score",       pa.float32()),      # Laplacian variance — filled by Module 2
    ("hair_removal_tier",   pa.string()),       # low | medium | high | none — filled by Module 2

    # ── LABEL ─────────────────────────────────────────────────────────────────
    ("label",               pa.string()),       # canonical disease name (leaf-level)
    ("main_class",          pa.string()),       # Bacterial | Viral | Autoimmune | Malignant …
    ("sub_class",           pa.string()),       # finer grouping within main_class
    ("icd11_code",          pa.string()),       # looked up from MongoDB during ETL
    ("label_confidence",    pa.float32()),      # 1.0=biopsy | 0.88=derm | 0.82=crowd
    ("systemic",            pa.bool_()),        # True → blood gate may activate
    ("confidence_score",    pa.float32()),      # model confidence at time of prediction (NULL for training)

    # ── SKIN TONE ─────────────────────────────────────────────────────────────
    ("fitzpatrick_type",    pa.int32()),        # 1–6 | -1 = unknown
    ("monk_skin_tone",      pa.int32()),        # 1–10 | -1 = unknown (DermaCon-IN MST labels)

    # ── LESION METADATA ───────────────────────────────────────────────────────
    ("body_part",           pa.string()),       # face | hand | trunk | lower_limb | unknown
    ("descriptors",         pa.string()),       # comma-separated morphological tags e.g. "plaque,scaling"

    # ── BLOOD MARKERS ─────────────────────────────────────────────────────────
    # Synthesised from MongoDB blood_logic at ingest (training data only)
    ("cbc_wbc",             pa.float32()),      # cells/uL   normal: 4500–11000
    ("cbc_hemoglobin",      pa.float32()),      # g/dL       normal: 12.0–17.5
    ("cbc_platelets",       pa.float32()),      # cells/uL   normal: 150k–400k
    ("cbc_eosinophils",     pa.float32()),      # 10^9/L     normal: 0.1–0.4
    ("inf_crp",             pa.float32()),      # mg/dL      normal: <0.5
    ("inf_esr",             pa.float32()),      # mm/hr      normal: <20 (M) / <30 (F)
    ("cmp_creatinine",      pa.float32()),      # mg/dL      normal: 0.6–1.2
    ("cmp_glucose",         pa.float32()),      # mg/dL      normal: 70–100

    # ── SYMPTOMS ──────────────────────────────────────────────────────────────
    ("symptom_pruritus",    pa.int32()),        # 0=none 1=mild 2=moderate 3=severe
    ("symptom_nociception", pa.int32()),        # 0–10 pain scale
    ("symptom_evolution",   pa.string()),       # rapid | chronic | stable | unknown

    # ── SFV — Shape Feature Vector ────────────────────────────────────────────
    # NULL at ingest. Module 2 fills after segmentation.
    ("sfv_border_irregularity", pa.float32()),  # 0–1    ABCDE → Border
    ("sfv_asymmetry_index",     pa.float32()),  # 0–1    ABCDE → Asymmetry
    ("sfv_fractal_dimension",   pa.float32()),  # 1.0–2.0
    ("sfv_color_gradient",      pa.float32()),  # 0–1    ABCDE → Color
])


def initialize_bedrock() -> str:
    """
    Creates skintel_bedrock.parquet with the 36-field flat schema.
    Safe to call multiple times — skips if file already exists.
    Returns the absolute path to the parquet file.
    """
    os.makedirs(DATA_PATH, exist_ok=True)

    if os.path.exists(PARQUET):
        existing = pq.read_table(PARQUET)
        logger.info(
            f"Parquet already exists — "
            f"{len(existing)} records | "
            f"{len(existing.schema)} columns — skipping"
        )
        return PARQUET

    empty_table = pa.table(
        {field.name: pa.array([], type=field.type) for field in SCHEMA},
        schema=SCHEMA,
    )
    pq.write_table(empty_table, PARQUET)

    logger.success(f"✅ Parquet bedrock initialised: {PARQUET}")
    logger.info(f"   Schema: {len(SCHEMA)} fields")

    return PARQUET


if __name__ == "__main__":
    initialize_bedrock()