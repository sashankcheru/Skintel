"""
Brick 1.3 — PyArrow ETL Ingestor
Extract  : Raw CSVs + MinIO image inventory
Transform: Unified clinical schema with symptom vectors
Load     : Appends new records to skintel_bedrock.parquet
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import asyncio
import os
from loguru import logger
from modules.module1_bedrock.minio_client import get_minio_client

# ── Paths ────────────────────────────────────────────────────────────────────
PARQUET_PATH  = "data/processed/skintel_bedrock.parquet"
RAW_BUCKET    = os.getenv("MINIO_BUCKET_RAW", "skintel-images")

FITZPATRICK_CSV = "data/raw/fitzpatrick17k/fitzpatrick17k.csv"
PAD_CSV         = "data/raw/pad_ufes_20/metadata.csv"
SCIN_CASES_CSV  = "data/raw/scin/scin_cases.csv"
SCIN_LABELS_CSV = "data/raw/scin/scin_labels.csv"

FITZPATRICK_IMG = "data/raw/fitzpatrick17k/images"
PAD_IMG         = "data/raw/pad_ufes_20/images"
SCIN_IMG        = "data/raw/scin/images"

# ── Parquet Schema ────────────────────────────────────────────────────────────
SCHEMA = pa.schema([
    ('patient_id',       pa.string()),
    ('source_dataset',   pa.string()),
    ('timestamp',        pa.timestamp('ms')),
    ('cbc', pa.struct([
        ('hemoglobin',   pa.float32()),
        ('wbc_total',    pa.float32()),
        ('neutrophils',  pa.float32()),
        ('lymphocytes',  pa.float32()),
        ('monocytes',    pa.float32()),
        ('eosinophils',  pa.float32()),
        ('basophils',    pa.float32()),
        ('platelets',    pa.float32()),
    ])),
    ('cmp', pa.struct([
        ('alt',          pa.float32()),
        ('ast',          pa.float32()),
        ('alp',          pa.float32()),
        ('creatinine',   pa.float32()),
        ('bun',          pa.float32()),
        ('glucose',      pa.float32()),
    ])),
    ('inflammatory', pa.struct([
        ('crp',          pa.float32()),
        ('esr',          pa.float32()),
    ])),
    ('symptoms',         pa.list_(pa.int32())),  # [pruritus(0-3), pain(0-10), evolution(0/1)]
    ('fitzpatrick_type', pa.int32()),
    ('image_path',       pa.string()),
    ('label',            pa.string()),
])
# ── Label Normalizer ──────────────────────────────────────────────────────────
# Maps all dataset-specific label variants to a single canonical name.
# Add new mappings here as new datasets are added — never touch ETL logic.

LABEL_MAP = {
    # ── Malignant ──────────────────────────────────────────────────────────
    "basal cell carcinoma":             "Basal Cell Carcinoma",
    "bcc":                              "Basal Cell Carcinoma",
    "solid cystic basal cell carcinoma":"Basal Cell Carcinoma",
    "basal cell carcinoma morpheiform": "Basal Cell Carcinoma",
    "scc/sccis":                        "Squamous Cell Carcinoma",
    "squamous cell carcinoma":          "Squamous Cell Carcinoma",
    "scc":                              "Squamous Cell Carcinoma",
    "melanoma":                         "Melanoma",
    "malignant melanoma":               "Melanoma",
    "superficial spreading melanoma ssm": "Melanoma",
    "lentigo maligna":                  "Melanoma",
    "mel":                              "Melanoma",
    "actinic keratosis":                "Actinic Keratosis",
    "ak":                               "Actinic Keratosis",
    "porokeratosis actinic":            "Actinic Keratosis",
    "disseminated actinic porokeratosis":"Actinic Keratosis",
    "kaposi sarcoma":                   "Kaposi Sarcoma",
    "mycosis fungoides":                "Mycosis Fungoides",
    "inflicted skin lesions":           "Inflicted Skin Lesions",

    # ── Autoimmune ─────────────────────────────────────────────────────────
    "psoriasis":                        "Psoriasis",
    "pustular psoriasis":               "Psoriasis",
    "lupus erythematosus":              "Lupus",
    "lupus subacute":                   "Lupus",
    "cutaneous lupus":                  "Lupus",
    "sle - systemic lupus erythematosus-related syndrome": "Lupus",
    "scleroderma":                      "Scleroderma",
    "scleromyxedema":                   "Scleroderma",
    "dermatomyositis":                  "Dermatomyositis",
    "acquired autoimmune bullous diseaseherpes gestationis": "Autoimmune Bullous Disease",
    "bullous pemphigoid":               "Autoimmune Bullous Disease",
    "epidermolysis bullosa":            "Autoimmune Bullous Disease",

    # ── Allergic / Eczema ──────────────────────────────────────────────────
    "eczema":                           "Eczema",
    "dyshidrotic eczema":               "Eczema",
    "neurodermatitis":                  "Eczema",
    "infected eczema":                  "Eczema",
    "acute constitutional eczema":      "Eczema",
    "lichenified eczematous dermatitis":"Eczema",
    "acute and chronic dermatitis":     "Eczema",
    "chronic dermatitis, nos":          "Eczema",
    "stasis dermatitis":                "Stasis Dermatitis",
    "allergic contact dermatitis":      "Contact Dermatitis",
    "irritant contact dermatitis":      "Contact Dermatitis",
    "contact dermatitis, nos":          "Contact Dermatitis",
    "cd - contact dermatitis":          "Contact Dermatitis",
    "acute dermatitis, nos":            "Contact Dermatitis",
    "contact dermatitis caused by rhus diversiloba": "Contact Dermatitis",
    "urticaria":                        "Urticaria",
    "urticaria pigmentosa":             "Urticaria",
    "drug eruption":                    "Drug Rash",
    "drug rash":                        "Drug Rash",
    "drug induced pigmentary changes":  "Drug Rash",
    "fixed eruptions":                  "Drug Rash",
    "photodermatoses":                  "Photodermatitis",
    "photodermatitis":                  "Photodermatitis",

    # ── Bacterial ──────────────────────────────────────────────────────────
    "cellulitis":                       "Cellulitis",
    "impetigo":                         "Impetigo",
    "folliculitis":                     "Folliculitis",
    "hidradenitis":                     "Hidradenitis Suppurativa",
    "paronychia":                       "Paronychia",
    "neutrophilic dermatoses":          "Neutrophilic Dermatosis",
    "abscess":                          "Abscess",
    "infection of skin":                "Skin Infection",
    "local infection of wound":         "Skin Infection",
    "localized skin infection":         "Skin Infection",
    "skin infection":                   "Skin Infection",

    # ── Viral ──────────────────────────────────────────────────────────────
    "shingles":                         "Shingles",
    "herpes zoster":                    "Shingles",
    "herpes simplex":                   "Herpes Simplex",
    "viral exanthem":                   "Viral Exanthem",
    "molluscum contagiosum":            "Molluscum Contagiosum",

    # ── Fungal ─────────────────────────────────────────────────────────────
    "tinea":                            "Tinea",
    "tinea versicolor":                 "Tinea Versicolor",
    "ringworm":                         "Tinea",
    "candida intertrigo":               "Candidiasis",
    "intertrigo":                       "Intertrigo",
    "deep fungal infection":            "Deep Fungal Infection",

    # ── Hormonal ───────────────────────────────────────────────────────────
    "acne":                             "Acne",
    "acne vulgaris":                    "Acne",
    "rosacea":                          "Rosacea",
    "perioral dermatitis":              "Perioral Dermatitis",
    "rhinophyma":                       "Rosacea",
    "seborrheic dermatitis":            "Seborrheic Dermatitis",

    # ── Nutritional / Systemic ─────────────────────────────────────────────
    "acanthosis nigricans":             "Acanthosis Nigricans",
    "xanthomas":                        "Xanthomas",
    "porphyria":                        "Porphyria",
    "necrobiosis lipoidica":            "Necrobiosis Lipoidica",
    "purpura":                          "Purpura",
    "leukocytoclastic vasculitis":      "Vasculitis",
    "pigmented purpuric eruption":      "Purpura",
    "erythema nodosum":                 "Erythema Nodosum",
    "erythema multiforme":              "Erythema Multiforme",
    "stevens johnson syndrome":         "Stevens Johnson Syndrome",
    "vitiligo":                         "Vitiligo",
    "melasma":                          "Melasma",

    # ── Benign / Structural ────────────────────────────────────────────────
    "nevus":                            "Nevus",
    "nev":                              "Nevus",
    "nevocytic nevus":                  "Nevus",
    "congenital nevus":                 "Nevus",
    "epidermal nevus":                  "Nevus",
    "becker nevus":                     "Nevus",
    "halo nevus":                       "Nevus",
    "melanocytic nevus":                "Nevus",
    "atypical nevus":                   "Atypical Nevus",
    "seborrheic keratosis":             "Seborrheic Keratosis",
    "sek":                              "Seborrheic Keratosis",
    "sk/isk":                           "Seborrheic Keratosis",
    "keratosis pilaris":                "Keratosis Pilaris",
    "lichen planus":                    "Lichen Planus",
    "lichen planus/lichenoid eruption": "Lichen Planus",
    "lichen simplex":                   "Lichen Simplex Chronicus",
    "lichen simplex chronicus":         "Lichen Simplex Chronicus",
    "prurigo nodularis":                "Prurigo Nodularis",
    "granuloma annulare":               "Granuloma Annulare",
    "sarcoidosis":                      "Sarcoidosis",
    "cutaneous sarcoidosis":            "Sarcoidosis",
    "keloid":                           "Keloid",
    "dermatofibroma":                   "Dermatofibroma",
    "pyogenic granuloma":               "Pyogenic Granuloma",
    "port wine stain":                  "Port Wine Stain",
    "hemangioma":                       "Hemangioma",
    "scabies":                          "Scabies",
    "insect bite":                      "Insect Bite",
    "pityriasis rosea":                 "Pityriasis Rosea",
    "pityriasis rubra pilaris":         "Pityriasis Rubra Pilaris",
    "ichthyosis vulgaris":              "Ichthyosis",
    "ichthyosis":                       "Ichthyosis",
    "striae":                           "Striae",
    "milia":                            "Milia",
    "syringoma":                        "Syringoma",
    "telangiectases":                   "Telangiectasia",
    "livedo reticularis":               "Livedo Reticularis",
    "scar condition":                   "Scar",
    "hypersensitivity":                 "Hypersensitivity",
    "skin ulcer":                       "Skin Ulcer",
}


def normalize_label(raw_label: str) -> str:
    """
    Normalizes any raw label string to a canonical disease name.

    Handles:
    - Simple strings: "basal cell carcinoma" → "Basal Cell Carcinoma"
    - SCIN probability dicts: "{'Eczema': 0.67, 'Contact Dermatitis': 0.33}"
      → picks highest confidence → normalizes
    - Unknown labels → returns title-cased original (never drops data)
    """
    if not raw_label or raw_label.strip() in ('{}', 'UNKNOWN', ''):
        return 'UNKNOWN'

    # Handle SCIN probability dict format
    if raw_label.strip().startswith('{'):
        try:
            import ast
            prob_dict = ast.literal_eval(raw_label.strip())
            if not prob_dict:
                return 'UNKNOWN'
            # Pick highest confidence label
            raw_label = max(prob_dict, key=prob_dict.get)
        except (ValueError, SyntaxError):
            pass

    # Normalize via map — case insensitive lookup
    key = raw_label.strip().lower()
    return LABEL_MAP.get(key, raw_label.strip().title())

# ── Helpers ───────────────────────────────────────────────────────────────────
def _empty_cbc():
    return {k: 0.0 for k in
            ['hemoglobin','wbc_total','neutrophils','lymphocytes',
             'monocytes','eosinophils','basophils','platelets']}

def _empty_cmp():
    return {k: 0.0 for k in
            ['alt','ast','alp','creatinine','bun','glucose']}

def _empty_inflammatory():
    return {'crp': 0.0, 'esr': 0.0}

def _load_existing_paths() -> set:
    """Returns set of image_paths already in Parquet."""
    if not os.path.exists(PARQUET_PATH):
        return set()
    df = pq.read_table(PARQUET_PATH, columns=['image_path']).to_pandas()
    return set(df['image_path'].tolist())

def _save_records(records: list, existing_paths: set):
    """Appends new records to Parquet, skipping already-loaded ones."""
    new = [r for r in records if r['image_path'] not in existing_paths]
    if not new:
        logger.info("No new records to append.")
        return 0

    df  = pd.DataFrame(new)
    tbl = pa.Table.from_pandas(df, schema=SCHEMA)

    if os.path.exists(PARQUET_PATH):
        existing = pq.read_table(PARQUET_PATH)
        tbl = pa.concat_tables([existing, tbl])

    # Deduplicate on image_path — guards against dataset-level duplicates
    df_final = tbl.to_pandas().drop_duplicates(subset=['image_path'], keep='first')
    tbl = pa.Table.from_pandas(df_final, schema=SCHEMA)

    pq.write_table(tbl, PARQUET_PATH)
    return len(new)


# ── Dataset Extractors ────────────────────────────────────────────────────────

def _extract_fitzpatrick(existing_paths: set) -> list:
    """
    Fitzpatrick17k:
    - Image filename = md5hash + .jpg
    - Label from 'label' column
    - Fitzpatrick scale directly available
    - No clinical blood data — stubs used
    """
    df = pd.read_csv(FITZPATRICK_CSV)
    records = []

    for _, row in df.iterrows():
        img_file = f"{row['md5hash']}.jpg"
        img_disk = os.path.join(FITZPATRICK_IMG, img_file)
        img_path = f"skintel-images/fitzpatrick17k/{img_file}"

        if img_path in existing_paths:
            continue
        if not os.path.exists(img_disk):
            continue  # Skip rows whose image wasn't downloaded

        # Symptom vector — no symptom data in this dataset
        symptoms = [0, 0, 0]

        # Fitzpatrick scale — clean to int, default 0 if missing
        fitz = int(row['fitzpatrick_scale']) if pd.notna(row['fitzpatrick_scale']) else 0

        records.append({
            'patient_id':       f"FTZ-{row['md5hash'][:8].upper()}",
            'source_dataset':   'fitzpatrick17k',
            'timestamp':        pd.Timestamp.now().floor('ms'),
            'cbc':              _empty_cbc(),
            'cmp':              _empty_cmp(),
            'inflammatory':     _empty_inflammatory(),
            'symptoms':         symptoms,
            'fitzpatrick_type': fitz,
            'image_path':       img_path,
            'label': normalize_label(str(row['label']) if pd.notna(row['label']) else ''),
        })

    logger.info(f"Fitzpatrick17k — {len(records)} new record(s) extracted")
    return records


def _extract_pad_ufes(existing_paths: set) -> list:
    """
    PAD-UFES-20:
    - Image filename = img_id column
    - Label from 'diagnostic' column
    - Rich symptom data: itch, hurt, grew, changed, bleed
    - Fitzpatrick from 'fitspatrick' column (note typo in dataset)
    - Age available for context
    """
    df = pd.read_csv(PAD_CSV)

    # PAD diagnostic codes → readable labels
    DIAG_MAP = {
        'ACK': 'Actinic Keratosis',
        'BCC': 'Basal Cell Carcinoma',
        'MEL': 'Melanoma',
        'NEV': 'Nevus',
        'SCC': 'Squamous Cell Carcinoma',
        'SEK': 'Seborrheic Keratosis',
    }

    records = []
    for _, row in df.iterrows():
        img_file = str(row['img_id'])
        img_disk = os.path.join(PAD_IMG, img_file)
        img_path = f"skintel-images/pad_ufes_20/{img_file}"

        if img_path in existing_paths:
            continue
        if not os.path.exists(img_disk):
            continue

        # Symptom vector — map PAD columns to [pruritus, pain, evolution]
        def to_bool(val):
            if isinstance(val, bool): return val
            if isinstance(val, str):  return val.strip().upper() == 'TRUE'
            return False

        itch      = 3 if to_bool(row.get('itch'))    else 0   # pruritus: 0 or 3
        hurt      = 7 if to_bool(row.get('hurt'))    else 0   # pain: 0 or 7
        evolution = 1 if to_bool(row.get('grew')) or \
                        to_bool(row.get('changed'))  else 0   # rapid=1, slow=0

        symptoms = [itch, hurt, evolution]

        fitz = int(row['fitspatrick']) if pd.notna(row.get('fitspatrick')) else 0
        diag = DIAG_MAP.get(str(row['diagnostic']), str(row['diagnostic']))

        records.append({
            'patient_id':       f"PAD-{row['patient_id']}",
            'source_dataset':   'pad_ufes_20',
            'timestamp':        pd.Timestamp.now().floor('ms'),
            'cbc':              _empty_cbc(),
            'cmp':              _empty_cmp(),
            'inflammatory':     _empty_inflammatory(),
            'symptoms':         symptoms,
            'fitzpatrick_type': fitz,
            'image_path':       img_path,
            'label': normalize_label(diag),
        })

    logger.info(f"PAD-UFES-20 — {len(records)} new record(s) extracted")
    return records


def _extract_scin(existing_paths: set) -> list:
    """
    SCIN:
    - Image filename matched via basename of image_1_path column
    - Label from scin_labels.csv 'weighted_skin_condition_label'
    - Rich symptom data from condition_symptoms_* columns
    - Fitzpatrick from dermatologist labels
    """
    cases  = pd.read_csv(SCIN_CASES_CSV)
    labels = pd.read_csv(SCIN_LABELS_CSV)[
        ['case_id', 'weighted_skin_condition_label',
         'dermatologist_fitzpatrick_skin_type_label_1']
    ]
    df = cases.merge(labels, on='case_id', how='left')

    # Build a lookup: basename → row, using image_1_path
    records = []
    for _, row in df.iterrows():
        img_1 = row.get('image_1_path')
        if pd.isna(img_1):
            continue

        img_file = os.path.basename(str(img_1))  # e.g. '-3205742176803893704.png'
        img_disk = os.path.join(SCIN_IMG, img_file)
        img_path = f"skintel-images/scin/{img_file}"

        if img_path in existing_paths:
            continue
        if not os.path.exists(img_disk):
            continue

        # Symptom vector
        itch = 3 if row.get('condition_symptoms_itching')       == True else 0
        pain = 7 if row.get('condition_symptoms_pain')          == True else 0
        evol = 1 if row.get('condition_symptoms_increasing_size') == True else 0
        symptoms = [itch, pain, evol]

        # Fitzpatrick
        fitz_raw = row.get('dermatologist_fitzpatrick_skin_type_label_1')
        try:
            fitz = int(float(fitz_raw)) if pd.notna(fitz_raw) else 0
        except (ValueError, TypeError):
            fitz = 0

        label = normalize_label(str(row.get('weighted_skin_condition_label', '')))
        
        # Skip unlabelled cases — no training value
        if label == 'UNKNOWN':
            continue
            
        case_id = str(row['case_id'])


        records.append({
            'patient_id':       f"SCIN-{img_file[:8]}",
            'source_dataset':   'scin',
            'timestamp':        pd.Timestamp.now().floor('ms'),
            'cbc':              _empty_cbc(),
            'cmp':              _empty_cmp(),
            'inflammatory':     _empty_inflammatory(),
            'symptoms':         symptoms,
            'fitzpatrick_type': fitz,
            'image_path':       img_path,
            'label':            label,
        })

    logger.info(f"SCIN — {len(records)} new record(s) extracted")
    return records

# ── Main ETL Pipeline ─────────────────────────────────────────────────────────

def _sync_run_etl() -> dict:
    """Full ETL pipeline — Extract all 3 datasets, Transform, Load to Parquet."""
    logger.info("🚀 Starting Skintel ETL Pipeline...")
    existing_paths = _load_existing_paths()
    logger.info(f"Existing Parquet records: {len(existing_paths)}")

    # Extract
    all_records = []
    all_records.extend(_extract_fitzpatrick(existing_paths))
    all_records.extend(_extract_pad_ufes(existing_paths))
    all_records.extend(_extract_scin(existing_paths))

    logger.info(f"Total new records to load: {len(all_records)}")

    # Load
    loaded = _save_records(all_records, existing_paths)

    # Final stats
    final = pq.read_table(PARQUET_PATH)
    result = {
        "new_records_loaded": loaded,
        "total_records":      final.num_rows,
        "datasets_processed": ["fitzpatrick17k", "pad_ufes_20", "scin"],
    }

    logger.success(
        f"✅ ETL Complete — "
        f"{loaded} new records loaded | "
        f"{final.num_rows} total in Parquet"
    )
    return result


async def run_etl() -> dict:
    """Async entry point — safe to call from FastAPI."""
    return await asyncio.to_thread(_sync_run_etl)


if __name__ == "__main__":
    result = asyncio.run(run_etl())
    logger.info(f"ETL Result: {result}")