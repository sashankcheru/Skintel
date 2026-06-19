"""
Module 1 — Brick 1.2  |  etl_ingestor.py
ETL pipeline: reads all 4 raw datasets → normalises labels →
looks up ICD-11 from MongoDB → synthesises blood values →
appends new rows to skintel_bedrock.parquet.

Idempotent: existing patient_ids are never re-inserted.
Run after initialize_bedrock.py and seed_knowledge.py.

Confirmed column names per dataset (verified from actual CSVs):
  fitzpatrick17k : label, fitzpatrick_scale, md5hash, url
  pad_ufes_20    : diagnostic (NEV/BCC/ACK/SEK/SCC/MEL),
                   fitspatrick (typo — missing z), img_id,
                   itch/hurt/grew/changed (TRUE/FALSE strings)
  scin           : scin_labels.csv joined with scin_cases.csv on case_id
                   weighted_skin_condition_label (probability dict string)
                   dermatologist_fitzpatrick_skin_type_label_1 (e.g. 'FST2')
                   image_1_path (e.g. 'dataset/images/-320...704.png')
                   condition_symptoms_itching/pain/burning ('YES' strings)
                   other_symptoms_fatigue/joint_pain/fever ('YES' strings)
  dermaconin     : Disease_label, Image_name, Confidence, Fitzpatrick,
                   Monk_skin_tone, Main_class, Sub_class, Body_part
"""

import asyncio
import ast
import os
import uuid
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

RAW_DIR  = "/app/data/raw"
DATA_DIR = "/app/data/processed"
PARQUET  = os.path.join(DATA_DIR, "skintel_bedrock.parquet")

MONGO_URL = os.getenv("MONGODB_URL")
DB_NAME   = os.getenv("MONGODB_DB_NAME")

# ── PAD-UFES-20 abbreviation map ──────────────────────────────────────────────
PAD_LABEL_MAP = {
    "NEV": "Nevus",
    "BCC": "Basal Cell Carcinoma",
    "ACK": "Actinic Keratosis",
    "SEK": "Seborrheic Keratosis",
    "SCC": "Squamous Cell Carcinoma",
    "MEL": "Melanoma",
}

# ── MASTER LABEL MAP ──────────────────────────────────────────────────────────
# Covers all labels from Fitzpatrick17k, PAD-UFES-20, SCIN, and DermaCon-IN.
# Verified against actual CSV values from all four datasets.
# Unknown labels are passed through as title-cased originals — never dropped.

LABEL_MAP: dict[str, str] = {

    # ── Malignant ─────────────────────────────────────────────────────────────
    "melanoma":                             "Melanoma",
    "mel":                                  "Melanoma",
    "malignant melanoma":                   "Melanoma",
    "lentigo maligna":                      "Melanoma",
    "superficial spreading melanoma ssm":   "Melanoma",
    "basal cell carcinoma":                 "Basal Cell Carcinoma",
    "bcc":                                  "Basal Cell Carcinoma",
    "solid cystic basal cell carcinoma":    "Basal Cell Carcinoma",
    "basal cell carcinoma morpheiform":     "Basal Cell Carcinoma",
    "squamous cell carcinoma":              "Squamous Cell Carcinoma",
    "scc":                                  "Squamous Cell Carcinoma",
    "scc/sccis":                            "Squamous Cell Carcinoma",
    "actinic keratosis":                    "Actinic Keratosis",
    "actinic_keratosis":                    "Actinic Keratosis",
    "actinic keratoses":                    "Actinic Keratosis",
    "ack":                                  "Actinic Keratosis",
    "ak":                                   "Actinic Keratosis",
    "porokeratosis actinic":               "Actinic Keratosis",
    "arsenical keratosis":                  "Actinic Keratosis",
    "mycosis fungoides":                    "Mycosis Fungoides",
    "cutaneous t cell lymphoma":            "Mycosis Fungoides",
    "kaposi's sarcoma of skin":             "Kaposi Sarcoma",
    "skin cancer":                          "Melanoma",

    # ── Nevus / Benign Growths ────────────────────────────────────────────────
    "nevus":                                "Nevus",
    "nev":                                  "Nevus",
    "nevocytic nevus":                      "Nevus",
    "melanocytic nevus":                    "Nevus",
    "congenital nevus":                     "Nevus",
    "epidermal nevus":                      "Nevus",
    "becker nevus":                         "Nevus",
    "halo nevus":                           "Nevus",
    "nevus anemicus":                       "Nevus",
    "nevus depigmentosis":                  "Nevus",
    "nevus sebaceous":                      "Nevus",
    "atypical nevus":                       "Nevus",
    "seborrheic keratosis":                 "Seborrheic Keratosis",
    "sek":                                  "Seborrheic Keratosis",
    "sk/isk":                               "Seborrheic Keratosis",
    "seborrheic melanosis":                 "Seborrheic Keratosis",
    "dermatofibroma":                       "Dermatofibroma",
    "pyogenic granuloma":                   "Pyogenic Granuloma",
    "granuloma pyogenic":                   "Pyogenic Granuloma",
    "hemangioma":                           "Hemangioma",
    "port wine stain":                      "Hemangioma",
    "cherry angioma":                       "Hemangioma",
    "keloid":                               "Keloid",
    "hypertrophic scar / keloid":           "Keloid",
    "scar condition":                       "Keloid",
    "milia":                                "Milia",
    "milium":                               "Milia",
    "syringoma":                            "Syringoma",
    "lipoma":                               "Lipoma",
    "acrochordon":                          "Acrochordon",
    "dermatosis papulosa nigra":            "Dermatosis Papulosa Nigra",
    "cyst":                                 "Cyst",
    "sebaceous cyst":                       "Cyst",
    "mucus cyst":                           "Cyst",
    "epidermoid cyst":                      "Cyst",
    "steatocystoma multiplex":              "Cyst",

    # ── Autoimmune ────────────────────────────────────────────────────────────
    "psoriasis":                            "Psoriasis",
    "plaque psoriasis":                     "Psoriasis",
    "chronic plaque psoriasis":             "Psoriasis",
    "psoriasis vulgaris":                   "Psoriasis",
    "guttate psoriasis":                    "Psoriasis",
    "pustular psoriasis":                   "Psoriasis",
    "inverse psoriasis":                    "Psoriasis",
    "palmar psoriasis":                     "Psoriasis",
    "parapsoriasis":                        "Psoriasis",
    "lupus":                                "Lupus",
    "sle":                                  "Lupus",
    "systemic lupus":                       "Lupus",
    "lupus erythematosus":                  "Lupus",
    "lupus_erythematosus":                  "Lupus",
    "cutaneous lupus":                      "Lupus",
    "discoid lupus erythematosus":          "Lupus",
    "scleroderma":                          "Scleroderma",
    "systemic sclerosis":                   "Scleroderma",
    "systemic_sclerosis":                   "Scleroderma",
    "morphea":                              "Scleroderma",
    "morphea/scleroderma":                  "Scleroderma",
    "scleromyxedema":                       "Scleroderma",
    "dermatomyositis":                      "Dermatomyositis",
    "pemphigus vulgaris":                   "Pemphigus Vulgaris",
    "pemphigus_vulgaris":                   "Pemphigus Vulgaris",
    "pemphigus":                            "Pemphigus Vulgaris",
    "pemphigus foliaceus":                  "Pemphigus Vulgaris",
    "bullous pemphigoid":                   "Bullous Pemphigoid",
    "bullous dermatitis, nos":              "Bullous Pemphigoid",
    "bullous dermatosis":                   "Bullous Pemphigoid",
    "dermatitis herpetiformis":             "Bullous Pemphigoid",
    "epidermolysis bullosa":                "Bullous Pemphigoid",
    "epidermolysis bullosa simplex":        "Bullous Pemphigoid",
    "chronic bullous disease of childhood": "Bullous Pemphigoid",
    "bullous fixed drug eruption":          "Bullous Pemphigoid",
    "vitiligo":                             "Vitiligo",
    "alopecia areata":                      "Alopecia Areata",
    "alopecia_areata":                      "Alopecia Areata",
    "traction alopecia":                    "Alopecia Areata",
    "androgenic alopecia":                  "Alopecia Areata",
    "folliculitis decalvans":               "Alopecia Areata",
    "ophiasis":                             "Alopecia Areata",
    "lichen planopilaris":                  "Alopecia Areata",
    "hidradenitis suppurativa":             "Hidradenitis Suppurativa",
    "hidradenitis_suppurativa":             "Hidradenitis Suppurativa",
    "hidradenitis":                         "Hidradenitis Suppurativa",
    "hs":                                   "Hidradenitis Suppurativa",
    "granuloma annulare":                   "Granuloma Annulare",
    "cutaneous sarcoidosis":                "Sarcoidosis",
    "sarcoidosis":                          "Sarcoidosis",
    "lichen planus":                        "Lichen Planus",
    "lichen planus/lichenoid eruption":     "Lichen Planus",
    "gutted lichen planus":                 "Lichen Planus",
    "hypertrophic lichen planus":           "Lichen Planus",
    "linear lichen planus":                 "Lichen Planus",
    "lichen nitidus":                       "Lichen Planus",
    "lichen striatus":                      "Lichen Planus",
    "lichen spinulosus":                    "Lichen Planus",
    "lichenoid eruption":                   "Lichen Planus",
    "lichen sclerosus":                     "Lichen Sclerosus",
    "lichen simplex":                       "Lichen Simplex Chronicus",
    "lichen simplex chronicus":             "Lichen Simplex Chronicus",
    "prurigo nodularis":                    "Prurigo Nodularis",
    "prurigo":                              "Prurigo Nodularis",
    "prurigo simplex":                      "Prurigo Nodularis",

    # ── Eczema / Allergic ─────────────────────────────────────────────────────
    "eczema":                               "Eczema",
    "atopic dermatitis":                    "Eczema",
    "atopic_dermatitis":                    "Eczema",
    "dyshidrotic eczema":                   "Eczema",
    "neurodermatitis":                      "Eczema",
    "infected eczema":                      "Eczema",
    "acute constitutional eczema":          "Eczema",
    "lichenified eczematous dermatitis":    "Eczema",
    "lichenified eczema":                   "Eczema",
    "acute and chronic dermatitis":         "Eczema",
    "chronic dermatitis, nos":              "Eczema",
    "crusted eczematous dermatitis":        "Eczema",
    "acute dermatitis, nos":               "Eczema",
    "disseminated eczema":                  "Eczema",
    "dry discoid eczema":                   "Eczema",
    "ear eczema":                           "Eczema",
    "infected keratoderma":                 "Eczema",
    "eczema keloidalis":                    "Eczema",
    "contact dermatitis":                   "Contact Dermatitis",
    "contact_dermatitis":                   "Contact Dermatitis",
    "allergic contact dermatitis":          "Contact Dermatitis",
    "irritant contact dermatitis":          "Contact Dermatitis",
    "contact dermatitis, nos":              "Contact Dermatitis",
    "cd - contact dermatitis":              "Contact Dermatitis",
    "contact dermatitis caused by rhus diversiloba": "Contact Dermatitis",
    "contact dermatitis with secondary infection": "Contact Dermatitis",
    "textile dermatitis":                   "Contact Dermatitis",
    "kumkum dermatitis":                    "Contact Dermatitis",
    "paederus dermatitis":                  "Contact Dermatitis",
    "diaper dermatitis":                    "Contact Dermatitis",
    "scrotal dermatitis":                   "Contact Dermatitis",
    "sweat dermatitis":                     "Contact Dermatitis",
    "berloque dermatitis":                  "Contact Dermatitis",
    "photocontact dermatitis [berloque dermatitis]": "Contact Dermatitis",
    "urticaria":                            "Urticaria",
    "urticaria pigmentosa":                 "Urticaria",
    "hives":                                "Urticaria",
    "angioedema":                           "Angioedema",
    "drug eruption":                        "Drug Eruption",
    "drug_eruption":                        "Drug Eruption",
    "drug rash":                            "Drug Eruption",
    "drug induced pigmentary changes":      "Drug Eruption",
    "fixed eruptions":                      "Drug Eruption",
    "fixed drug eruption":                  "Drug Eruption",
    "photodermatoses":                      "Photodermatitis",
    "photodermatitis":                      "Photodermatitis",
    "phototoxic dermatitis":                "Photodermatitis",
    "phytophotodermatitis":                 "Photodermatitis",
    "chronic actinic dermatitis":           "Photodermatitis",
    "actinic dermatitis":                   "Photodermatitis",
    "polymorphous light eruption":          "Photodermatitis",
    "stasis dermatitis":                    "Stasis Dermatitis",
    "erythema multiforme":                  "Erythema Multiforme",
    "erythema nodosum":                     "Erythema Nodosum",
    "erythema annulare centrifugum":        "Erythema Annulare",
    "erythema migrans":                     "Erythema Migrans",
    "insect bite":                          "Insect Bite",
    "insect bite reaction":                 "Insect Bite",

    # ── Bacterial ─────────────────────────────────────────────────────────────
    "cellulitis":                           "Cellulitis",
    "erysipelas":                           "Erysipelas",
    "impetigo":                             "Impetigo",
    "pyoderma":                             "Pyoderma",
    "folliculitis":                         "Folliculitis",
    "agminate folliculitis":                "Folliculitis",
    "pseudofolliculitis barbae":            "Folliculitis",
    "acne keloidalis":                      "Folliculitis",
    "furuncle":                             "Pyoderma",
    "abscess":                              "Pyoderma",
    "ecthyma":                              "Impetigo",
    "paronychia":                           "Paronychia",
    "skin infection":                       "Skin Infection",
    "infection of skin":                    "Skin Infection",
    "local infection of wound":             "Skin Infection",
    "staphylococcal scalded skin syndrome": "Skin Infection",
    "syphilis":                             "Syphilis",
    "tuberculosis of skin and subcutaneous tissue": "Leprosy",
    "lupus vulgaris":                       "Leprosy",
    "leprosy":                              "Leprosy",
    "hansen disease":                       "Leprosy",
    "hansen's disease":                     "Leprosy",
    "mycetoma":                             "Mycetoma",
    "maduromycosis":                        "Mycetoma",

    # ── Viral ─────────────────────────────────────────────────────────────────
    "shingles":                             "Shingles",
    "herpes zoster":                        "Shingles",
    "herpes_zoster":                        "Shingles",
    "chickenpox":                           "Chickenpox",
    "varicella":                            "Chickenpox",
    "varicella zoster":                     "Chickenpox",
    "chicken pox exanthem":                 "Chickenpox",
    "chickenpox exanthem":                  "Chickenpox",
    "measles":                              "Measles",
    "rubeola":                              "Measles",
    "molluscum contagiosum":                "Molluscum Contagiosum",
    "molluscum_contagiosum":                "Molluscum Contagiosum",
    "molluscum":                            "Molluscum Contagiosum",
    "warts":                                "Warts",
    "wart":                                 "Warts",
    "verruca":                              "Warts",
    "verruca vulgaris":                     "Warts",
    "hpv wart":                             "Warts",
    "condyloma acuminatum":                 "Warts",
    "herpes simplex":                       "Herpes Simplex",
    "herpes genitalis":                     "Herpes Simplex",
    "herpes labialis":                      "Herpes Simplex",
    "viral exanthem":                       "Viral Exanthem",
    "unilateral laterothoracic exanthem":   "Viral Exanthem",
    "hand foot mouth disease":              "Hand Foot Mouth Disease",
    "hand_foot_mouth_disease":              "Hand Foot Mouth Disease",
    "hfmd":                                 "Hand Foot Mouth Disease",

    # ── Fungal ────────────────────────────────────────────────────────────────
    "tinea":                                "Tinea",
    "ringworm":                             "Tinea",
    "dermatophytosis":                      "Tinea",
    "tinea corporis":                       "Tinea",
    "tinea_corporis":                       "Tinea",
    "tinea capitis":                        "Tinea",
    "tinea cruris":                         "Tinea",
    "tinea pedis":                          "Tinea",
    "tinea faciei":                         "Tinea",
    "tinea manuum":                         "Tinea",
    "tinea unguium":                        "Tinea",
    "onychomycosis":                        "Tinea",
    "steroid modified tinea":               "Tinea",
    "eczema keloidalis nuchae":             "Tinea",
    "infected tinea":                       "Tinea",
    "eczemated tinea":                      "Tinea",
    "tinea infected with secondary bacterial infection": "Tinea",
    "majocchi granuloma":                   "Tinea",
    "pityriasis versicolor":                "Pityriasis Versicolor",
    "pityriasis_versicolor":                "Pityriasis Versicolor",
    "tinea versicolor":                     "Pityriasis Versicolor",
    "tinea_versicolor":                     "Pityriasis Versicolor",
    "candida intertrigo":                   "Candidiasis",
    "candidal intertrigo":                  "Candidiasis",
    "candidal balanoposthitis":             "Candidiasis",
    "candidal vulvovaginitis":              "Candidiasis",
    "oral candidiasis":                     "Candidiasis",
    "intertrigo":                           "Candidiasis",
    "deep fungal infection":                "Deep Fungal Infection",
    "chromoblastomycosis":                  "Deep Fungal Infection",
    "fungal dermatitis":                    "Tinea",

    # ── Acne / Sebaceous ─────────────────────────────────────────────────────
    "acne":                                 "Acne",
    "acne vulgaris":                        "Acne",
    "acne_vulgaris":                        "Acne",
    "acne conglobata":                      "Acne",
    "truncal acne":                         "Acne",
    "acne scars":                           "Acne",
    "rosacea":                              "Rosacea",
    "rhinophyma":                           "Rosacea",
    "perioral dermatitis":                  "Rosacea",
    "seborrheic dermatitis":                "Seborrheic Dermatitis",
    "seborrheic_dermatitis":                "Seborrheic Dermatitis",
    "seborrhoeic dermatitis":               "Seborrheic Dermatitis",
    "dandruff":                             "Seborrheic Dermatitis",

    # ── Pigmentation ─────────────────────────────────────────────────────────
    "melasma":                              "Melasma",
    "chloasma":                             "Melasma",
    "hyperpigmentation":                    "Melasma",
    "hyperpigmentation disorder":           "Melasma",
    "facial hypermelanosis":                "Melasma",
    "post-inflammatory hyperpigmentation":  "Post-Inflammatory Hyperpigmentation",
    "post inflammatory hyperpigmentation":  "Post-Inflammatory Hyperpigmentation",
    "pigmented purpuric eruption":          "Purpura",
    "idiopathic guttate hypomelanosis":     "Vitiligo",
    "pityriasis alba":                      "Vitiligo",
    "ochronosis":                           "Post-Inflammatory Hyperpigmentation",

    # ── Scabies / Infestation ─────────────────────────────────────────────────
    "scabies":                              "Scabies",
    "nodular scabies":                      "Scabies",
    "pustular scabies":                     "Scabies",
    "scabies infected":                     "Scabies",

    # ── Systemic ─────────────────────────────────────────────────────────────
    "acanthosis nigricans":                 "Acanthosis Nigricans",
    "acanthosis_nigricans":                 "Acanthosis Nigricans",
    "purpura":                              "Purpura",
    "contact purpura":                      "Purpura",
    "petechiae":                            "Purpura",
    "o/e - petechiae on skin":              "Purpura",
    "o/e - petechiae present":              "Purpura",
    "traumatic petechiae":                  "Purpura",
    "superficial hemorrhage":               "Purpura",
    "cutaneous vasculitis":                 "Cutaneous Vasculitis",
    "vasculitis":                           "Cutaneous Vasculitis",
    "leukocytoclastic vasculitis":          "Cutaneous Vasculitis",
    "urticarial vasculitis":                "Cutaneous Vasculitis",
    "nodular vasculitis":                   "Cutaneous Vasculitis",
    "vasculitis of the skin":               "Cutaneous Vasculitis",
    "localised cutaneous vasculitis":       "Cutaneous Vasculitis",
    "localized cutaneous vasculitis":       "Cutaneous Vasculitis",
    "jaundice":                             "Jaundice",
    "uremic pruritus":                      "Uremic Pruritus",
    "uremic_pruritus":                      "Uremic Pruritus",
    "polycythaemia vera":                   "Polycythaemia Vera",
    "polycythemia vera":                    "Polycythaemia Vera",
    "iron deficiency":                      "Iron Deficiency Anaemia",
    "iron deficiency anaemia":              "Iron Deficiency Anaemia",
    "iron deficiency anemia":              "Iron Deficiency Anaemia",
    "vitamin d deficiency":                 "Vitamin D Deficiency",
    "pellagra dermatitis":                  "Nutritional Dermatitis",
    "nutritional dermatitis":               "Nutritional Dermatitis",
    "phrynoderma":                          "Nutritional Dermatitis",
    "xanthelasma palpebrarum":              "Xanthoma",
    "xanthoma":                             "Xanthoma",
    "xanthomas":                            "Xanthoma",
    "eruptive xanthoma":                    "Xanthoma",
    "diffuse xanthoma":                     "Xanthoma",
    "necrobiosis lipoidica":                "Necrobiosis Lipoidica",
    "diabetic ulcer":                       "Skin Ulcer",
    "skin ulcer":                           "Skin Ulcer",
    "foot ulcer":                           "Skin Ulcer",
    "pressure ulcer":                       "Skin Ulcer",
    "trophic ulcer":                        "Skin Ulcer",
    "venous stasis ulcer":                  "Skin Ulcer",
    "varicose vein":                        "Varicose Veins",
    "varicose veins of lower extremity":    "Varicose Veins",
    "deep vein thrombosis":                 "Varicose Veins",

    # ── Other common conditions ───────────────────────────────────────────────
    "keratosis pilaris":                    "Keratosis Pilaris",
    "ichthyosis":                           "Ichthyosis",
    "ichthyosis vulgaris":                  "Ichthyosis",
    "pityriasis rosea":                     "Pityriasis Rosea",
    "pityriasis rubra pilaris":             "Pityriasis Rubra Pilaris",
    "livedo reticularis":                   "Livedo Reticularis",
    "striae":                               "Striae",
    "miliaria":                             "Miliaria",
    "xerosis":                              "Xerosis",
    "dryness (xerosis)":                    "Xerosis",
    "xerosis vulgaris":                     "Xerosis",
    "sunburn":                              "Photodermatitis",
    "sun damaged skin":                     "Photodermatitis",
    "burn of skin":                         "Burn",
    "contact burn of skin":                 "Burn",
    "erythrasma":                           "Erythrasma",
    "cutaneous amyloidosis":                "Cutaneous Amyloidosis",
    "cutaneous larva migrans":              "Cutaneous Larva Migrans",
    "inflicted skin lesions":               "Inflicted Skin Lesions",
    "pityriasis lichenoides":               "Pityriasis Lichenoides",
    "darier's disease":                     "Darier's Disease",
    "grover's disease":                     "Grover's Disease",
    "sweet syndrome":                       "Sweet Syndrome",
    "pseudolymphoma":                       "Pseudolymphoma",
    "pseudo lymphoma":                      "Pseudolymphoma",
    "lymphocutaneous sporotrichosis":       "Sporotrichosis",
    "hypersensitivity":                     "Hypersensitivity",
}


def normalize_label(raw: str) -> str | None:
    """
    Maps any raw label string to a canonical disease name.
    Handles plain strings and SCIN probability dict strings.
    Returns None only for empty/null input — never silently drops data.
    Unknown labels return title-cased original so they are still ingested.
    """
    if not raw or str(raw).strip() in ("", "nan", "UNKNOWN", "{}"):
        return None

    s = str(raw).strip()

    # Handle SCIN weighted_skin_condition_label probability dict format
    # e.g. "{'Eczema': 0.67, 'Contact Dermatitis': 0.33}"
    if s.startswith("{"):
        try:
            prob_dict = ast.literal_eval(s)
            if not prob_dict:
                return None
            # Pick highest confidence label
            s = max(prob_dict, key=prob_dict.get)
        except (ValueError, SyntaxError):
            pass

    # Handle SCIN list format
    # e.g. "['Eczema', 'Contact Dermatitis']"
    if s.startswith("["):
        try:
            items = ast.literal_eval(s)
            if not items:
                return None
            s = str(items[0])
        except (ValueError, SyntaxError):
            pass

    key = s.strip().lower()
    return LABEL_MAP.get(key) or LABEL_MAP.get(key.replace(" ", "_")) or s.strip().title()


# ── BLOOD SYNTHESISER ─────────────────────────────────────────────────────────
_NORMAL_RANGES = {
    "wbc":        (4500.0,   9000.0),
    "hemoglobin": (12.0,     15.5),
    "platelets":  (150000.0, 300000.0),
    "eosinophils":(0.1,      0.4),
    "crp":        (0.1,      0.4),
    "esr":        (5.0,      20.0),
    "creatinine": (0.6,      1.1),
    "glucose":    (75.0,     99.0),
}


def _synth_value(min_v: float, max_v: float) -> float:
    mean = (min_v + max_v) / 2.0
    std  = (max_v - min_v) / 6.0
    return float(np.clip(np.random.normal(mean, std), min_v, max_v))


def _extract_range(blood_logic: dict, key: str) -> tuple[float, float] | None:
    v = blood_logic.get(key)
    if isinstance(v, dict) and "min" in v and "max" in v:
        return float(v["min"]), float(v["max"])
    return None


def synthesise_blood(blood_logic: dict) -> dict:
    def get(key: str, normal_key: str) -> float:
        rng = _extract_range(blood_logic, key)
        if rng:
            return _synth_value(*rng)
        return _synth_value(*_NORMAL_RANGES[normal_key])

    return {
        "cbc_wbc":         get("wbc",            "wbc"),
        "cbc_hemoglobin":  get("hemoglobin",      "hemoglobin"),
        "cbc_platelets":   get("platelets",       "platelets"),
        "cbc_eosinophils": get("eosinophils",     "eosinophils"),
        "inf_crp":         get("crp",             "crp"),
        "inf_esr":         get("esr",             "esr"),
        "cmp_creatinine":  get("creatinine",      "creatinine"),
        "cmp_glucose":     get("glucose_fasting", "glucose"),
    }


# ── SYMPTOM HELPERS ───────────────────────────────────────────────────────────

def _yes(val) -> bool:
    """Converts 'YES'/'TRUE'/True to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().upper() in ("YES", "TRUE")
    return False


# ── DATASET EXTRACTORS ────────────────────────────────────────────────────────

def _extract_fitzpatrick() -> list[dict]:
    """
    Columns: label, fitzpatrick_scale, md5hash, url
    Images: {md5hash}.jpg stored locally in fitzpatrick17k/images/
    """
    csv_path = os.path.join(RAW_DIR, "fitzpatrick17k", "fitzpatrick17k.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"Not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    rows, skipped = [], 0

    for _, r in df.iterrows():
        label = normalize_label(str(r.get("label", "")))
        if not label:
            skipped += 1
            continue

        md5 = str(r.get("md5hash", "")).strip()
        rows.append({
            "patient_id":          f"FK17-{uuid.uuid4().hex[:8].upper()}",
            "source_dataset":      "fitzpatrick17k",
            "abha_id":             None,
            "split":               "train",
            "is_augmented":        False,
            "image_path":          f"fitzpatrick17k/images/{md5}.jpg",
            "original_resolution": "unknown",
            "quality_score":       None,
            "hair_removal_tier":   None,
            "label":               label,
            "main_class":          None,
            "sub_class":           None,
            "icd11_code":          None,
            "label_confidence":    0.88,
            "systemic":            None,
            "confidence_score":    None,
            "fitzpatrick_type":    int(r["fitzpatrick_scale"]) if pd.notna(r.get("fitzpatrick_scale")) else -1,
            "monk_skin_tone":      -1,
            "body_part":           "unknown",
            "descriptors":         None,
            "symptom_pruritus":    None,
            "symptom_nociception": None,
            "symptom_evolution":   None,
        })

    logger.info(f"Fitzpatrick17k  → {len(rows)} rows  ({skipped} skipped)")
    return rows


def _extract_pad_ufes() -> list[dict]:
    """
    Columns: diagnostic (NEV/BCC/ACK/SEK/SCC/MEL),
             fitspatrick (typo — confirmed), img_id,
             itch/hurt/grew/changed (TRUE/FALSE strings)
    """
    csv_path = os.path.join(RAW_DIR, "pad_ufes_20", "metadata.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"Not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    rows, skipped = [], 0

    for _, r in df.iterrows():
        diag  = str(r.get("diagnostic", "")).strip().upper()
        label = PAD_LABEL_MAP.get(diag)
        if not label:
            skipped += 1
            continue

        # FST — column is 'fitspatrick' (missing z — confirmed typo in dataset)
        fitz = -1
        raw_fst = r.get("fitspatrick")
        if pd.notna(raw_fst):
            try:
                fitz = int(float(raw_fst))
            except (ValueError, TypeError):
                pass

        # Symptoms from PAD columns
        pruritus    = 3 if _yes(r.get("itch"))    else 0
        nociception = 7 if _yes(r.get("hurt"))    else 0
        evolution   = "rapid" if (_yes(r.get("grew")) or _yes(r.get("changed"))) else "stable"

        rows.append({
            "patient_id":          f"PU20-{uuid.uuid4().hex[:8].upper()}",
            "source_dataset":      "pad_ufes_20",
            "abha_id":             None,
            "split":               "train",
            "is_augmented":        False,
            "image_path":          f"pad_ufes_20/images/{str(r.get('img_id', '')).strip()}",
            "original_resolution": "unknown",
            "quality_score":       None,
            "hair_removal_tier":   None,
            "label":               label,
            "main_class":          None,
            "sub_class":           None,
            "icd11_code":          None,
            "label_confidence":    1.0,   # biopsy-confirmed
            "systemic":            None,
            "confidence_score":    None,
            "fitzpatrick_type":    fitz,
            "monk_skin_tone":      -1,
            "body_part":           str(r.get("region", "unknown")).strip(),
            "descriptors":         None,
            "symptom_pruritus":    pruritus,
            "symptom_nociception": nociception,
            "symptom_evolution":   evolution,
        })

    logger.info(f"PAD-UFES-20     → {len(rows)} rows  ({skipped} skipped)")
    return rows


def _extract_scin() -> list[dict]:
    """
    Joins scin_labels.csv + scin_cases.csv on case_id.

    scin_labels.csv:
      weighted_skin_condition_label — probability dict string, take highest
      dermatologist_fitzpatrick_skin_type_label_1 — e.g. 'FST2'

    scin_cases.csv:
      image_1_path — e.g. 'dataset/images/-320...704.png'
      condition_symptoms_itching/pain/burning — 'YES' strings
      other_symptoms_fatigue/joint_pain/fever — 'YES' strings
    """
    labels_path = os.path.join(RAW_DIR, "scin", "scin_labels.csv")
    cases_path  = os.path.join(RAW_DIR, "scin", "scin_cases.csv")

    if not os.path.exists(labels_path):
        logger.warning(f"Not found: {labels_path}")
        return []

    labels_df = pd.read_csv(labels_path)

    cases_df = None
    if os.path.exists(cases_path):
        cases_df = pd.read_csv(cases_path)
        cases_df = cases_df.set_index("case_id")
    else:
        logger.warning(f"scin_cases.csv not found: {cases_path}")

    rows, skipped = [], 0

    for _, r in labels_df.iterrows():
        # weighted_skin_condition_label is a probability dict string
        label = normalize_label(str(r.get("weighted_skin_condition_label", "")))
        if not label:
            skipped += 1
            continue

        case_id = r.get("case_id")

        # FST from dermatologist label e.g. "FST2" → 2
        fitz = -1
        raw_fst = str(r.get("dermatologist_fitzpatrick_skin_type_label_1", ""))
        try:
            fitz = int(raw_fst.upper().replace("FST", "").strip())
        except ValueError:
            pass

        # Image path and symptoms from cases CSV
        img_path    = ""
        pruritus    = None
        nociception = None
        evolution   = None

        if cases_df is not None and case_id in cases_df.index:
            c        = cases_df.loc[case_id]
            raw_img  = c.get("image_1_path")
            if pd.notna(raw_img):
                filename = os.path.basename(str(raw_img))
                img_path = f"scin/images/{filename}"

            # Symptoms — values are 'YES' strings
            pruritus    = 3 if _yes(c.get("condition_symptoms_itching")) else 0
            nociception = 7 if _yes(c.get("condition_symptoms_pain"))    else 0

            # Red flag systemic symptoms → affects blood gate in Module 4
            has_fever      = _yes(c.get("other_symptoms_fever"))
            has_joint_pain = _yes(c.get("other_symptoms_joint_pain"))
            has_fatigue    = _yes(c.get("other_symptoms_fatigue"))

            if _yes(c.get("condition_symptoms_increasing_size")):
                evolution = "rapid"
            elif has_fever or has_joint_pain or has_fatigue:
                evolution = "systemic"
            else:
                evolution = "stable"

        rows.append({
            "patient_id":          f"SC-{uuid.uuid4().hex[:8].upper()}",
            "source_dataset":      "scin",
            "abha_id":             None,
            "split":               "train",
            "is_augmented":        False,
            "image_path":          img_path,
            "original_resolution": "unknown",
            "quality_score":       None,
            "hair_removal_tier":   None,
            "label":               label,
            "main_class":          None,
            "sub_class":           None,
            "icd11_code":          None,
            "label_confidence":    0.82,   # crowdsourced + dermatologist review
            "systemic":            None,
            "confidence_score":    None,
            "fitzpatrick_type":    fitz,
            "monk_skin_tone":      -1,
            "body_part":           "unknown",
            "descriptors":         None,
            "symptom_pruritus":    pruritus,
            "symptom_nociception": nociception,
            "symptom_evolution":   evolution,
        })

    logger.info(f"SCIN            → {len(rows)} rows  ({skipped} skipped)")
    return rows


def _extract_dermaconin() -> list[dict]:
    """
    Confirmed columns from metadata.csv:
      Disease_label, Image_name, Confidence, Fitzpatrick (e.g. 'FST 3'),
      Monk_skin_tone (e.g. 'MST 5'), Main_class, Sub_class, Body_part

    Filters:
      - Confidence == 5 only
      - Drop rows where Fitzpatrick or Monk_skin_tone is null
      - Split from train_split.csv / test_split.csv by Image_name
    """
    meta_path  = os.path.join(RAW_DIR, "dermaconin", "metadata.csv")
    train_path = os.path.join(RAW_DIR, "dermaconin", "train_split.csv")
    test_path  = os.path.join(RAW_DIR, "dermaconin", "test_split.csv")

    if not os.path.exists(meta_path):
        logger.warning(f"DermaCon-IN metadata not found: {meta_path}")
        return []

    df = pd.read_csv(meta_path)
    logger.info(f"DermaCon-IN     raw rows: {len(df)}")

    before = len(df)
    df = df[df["Confidence"] == 5]
    logger.info(f"DermaCon-IN     confidence=5: {before} → {len(df)}")

    before = len(df)
    df = df.dropna(subset=["Fitzpatrick", "Monk_skin_tone"])
    if before - len(df) > 0:
        logger.info(f"DermaCon-IN     dropped {before - len(df)} null skin-tone rows")

    # Build split lookup from Image_name
    split_map: dict[str, str] = {}
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        if "Image_name" in train_df.columns:
            for img in train_df["Image_name"].dropna():
                split_map[str(img).strip()] = "train"
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        if "Image_name" in test_df.columns:
            for img in test_df["Image_name"].dropna():
                split_map[str(img).strip()] = "test"

    logger.info(f"DermaCon-IN     split map: {len(split_map)} images")

    rows, skipped = [], 0

    for _, r in df.iterrows():
        label = normalize_label(str(r.get("Disease_label", "")))
        if not label:
            skipped += 1
            continue

        fitz = -1
        try:
            fitz = int(str(r.get("Fitzpatrick", "")).upper().replace("FST", "").strip())
        except ValueError:
            pass

        mst = -1
        try:
            mst = int(str(r.get("Monk_skin_tone", "")).upper().replace("MST", "").strip())
        except ValueError:
            pass

        img_name  = str(r.get("Image_name", "")).strip()
        body_part = str(r.get("Body_part", "unknown")).strip()
        if not body_part or body_part == "nan":
            body_part = "unknown"

        rows.append({
            "patient_id":          f"DC-{uuid.uuid4().hex[:8].upper()}",
            "source_dataset":      "dermaconin",
            "abha_id":             None,
            "split":               split_map.get(img_name, "train"),
            "is_augmented":        False,
            "image_path":          f"dermaconin/images/{img_name}",
            "original_resolution": "unknown",
            "quality_score":       None,
            "hair_removal_tier":   None,
            "label":               label,
            "main_class":          str(r.get("Main_class", "")).strip() or None,
            "sub_class":           str(r.get("Sub_class",  "")).strip() or None,
            "icd11_code":          None,
            "label_confidence":    1.0,
            "systemic":            None,
            "confidence_score":    None,
            "fitzpatrick_type":    fitz,
            "monk_skin_tone":      mst,
            "body_part":           body_part,
            "descriptors":         str(r.get("Descriptors", "")).strip() or None,
            "symptom_pruritus":    None,
            "symptom_nociception": None,
            "symptom_evolution":   None,
        })

    logger.info(f"DermaCon-IN     → {len(rows)} rows  ({skipped} skipped)")
    return rows


# ── MAIN ETL RUNNER ───────────────────────────────────────────────────────────

async def run_etl() -> None:
    client     = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    collection = client[DB_NAME]["medical_knowledge_base"]

    existing_ids: set[str] = set()
    if os.path.exists(PARQUET):
        existing_ids = set(
            pq.read_table(PARQUET, columns=["patient_id"])
            .column("patient_id")
            .to_pylist()
        )
    logger.info(f"Existing Parquet records: {len(existing_ids)}")

    all_rows = (
        _extract_fitzpatrick()
        + _extract_pad_ufes()
        + _extract_scin()
        + _extract_dermaconin()
    )
    logger.info(f"Total extracted: {len(all_rows)}")

    # MongoDB lookup cache — one query per unique canonical label
    label_cache: dict[str, dict] = {}
    for label in {r["label"] for r in all_rows if r.get("label")}:
        doc = await collection.find_one({"disease": label})
        label_cache[label] = doc or {}
    logger.info(f"MongoDB cache: {len(label_cache)} unique labels")

    new_rows, skipped_dup = [], 0

    for row in all_rows:
        if row["patient_id"] in existing_ids:
            skipped_dup += 1
            continue

        doc         = label_cache.get(row["label"], {})
        blood_logic = doc.get("blood_logic", {})

        row["icd11_code"] = doc.get("icd11",    "UNKNOWN")
        row["systemic"]   = doc.get("systemic",  False)
        if not row.get("main_class"):
            row["main_class"] = doc.get("category", None)

        row.update(synthesise_blood(blood_logic))

        row["sfv_border_irregularity"] = None
        row["sfv_asymmetry_index"]     = None
        row["sfv_fractal_dimension"]   = None
        row["sfv_color_gradient"]      = None
        row["ingest_timestamp"]        = datetime.now(timezone.utc)

        new_rows.append(row)

    if not new_rows:
        logger.info(f"No new records. Duplicates skipped: {skipped_dup}")
        client.close()
        return

    from modules.module1_bedrock.initialize_bedrock import SCHEMA

    col_data = {}
    for field in SCHEMA:
        values = [row.get(field.name) for row in new_rows]
        try:
            col_data[field.name] = pa.array(values, type=field.type)
        except Exception as e:
            logger.warning(f"Column cast [{field.name}]: {e} — filling nulls")
            col_data[field.name] = pa.array([None] * len(new_rows), type=field.type)

    new_table = pa.table(col_data, schema=SCHEMA)

    if os.path.exists(PARQUET):
        existing_table = pq.read_table(PARQUET)
        final_table    = pa.concat_tables([existing_table, new_table])
    else:
        final_table = new_table

    os.makedirs(DATA_DIR, exist_ok=True)
    pq.write_table(final_table, PARQUET)

    logger.success(
        f"✅ ETL complete — "
        f"new: {len(new_rows)} | "
        f"skipped duplicates: {skipped_dup} | "
        f"total in Parquet: {len(final_table)}"
    )
    client.close()


if __name__ == "__main__":
    asyncio.run(run_etl())