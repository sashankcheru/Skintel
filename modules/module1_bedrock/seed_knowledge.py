import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger
from dotenv import load_dotenv
from pymongo import UpdateOne

load_dotenv()

SKINTEL_RULEBOOK = [
    # ── BACTERIAL ──────────────────────────────────────────────────────────────
    {
        "category": "Bacterial",
        "icd11": "L03",
        "disease": "Cellulitis",
        "systemic": True,
        "blood_logic": {
            "wbc":         {"min": 11000, "max": 16000, "unit": "cells/uL"},
            "crp":         {"min": 2.0,   "max": 15.0,  "unit": "mg/dL"},
            "neutrophils": "elevated"
        },
        "symptom_logic": {"pain": 8, "warmth": True, "itch": 0, "evolution": "rapid"}
    },
    {
        "category": "Bacterial",
        "icd11": "L01",
        "disease": "Impetigo",
        "systemic": True,
        "blood_logic": {
            "wbc": {"min": 10000, "max": 12000, "unit": "cells/uL"},
            "crp": {"min": 0.5,   "max": 2.0,   "unit": "mg/dL"}
        },
        "symptom_logic": {"pain": 2, "itch": 2, "crust": "honey-colored", "evolution": "rapid"}
    },

    # ── VIRAL ──────────────────────────────────────────────────────────────────
    {
        "category": "Viral",
        "icd11": "1D61",
        "disease": "Shingles",
        "systemic": True,
        "blood_logic": {
            "wbc":         "normal",
            "lymphocytes": "elevated"
        },
        "symptom_logic": {"pain": 10, "itch": 0, "evolution": "clusters", "sensation": "burning"}
    },

    # ── ALLERGIC ───────────────────────────────────────────────────────────────
    {
        "category": "Allergic",
        "icd11": "EA80",
        "disease": "Eczema",
        "systemic": True,
        "blood_logic": {
            "ige":         {"min": 200,  "max": 1000, "unit": "kU/L"},
            "eosinophils": {"min": 0.5,  "max": 1.5,  "unit": "10^9/L"}
        },
        "symptom_logic": {"pain": 1, "itch": 10, "evolution": "chronic", "dryness": True}
    },
    {
        "category": "Allergic",
        "icd11": "EA81",
        "disease": "Urticaria",
        "systemic": True,
        "blood_logic": {
            "ige":         {"min": 150,  "max": 800,  "unit": "kU/L"},
            "eosinophils": {"min": 0.4,  "max": 1.2,  "unit": "10^9/L"},
            "crp":         "mildly_elevated"
        },
        "symptom_logic": {"pain": 1, "itch": 9, "evolution": "rapid", "wheals": True}
    },
    {
        "category": "Allergic",
        "icd11": "EK00",
        "disease": "Contact Dermatitis",
        "systemic": False,
        "blood_logic": {
            "status": "usually_normal",
            "note":   "Blood normal — purely local inflammatory response"
        },
        "symptom_logic": {"pain": 3, "itch": 7, "evolution": "rapid", "trigger": "contact"}
    },

    # ── AUTOIMMUNE ─────────────────────────────────────────────────────────────
    {
        "category": "Autoimmune",
        "icd11": "EA90",
        "disease": "Psoriasis",
        "systemic": True,
        "blood_logic": {
            "crp": {"min": 1.0, "max": 5.0,  "unit": "mg/dL"},
            "esr": {"min": 15,  "max": 40,   "unit": "mm/hr"},
            "nlr": ">2.0"
        },
        "symptom_logic": {"pain": 3, "itch": 5, "scaling": "silvery", "evolution": "persistent"}
    },
    {
        "category": "Autoimmune",
        "icd11": "6D91",
        "disease": "Lupus",
        "systemic": True,
        "blood_logic": {
            "ana":       "positive",
            "esr":       "very_high",
            "platelets": "low"
        },
        "symptom_logic": {"pain": 4, "itch": 0, "evolution": "sun-sensitive"}
    },
    {
        "category": "Autoimmune",
        "icd11": "LD27",
        "disease": "Scleroderma",
        "systemic": True,
        "blood_logic": {
            "ana":         "positive",
            "esr":         {"min": 20,  "max": 60,   "unit": "mm/hr"},
            "crp":         {"min": 1.0, "max": 4.0,  "unit": "mg/dL"},
            "eosinophils": "elevated"
        },
        "symptom_logic": {"pain": 5, "itch": 2, "evolution": "slow", "texture": "hardening"}
    },

    # ── HORMONAL ───────────────────────────────────────────────────────────────
    {
        "category": "Hormonal",
        "icd11": "ED56",
        "disease": "Acne",
        "systemic": True,
        "blood_logic": {
            "testosterone": "high",
            "crp":          "mildly_elevated"
        },
        "symptom_logic": {"pain": 4, "itch": 0, "evolution": "periodic"}
    },

    # ── MALIGNANT ──────────────────────────────────────────────────────────────
    {
        "category": "Malignant",
        "icd11": "2C30",
        "disease": "Melanoma",
        "systemic": True,
        "blood_logic": {
            "ldh":    "high_if_late_stage",
            "marker": "S100"
        },
        "symptom_logic": {"pain": 0, "itch": 2, "bleeding": True, "evolution": "changing"}
    },
    {
        "category": "Malignant",
        "icd11": "2C31",
        "disease": "BCC",
        "systemic": False,
        "blood_logic": {
            "status": "usually_normal",
            "note":   "Blood normal in early BCC — systemic only in rare metastatic cases"
        },
        "symptom_logic": {"pain": 1, "itch": 0, "evolution": "very_slow", "texture": "waxy"}
    },

    # ── NUTRITIONAL ────────────────────────────────────────────────────────────
    {
        "category": "Nutritional",
        "icd11": "EB90",
        "disease": "Vit D Deficiency",
        "systemic": True,
        "blood_logic": {
            "serum_25_oh_d": "<20 ng/mL"
        },
        "symptom_logic": {"pain": 0, "itch": 3, "scaling": "diffuse", "evolution": "seasonal"}
    },
    {
        "category": "Nutritional",
        "icd11": "5B55",
        "disease": "Iron Deficiency",
        "systemic": True,
        "blood_logic": {
            "hemoglobin": {"max": 12.0,  "unit": "g/dL"},
            "ferritin":   {"max": 12.0,  "unit": "ng/mL"},
            "wbc":        "normal"
        },
        "symptom_logic": {"pain": 0, "itch": 4, "evolution": "chronic",
                          "signs": "pallor, brittle nails, hair loss"}
    },

    # ── BLOOD → SKIN (Systemic origin) ─────────────────────────────────────────
    {
        "category": "Systemic",
        "icd11": "FA24",
        "disease": "Uremic Pruritus",
        "systemic": True,
        "blood_logic": {
            "creatinine": {"min": 3.0,  "max": 10.0, "unit": "mg/dL"},
            "bun":        {"min": 40,   "max": 100,  "unit": "mg/dL"},
            "note":       "Kidney failure → extreme skin itch"
        },
        "symptom_logic": {"pain": 0, "itch": 10, "evolution": "chronic", "trigger": "renal"}
    },
    {
        "category": "Systemic",
        "icd11": "5C56",
        "disease": "Jaundice (Skin)",
        "systemic": True,
        "blood_logic": {
            "bilirubin": {"min": 2.5,  "unit": "mg/dL"},
            "alt":       "elevated",
            "ast":       "elevated",
            "alp":       "elevated",
            "note":      "Liver failure → yellowing of skin"
        },
        "symptom_logic": {"pain": 2, "itch": 6, "evolution": "rapid", "color": "yellow"}
    },
    {
        "category": "Systemic",
        "icd11": "3B64",
        "disease": "Purpura",
        "systemic": True,
        "blood_logic": {
            "platelets": {"max": 50000, "unit": "cells/uL"},
            "note":      "Low platelets → bleeding under skin"
        },
        "symptom_logic": {"pain": 1, "itch": 0, "evolution": "rapid", "bleeding": True}
    },

    # ── FUNGAL (Blood-negative — teaches gate suppression) ─────────────────────
    {
        "category": "Fungal",
        "icd11": "EA60",
        "disease": "Tinea (Ringworm)",
        "systemic": False,
        "blood_logic": {
            "status": "normal",
            "note":   "Purely local — blood normal, gate should suppress clinical stream"
        },
        "symptom_logic": {"pain": 0, "itch": 6, "evolution": "slow", "pattern": "ring-shaped"}
    },
]


async def seed_matrix():
    """
    Idempotent upsert — safe to run multiple times.
    Keyed on icd11: updates existing, inserts new, never deletes.
    """
    mongo_url = os.getenv("MONGODB_URL")
    db_name   = os.getenv("MONGODB_DB_NAME")

    client = AsyncIOMotorClient(
        mongo_url,
        serverSelectionTimeoutMS=5000,
    )
    db         = client[db_name]
    collection = db["medical_knowledge_base"]

    try:
        operations = [
            UpdateOne(
                {"icd11": rule["icd11"]},
                {"$set": rule},
                upsert=True
            )
            for rule in SKINTEL_RULEBOOK
        ]

        result = await collection.bulk_write(operations, ordered=False)

        logger.success(
            f"✅ Knowledge base synced — "
            f"inserted: {result.upserted_count}, "
            f"updated: {result.modified_count}, "
            f"matched: {result.matched_count} | "
            f"Total diseases: {len(SKINTEL_RULEBOOK)}"
        )

        # Summary by category
        pipeline = [{"$group": {"_id": "$category", "count": {"$sum": 1}}}]
        cursor   = collection.aggregate(pipeline)
        cats     = await cursor.to_list(length=100)
        logger.info("Category breakdown:")
        for cat in sorted(cats, key=lambda x: x['_id']):
            logger.info(f"  {cat['_id']:<15}: {cat['count']} disease(s)")

        # Systemic vs local summary
        systemic = sum(1 for r in SKINTEL_RULEBOOK if r.get('systemic'))
        local    = sum(1 for r in SKINTEL_RULEBOOK if not r.get('systemic'))
        logger.info(f"  Systemic (blood linked) : {systemic}")
        logger.info(f"  Local (blood normal)    : {local}")

    except Exception as e:
        logger.error(f"❌ Failed to seed Rulebook: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(seed_matrix())