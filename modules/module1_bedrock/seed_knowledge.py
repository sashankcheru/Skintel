import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger
from dotenv import load_dotenv
from pymongo import UpdateOne

load_dotenv()

SKINTEL_RULEBOOK = [
    {
        "category": "Bacterial",
        "icd11": "L03",
        "disease": "Cellulitis",
        "blood_logic": {
            "wbc": {"min": 11000, "max": 16000, "unit": "cells/uL"},
            "crp": {"min": 2.0, "max": 15.0, "unit": "mg/dL"},
            "neutrophils": "elevated"
        },
        "symptom_logic": {"pain": 8, "warmth": True, "itch": 0, "evolution": "rapid"}
    },
    {
        "category": "Bacterial",
        "icd11": "L01",
        "disease": "Impetigo",
        "blood_logic": {
            "wbc": {"min": 10000, "max": 12000, "unit": "cells/uL"},
            "crp": {"min": 0.5, "max": 2.0, "unit": "mg/dL"}
        },
        "symptom_logic": {"pain": 2, "itch": 2, "crust": "honey-colored", "evolution": "rapid"}
    },
    {
        "category": "Viral",
        "icd11": "1D61",
        "disease": "Shingles",
        "blood_logic": {"wbc": "normal", "lymphocytes": "elevated"},
        "symptom_logic": {"pain": 10, "itch": 0, "evolution": "clusters", "sensation": "burning"}
    },
    {
        "category": "Allergic",
        "icd11": "EA80",
        "disease": "Eczema",
        "blood_logic": {
            "ige": {"min": 200, "max": 1000, "unit": "kU/L"},
            "eosinophils": {"min": 0.5, "max": 1.5, "unit": "10^9/L"}
        },
        "symptom_logic": {"pain": 1, "itch": 10, "evolution": "chronic", "dryness": True}
    },
    {
        "category": "Autoimmune",
        "icd11": "EA90",
        "disease": "Psoriasis",
        "blood_logic": {
            "crp": {"min": 1.0, "max": 5.0, "unit": "mg/dL"},
            "esr": {"min": 15, "max": 40, "unit": "mm/hr"},
            "nlr": ">2.0"
        },
        "symptom_logic": {"pain": 3, "itch": 5, "scaling": "silvery", "evolution": "persistent"}
    },
    {
        "category": "Autoimmune",
        "icd11": "6D91",
        "disease": "Lupus",
        "blood_logic": {"ana": "positive", "esr": "very_high", "platelets": "low"},
        "symptom_logic": {"pain": 4, "itch": 0, "evolution": "sun-sensitive"}
    },
    {
        "category": "Hormonal",
        "icd11": "ED56",
        "disease": "Acne",
        "blood_logic": {"testosterone": "high", "crp": "mildly_elevated"},
        "symptom_logic": {"pain": 4, "itch": 0, "evolution": "periodic"}
    },
    {
        "category": "Malignant",
        "icd11": "2C30",
        "disease": "Melanoma",
        "blood_logic": {"ldh": "high_if_late_stage", "marker": "S100"},
        "symptom_logic": {"pain": 0, "itch": 2, "bleeding": True, "evolution": "changing"}
    },
    {
        "category": "Malignant",
        "icd11": "2C31",
        "disease": "BCC",
        "blood_logic": {"status": "usually_normal"},
        "symptom_logic": {"pain": 1, "itch": 0, "evolution": "very_slow", "texture": "waxy"}
    },
    {
        "category": "Nutritional",
        "icd11": "EB90",
        "disease": "Vit D Deficiency",
        "blood_logic": {"serum_25_oh_d": "<20 ng/mL"},
        "symptom_logic": {"pain": 0, "itch": 3, "scaling": "diffuse", "evolution": "seasonal"}
    }
]

async def seed_matrix():
    """
    Idempotent upsert — safe to run multiple times.
    Keyed on icd11: updates existing records, inserts new ones, never deletes.
    """
    mongo_url = os.getenv("MONGODB_URL")
    db_name = os.getenv("MONGODB_DB_NAME")

    client = AsyncIOMotorClient(
        mongo_url,
        serverSelectionTimeoutMS=5000,
    )
    db = client[db_name]
    collection = db["medical_knowledge_base"]

    try:
        # Build upsert operations — one per disease, keyed on icd11
        operations = [
            UpdateOne(
                {"icd11": rule["icd11"]},   # match filter
                {"$set": rule},              # update payload
                upsert=True                  # insert if not found
            )
            for rule in SKINTEL_RULEBOOK
        ]

        result = await collection.bulk_write(operations, ordered=False)

        logger.success(
            f"✅ Knowledge base synced — "
            f"inserted: {result.upserted_count}, "
            f"updated: {result.modified_count}, "
            f"matched: {result.matched_count}"
        )
    except Exception as e:
        logger.error(f"❌ Failed to seed Rulebook: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(seed_matrix())