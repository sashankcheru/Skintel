"""
Module 4 — clinical_engine.py
Construction phase: core rule-based logic only. No OCR, no real patient
questionnaire yet (Module 6) — that's furniture, added later.

Two functions:
- confirm_symptoms(): checks top-3 predictions against the rulebook's
  symptom_logic. Honest about gaps — if a disease has no rulebook entry,
  returns "insufficient_rule_data", never silently treated as a match.
- check_systemic_gate(): the 3-trigger blood-request decision.

DESIGN DECISION (flagging explicitly, not deciding silently):
If a predicted disease has no rulebook entry, its "systemic" status is
unknown. I default to TRIGGERING the gate in that case — erring toward
asking for more info rather than silently assuming "not systemic."
Change SYSTEMIC_UNKNOWN_DEFAULT below if you want the opposite default.
"""

import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient

SYSTEMIC_UNKNOWN_DEFAULT = True  # trigger gate when systemic status is unknown
CONFIDENCE_THRESHOLD = 0.75
RED_FLAG_FIELDS = ["fatigue", "joint_pain", "fever", "weight_loss"]


async def fetch_rule(disease_name: str) -> dict | None:
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL"), serverSelectionTimeoutMS=5000)
    coll = client[os.getenv("MONGODB_DB_NAME")]["medical_knowledge_base"]
    rule = await coll.find_one({"disease": disease_name})
    client.close()
    return rule


def _compare_field(rule_value, patient_value) -> str:
    """Compares one symptom field. Returns 'support', 'contradict', or 'skip'."""
    if isinstance(rule_value, bool):
        if not isinstance(patient_value, bool):
            return "skip"
        return "support" if rule_value == patient_value else "contradict"

    if isinstance(rule_value, (int, float)):
        if not isinstance(patient_value, (int, float)):
            return "skip"
        # Within 3 points on a 0-10 scale counts as supporting — a coarse,
        # deliberately forgiving threshold since patient-reported severity
        # is inherently fuzzy. Tune this once real questionnaire data exists.
        return "support" if abs(rule_value - patient_value) <= 3 else "contradict"

    if isinstance(rule_value, str):
        if not isinstance(patient_value, str):
            return "skip"
        return "support" if rule_value.lower() == patient_value.lower() else "contradict"

    return "skip"


async def confirm_symptoms(top_predictions: list[tuple[str, float]], patient_symptoms: dict) -> list[dict]:
    """
    top_predictions: [(disease_name, confidence), ...] — top 3 from Module 3
    patient_symptoms: flat dict, e.g. {"itch": 9, "pain": 1, "fever": True, ...}
    Field names should align with whatever symptom_logic keys the rulebook
    uses for that disease — partial overlap is fine, fields just get skipped.
    """
    results = []
    for disease, confidence in top_predictions:
        rule = await fetch_rule(disease)

        if rule is None or "symptom_logic" not in rule:
            results.append({
                "disease": disease,
                "confidence": confidence,
                "signal": "insufficient_rule_data",
                "matched_fields": [],
                "contradicted_fields": [],
            })
            continue

        supports, contradicts = [], []
        for field, rule_value in rule["symptom_logic"].items():
            if field not in patient_symptoms:
                continue
            outcome = _compare_field(rule_value, patient_symptoms[field])
            if outcome == "support":
                supports.append(field)
            elif outcome == "contradict":
                contradicts.append(field)

        if not supports and not contradicts:
            signal = "neutral"  # no overlapping fields reported at all
        elif len(supports) > len(contradicts):
            signal = "supporting"
        elif len(contradicts) > len(supports):
            signal = "contradicting"
        else:
            signal = "neutral"

        results.append({
            "disease": disease,
            "confidence": confidence,
            "signal": signal,
            "support_strength": len(supports),       # NEW — lets Module 5 rank
            "contradict_strength": len(contradicts),  # candidates by HOW well-
            "matched_fields": supports,                # supported, not just
            "contradicted_fields": contradicts,        # whether they're positive
        })
    return results


async def check_systemic_gate(top1_disease: str, top1_confidence: float, patient_symptoms: dict) -> dict:
    reasons = []

    rule = await fetch_rule(top1_disease)
    if rule is None:
        is_systemic = SYSTEMIC_UNKNOWN_DEFAULT
        reasons.append(f"No rulebook entry for '{top1_disease}' — systemic status unknown, defaulting to {SYSTEMIC_UNKNOWN_DEFAULT}")
    else:
        is_systemic = rule.get("systemic", SYSTEMIC_UNKNOWN_DEFAULT)
        if is_systemic:
            reasons.append(f"'{top1_disease}' is classified systemic in the knowledge base")

    low_confidence = top1_confidence < CONFIDENCE_THRESHOLD
    if low_confidence:
        reasons.append(f"Confidence {top1_confidence:.1%} is below the {CONFIDENCE_THRESHOLD:.0%} threshold")

    red_flags_present = [f for f in RED_FLAG_FIELDS if patient_symptoms.get(f)]
    if red_flags_present:
        reasons.append(f"Red-flag symptom(s) reported: {', '.join(red_flags_present)}")

    triggered = is_systemic or low_confidence or bool(red_flags_present)

    return {
        "gate_triggered": triggered,
        "reasons": reasons,
    }