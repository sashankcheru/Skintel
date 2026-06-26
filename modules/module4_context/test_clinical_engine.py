import asyncio
from modules.module4_context.clinical_engine import confirm_symptoms, check_systemic_gate

async def main():
    # Simulated Module 3 output — a real disease with a rulebook entry,
    # one without, mimicking actual top-3 prediction shape
    top3 = [("Scabies", 0.46), ("Eczema", 0.30), ("Pityriasis Rubra Pilaris", 0.10)]

    patient_symptoms = {
        "itch": 9,
        "timing": "night_itch_severe",
        "fatigue": False,
        "joint_pain": False,
        "fever": False,
        "weight_loss": False,
    }

    symptom_results = await confirm_symptoms(top3, patient_symptoms)
    print("=== Symptom Confirmation ===")
    for r in symptom_results:
        print(r)

    print()
    gate = await check_systemic_gate(top3[0][0], top3[0][1], patient_symptoms)
    print("=== Systemic Gate ===")
    print(gate)

asyncio.run(main())