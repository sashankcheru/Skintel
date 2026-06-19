import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger
from dotenv import load_dotenv
from pymongo import UpdateOne

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# UNIT CONVENTIONS  (every value in this file uses these units consistently)
# ─────────────────────────────────────────────────────────────────────────────
#
#  WBC / Neutrophils / Lymphocytes   cells/uL   normal: 4500 – 11000
#  Eosinophils                       10^9/L     normal: 0.1  – 0.4
#  Hemoglobin                        g/dL       normal: M 13.5–17.5 | F 12.0–15.5
#  Platelets                         cells/uL   normal: 150,000 – 400,000
#  CRP (C-Reactive Protein)          mg/dL      normal: < 0.5
#  ESR                               mm/hr      normal: M < 20 | F < 30
#  IgE (total)                       kU/L       normal: < 100
#  Creatinine                        mg/dL      normal: 0.6 – 1.2
#  BUN (Blood Urea Nitrogen)         mg/dL      normal: 7  – 20
#  Bilirubin (total)                 mg/dL      normal: < 1.2
#  ALT / AST                         U/L        normal: ALT 7–56 | AST 10–40
#  CK  (Creatine Kinase)             IU/L       normal: 22 – 198
#  LDH (Lactate Dehydrogenase)       U/L        normal: 140 – 280
#  Ferritin                          ng/mL      normal: M 24–336 | F 11–307
#  Glucose (fasting)                 mg/dL      normal: 70 – 100
#  HbA1c                             %          normal: < 5.7
#  TSH                               mIU/L      normal: 0.4 – 4.0
#  25-OH Vitamin D                   ng/mL      normal: 30 – 100
#  NLR (Neutrophil-Lymphocyte Ratio) unitless   normal: < 2.0
#
#  ALL RANGES SOURCED FROM:
#  Harrison's Principles of Internal Medicine (21st ed.)
#  Fitzpatrick's Dermatology (9th ed.)
#  WHO Laboratory Reference Ranges 2024
#  Labcorp / Mayo Clinic reference intervals
#  ADA Standards of Care 2024
#  Endocrine Society Vitamin D Guidelines 2024
# ─────────────────────────────────────────────────────────────────────────────


SKINTEL_RULEBOOK = [

    # ══════════════════════════════════════════════════════════════════════════
    # BACTERIAL
    # Pattern: WBC elevated + Neutrophils elevated + CRP elevated
    # Deeper infection = higher all three values
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Bacterial",
        "disease":  "Cellulitis",
        "icd11":    "L03",
        "systemic": True,
        "blood_logic": {
            # Deep dermis + subcutaneous bacterial invasion — Strep/Staph
            # WBC 11k–16k: standard published range for moderate-severe cellulitis
            # CRP 2–15 mg/dL: rises proportionally with infection depth
            "wbc":         {"min": 11000, "max": 16000, "unit": "cells/uL"},
            "crp":         {"min": 2.0,   "max": 15.0,  "unit": "mg/dL"},
            "neutrophils": "elevated",
            "note":        "WBC + CRP rise together. CRP level correlates with infection severity."
        },
        "symptom_logic": {
            "pain": 8, "warmth": True, "itch": 0,
            "swelling": True, "evolution": "rapid"
        }
    },

    {
        "category": "Bacterial",
        "disease":  "Erysipelas",
        "icd11":    "1B73",
        "systemic": True,
        "blood_logic": {
            # Streptococcal dermis + upper lymphatic infection
            # More severe than cellulitis — higher WBC, ESR markedly elevated
            # WBC 15k–20k, CRP 5–20, ESR 40–80
            "wbc":         {"min": 15000, "max": 20000, "unit": "cells/uL"},
            "crp":         {"min": 5.0,   "max": 20.0,  "unit": "mg/dL"},
            "esr":         {"min": 40,    "max": 80,    "unit": "mm/hr"},
            "neutrophils": "very_elevated",
            "note":        "Higher WBC+CRP than cellulitis. Sharply demarcated raised border distinguishes visually."
        },
        "symptom_logic": {
            "pain": 9, "warmth": True, "itch": 0,
            "border": "sharply_demarcated", "evolution": "rapid"
        }
    },

    {
        "category": "Bacterial",
        "disease":  "Impetigo",
        "icd11":    "L01",
        "systemic": True,
        "blood_logic": {
            # Superficial epidermal bacterial infection — Staph aureus / Strep pyogenes
            # Milder than cellulitis — WBC mildly elevated
            "wbc":  {"min": 10000, "max": 12000, "unit": "cells/uL"},
            "crp":  {"min": 0.5,   "max": 2.0,   "unit": "mg/dL"},
            "note": "Superficial infection — WBC/CRP less elevated than deep infections."
        },
        "symptom_logic": {
            "pain": 2, "itch": 2,
            "crust": "honey_colored", "evolution": "rapid"
        }
    },

    {
        "category": "Bacterial",
        "disease":  "Pyoderma",
        "icd11":    "EK70",
        "systemic": True,
        "blood_logic": {
            # Pus-forming bacterial skin infection
            # Most common bacterial skin disease in Indian children
            # Deeper than impetigo — higher WBC range
            "wbc":         {"min": 12000, "max": 18000, "unit": "cells/uL"},
            "crp":         {"min": 2.0,   "max": 10.0,  "unit": "mg/dL"},
            "neutrophils": "elevated",
            "note":        "Deeper than impetigo — more elevated markers. India: most common bacterial skin disease in children."
        },
        "symptom_logic": {
            "pain": 6, "pus": True, "crust": True,
            "warmth": True, "evolution": "rapid"
        }
    },

    {
        "category": "Bacterial",
        "disease":  "Folliculitis",
        "icd11":    "EK70.0",
        "systemic": True,
        "blood_logic": {
            # Staph aureus infection of hair follicle — focal, contained
            # Mildest bacterial blood signal — infection does not spread beyond follicle
            "wbc":         {"min": 9000, "max": 13000, "unit": "cells/uL"},
            "crp":         {"min": 0.5,  "max": 3.0,   "unit": "mg/dL"},
            "neutrophils": "mildly_elevated",
            "note":        "Mild systemic response — infection confined to follicle unit."
        },
        "symptom_logic": {
            "pain": 3, "itch": 4, "pus": True,
            "pattern": "follicle_centered", "evolution": "rapid"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # VIRAL
    # Pattern: WBC normal or LOW + Lymphocytes ELEVATED
    # This is the OPPOSITE of bacterial — Module 5 gate uses this separation
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Viral",
        "disease":  "Shingles",
        "icd11":    "1D61",
        "systemic": True,
        "blood_logic": {
            # Varicella-Zoster reactivation along sensory dermatome
            # WBC normal or low, lymphocytes elevated
            "wbc":         {"min": 3500, "max": 9000, "unit": "cells/uL"},
            "lymphocytes": "elevated",
            "note":        "Classic viral pattern: lymphocytosis + normal/low WBC. Opposite of bacterial."
        },
        "symptom_logic": {
            "pain": 10, "itch": 0, "sensation": "burning",
            "pattern": "dermatomal", "evolution": "clusters"
        }
    },

    {
        "category": "Viral",
        "disease":  "Chickenpox",
        "icd11":    "1E90",
        "systemic": True,
        "blood_logic": {
            # Varicella primary infection — leukopenia + lymphocytosis
            "wbc":         {"min": 3000, "max": 9000, "unit": "cells/uL"},
            "lymphocytes": "elevated",
            "crp":         {"min": 0.5,  "max": 2.5,  "unit": "mg/dL"},
            "note":        "Leukopenia (WBC often below normal) is common in viral exanthem. Lymphocytosis confirms viral."
        },
        "symptom_logic": {
            "pain": 3, "itch": 8, "fever": True,
            "morphology": "vesicles_all_stages", "evolution": "centripetal"
        }
    },

    {
        "category": "Viral",
        "disease":  "Measles",
        "icd11":    "1F03",
        "systemic": True,
        "blood_logic": {
            # Paramyxovirus — pronounced leukopenia is the diagnostic hallmark
            # WBC 2000–5000 is the well-established measles finding
            "wbc":         {"min": 2000, "max": 5000, "unit": "cells/uL"},
            "lymphocytes": "elevated",
            "note":        "Pronounced leukopenia (WBC 2000–5000) is a diagnostic hallmark of measles."
        },
        "symptom_logic": {
            "pain": 2, "itch": 4, "fever": True,
            "koplik_spots": True, "evolution": "cephalocaudal"
        }
    },

    {
        "category": "Viral",
        "disease":  "Molluscum Contagiosum",
        "icd11":    "1E71",
        "systemic": False,
        "blood_logic": {
            # Poxvirus — local epidermal infection only
            "status": "normal",
            "note":   "Local viral infection — blood completely normal. Gate suppresses clinical stream."
        },
        "symptom_logic": {
            "pain": 0, "itch": 2,
            "morphology": "pearly_umbilicated", "evolution": "slow"
        }
    },

    {
        "category": "Viral",
        "disease":  "Warts",
        "icd11":    "EL70",
        "systemic": False,
        "blood_logic": {
            # HPV infects keratinocytes — no systemic involvement
            "status": "normal",
            "note":   "HPV infects keratinocytes only — no systemic blood changes at all."
        },
        "symptom_logic": {
            "pain": 2, "itch": 1,
            "morphology": "rough_verrucous",
            "pattern": "hands_feet_face", "evolution": "chronic"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # INFESTATION
    # Pattern: IgE + Eosinophils elevated — allergic response to parasite
    # WBC and CRP NORMAL — this is not infection
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Infestation",
        "disease":  "Scabies",
        "icd11":    "1G04",
        "systemic": False,
        "blood_logic": {
            # Sarcoptes scabiei mite protein triggers IgE-mediated allergic response
            # IgE 150–500, Eosinophils 0.6–2.0
            # WBC + CRP: NORMAL — critical distinction from bacterial infections
            "ige":         {"min": 150, "max": 500, "unit": "kU/L"},
            "eosinophils": {"min": 0.6, "max": 2.0, "unit": "10^9/L"},
            "wbc":         "normal",
            "crp":         "normal",
            "note":        "Allergic response to mite protein — NOT infection. Normal WBC+CRP distinguishes from cellulitis."
        },
        "symptom_logic": {
            "pain": 1, "itch": 10,
            "timing": "night_itch_severe",
            "pattern": "finger_webs_wrists",
            "morphology": "burrow_tracks", "evolution": "chronic"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # ALLERGIC
    # Pattern: IgE + Eosinophils elevated
    # IgE > 200 + chronic = Eczema
    # IgE moderate + wheals + rapid = Urticaria
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Allergic",
        "disease":  "Eczema",
        "icd11":    "EA80",
        "systemic": True,
        "blood_logic": {
            # IgE > 200 kU/L with eosinophilia strongly supports atopic diagnosis
            # Normal IgE < 100 kU/L — these values are well above normal
            "ige":         {"min": 200,  "max": 1000, "unit": "kU/L"},
            "eosinophils": {"min": 0.5,  "max": 1.5,  "unit": "10^9/L"},
            "note":        "IgE > 200 kU/L with eosinophilia is strongly diagnostic of atopic state."
        },
        "symptom_logic": {
            "pain": 1, "itch": 10, "dryness": True,
            "trigger": "allergen_or_irritant", "evolution": "chronic"
        }
    },

    {
        "category": "Allergic",
        "disease":  "Urticaria",
        "icd11":    "EA81",
        "systemic": True,
        "blood_logic": {
            # Mast cell IgE-mediated acute release — lower IgE range than Eczema
            "ige":         {"min": 150,  "max": 800,  "unit": "kU/L"},
            "eosinophils": {"min": 0.4,  "max": 1.2,  "unit": "10^9/L"},
            "crp":         "mildly_elevated",
            "note":        "Lower IgE than Eczema. Wheals appear in minutes and resolve in hours — key distinguishing feature."
        },
        "symptom_logic": {
            "pain": 1, "itch": 9, "wheals": True,
            "duration": "hours_then_resolves", "evolution": "rapid"
        }
    },

    {
        "category": "Allergic",
        "disease":  "Contact Dermatitis",
        "icd11":    "EK00",
        "systemic": False,
        "blood_logic": {
            # T-cell mediated delayed hypersensitivity — no systemic blood activation
            "status": "normal",
            "note":   "Purely local T-cell response — blood completely normal. Gate must suppress clinical stream."
        },
        "symptom_logic": {
            "pain": 3, "itch": 7,
            "trigger": "contact", "pattern": "contact_site_only",
            "evolution": "rapid"
        }
    },

    {
        "category": "Allergic",
        "disease":  "Drug Eruption",
        "icd11":    "EK90",
        "systemic": True,
        "blood_logic": {
            # Drug hypersensitivity — eosinophilia is the hallmark diagnostic marker
            # Critical in India due to extremely high self-medication culture
            "eosinophils": {"min": 0.5,  "max": 2.0,  "unit": "10^9/L"},
            "wbc":         {"min": 9000, "max": 14000, "unit": "cells/uL"},
            "crp":         {"min": 1.0,  "max": 5.0,  "unit": "mg/dL"},
            "note":        "Eosinophilia is the hallmark of drug hypersensitivity. India priority: self-medication very common."
        },
        "symptom_logic": {
            "pain": 2, "itch": 7, "trigger": "medication",
            "widespread": True, "evolution": "rapid"
        }
    },

    {
        "category": "Allergic",
        "disease":  "Angioedema",
        "icd11":    "4A84",
        "systemic": True,
        "blood_logic": {
            # IgE-mediated deep tissue swelling — deeper than urticaria
            # C4 complement low: consumed in IgE-mediated and hereditary forms
            "ige":           {"min": 200, "max": 800, "unit": "kU/L"},
            "eosinophils":   {"min": 0.4, "max": 1.2, "unit": "10^9/L"},
            "complement_c4": "low",
            "note":          "Low C4 is key. Hereditary type: C4 always low. Acquired: C4 low during attacks."
        },
        "symptom_logic": {
            "pain": 3, "itch": 5, "swelling": "deep_tissue",
            "trigger": "allergen_or_unknown", "evolution": "rapid"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # AUTOIMMUNE
    # Most complex blood patterns — antibodies + complement + special markers
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Autoimmune",
        "disease":  "Psoriasis",
        "icd11":    "EA90",
        "systemic": True,
        "blood_logic": {
            # Th17-driven chronic inflammation
            # NLR > 2.0 is a validated psoriasis severity marker
            "crp": {"min": 1.0, "max": 5.0, "unit": "mg/dL"},
            "esr": {"min": 15,  "max": 40,  "unit": "mm/hr"},
            "nlr": ">2.0",
            "note": "NLR > 2.0 is validated psoriasis severity biomarker. Reflects Th17 chronic inflammation."
        },
        "symptom_logic": {
            "pain": 3, "itch": 5, "scaling": "silvery_white",
            "pattern": "plaques_elbows_knees_scalp",
            "evolution": "persistent"
        }
    },

    {
        "category": "Autoimmune",
        "disease":  "Lupus",
        "icd11":    "4A40.00",  # ICD-11 SLE with skin involvement
                                 # Your old code had 6D91 — that is WRONG
        "systemic": True,
        "blood_logic": {
            # ANA positive > 95% cases (ACR diagnostic criterion)
            # LEUKOPENIA: WBC 2000–4000 — immune complex consumption
            # This is the OPPOSITE of bacterial — WBC falls, not rises
            "ana":           "positive",
            "wbc":           {"min": 2000, "max": 4000, "unit": "cells/uL"},
            "esr":           "very_high",
            "platelets":     "low",
            "complement_c3": "low",
            "complement_c4": "low",
            "note":          "ANA + leukopenia (WBC FALLS to 2000–4000) + low complement = SLE. WBC falling is opposite of all bacterial."
        },
        "symptom_logic": {
            "pain": 4, "itch": 0,
            "pattern": "butterfly_malar_rash",
            "trigger": "sun_exposure",
            "evolution": "flares_and_remits"
        }
    },

    {
        "category": "Autoimmune",
        "disease":  "Scleroderma",
        "icd11":    "LD27",
        "systemic": True,
        "blood_logic": {
            # Anti-Scl70 (anti-topoisomerase I) highly specific for diffuse scleroderma
            # ANA positive > 95%, ESR + CRP elevated
            "ana":        "positive",
            "anti_scl70": "positive_in_diffuse_type",
            "esr":        {"min": 20,  "max": 60,  "unit": "mm/hr"},
            "crp":        {"min": 1.0, "max": 4.0, "unit": "mg/dL"},
            "note":       "Anti-Scl70 is highly specific for diffuse scleroderma. ANA near-universal."
        },
        "symptom_logic": {
            "pain": 5, "itch": 2, "texture": "skin_hardening",
            "pattern": "fingers_face_trunk",
            "evolution": "slow_progressive"
        }
    },

    {
        "category": "Autoimmune",
        "disease":  "Dermatomyositis",
        "icd11":    "5C01",
        "systemic": True,
        "blood_logic": {
            # CK spikes 5–50x normal: most unique blood marker in all skin diseases
            # Normal CK = 22–198 IU/L → elevated = 1000–10000 IU/L
            # CK + skin rash = near-pathognomonic combination
            "ck":       {"min": 1000, "max": 10000, "unit": "IU/L"},
            "ldh":      {"min": 280,  "max": 600,   "unit": "U/L"},
            "aldolase": "elevated",
            "esr":      {"min": 20,   "max": 60,    "unit": "mm/hr"},
            "ana":      "positive_in_30_percent",
            "note":     "CK 5–50x normal. No other common skin disease spikes CK. Gottron papules + elevated CK = near-pathognomonic."
        },
        "symptom_logic": {
            "pain": 6, "muscle_weakness": True,
            "rash": "gottron_papules",
            "heliotrope_rash": True, "evolution": "subacute"
        }
    },

    {
        "category": "Autoimmune",
        "disease":  "Pemphigus Vulgaris",
        "icd11":    "EE40",
        "systemic": True,
        "blood_logic": {
            # Anti-Dsg3: > 95% sensitivity for PV (Labcorp: > 20 RU/mL = positive)
            # Anti-Dsg1: positive in ~50–60%, correlates with skin lesion severity
            # India has the world's highest prevalence of Pemphigus
            "anti_dsg3":   "positive",
            "anti_dsg1":   "positive_in_50_to_60_percent",
            "eosinophils": "mildly_elevated_in_some_patients",
            "note":        "Anti-Dsg3 > 95% sensitivity. India has world's highest prevalence. Mucosal involvement = early sign."
        },
        "symptom_logic": {
            "pain": 8, "itch": 3,
            "morphology": "flaccid_blisters",
            "mucosal_involvement": True, "evolution": "subacute"
        }
    },

    {
        "category": "Autoimmune",
        "disease":  "Vitiligo",
        "icd11":    "ED63",
        "systemic": False,
        "blood_logic": {
            # Melanocyte destruction — blood CBC + CRP normal
            # ~30% have thyroid antibodies (associated autoimmune condition)
            "thyroid_antibodies": "elevated_in_30_percent",
            "ana":                "possibly_positive",
            "wbc":                "normal",
            "crp":                "normal",
            "note":               "Blood generally normal. Thyroid antibody screening recommended. Gate mostly suppresses."
        },
        "symptom_logic": {
            "pain": 0, "itch": 0, "depigmentation": True,
            "pattern": "face_hands_genitals",
            "evolution": "chronic_stable"
        }
    },

    {
        "category": "Autoimmune",
        "disease":  "Alopecia Areata",
        "icd11":    "ED70",
        "systemic": False,
        "blood_logic": {
            # T-cell attack on hair follicle — blood generally normal
            "thyroid_antibodies": "elevated_in_30_percent",
            "ana":                "possibly_positive",
            "wbc":                "normal",
            "crp":                "normal",
            "note":               "Blood generally normal. Screen thyroid. Gate suppresses; visual pattern is primary signal."
        },
        "symptom_logic": {
            "pain": 0, "itch": 0, "hair_loss": True,
            "pattern": "smooth_round_patches",
            "evolution": "patchy"
        }
    },

    {
        "category": "Autoimmune",
        "disease":  "Hidradenitis Suppurativa",
        "icd11":    "EK50",
        "systemic": True,
        "blood_logic": {
            # TNF-α driven chronic follicular occlusion
            # CRP correlates with IHS4 disease activity score
            "crp":         {"min": 2.0,   "max": 8.0,  "unit": "mg/dL"},
            "wbc":         {"min": 10000, "max": 14000, "unit": "cells/uL"},
            "esr":         {"min": 20,    "max": 50,   "unit": "mm/hr"},
            "note":        "CRP correlates with disease activity. TNF-α is primary cytokine driver."
        },
        "symptom_logic": {
            "pain": 8, "itch": 2, "pus": True,
            "pattern": "axillae_groin_inframammary",
            "evolution": "recurrent_chronic"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HORMONAL
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Hormonal",
        "disease":  "Acne",
        "icd11":    "EE60",    # Correct ICD-11 for acne vulgaris
                                # Your old code had ED56 — that is WRONG
        "systemic": False,      # Acne is LOCAL — blood normal in most cases
        "blood_logic": {
            "testosterone": "elevated_in_hormonal_acne_subset",
            "crp":          "mildly_elevated_in_inflammatory_type_only",
            "status":       "mostly_normal",
            "note":         "Predominantly local. Testosterone only relevant in female hormonal acne. Gate suppresses."
        },
        "symptom_logic": {
            "pain": 4, "itch": 0,
            "pattern": "face_back_chest",
            "evolution": "periodic_relapsing"
        }
    },

    {
        "category": "Hormonal",
        "disease":  "Melasma",
        "icd11":    "ED55",
        "systemic": False,
        "blood_logic": {
            "estrogen":     "elevated_in_pregnancy_or_ocp",
            "progesterone": "elevated_in_pregnancy",
            "tsh":          "check_if_thyroid_coexists",
            "status":       "otherwise_normal",
            "note":         "Hormonal trigger only — blood otherwise normal. Very common Indian women."
        },
        "symptom_logic": {
            "pain": 0, "itch": 0,
            "trigger": "sun_and_hormones",
            "pattern": "symmetric_face", "evolution": "chronic"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # FUNGAL
    # Superficial keratin infection — blood ALWAYS normal
    # These entries teach Module 5 gate to suppress clinical stream
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Fungal",
        "disease":  "Tinea",
        "icd11":    "1F28",    # Confirmed ICD-11 for dermatophytosis
                                # Your old code had EA60 — that is WRONG
        "systemic": False,
        "blood_logic": {
            "status": "normal",
            "note":   "Dermatophyte infects keratin only — no systemic immune activation. Gate must suppress."
        },
        "symptom_logic": {
            "pain": 0, "itch": 6,
            "pattern": "ring_shaped_peripheral_scaling",
            "evolution": "slow"
        }
    },

    {
        "category": "Fungal",
        "disease":  "Pityriasis Versicolor",
        "icd11":    "1F27",
        "systemic": False,
        "blood_logic": {
            "status": "normal",
            "note":   "Superficial yeast — blood normal. Extremely common South India + monsoon season."
        },
        "symptom_logic": {
            "pain": 0, "itch": 3,
            "pattern": "hypo_hyperpigmented_patches",
            "location": "chest_back_shoulders",
            "evolution": "chronic"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SEBACEOUS
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Sebaceous",
        "disease":  "Seborrheic Dermatitis",
        "icd11":    "EA85",
        "systemic": False,
        "blood_logic": {
            "status": "normal",
            "note":   "Yeast-triggered local condition — blood completely normal. Dandruff is same condition on scalp."
        },
        "symptom_logic": {
            "pain": 1, "itch": 5, "scaling": "greasy_yellow",
            "pattern": "scalp_face_chest",
            "evolution": "chronic_relapsing"
        }
    },

    {
        "category": "Sebaceous",
        "disease":  "Rosacea",
        "icd11":    "EE21",
        "systemic": False,
        "blood_logic": {
            "crp":    "mildly_elevated_0.5_to_1.5_in_some",
            "status": "mostly_normal",
            "note":   "Very weak blood signal — inconsistent across patients. Vision + TNF features carry primary weight."
        },
        "symptom_logic": {
            "pain": 2, "itch": 3,
            "trigger": "sun_alcohol_spicy_food",
            "pattern": "central_face_nose_cheeks",
            "evolution": "chronic_relapsing"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MALIGNANT
    # Early stage: blood NORMAL — vision + TNF are the only signals
    # Advanced stage: blood changes appear
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Malignant",
        "disease":  "Melanoma",
        "icd11":    "2C30",
        "systemic": True,
        "blood_logic": {
            # Early melanoma: blood NORMAL — visual + TNF geometry are primary signals
            # Advanced/metastatic: LDH > 280 U/L (AJCC staging criterion)
            "ldh":   "elevated_if_metastatic",
            "s100b": "elevated_if_advanced",
            "wbc":   "normal_in_early_stage",
            "note":  "Blood normal in early melanoma — visual + TNF geometry are primary. LDH used in AJCC staging."
        },
        "symptom_logic": {
            "pain": 0, "itch": 2, "bleeding": True,
            "asymmetry": True, "evolution": "changing_over_weeks"
        }
    },

    {
        "category": "Malignant",
        "disease":  "Basal Cell Carcinoma",
        "icd11":    "2C31",
        "systemic": False,
        "blood_logic": {
            "status": "normal",
            "note":   "Local invasion only — blood normal. Visual + TNF geometry are the only diagnostic signals."
        },
        "symptom_logic": {
            "pain": 1, "itch": 0, "texture": "pearly_waxy",
            "bleeding": True, "evolution": "very_slow"
        }
    },

    {
        "category": "Malignant",
        "disease":  "Actinic Keratosis",
        "icd11":    "2F03",
        "systemic": False,
        "blood_logic": {
            "status": "normal",
            "note":   "Pre-malignant — blood normal. Visual detection on sun-exposed areas is the only signal."
        },
        "symptom_logic": {
            "pain": 1, "itch": 2, "texture": "rough_scaly",
            "pattern": "sun_exposed_areas", "evolution": "slow"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NUTRITIONAL
    # Blood deficiency creates specific skin manifestation
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Nutritional",
        "disease":  "Iron Deficiency Anaemia",
        "icd11":    "5B55",
        "systemic": True,
        "blood_logic": {
            # Low Hb + low MCV + low ferritin = confirmed IDA triad
            # Hb < 12.0 g/dL = anaemia by WHO (women) | < 13.5 (men)
            # MCV < 80 fL = microcytic
            # Ferritin < 12 ng/mL = depleted iron stores
            "hemoglobin": {"max": 12.0, "unit": "g/dL"},
            "mcv":        {"max": 80,   "unit": "fL"},
            "ferritin":   {"max": 12.0, "unit": "ng/mL"},
            "wbc":        "normal",
            "note":       "Hb + MCV + ferritin all low = confirmed IDA. Skin: pallor, koilonychia, brittle nails. Very common Indian women."
        },
        "symptom_logic": {
            "pain": 0, "itch": 4,
            "signs": "pallor_koilonychia_brittle_nails_hair_loss",
            "fatigue": True, "evolution": "chronic"
        }
    },

    {
        "category": "Nutritional",
        "disease":  "Vitamin D Deficiency",
        "icd11":    "EB90",
        "systemic": True,
        "blood_logic": {
            # 25-OH-D < 20 ng/mL = deficient (Endocrine Society 2024)
            # 20–30 ng/mL = insufficient
            "serum_25_oh_d": {"max": 20, "unit": "ng/mL"},
            "calcium":       "low_in_severe_deficiency",
            "note":          "25-OH-D < 20 ng/mL = WHO deficiency cutoff. Endemic in India despite sun — melanin + clothing."
        },
        "symptom_logic": {
            "pain": 0, "itch": 3, "scaling": "diffuse",
            "fatigue": True, "evolution": "seasonal"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SYSTEMIC — Blood disease causes skin manifestation
    # Blood problem comes FIRST, skin sign is the visible symptom
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Systemic",
        "disease":  "Uremic Pruritus",
        "icd11":    "FA24",
        "systemic": True,
        "blood_logic": {
            # Chronic kidney disease → toxin accumulation → extreme itch
            "creatinine": {"min": 3.0,  "max": 10.0, "unit": "mg/dL"},
            "bun":        {"min": 40,   "max": 100,  "unit": "mg/dL"},
            "note":       "Creatinine level directly correlates with itch severity. Blood disease causes skin symptom."
        },
        "symptom_logic": {
            "pain": 0, "itch": 10,
            "trigger": "renal_failure", "evolution": "chronic"
        }
    },

    {
        "category": "Systemic",
        "disease":  "Jaundice",
        "icd11":    "5C56",
        "systemic": True,
        "blood_logic": {
            # Bilirubin > 2.5 mg/dL = clinically visible yellowing of skin + sclera
            "bilirubin_total": {"min": 2.5, "unit": "mg/dL"},
            "alt":             "elevated",
            "ast":             "elevated",
            "alp":             "elevated",
            "note":            "Bilirubin > 2.5 mg/dL = visible yellow skin. Liver damage drives bilirubin accumulation."
        },
        "symptom_logic": {
            "pain": 2, "itch": 6,
            "color": "yellow_skin_and_eyes", "evolution": "rapid"
        }
    },

    {
        "category": "Systemic",
        "disease":  "Purpura",
        "icd11":    "3B64",
        "systemic": True,
        "blood_logic": {
            # Platelets < 50,000 = high purpura risk
            # Platelets < 20,000 = critical spontaneous bleeding
            "platelets": {"max": 50000, "unit": "cells/uL"},
            "note":      "Platelets < 50k = purpura risk. Non-blanching on pressure = distinguishes from rash."
        },
        "symptom_logic": {
            "pain": 1, "itch": 0, "bleeding": True,
            "blanching": False, "evolution": "rapid"
        }
    },

    {
        "category": "Systemic",
        "disease":  "Acanthosis Nigricans",
        "icd11":    "EE10.0",
        "systemic": True,
        "blood_logic": {
            # Insulin resistance → dark velvety skin on neck/axillae
            # Fasting glucose > 126 mg/dL = diabetes (ADA 2024)
            # HbA1c > 6.5% = diabetes (ADA 2024)
            "glucose_fasting": {"min": 126, "unit": "mg/dL"},
            "hba1c":           {"min": 6.5, "unit": "percent"},
            "insulin":         "elevated",
            "note":            "Dark velvety patches = visible skin sign of insulin resistance. 400M diabetics in India."
        },
        "symptom_logic": {
            "pain": 0, "itch": 2, "texture": "velvety_dark",
            "pattern": "neck_axillae_groin", "evolution": "chronic"
        }
    },

    {
        "category": "Systemic",
        "disease":  "Cutaneous Vasculitis",
        "icd11":    "4A44",
        "systemic": True,
        "blood_logic": {
            # Inflamed blood vessel walls → palpable purpura
            # ANCA positive in > 90% of systemic vasculitis
            "anca":          "positive",
            "esr":           {"min": 50,  "max": 100,  "unit": "mm/hr"},
            "crp":           {"min": 3.0, "max": 10.0, "unit": "mg/dL"},
            "complement_c3": "low_in_immune_complex_type",
            "note":          "ANCA + high ESR + palpable purpura = vasculitis. Palpable (raised) purpura is hallmark."
        },
        "symptom_logic": {
            "pain": 4, "itch": 2,
            "morphology": "palpable_purpura",
            "pattern": "lower_legs_feet", "evolution": "rapid"
        }
    },

    {
        "category": "Systemic",
        "disease":  "Polycythaemia Vera",
        "icd11":    "2A20",
        "systemic": True,
        "blood_logic": {
            # Excess RBCs → hyperviscous blood → aquagenic pruritus
            # Hematocrit > 52% men / > 48% women (WHO PV criteria 2022)
            # JAK2 V617F positive in > 95% of PV
            "rbc":        "very_elevated",
            "hematocrit": {"min": 52,   "unit": "percent"},
            "hemoglobin": {"min": 18.5, "unit": "g/dL"},
            "jak2_v617f": "positive",
            "note":       "Aquagenic pruritus (itch after hot bath) + high Hb + JAK2 = PV triad."
        },
        "symptom_logic": {
            "pain": 2, "itch": 9,
            "trigger": "hot_water_contact",
            "color": "deep_red_plethoric_face", "evolution": "chronic"
        }
    },

    # ══════════════════════════════════════════════════════════════════════════
    # INDIA-PRIORITY NEGLECTED DISEASES
    # ══════════════════════════════════════════════════════════════════════════

    {
        "category": "Neglected",
        "disease":  "Leprosy",
        "icd11":    "1B20",
        "systemic": True,
        "blood_logic": {
            # Mycobacterium leprae — peripheral nerve + skin
            # India: ~127,000 new cases/year = ~50% of global burden
            "esr":  {"min": 15, "max": 40, "unit": "mm/hr"},
            "crp":  "mildly_elevated",
            "wbc":  "normal",
            "note": "Blood changes subtle — diagnosis primarily clinical. ESR mildly elevated. India priority."
        },
        "symptom_logic": {
            "pain": 0, "itch": 0, "sensation_loss": True,
            "pattern": "hypopigmented_anaesthetic_patches",
            "evolution": "very_slow"
        }
    },

    {
        "category": "Neglected",
        "disease":  "Hand Foot Mouth Disease",
        "icd11":    "1F05",
        "systemic": True,
        "blood_logic": {
            # Coxsackievirus — common in Indian children under 5
            "wbc":         {"min": 6000, "max": 11000, "unit": "cells/uL"},
            "lymphocytes": "mildly_elevated",
            "crp":         {"min": 0.5,  "max": 2.0,   "unit": "mg/dL"},
            "note":        "Mild viral response. Vesicles on palms + soles + oral ulcers = diagnostic triad."
        },
        "symptom_logic": {
            "pain": 5, "itch": 4, "fever": True,
            "pattern": "palms_soles_mouth", "evolution": "rapid"
        }
    },
]


async def seed_matrix():
    """
    Idempotent upsert keyed on icd11.
    Safe to run any number of times.
    Updates existing entries, inserts new ones, never deletes.
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
                {"$set":  rule},
                upsert=True,
            )
            for rule in SKINTEL_RULEBOOK
        ]

        result = await collection.bulk_write(operations, ordered=False)

        logger.success(
            f"✅ Knowledge base synced — "
            f"inserted: {result.upserted_count}  "
            f"updated: {result.modified_count}  "
            f"matched: {result.matched_count}  "
            f"total diseases: {len(SKINTEL_RULEBOOK)}"
        )

        # ── Category breakdown ────────────────────────────────────────────────
        pipeline = [{"$group": {"_id": "$category", "count": {"$sum": 1}}}]
        cursor   = collection.aggregate(pipeline)
        cats     = await cursor.to_list(length=100)
        logger.info("─── Category breakdown ──────────────────────────")
        for cat in sorted(cats, key=lambda x: x["_id"]):
            logger.info(f"  {cat['_id']:<22}: {cat['count']} disease(s)")

        # ── Systemic vs local ─────────────────────────────────────────────────
        systemic = sum(1 for r in SKINTEL_RULEBOOK if r.get("systemic"))
        local    = sum(1 for r in SKINTEL_RULEBOOK if not r.get("systemic"))
        logger.info("─── Blood gate summary ──────────────────────────")
        logger.info(f"  Systemic (gate AMPLIFIES) : {systemic}")
        logger.info(f"  Local    (gate SUPPRESSES): {local}")
        logger.info(f"  Total diseases             : {len(SKINTEL_RULEBOOK)}")

    except Exception as e:
        logger.error(f"❌ Failed to seed rulebook: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(seed_matrix())