import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime
import os
from loguru import logger

DATA_PATH = 'data/processed'
os.makedirs(DATA_PATH, exist_ok=True)

# FULL SCHEMA — CBC + CMP aligned with Module 4 Roadmap
schema = pa.schema([
    ('patient_id', pa.string()),
    ('timestamp', pa.timestamp('ms')),

    # CBC — Complete Blood Count
    ('cbc', pa.struct([
        ('hemoglobin',      pa.float32()),   # Hb — g/dL
        ('wbc_total',       pa.float32()),   # Total WBC — cells/uL
        ('neutrophils',     pa.float32()),   # WBC Diff — %
        ('lymphocytes',     pa.float32()),   # WBC Diff — %
        ('monocytes',       pa.float32()),   # WBC Diff — %
        ('eosinophils',     pa.float32()),   # WBC Diff — %
        ('basophils',       pa.float32()),   # WBC Diff — %
        ('platelets',       pa.float32()),   # PLT — 10^3/uL
    ])),

    # CMP — Comprehensive Metabolic Panel
    ('cmp', pa.struct([
        ('alt',             pa.float32()),   # Liver enzyme — U/L
        ('ast',             pa.float32()),   # Liver enzyme — U/L
        ('alp',             pa.float32()),   # Liver enzyme — U/L
        ('creatinine',      pa.float32()),   # Kidney — mg/dL
        ('bun',             pa.float32()),   # Blood Urea Nitrogen — mg/dL
        ('glucose',         pa.float32()),   # mg/dL
    ])),

    # Inflammatory Markers
    ('inflammatory', pa.struct([
        ('crp',             pa.float32()),   # C-Reactive Protein — mg/dL
        ('esr',             pa.float32()),   # Erythrocyte Sedimentation Rate — mm/hr
    ])),

    # Symptom Matrix (Brick 1.2) — [Pruritus(0-3), Nociception(0-10), Evolution(0=slow,1=rapid)]
    ('symptoms', pa.list_(pa.int32())),

    # Fitzpatrick Skin Type (I–VI) for bias-aware training
    ('fitzpatrick_type',    pa.int32()),

    # MinIO image pointer
    ('image_path',          pa.string()),
    ('label',               pa.string()),
])

def create_foundation():
    """Creates the Parquet foundation with the full clinical schema."""
    data = [{
        'patient_id': 'SK-ALPHA-01',
        'timestamp': pd.Timestamp.now().floor('ms'),
        'cbc': {
            'hemoglobin':   13.5,
            'wbc_total':    7200.0,
            'neutrophils':  58.0,
            'lymphocytes':  32.0,
            'monocytes':    6.0,
            'eosinophils':  3.0,
            'basophils':    1.0,
            'platelets':    250.0,
        },
        'cmp': {
            'alt':          22.0,
            'ast':          19.0,
            'alp':          75.0,
            'creatinine':   0.9,
            'bun':          14.0,
            'glucose':      92.0,
        },
        'inflammatory': {
            'crp':          0.4,
            'esr':          8.0,
        },
        'symptoms':         [1, 0, 0],
        'fitzpatrick_type': 3,
        'image_path':       'skintel-images/raw/sample.jpg',
        'label':            'Healthy',
    }]

    try:
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df, schema=schema)
        file_path = os.path.join(DATA_PATH, 'skintel_bedrock.parquet')
        pq.write_table(table, file_path)
        logger.success(f"✅ Full clinical schema written to: {file_path}")

        # Verify — read back and confirm shape
        verified = pq.read_table(file_path)
        logger.info(f"Schema verified — {verified.num_rows} row(s), {verified.num_columns} columns")
    except Exception as e:
        logger.error(f"❌ Failed to write schema: {e}")
        raise

if __name__ == "__main__":
    create_foundation()