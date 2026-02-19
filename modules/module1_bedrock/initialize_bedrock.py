import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime
import os
from loguru import logger

# 1. Align with your Docker directory structure
DATA_PATH = 'data/processed'
os.makedirs(DATA_PATH, exist_ok=True)

# THE SCHEMA: Optimized for Skintel's Multimodal Brain
schema = pa.schema([
    ('patient_id', pa.string()),
    ('timestamp', pa.timestamp('ms')),
    
    # Hematological Data (Module 4) - Using Structs for "Feature Fusion"
    ('blood_markers', pa.struct([
        ('wbc', pa.float32()),  
        ('crp', pa.float32()),  
        ('esr', pa.float32()),  
    ])),
    
    # Symptom Matrix (Brick 1.2) - [Itch, Pain, Rapid Growth]
    ('symptoms', pa.list_(pa.int32())),
    
    # Path to high-res images in MinIO
    ('image_path', pa.string()),
    ('label', pa.string())
])

def create_foundation():
    # Example data aligned with your Clinical Reference rules
    data = [{
        'patient_id': 'SK-ALPHA-01',
        'timestamp': datetime.now(),
        'blood_markers': {'wbc': 7.2, 'crp': 2.1, 'esr': 8.0},
        'symptoms': [1, 0, 0], 
        'image_path': 'skintel-images/raw/sample.jpg', # MinIO path
        'label': 'Healthy'
    }]
    
    try:
        table = pa.Table.from_pandas(pd.DataFrame(data), schema=schema)
        file_path = os.path.join(DATA_PATH, 'skintel_bedrock.parquet')
        pq.write_table(table, file_path)
        logger.success(f"✅ Foundation stone set at: {file_path}")
    except Exception as e:
        logger.error(f"❌ Failed to set foundation: {e}")

if __name__ == "__main__":
    create_foundation()