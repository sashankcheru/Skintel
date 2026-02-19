from minio import Minio
from loguru import logger
import os

def get_minio_client():
    """Returns a configured MinIO client."""
    return Minio(
        os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ROOT_USER"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
    )

async def initialize_minio():
    """Creates required buckets if they do not exist."""
    client = get_minio_client()
    buckets = [
        os.getenv("MINIO_BUCKET_RAW"),
        os.getenv("MINIO_BUCKET_PROCESSED"),
        os.getenv("MINIO_BUCKET_MODELS")
    ]
    
    for bucket in buckets:
        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info(f"Created MinIO bucket: {bucket}")
        except Exception as e:
            logger.error(f"Error initializing bucket {bucket}: {e}")