"""
Module 1 — Brick 1.1  |  minio_client.py
MinIO object storage client.

The Minio SDK is entirely synchronous.
All blocking calls run inside run_in_executor so they never
block the FastAPI async event loop.

Bucket names match docker-compose minio-setup service + .env exactly:
  skintel-images    → raw dataset images (MINIO_BUCKET_RAW)
  processed-images  → Module 2 output   (MINIO_BUCKET_PROCESSED)
  skintel-models    → model weights     (MINIO_BUCKET_MODELS)
  skintel-runtime   → patient uploads at inference (created here only)
"""

import asyncio
import os
from minio import Minio
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# All buckets Skintel needs — env vars match docker-compose and .env exactly
_BUCKETS = [
    os.getenv("MINIO_BUCKET_RAW",       "skintel-images"),
    os.getenv("MINIO_BUCKET_PROCESSED",  "processed-images"),
    os.getenv("MINIO_BUCKET_MODELS",     "skintel-models"),
    "skintel-runtime",   # patient uploads at inference — not in docker-compose
]


def get_minio_client() -> Minio:
    """Returns a configured synchronous Minio client."""
    endpoint = os.getenv("MINIO_ENDPOINT",      "minio:9000")
    access   = os.getenv("MINIO_ROOT_USER",     "minioadmin")
    secret   = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123")
    secure   = os.getenv("MINIO_SECURE",        "false").lower() == "true"

    if not access or not secret:
        raise ValueError(
            "MINIO_ROOT_USER and MINIO_ROOT_PASSWORD must be set in .env"
        )

    return Minio(endpoint, access_key=access, secret_key=secret, secure=secure)


def _create_buckets() -> None:
    """
    Synchronous bucket setup — runs inside a thread pool executor.
    Creates any bucket that does not already exist.
    Skips empty bucket names (guards against missing .env vars).
    """
    client = get_minio_client()

    for bucket in _BUCKETS:
        if not bucket:
            logger.warning("Empty bucket name — check .env (skipping)")
            continue
        try:
            if client.bucket_exists(bucket):
                logger.info(f"MinIO  exists  : {bucket}")
            else:
                client.make_bucket(bucket)
                logger.info(f"MinIO  created : {bucket}")
        except Exception as exc:
            logger.error(f"MinIO  error [{bucket}]: {exc}")


async def initialize_minio() -> None:
    """
    Async entry point — runs the blocking _create_buckets call
    in a thread pool so the event loop is never blocked.
    Called once at FastAPI startup inside the lifespan manager.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _create_buckets)
    logger.success("✅ MinIO initialised — all buckets ready")


async def upload_image(bucket: str, object_name: str, local_path: str) -> str:
    """
    Uploads a local file to MinIO.
    Returns the full object reference: "bucket/object_name"
    Uses run_in_executor to avoid blocking the event loop.
    """
    def _put():
        client = get_minio_client()
        client.fput_object(bucket, object_name, local_path)
        return f"{bucket}/{object_name}"

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _put)
    logger.info(f"Uploaded → {result}")
    return result
