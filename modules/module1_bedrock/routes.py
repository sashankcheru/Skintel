"""
Module 1 — Brick 1.3  |  routes.py
FastAPI router exposing Module 1 functionality.

Endpoints:
  GET  /api/v1/bedrock/status          → Parquet record count + schema info
  GET  /api/v1/bedrock/task/{task_id}  → Poll Celery task status
  POST /api/v1/bedrock/init            → Initialise Parquet schema
  POST /api/v1/bedrock/seed            → Seed MongoDB knowledge base
  POST /api/v1/bedrock/ingest          → Dispatch ETL task (returns immediately)
"""

import os
import pyarrow.parquet as pq
from fastapi import APIRouter
from loguru import logger

from modules.module1_bedrock.initialize_bedrock import initialize_bedrock, PARQUET
from modules.module1_bedrock.seed_knowledge     import seed_matrix, SKINTEL_RULEBOOK
from modules.module1_bedrock.tasks              import run_etl_task

router = APIRouter()


@router.get("/status")
async def bedrock_status():
    """
    Returns the current state of the Parquet bedrock file.
    Shows record count, schema field count, label/source/split distribution.
    """
    if not os.path.exists(PARQUET):
        return {
            "status":         "not_initialised",
            "parquet_exists": False,
            "records":        0,
            "schema_fields":  0,
        }

    table         = pq.read_table(PARQUET)
    label_counts  = {}
    source_counts = {}
    split_counts  = {}

    if "label" in table.schema.names:
        for lbl in table.column("label").to_pylist():
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    if "source_dataset" in table.schema.names:
        for src in table.column("source_dataset").to_pylist():
            source_counts[src] = source_counts.get(src, 0) + 1

    if "split" in table.schema.names:
        for sp in table.column("split").to_pylist():
            split_counts[sp] = split_counts.get(sp, 0) + 1

    return {
        "status":         "ready",
        "parquet_exists": True,
        "records":        len(table),
        "schema_fields":  len(table.schema),
        "diseases":       len(label_counts),
        "label_counts":   label_counts,
        "source_counts":  source_counts,
        "split_counts":   split_counts,
    }


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    Polls the status of a Celery task by ID.
    States: PENDING → STARTED → SUCCESS | FAILURE
    """
    from modules.celery_app import celery_app
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status":  result.status,
        "result":  result.result if result.ready() else None,
    }


@router.post("/init")
async def init_bedrock():
    """
    Initialises skintel_bedrock.parquet with the 36-field flat schema.
    Idempotent — skips safely if file already exists.
    """
    path = initialize_bedrock()
    return {
        "status":  "ok",
        "message": "Parquet schema initialised",
        "path":    path,
    }


@router.post("/seed")
async def seed_knowledge():
    """
    Seeds the MongoDB medical_knowledge_base collection with disease profiles.
    Idempotent — upserts on ICD-11 code, never deletes existing records.
    """
    await seed_matrix()
    return {
        "status":  "ok",
        "message": f"Knowledge base seeded — {len(SKINTEL_RULEBOOK)} disease profiles written to MongoDB",
    }


@router.post("/ingest", status_code=202)
async def ingest_datasets():
    """
    Dispatches the ETL task to the Celery GPU worker.
    Returns immediately with a task_id — does NOT block the API.
    Poll GET /api/v1/bedrock/task/{task_id} to check progress.

    Datasets ingested: Fitzpatrick17k + PAD-UFES-20 + SCIN + DermaCon-IN
    """
    task = run_etl_task.delay()
    return {
        "status":  "accepted",
        "message": "ETL task dispatched to worker",
        "task_id": task.id,
        "poll":    f"/api/v1/bedrock/task/{task.id}",
    }