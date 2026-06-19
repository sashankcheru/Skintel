"""
modules/module1_bedrock/tasks.py
Celery tasks for Module 1 — Data Bedrock.

Tasks:
  run_etl_task   — runs the full ETL pipeline asynchronously
                   dispatched by POST /api/v1/bedrock/ingest
                   returns record counts on completion
"""

import asyncio
from loguru import logger
from modules.celery_app import celery_app
from modules.module1_bedrock.etl_ingestor import run_etl


@celery_app.task(
    name="module1.run_etl",
    bind=True,
    max_retries=1,
    queue="skintel",
)
def run_etl_task(self):
    """
    Runs the full ETL pipeline inside the Celery GPU worker.
    FastAPI route dispatches this and returns a task_id immediately.
    The client polls GET /api/v1/bedrock/task/{task_id} for status.
    """
    try:
        logger.info(f"[Celery] ETL task started — task_id: {self.request.id}")
        asyncio.run(run_etl())
        logger.success(f"[Celery] ETL task complete — task_id: {self.request.id}")
        return {"status": "complete", "task_id": self.request.id}
    except Exception as exc:
        logger.error(f"[Celery] ETL task failed: {exc}")
        raise self.retry(exc=exc, countdown=10)