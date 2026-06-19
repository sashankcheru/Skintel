"""
modules/celery_app.py
Celery application instance for SkinTel.

All heavy async tasks (ETL, preprocessing, inference) are dispatched
through this broker so FastAPI routes return immediately with a task_id.
The GPU worker container runs this with:
  celery -A modules.celery_app worker --loglevel=info --concurrency=1 -Q skintel

Queues:
  skintel   — all tasks (single queue for now; split by module later)
"""

from celery import Celery
from modules.config.settings import settings

celery_app = Celery(
    "skintel",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "modules.module1_bedrock.tasks",   # ETL task
        # "modules.module2_gateway.tasks", # preprocessing — add when Module 2 complete
        # "modules.module3_vision.tasks",  # inference — add when Module 3 complete
    ],
)

celery_app.conf.update(
    task_serializer        = "json",
    result_serializer      = "json",
    accept_content         = ["json"],
    timezone               = "UTC",
    enable_utc             = True,
    task_default_queue     = "skintel",
    task_track_started     = True,       # task moves to STARTED state when worker picks it up
    result_expires         = 86400,      # results kept in Redis for 24 hours
    worker_prefetch_multiplier = 1,      # GPU tasks: never prefetch — one at a time
    broker_connection_retry_on_startup = True,
)