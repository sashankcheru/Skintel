"""
Skintel — AI-Powered Dermatological Screening System
Main FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from modules.config.settings import settings
from modules.module1_bedrock.db_client    import initialize_mongodb, close_mongodb
from modules.module1_bedrock.minio_client import initialize_minio
from modules.module1_bedrock.routes       import router as bedrock_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────
    logger.info("🚀 Starting Skintel...")

    # MongoDB — must succeed before the app accepts requests
    try:
        await initialize_mongodb()
    except Exception as exc:
        logger.error(f"❌ MongoDB init failed: {exc}")

    # MinIO — creates buckets if they do not exist
    try:
        await initialize_minio()
    except Exception as exc:
        logger.error(f"❌ MinIO init failed: {exc}")

    # Data directories — mounted from host via docker-compose volumes
    for directory in [
        "/app/data/raw",
        "/app/data/processed",
        "/app/models/checkpoints",
        "/app/logs",
    ]:
        os.makedirs(directory, exist_ok=True)

    logger.success("✅ Skintel started successfully")

    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────
    logger.info("Shutting down Skintel...")
    await close_mongodb()
    logger.success("✅ Shutdown complete")


# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Skintel API",
    description="Multimodal AI-Driven Skin Disease Risk Screening",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ROUTERS ───────────────────────────────────────────────────────────────────
app.include_router(
    bedrock_router,
    prefix="/api/v1/bedrock",
    tags=["Module 1 — Data Bedrock"],
)

# Uncomment as each module is built:
from modules.module2_gateway.routes  import router as gateway_router
# from modules.module3_vision.routes   import router as vision_router
# from modules.module4_context.routes  import router as context_router
# from modules.module5_brain.routes    import router as brain_router
# from modules.module6_standards.routes import router as standards_router
app.include_router(gateway_router,  prefix="/api/v1/gateway",   tags=["Module 2"])
# app.include_router(vision_router,   prefix="/api/v1/vision",    tags=["Module 3"])
# app.include_router(context_router,  prefix="/api/v1/context",   tags=["Module 4"])
# app.include_router(brain_router,    prefix="/api/v1/brain",     tags=["Module 5"])
# app.include_router(standards_router,prefix="/api/v1/standards", tags=["Module 6"])


# ── ROOT ENDPOINTS ────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to Skintel API",
        "version": "1.0.0",
        "docs":    "/docs",
        "status":  "/health",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Docker health check endpoint — must return 200 for the container to be healthy."""
    return {
        "status":      "healthy",
        "service":     "skintel-api",
        "environment": settings.APP_ENV,
    }


@app.get("/api/v1/status", tags=["Status"])
async def system_status():
    """Returns the operational status of every module."""
    return {
        "api": "online",
        "modules": {
            "module1_bedrock":     "active",
            "module2_gateway":     "pending",
            "module3_vision":      "pending",
            "module4_context":     "pending",
            "module5_brain":       "pending",
            "module6_standards":   "pending",
        },
        "storage": {
            "mongodb": "connected",
            "minio":   "connected",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
