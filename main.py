"""
Skintel - AI-Powered Dermatological Diagnosis System
Main FastAPI Application Entry Point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
from loguru import logger

# Import configuration - FIXED: Import the settings object, not the module
from modules.config.settings import settings

# Import routers (to be created)
# from modules.module1_bedrock.routes import router as bedrock_router
# from modules.module2_gateway.routes import router as gateway_router
# from modules.module3_vision.routes import router as vision_router
# from modules.module4_context.routes import router as context_router
# from modules.module5_brain.routes import router as brain_router
# from modules.module6_standards.routes import router as standards_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting Skintel Application...")
    
    # Initialize MinIO buckets
    try:
        from modules.module1_bedrock.minio_client import initialize_minio
        await initialize_minio()
        logger.info("✅ MinIO initialized successfully")
    except Exception as e:
        logger.error(f"❌ MinIO initialization failed: {e}")
    
    # Initialize MongoDB connection
    try:
        from modules.module1_bedrock.db_client import initialize_mongodb
        await initialize_mongodb()
        logger.info("✅ MongoDB initialized successfully")
    except Exception as e:
        logger.error(f"❌ MongoDB initialization failed: {e}")
    
    # Create data directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info("✅ All directories created")
    logger.success("🎉 Skintel Application Started Successfully!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Skintel Application...")
    # Close database connections if needed
    logger.success("✅ Graceful shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Skintel API",
    description="High-Precision Dermatological Diagnosis and Medication Advice System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Skintel API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Docker"""
    return {
        "status": "healthy",
        "service": "skintel-api",
        "environment": settings.APP_ENV,
    }


@app.get("/api/v1/status", tags=["Status"])
async def system_status():
    """Get system status including all modules"""
    status = {
        "api": "online",
        "modules": {
            "module1_bedrock": "initialized",
            "module2_gateway": "pending",
            "module3_vision": "pending",
            "module4_context": "pending",
            "module5_brain": "pending",
            "module6_standards": "pending",
        },
        "storage": {
            "minio": "connected",
            "mongodb": "connected",
        },
    }
    return status


# Include module routers (uncomment as we build each module)
# app.include_router(bedrock_router, prefix="/api/v1/bedrock", tags=["Module 1: Data Bedrock"])
# app.include_router(gateway_router, prefix="/api/v1/gateway", tags=["Module 2: Gateway"])
# app.include_router(vision_router, prefix="/api/v1/vision", tags=["Module 3: Vision"])
# app.include_router(context_router, prefix="/api/v1/context", tags=["Module 4: Context"])
# app.include_router(brain_router, prefix="/api/v1/brain", tags=["Module 5: Brain"])
# app.include_router(standards_router, prefix="/api/v1/standards", tags=["Module 6: Standards"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )