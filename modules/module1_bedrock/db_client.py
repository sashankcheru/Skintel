"""
Module 1 — Brick 1.1  |  db_client.py
Async MongoDB connection manager using Motor.
Single connection pool shared across all FastAPI requests.
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class _MongoDB:
    """Internal singleton — holds the Motor client and DB handle."""
    client: AsyncIOMotorClient   = None
    db:     AsyncIOMotorDatabase = None


_mongo = _MongoDB()


async def initialize_mongodb() -> None:
    """
    Opens the MongoDB connection pool.
    Called once at FastAPI startup inside the lifespan context manager.

    serverSelectionTimeoutMS=5000  → fails fast if MongoDB is not reachable
    maxPoolSize=10                 → max concurrent connections
    minPoolSize=2                  → always keep 2 warm connections ready
    """
    mongo_url = os.getenv("MONGODB_URL")
    db_name   = os.getenv("MONGODB_DB_NAME")

    if not mongo_url or not db_name:
        raise ValueError(
            "MONGODB_URL and MONGODB_DB_NAME must be set in .env"
        )

    _mongo.client = AsyncIOMotorClient(
        mongo_url,
        serverSelectionTimeoutMS=5000,
        maxPoolSize=10,
        minPoolSize=2,
    )
    _mongo.db = _mongo.client[db_name]

    # Confirm connection is live before returning
    await _mongo.client.admin.command("ping")
    logger.success(f"✅ MongoDB connected → {db_name}")


async def close_mongodb() -> None:
    """
    Closes the connection pool gracefully.
    Called at FastAPI shutdown inside the lifespan context manager.
    """
    if _mongo.client:
        _mongo.client.close()
        _mongo.client = None
        _mongo.db     = None
        logger.info("MongoDB connection pool closed")


async def get_db() -> AsyncIOMotorDatabase:
    """
    Returns the active database handle.
    Lazily initialises if called before the startup event (e.g. in scripts).
    """
    if _mongo.db is None:
        await initialize_mongodb()
    return _mongo.db
