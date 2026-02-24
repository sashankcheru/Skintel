from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger
import os

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

db = MongoDB()

async def initialize_mongodb():
    """Initializes the MongoDB connection pool."""
    mongo_url = os.getenv("MONGODB_URL")
    db_name = os.getenv("MONGODB_DB_NAME")

    try:
        db.client = AsyncIOMotorClient(
            mongo_url,
            maxPoolSize=10,
            minPoolSize=2,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        db.db = db.client[db_name]
        await db.client.admin.command('ping')
        logger.info(f"Successfully connected to MongoDB: {db_name}")
    except Exception as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        raise

async def get_mongodb_db():
    if db.db is None:
        await initialize_mongodb()
    return db.db