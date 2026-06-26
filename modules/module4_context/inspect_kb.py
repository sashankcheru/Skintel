import asyncio
import json
from motor.motor_asyncio import AsyncIOMotorClient
import os

async def main():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL"), serverSelectionTimeoutMS=5000)
    coll = client[os.getenv("MONGODB_DB_NAME")]["medical_knowledge_base"]
    count = await coll.count_documents({})
    print("Total documents:", count)
    sample = await coll.find_one({})
    print(json.dumps(sample, indent=2, default=str))
    client.close()

asyncio.run(main())