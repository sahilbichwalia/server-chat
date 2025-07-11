# src/data_processing/initialize_mongodb.py
import json
from pathlib import Path
from pymongo import MongoClient
from src.common.constant import MONGO_URI, DB_NAME, DEFAULT_COLLECTION
import logging
from src.data_processing.mongodb_loader import MongoDBLoader

logger = logging.getLogger(__name__)

def load_json_to_mongodb(json_path: str):
    """Load JSON to MongoDB with SSL handling"""
    loader = MongoDBLoader()
    try:
        with open(json_path) as f:
            data = json.load(f)
            
            # Skip if collection has data
            if loader.collection.count_documents({}) > 0:
                logger.info("Data already exists in MongoDB")
                return False
                
            # Insert data
            if isinstance(data, list):
                loader.collection.insert_many(data)
            else:
                loader.collection.insert_one(data)
                
            logger.info(f"Inserted {len(data) if isinstance(data, list) else 1} documents")
            return True
            
    except Exception as e:
        logger.error(f"Failed to load to MongoDB: {e}")
        raise
    finally:
        loader.close()
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python initialize_mongodb.py <path_to_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    try:
        load_json_to_mongodb(json_file)
        print(f"Successfully loaded data from {json_file} to MongoDB")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)