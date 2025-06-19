# src/data_processing/loader.py
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from .mongodb_loader import MongoDBLoader

logger = logging.getLogger(__name__)

def load_data(source_type: str = "mongodb", **kwargs):
    if source_type == "mongodb":
        try:
            # Get collection name with fallback
            collection_name = kwargs.get("collection_name")
            if not collection_name:
                from src.common.constant import DEFAULT_COLLECTION
                collection_name = DEFAULT_COLLECTION
                
            if not isinstance(collection_name, str):
                raise ValueError("Collection name must be a string")
                
            loader = MongoDBLoader(collection_name)
            try:
                if "query" in kwargs:
                    return loader.load_servers_by_query(kwargs["query"])
                return loader.load_all_servers()
            finally:
                loader.close()
                
        except Exception as e:
            logger.error(f"MongoDB failed: {e}")
            # Fallback to JSON
            from src.common.constant import JSON_DATA_PATH
            if JSON_DATA_PATH:
                return _load_json(JSON_DATA_PATH)
            raise
            
def _load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file (kept for backward compatibility)"""
    path = Path(file_path)
    try:
        with open(path, "r") as file:
            data = json.load(file)
            logger.info(f"Loaded data for {len(data)} servers from JSON")
            return data
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
        raise