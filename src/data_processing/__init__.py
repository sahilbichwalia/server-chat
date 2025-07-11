# src/data_processing/__init__.py
from .initialize_mongodb import load_json_to_mongodb
from .loader import load_data
from src.common.constant import JSON_DATA_PATH

def initialize_data():
    """Initialize data from JSON to MongoDB if needed"""
    try:
        # Try loading from MongoDB first
        data = load_data(source_type="mongodb")
        if not data:
            # If empty, load from JSON
            load_json_to_mongodb(JSON_DATA_PATH)
            data = load_data(source_type="mongodb")
        return data
    except Exception as e:
        # Fallback to JSON if MongoDB fails
        return load_data(source_type="json", file_path=JSON_DATA_PATH)