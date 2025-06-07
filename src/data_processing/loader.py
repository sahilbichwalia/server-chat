import json
from src.config.logging_config import setup_logging
from src.common.constant import Data_PATH
logger = setup_logging()
try:
    with open(Data_PATH, "r") as file:
        server_data_raw = json.load(file)
        logger.info(f"Loaded data for {len(server_data_raw)} servers")
except FileNotFoundError:
    logger.error("Server data file 'new_dump1.json' not found. Please ensure it's in the same directory.")
    server_data_raw = []
except json.JSONDecodeError:
    logger.error("Invalid JSON format in server data file 'new_dump1.json'.")
    raise Exception("Invalid JSON format in server data file.")