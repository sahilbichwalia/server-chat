"""Logging configuration setup."""

import os
import logging

def setup_logging():
    """Configure application logging."""
    # Create 'logs' directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        handlers=[
            logging.FileHandler("logs/server_monitoring.log", mode='a'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)