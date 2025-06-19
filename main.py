import logging
from src.data_processing import initialize_data
import os
from src.config.logging_config import setup_logging
from src.ui.ui import create_gradio_interface
logger=setup_logging()

# Main execution

def initialize_app():
    """Handle all initialization"""
    try:
        # Try MongoDB first
        from src.data_processing.loader import load_data
        return load_data(source_type="mongodb")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}")
        
        # Fallback to JSON
        from src.common.constant import JSON_DATA_PATH
        if JSON_DATA_PATH:
            logger.warning("Falling back to JSON data")
            from src.data_processing.loader import _load_json
            return _load_json(JSON_DATA_PATH)
        raise
if __name__ == "__main__":
    logger.info("Starting Server Monitoring Assistant Application...")

    server_data = initialize_app()
    # Create and launch Gradio interface
    gradio_app = create_gradio_interface(server_data)
    try:
        gradio_app.launch(share=True, debug=False)
        logger.info("Application launched successfully. Access via the URL printed above.")
    except Exception as e:
        logger.critical(f"Failed to launch Gradio application: {e}", exc_info=True)