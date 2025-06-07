import logging
import os
from src.config.logging_config import setup_logging
from src.ui.ui import create_gradio_interface
logger=setup_logging()

# Define paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'new_dump1.json')

# Main execution
if __name__ == "__main__":
    logger.info("Starting Server Monitoring Assistant Application...")

    
    # Create and launch Gradio interface
    gradio_app = create_gradio_interface()
    try:
        gradio_app.launch(share=False, debug=False)
        logger.info("Application launched successfully. Access via the URL printed above.")
    except Exception as e:
        logger.critical(f"Failed to launch Gradio application: {e}", exc_info=True)