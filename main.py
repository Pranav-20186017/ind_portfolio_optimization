import os
import sys
import logging
import uvicorn
import logfire
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

def setup_logging():
    """Configure logging to capture all Uvicorn logs and send to Logfire only"""
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure Logfire
    logfire.configure(token=LOGFIRE_TOKEN, environment="production")
    
    # Create Logfire handler
    lf_handler = logfire.LogfireLoggingHandler()
    lf_handler.setFormatter(formatter)
    
    # Configure loggers
    loggers = [
        logging.getLogger(),  # Root logger
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.access"),
        logging.getLogger("uvicorn.error"),
        logging.getLogger("fastapi")
    ]
    
    for logger in loggers:
        logger.setLevel(logging.INFO)
        # Remove any existing handlers
        for handler in logger.handlers:
            logger.removeHandler(handler)
        # Add only the Logfire handler
        logger.addHandler(lf_handler)
        logger.propagate = False
    
    return lf_handler

if __name__ == "__main__":
    # Setup logging before starting Uvicorn
    lf_handler = setup_logging()
    
    # Get port from environment or use default of 80
    port = int(os.getenv("PORT", 80))
    
    try:
        # Run Uvicorn with proper logging configuration
        uvicorn.run(
            "srv:app",
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    finally:
        # Ensure all logs are flushed on shutdown
        lf_handler.flush() 