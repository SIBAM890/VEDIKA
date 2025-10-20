import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standard logger.
    """
    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout) # Log to console
        ]
    )
    
    # Get and return the logger instance
    logger = logging.getLogger(name)
    return logger