"""Logging utility for the fraud detection system."""
import logging
import sys
from pythonjsonlogger import jsonlogger


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a JSON logger for the application."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

