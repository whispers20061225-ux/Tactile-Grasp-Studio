"""
Logging configuration helpers.
"""

import logging


def configure_logging(level=logging.INFO):
    """Configure root logging and return the tactile_system logger."""
    logging.basicConfig(level=level)
    return logging.getLogger("tactile_system")


def get_logger(name: str = "tactile_system", level=logging.INFO):
    """
    Get a configured logger by name.
    
    Args:
        name: Logger name.
        level: Log level.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        configure_logging(level)
    return logger
