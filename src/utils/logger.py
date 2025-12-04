import logging
from typing import Optional


def get_logger(name: str = "nlp_project", level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: logger name
        level: optional level string e.g. 'INFO', 'DEBUG'. If None uses INFO.
    """
    logger = logging.getLogger(name)
    if level is None:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
